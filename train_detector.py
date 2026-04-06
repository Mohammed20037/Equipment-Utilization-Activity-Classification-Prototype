#!/usr/bin/env python3
"""
Construction Equipment Detection Training Pipeline

Downloads open-source construction equipment datasets and trains
a YOLOv8 model for domain-specific detection and segmentation.

Supported dataset sources:
  - Roboflow Universe (requires ROBOFLOW_API_KEY)
  - Open Images v6  (auto-downloaded, no API key)
  - COCO 2017 vehicles subset (auto-downloaded, no API key)

Usage:
    # Detection model with all available sources
    python train_detector.py

    # Specific model size
    python train_detector.py --model yolov8m --epochs 150

    # Segmentation mode
    python train_detector.py --model yolov8s-seg --mode segmentation

    # Single dataset source
    python train_detector.py --dataset roboflow
    python train_detector.py --dataset open_images
    python train_detector.py --dataset coco

Environment variables:
    ROBOFLOW_API_KEY        Roboflow API key (roboflow.com → account settings)
    ROBOFLOW_WORKSPACE      Workspace slug (default: construction-equipment-site)
    ROBOFLOW_PROJECT        Project slug   (default: construction-equipment-detection)
    ROBOFLOW_VERSION        Dataset version number (default: 1)
    TRAINING_EPOCHS         Max epochs (default: 100)
    TRAINING_BATCH_SIZE     Batch size; -1 = auto (default: -1)
    TRAINING_IMG_SIZE       Input resolution (default: 640)
    TRAINING_DEVICE         cpu | 0 | 0,1 | mps  (default: auto)
    TRAINING_PATIENCE       Early-stopping patience epochs (default: 20)
    MAX_IMAGES_PER_SOURCE   Cap per dataset source (default: 3000)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="a"),
    ],
)
log = logging.getLogger("train_detector")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "training"
RUNS_DIR = ROOT / "runs" / "train"

# ── Equipment class taxonomy ──────────────────────────────────────────────────
EQUIPMENT_CLASSES: List[str] = [
    "excavator",
    "dump_truck",
    "wheel_loader",
    "bulldozer",
    "crane",
    "concrete_mixer",
    "forklift",
    "compactor",
]
CLASS_INDEX: Dict[str, int] = {cls: i for i, cls in enumerate(EQUIPMENT_CLASSES)}

# Open Images label names → our class names
OPEN_IMAGES_CLASS_MAP: Dict[str, str] = {
    "Excavator": "excavator",
    "Bulldozer": "bulldozer",
    "Crane": "crane",
    "Forklift": "forklift",
    "Truck": "dump_truck",
}

# COCO category id → our class name (only construction-relevant)
COCO_CATEGORY_MAP: Dict[int, str] = {
    8: "dump_truck",    # truck
    7: "dump_truck",    # train (misuse avoidance; skip in practice)
}

# ── Open Images URLs ──────────────────────────────────────────────────────────
OI_BASE = "https://storage.googleapis.com/openimages"
OI_CLASS_DESC_URL = f"{OI_BASE}/v6/oidv6-class-descriptions.csv"
OI_VAL_BBOX_URL = f"{OI_BASE}/v5/validation-annotations-bbox.csv"
OI_VAL_IMAGE_LIST_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-boxable.csv"
)
OI_IMAGE_TEMPLATE = (
    "https://storage.googleapis.com/open-images-dataset/validation/{image_id}.jpg"
)

# COCO URLs
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_VAL_IMAGE_TEMPLATE = "http://images.cocodataset.org/val2017/{image_id:012d}.jpg"


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download *url* to *dest*, resuming is not supported but shows progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    log.info("Downloading %s → %s", label, dest)
    try:
        urllib.request.urlretrieve(url, dest, _progress_hook(label))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    print()  # newline after progress
    return dest


def _progress_hook(label: str):
    """Return urlretrieve reporthook that prints a simple progress bar."""
    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            bar = "#" * int(pct / 4)
            print(f"\r  {label}: [{bar:<25}] {pct:5.1f}%", end="", flush=True)
        else:
            mb = downloaded / 1_048_576
            print(f"\r  {label}: {mb:.1f} MB", end="", flush=True)
    return hook


def yolo_bbox(x_min: float, x_max: float, y_min: float, y_max: float) -> Tuple[float, float, float, float]:
    """Convert [0,1]-normalised corner coords → YOLO cx,cy,w,h."""
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def write_yolo_label(path: Path, annotations: List[Tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls_id, cx, cy, w, h in annotations]
    path.write_text("\n".join(lines))


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ── Dataset downloader ────────────────────────────────────────────────────────

class DatasetDownloader:
    """Downloads and converts construction equipment datasets to YOLOv8 format."""

    def __init__(self, base_dir: Path, max_images: int = 3000):
        self.base_dir = base_dir
        self.max_images = max_images

    # ── Roboflow ──────────────────────────────────────────────────────────────

    def download_roboflow(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        mode: str = "detect",
    ) -> Optional[Path]:
        """
        Download a Roboflow Universe dataset in YOLOv8 format.

        The dataset must already exist in your workspace or be a public dataset
        accessible under the provided API key.
        """
        try:
            from roboflow import Roboflow  # noqa: PLC0415
        except ImportError:
            log.error("roboflow package not installed. Run: pip install roboflow>=1.1.0")
            return None

        fmt = "yolov8" if mode == "detect" else "yolov8-seg"
        target = self.base_dir / "roboflow"
        target.mkdir(parents=True, exist_ok=True)

        log.info("Connecting to Roboflow: workspace=%s project=%s version=%s", workspace, project, version)
        try:
            rf = Roboflow(api_key=api_key)
            proj = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download(fmt, location=str(target), overwrite=True)
            log.info("Roboflow dataset downloaded to: %s", dataset.location)
            return Path(dataset.location)
        except Exception as exc:
            log.error("Roboflow download failed: %s", exc)
            return None

    # ── Open Images ───────────────────────────────────────────────────────────

    def download_open_images(self) -> Optional[Path]:
        """
        Download construction-relevant classes from Open Images v6 validation split.

        Classes: Excavator, Bulldozer, Crane, Forklift, Truck
        Uses the public annotation CSVs + per-image HTTP download.
        No API key required.
        """
        oi_dir = self.base_dir / "open_images"
        cache_dir = oi_dir / "_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Class descriptions ────────────────────────────────────────────
        class_csv = cache_dir / "oidv6-class-descriptions.csv"
        if not class_csv.exists():
            download_file(OI_CLASS_DESC_URL, class_csv, "OI class descriptions")

        class_id_to_name: Dict[str, str] = {}
        wanted_ids: Dict[str, str] = {}  # oi_label_id → our class name
        with class_csv.open() as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                label_id, label_name = row[0], row[1]
                if label_name in OPEN_IMAGES_CLASS_MAP:
                    our_name = OPEN_IMAGES_CLASS_MAP[label_name]
                    wanted_ids[label_id] = our_name
                    class_id_to_name[label_id] = label_name

        if not wanted_ids:
            log.error("No matching Open Images classes found. Skipping OI source.")
            return None

        log.info("Open Images target classes: %s", list(wanted_ids.values()))

        # ── 2. Bbox annotations ───────────────────────────────────────────────
        bbox_csv = cache_dir / "validation-annotations-bbox.csv"
        if not bbox_csv.exists():
            download_file(OI_VAL_BBOX_URL, bbox_csv, "OI bbox annotations")

        # image_id → [(our_class_name, xmin, xmax, ymin, ymax)]
        image_annotations: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
        log.info("Parsing Open Images bbox annotations …")
        with bbox_csv.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label_id = row.get("LabelName", "")
                if label_id not in wanted_ids:
                    continue
                img_id = row["ImageID"]
                our_name = wanted_ids[label_id]
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])
                image_annotations.setdefault(img_id, []).append((our_name, xmin, xmax, ymin, ymax))

        log.info("Found %d Open Images images with target annotations", len(image_annotations))

        # Cap to max_images
        all_ids = list(image_annotations.keys())
        random.shuffle(all_ids)
        selected_ids = set(all_ids[: self.max_images])

        # ── 3. Train/val split (80/20) ────────────────────────────────────────
        selected_list = list(selected_ids)
        split_idx = int(len(selected_list) * 0.8)
        train_ids = set(selected_list[:split_idx])
        val_ids = set(selected_list[split_idx:])

        # ── 4. Download images & write labels ─────────────────────────────────
        img_dir_train = oi_dir / "train" / "images"
        lbl_dir_train = oi_dir / "train" / "labels"
        img_dir_val = oi_dir / "valid" / "images"
        lbl_dir_val = oi_dir / "valid" / "labels"
        for d in [img_dir_train, lbl_dir_train, img_dir_val, lbl_dir_val]:
            d.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        failed = 0
        for img_id in selected_ids:
            url = OI_IMAGE_TEMPLATE.format(image_id=img_id)
            split = "train" if img_id in train_ids else "valid"
            img_path = oi_dir / split / "images" / f"{img_id}.jpg"
            lbl_path = oi_dir / split / "labels" / f"{img_id}.txt"

            if not img_path.exists():
                try:
                    urllib.request.urlretrieve(url, img_path)
                    downloaded += 1
                except Exception as exc:
                    log.debug("Failed to download OI image %s: %s", img_id, exc)
                    failed += 1
                    continue
            else:
                downloaded += 1

            # Write YOLO label
            annots: List[Tuple[int, float, float, float, float]] = []
            for our_name, xmin, xmax, ymin, ymax in image_annotations[img_id]:
                if our_name not in CLASS_INDEX:
                    continue
                cx, cy, w, h = yolo_bbox(xmin, xmax, ymin, ymax)
                annots.append((CLASS_INDEX[our_name], cx, cy, w, h))
            if annots:
                write_yolo_label(lbl_path, annots)

            if downloaded % 100 == 0:
                log.info("  OI progress: %d downloaded, %d failed", downloaded, failed)

        log.info("Open Images: %d images downloaded, %d failed", downloaded, failed)
        if downloaded == 0:
            return None
        return oi_dir

    # ── COCO ──────────────────────────────────────────────────────────────────

    def download_coco_subset(self) -> Optional[Path]:
        """
        Download COCO 2017 val annotations, filter for vehicle categories, and
        download only the matching images.
        """
        coco_dir = self.base_dir / "coco"
        cache_dir = coco_dir / "_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Download annotations zip ───────────────────────────────────────
        ann_zip = cache_dir / "annotations_trainval2017.zip"
        if not ann_zip.exists():
            log.info("Downloading COCO 2017 annotations (~241 MB) …")
            download_file(COCO_ANNOTATIONS_URL, ann_zip, "COCO annotations")

        ann_json = cache_dir / "annotations" / "instances_val2017.json"
        if not ann_json.exists():
            log.info("Extracting COCO annotations …")
            with zipfile.ZipFile(ann_zip, "r") as zf:
                zf.extractall(cache_dir)

        # ── 2. Parse annotations ──────────────────────────────────────────────
        log.info("Parsing COCO instances …")
        with ann_json.open() as fh:
            coco_data = json.load(fh)

        # Filter for wanted category IDs
        wanted_cat_ids = set(COCO_CATEGORY_MAP.keys())
        image_meta: Dict[int, dict] = {img["id"]: img for img in coco_data["images"]}

        # image_id → [(our_class_name, cx, cy, w_norm, h_norm)]
        image_annotations: Dict[int, List[Tuple[str, float, float, float, float]]] = {}
        for ann in coco_data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in wanted_cat_ids:
                continue
            img_id = ann["image_id"]
            meta = image_meta.get(img_id)
            if meta is None:
                continue
            img_w = meta["width"]
            img_h = meta["height"]
            x, y, bw, bh = ann["bbox"]  # COCO: x,y,w,h in pixels
            cx = (x + bw / 2.0) / img_w
            cy = (y + bh / 2.0) / img_h
            w_norm = bw / img_w
            h_norm = bh / img_h
            our_name = COCO_CATEGORY_MAP[cat_id]
            image_annotations.setdefault(img_id, []).append((our_name, cx, cy, w_norm, h_norm))

        log.info("COCO: found %d images with vehicle annotations", len(image_annotations))

        # Cap to max_images
        all_ids = list(image_annotations.keys())
        random.shuffle(all_ids)
        selected_ids = all_ids[: self.max_images]

        split_idx = int(len(selected_ids) * 0.8)
        train_ids = set(selected_ids[:split_idx])

        img_dir_train = coco_dir / "train" / "images"
        lbl_dir_train = coco_dir / "train" / "labels"
        img_dir_val = coco_dir / "valid" / "images"
        lbl_dir_val = coco_dir / "valid" / "labels"
        for d in [img_dir_train, lbl_dir_train, img_dir_val, lbl_dir_val]:
            d.mkdir(parents=True, exist_ok=True)

        # ── 3. Download images ────────────────────────────────────────────────
        downloaded = 0
        failed = 0
        for img_id in selected_ids:
            meta = image_meta[img_id]
            file_name = meta["file_name"]  # e.g. "000000391895.jpg"
            coco_id_int = int(Path(file_name).stem)
            url = COCO_VAL_IMAGE_TEMPLATE.format(image_id=coco_id_int)
            split = "train" if img_id in train_ids else "valid"
            img_path = coco_dir / split / "images" / file_name
            lbl_path = coco_dir / split / "labels" / Path(file_name).with_suffix(".txt").name

            if not img_path.exists():
                try:
                    urllib.request.urlretrieve(url, img_path)
                    downloaded += 1
                except Exception as exc:
                    log.debug("COCO image download failed %s: %s", url, exc)
                    failed += 1
                    continue
            else:
                downloaded += 1

            annots: List[Tuple[int, float, float, float, float]] = []
            for our_name, cx, cy, wn, hn in image_annotations[img_id]:
                if our_name not in CLASS_INDEX:
                    continue
                annots.append((CLASS_INDEX[our_name], cx, cy, wn, hn))
            if annots:
                write_yolo_label(lbl_path, annots)

            if downloaded % 100 == 0:
                log.info("  COCO progress: %d downloaded, %d failed", downloaded, failed)

        log.info("COCO: %d images downloaded, %d failed", downloaded, failed)
        if downloaded == 0:
            return None
        return coco_dir

    # ── Merge ──────────────────────────────────────────────────────────────────

    def merge_datasets(self, source_dirs: List[Path], output_dir: Path) -> Path:
        """
        Merge multiple YOLOv8-format dataset directories into one unified dataset.
        Expects each source to have train/images, train/labels, valid/images, valid/labels.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid"):
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        total = 0
        for src in source_dirs:
            for split in ("train", "valid"):
                img_src = src / split / "images"
                lbl_src = src / split / "labels"
                if not img_src.exists():
                    continue
                for img_path in img_src.glob("*.*"):
                    stem = img_path.stem
                    lbl_path = lbl_src / f"{stem}.txt"
                    if not lbl_path.exists():
                        continue
                    # Prefix filename with source directory name to avoid collisions
                    prefix = src.name
                    dst_img = output_dir / split / "images" / f"{prefix}_{img_path.name}"
                    dst_lbl = output_dir / split / "labels" / f"{prefix}_{stem}.txt"
                    safe_copy(img_path, dst_img)
                    safe_copy(lbl_path, dst_lbl)
                    total += 1

        log.info("Merged %d image/label pairs into %s", total, output_dir)
        return output_dir


# ── Dataset validator ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    train_images: int = 0
    val_images: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class DatasetValidator:
    """Checks a YOLOv8-format dataset for completeness and class balance."""

    def validate(self, dataset_dir: Path, class_names: List[str]) -> ValidationResult:
        result = ValidationResult(valid=True)

        for split, attr in [("train", "train_images"), ("valid", "val_images")]:
            img_dir = dataset_dir / split / "images"
            lbl_dir = dataset_dir / split / "labels"

            if not img_dir.exists():
                result.issues.append(f"Missing directory: {img_dir}")
                result.valid = False
                continue

            images = list(img_dir.glob("*.*"))
            setattr(result, attr, len(images))

            if len(images) == 0:
                result.issues.append(f"No images in {split} split")
                result.valid = False
                continue

            missing_labels = 0
            for img in images:
                lbl = lbl_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    missing_labels += 1
                else:
                    for line in lbl.read_text().splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(class_names):
                            name = class_names[cls_id]
                            result.class_counts[name] = result.class_counts.get(name, 0) + 1

            if missing_labels > 0:
                pct = missing_labels / max(1, len(images)) * 100
                result.issues.append(
                    f"{split}: {missing_labels}/{len(images)} images lack labels ({pct:.1f}%)"
                )

        if result.train_images < 10:
            result.issues.append(f"Very few training images: {result.train_images}")
            result.valid = False

        return result

    def report(self, result: ValidationResult) -> None:
        status = "PASS" if result.valid else "FAIL"
        log.info("── Dataset validation: %s ──", status)
        log.info("  Train images : %d", result.train_images)
        log.info("  Val   images : %d", result.val_images)
        if result.class_counts:
            log.info("  Class distribution:")
            for cls, count in sorted(result.class_counts.items(), key=lambda x: -x[1]):
                log.info("    %-20s %d", cls, count)
        for issue in result.issues:
            log.warning("  ISSUE: %s", issue)


# ── Training pipeline ─────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model: str = "yolov8s"
    mode: str = "detect"            # detect | segment
    dataset_sources: List[str] = field(default_factory=lambda: ["open_images", "coco"])
    epochs: int = 100
    patience: int = 20
    batch_size: int = -1            # -1 = auto
    img_size: int = 640
    device: str = ""                # empty = auto-select
    max_images_per_source: int = 3000
    roboflow_api_key: str = ""
    roboflow_workspace: str = "construction-equipment-site"
    roboflow_project: str = "construction-equipment-detection"
    roboflow_version: int = 1
    output_model: Path = MODELS_DIR / "equipment_detector.pt"
    data_dir: Path = DATA_DIR
    runs_dir: Path = RUNS_DIR
    run_name: str = "equipment_detector"


class TrainingPipeline:
    """Orchestrates dataset download → validation → training → export."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.downloader = DatasetDownloader(cfg.data_dir / "raw", cfg.max_images_per_source)
        self.validator = DatasetValidator()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Dataset preparation ─────────────────────────────────────────

    def prepare_dataset(self) -> Tuple[Path, Path]:
        """Download, merge, and validate dataset. Returns (dataset_dir, data_yaml)."""
        source_dirs: List[Path] = []
        cfg = self.cfg

        for source in cfg.dataset_sources:
            log.info("═══ Downloading source: %s ═══", source)

            if source == "roboflow":
                if not cfg.roboflow_api_key:
                    log.warning("ROBOFLOW_API_KEY not set; skipping Roboflow source")
                    continue
                # Determine format
                fmt_mode = "segment" if cfg.mode == "segment" else "detect"
                result = self.downloader.download_roboflow(
                    api_key=cfg.roboflow_api_key,
                    workspace=cfg.roboflow_workspace,
                    project=cfg.roboflow_project,
                    version=cfg.roboflow_version,
                    mode=fmt_mode,
                )
                if result:
                    source_dirs.append(result)

            elif source == "open_images":
                result = self.downloader.download_open_images()
                if result:
                    source_dirs.append(result)

            elif source == "coco":
                result = self.downloader.download_coco_subset()
                if result:
                    source_dirs.append(result)

            else:
                log.warning("Unknown dataset source '%s'; skipping", source)

        if not source_dirs:
            raise RuntimeError(
                "No datasets were successfully downloaded. "
                "Set ROBOFLOW_API_KEY or ensure network access for Open Images / COCO."
            )

        # ── Merge ────────────────────────────────────────────────────────────
        if len(source_dirs) == 1:
            merged_dir = source_dirs[0]
        else:
            merged_dir = cfg.data_dir / "merged"
            self.downloader.merge_datasets(source_dirs, merged_dir)

        # ── Write data.yaml ───────────────────────────────────────────────────
        data_yaml = merged_dir / "data.yaml"
        dataset_config = {
            "path": str(merged_dir.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "nc": len(EQUIPMENT_CLASSES),
            "names": EQUIPMENT_CLASSES,
        }
        # Add test split if present
        if (merged_dir / "test" / "images").exists():
            dataset_config["test"] = "test/images"

        with data_yaml.open("w") as fh:
            yaml.dump(dataset_config, fh, default_flow_style=False)
        log.info("Wrote data.yaml: %s", data_yaml)

        # ── Validate ──────────────────────────────────────────────────────────
        vresult = self.validator.validate(merged_dir, EQUIPMENT_CLASSES)
        self.validator.report(vresult)
        if not vresult.valid:
            log.warning("Dataset validation issues detected; training may yield poor results.")

        return merged_dir, data_yaml

    # ── Phase 2: Training ─────────────────────────────────────────────────────

    def train(self, data_yaml: Path) -> Path:
        """Run YOLOv8 training. Returns path to best.pt checkpoint."""
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("ultralytics package is required for training") from exc

        cfg = self.cfg

        # Resolve model name: append -seg suffix if segmentation mode
        model_name = cfg.model
        if cfg.mode == "segment" and not model_name.endswith("-seg"):
            model_name = model_name + "-seg"
        log.info("Loading base model: %s", model_name)

        model = YOLO(model_name)

        # Build training kwargs
        train_kwargs = dict(
            data=str(data_yaml),
            epochs=cfg.epochs,
            patience=cfg.patience,
            imgsz=cfg.img_size,
            batch=cfg.batch_size,
            project=str(cfg.runs_dir),
            name=cfg.run_name,
            exist_ok=True,
            val=True,
            plots=True,
            save=True,
            # Augmentation tuned for construction site imagery
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1 if cfg.mode == "segment" else 0.0,
            flipud=0.0,       # vertical flip rarely valid for construction
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,     # allow moderate rotation
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0005,
            # Optimiser
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            cos_lr=True,
            weight_decay=0.0005,
            # Logging
            verbose=True,
        )

        if cfg.device:
            train_kwargs["device"] = cfg.device

        log.info("Starting training: epochs=%d patience=%d imgsz=%d", cfg.epochs, cfg.patience, cfg.img_size)
        results = model.train(**train_kwargs)

        # Locate best.pt
        run_dir = cfg.runs_dir / cfg.run_name
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            # Fallback: last.pt
            best_pt = run_dir / "weights" / "last.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Could not locate trained weights in {run_dir}")

        log.info("Training complete. Best checkpoint: %s", best_pt)
        return best_pt

    # ── Phase 3: Export ───────────────────────────────────────────────────────

    def export_model(self, best_pt: Path) -> Path:
        """Copy best checkpoint to models/equipment_detector.pt."""
        dest = self.cfg.output_model
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dest)
        size_mb = dest.stat().st_size / 1_048_576
        log.info("Model exported: %s  (%.1f MB)", dest, size_mb)
        return dest

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full pipeline: download → validate → train → export."""
        log.info("╔══════════════════════════════════════════════════════╗")
        log.info("║  Construction Equipment Detector – Training Pipeline  ║")
        log.info("╚══════════════════════════════════════════════════════╝")
        log.info("Model : %s  |  Mode : %s", self.cfg.model, self.cfg.mode)
        log.info("Sources: %s", self.cfg.dataset_sources)

        _, data_yaml = self.prepare_dataset()
        best_pt = self.train(data_yaml)
        exported = self.export_model(best_pt)

        log.info("══════════════════════════════════════════════")
        log.info("Training pipeline finished successfully.")
        log.info("Model saved to: %s", exported)
        log.info("══════════════════════════════════════════════")
        return exported


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Merge CLI args + environment variables into a TrainingConfig."""

    def env_int(key: str, default: int) -> int:
        return int(os.environ.get(key, default))

    dataset_sources = [s.strip() for s in args.dataset.split(",") if s.strip()]
    if dataset_sources == ["all"]:
        dataset_sources = ["roboflow", "open_images", "coco"]

    return TrainingConfig(
        model=args.model,
        mode=args.mode,
        dataset_sources=dataset_sources,
        epochs=env_int("TRAINING_EPOCHS", args.epochs),
        patience=env_int("TRAINING_PATIENCE", args.patience),
        batch_size=env_int("TRAINING_BATCH_SIZE", args.batch),
        img_size=env_int("TRAINING_IMG_SIZE", args.img_size),
        device=os.environ.get("TRAINING_DEVICE", args.device),
        max_images_per_source=env_int("MAX_IMAGES_PER_SOURCE", args.max_images),
        roboflow_api_key=os.environ.get("ROBOFLOW_API_KEY", ""),
        roboflow_workspace=os.environ.get("ROBOFLOW_WORKSPACE", "construction-equipment-site"),
        roboflow_project=os.environ.get("ROBOFLOW_PROJECT", "construction-equipment-detection"),
        roboflow_version=int(os.environ.get("ROBOFLOW_VERSION", "1")),
        output_model=MODELS_DIR / "equipment_detector.pt",
        data_dir=DATA_DIR,
        runs_dir=RUNS_DIR,
        run_name="equipment_detector",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a construction equipment detector with YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="yolov8s",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l",
                 "yolov8n-seg", "yolov8s-seg", "yolov8m-seg"],
        help="YOLOv8 model variant (default: yolov8s)",
    )
    parser.add_argument(
        "--mode",
        default="detect",
        choices=["detect", "segment"],
        help="Training mode: detect=bboxes only, segment=instance masks (default: detect)",
    )
    parser.add_argument(
        "--dataset",
        default="open_images,coco",
        help=(
            "Comma-separated dataset sources: roboflow, open_images, coco, all "
            "(default: open_images,coco)"
        ),
    )
    parser.add_argument("--epochs",    type=int, default=100, help="Max training epochs (default: 100)")
    parser.add_argument("--patience",  type=int, default=20,  help="Early-stopping patience (default: 20)")
    parser.add_argument("--batch",     type=int, default=-1,  help="Batch size; -1=auto (default: -1)")
    parser.add_argument("--img-size",  type=int, default=640, help="Input resolution (default: 640)")
    parser.add_argument("--device",    type=str, default="",  help="Device: cpu|0|0,1 (default: auto)")
    parser.add_argument("--max-images",type=int, default=3000,help="Max images per source (default: 3000)")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and validate datasets; skip training",
    )

    args = parser.parse_args()
    cfg = _build_config_from_args(args)
    pipeline = TrainingPipeline(cfg)

    if args.download_only:
        _, data_yaml = pipeline.prepare_dataset()
        vresult = pipeline.validator.validate(data_yaml.parent, EQUIPMENT_CLASSES)
        pipeline.validator.report(vresult)
        log.info("Dataset ready at: %s", data_yaml)
        return

    pipeline.run()


if __name__ == "__main__":
    main()
