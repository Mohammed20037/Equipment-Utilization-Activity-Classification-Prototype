#!/usr/bin/env python3
"""
Construction Equipment Detector – Evaluation Pipeline

Evaluates a trained YOLOv8 model against a validation or test dataset and
produces comprehensive metrics including mAP, precision, recall, per-class
accuracy, and an ASCII confusion matrix.

Usage:
    # Evaluate default model against training dataset
    python evaluate_model.py

    # Evaluate specific model checkpoint
    python evaluate_model.py --model models/equipment_detector.pt

    # Evaluate against a custom dataset
    python evaluate_model.py --data data/training/merged/data.yaml

    # Run on a directory of images (no ground-truth metrics)
    python evaluate_model.py --source data/raw_videos/ --predict-only

    # Change IoU threshold for mAP
    python evaluate_model.py --iou 0.6

    # Save annotated prediction images
    python evaluate_model.py --save-images
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("evaluate_model")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "models" / "equipment_detector.pt"
DEFAULT_DATA = ROOT / "data" / "training" / "merged" / "data.yaml"
RESULTS_DIR = ROOT / "models" / "eval_results"

EQUIPMENT_CLASSES = [
    "excavator",
    "dump_truck",
    "wheel_loader",
    "bulldozer",
    "crane",
    "concrete_mixer",
    "forklift",
    "compactor",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model(model_arg: str) -> str:
    """Resolve model path: prefer custom trained model, then user arg."""
    if Path(model_arg).exists():
        return model_arg
    if DEFAULT_MODEL.exists():
        log.info("Using default trained model: %s", DEFAULT_MODEL)
        return str(DEFAULT_MODEL)
    log.warning(
        "Model '%s' not found and no custom model at %s. Falling back to yolov8s.pt",
        model_arg,
        DEFAULT_MODEL,
    )
    return "yolov8s.pt"


def _resolve_data(data_arg: str) -> Optional[str]:
    """Return data.yaml path if it exists, else None."""
    p = Path(data_arg)
    if p.exists():
        return str(p)
    if DEFAULT_DATA.exists():
        log.info("Using default data.yaml: %s", DEFAULT_DATA)
        return str(DEFAULT_DATA)
    return None


# ── Confusion matrix printer ──────────────────────────────────────────────────

def print_confusion_matrix(matrix: np.ndarray, class_names: List[str], title: str = "Confusion Matrix") -> None:
    """Pretty-print a normalised confusion matrix to stdout."""
    n = len(class_names)
    max_name = max(len(c) for c in class_names)
    col_w = max(6, max_name)

    print(f"\n{'─' * 4} {title} {'─' * 4}")
    header_pad = " " * (max_name + 2)
    print(header_pad + "  ".join(f"{c[:col_w]:>{col_w}}" for c in class_names) + "  <- predicted")
    print(header_pad + "  ".join("─" * col_w for _ in class_names))

    for i, row_name in enumerate(class_names):
        row_str = "  ".join(f"{matrix[i, j]:>{col_w}.2f}" for j in range(n))
        print(f"  {row_name:<{max_name}}  {row_str}")
    print()


# ── Per-class accuracy ────────────────────────────────────────────────────────

def compute_per_class_accuracy(conf_matrix: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """Compute per-class recall (true-positive rate) from a confusion matrix."""
    accuracy: Dict[str, float] = {}
    for i, cls in enumerate(class_names):
        row_sum = conf_matrix[i].sum()
        if row_sum > 0:
            accuracy[cls] = float(conf_matrix[i, i] / row_sum)
        else:
            accuracy[cls] = float("nan")
    return accuracy


# ── Validation run ────────────────────────────────────────────────────────────

def run_validation(model_path: str, data_yaml: str, iou: float, conf: float, split: str) -> dict:
    """Run YOLOv8 .val() and return structured metrics dict."""
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("ultralytics package is required. pip install ultralytics") from exc

    log.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    log.info("Running validation on split='%s'  iou=%.2f  conf=%.2f", split, iou, conf)
    results = model.val(
        data=data_yaml,
        iou=iou,
        conf=conf,
        split=split,
        plots=True,
        save_json=True,
        verbose=False,
    )

    # ── Extract metrics ───────────────────────────────────────────────────────
    metrics = results.results_dict
    names = results.names  # {int: str}

    # mAP values
    map50 = float(metrics.get("metrics/mAP50(B)", metrics.get("metrics/mAP50", 0.0)))
    map5095 = float(metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95", 0.0)))
    precision = float(metrics.get("metrics/precision(B)", metrics.get("metrics/precision", 0.0)))
    recall = float(metrics.get("metrics/recall(B)", metrics.get("metrics/recall", 0.0)))

    # Per-class AP (from results.box.ap_class_index + results.box.ap50)
    per_class: Dict[str, Dict[str, float]] = {}
    try:
        ap_indices = results.box.ap_class_index.tolist()
        ap50_vals = results.box.ap50.tolist()
        ap_vals = results.box.ap.tolist()
        p_vals = results.box.p.tolist()
        r_vals = results.box.r.tolist()
        for idx, cls_idx in enumerate(ap_indices):
            cls_name = names.get(int(cls_idx), f"class_{cls_idx}")
            per_class[cls_name] = {
                "precision": round(p_vals[idx], 4) if idx < len(p_vals) else float("nan"),
                "recall": round(r_vals[idx], 4) if idx < len(r_vals) else float("nan"),
                "ap50": round(ap50_vals[idx], 4) if idx < len(ap50_vals) else float("nan"),
                "ap50_95": round(ap_vals[idx], 4) if idx < len(ap_vals) else float("nan"),
            }
    except AttributeError:
        log.debug("Per-class metrics not available from this results object")

    # Confusion matrix
    conf_matrix_data: Optional[np.ndarray] = None
    class_names_ordered: List[str] = []
    try:
        cm = results.confusion_matrix
        conf_matrix_data = cm.matrix
        class_names_ordered = [names.get(i, f"class_{i}") for i in range(len(names))]
    except AttributeError:
        log.debug("Confusion matrix not available")

    return {
        "model": model_path,
        "data_yaml": data_yaml,
        "split": split,
        "iou_threshold": iou,
        "conf_threshold": conf,
        "summary": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map5095, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
        "per_class": per_class,
        "confusion_matrix": conf_matrix_data.tolist() if conf_matrix_data is not None else None,
        "confusion_matrix_class_names": class_names_ordered,
    }


# ── Predict-only run ──────────────────────────────────────────────────────────

def run_predict(model_path: str, source: str, conf: float, save_images: bool) -> None:
    """Run inference on a directory/video without ground-truth evaluation."""
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("ultralytics package is required") from exc

    log.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    log.info("Running inference on: %s  conf=%.2f", source, conf)
    results = model.predict(
        source=source,
        conf=conf,
        save=save_images,
        verbose=True,
    )

    total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
    log.info("Prediction complete: %d frames, %d total detections", len(results), total_dets)


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> None:
    summary = metrics["summary"]
    per_class = metrics.get("per_class", {})
    conf_matrix = metrics.get("confusion_matrix")
    cm_names = metrics.get("confusion_matrix_class_names", [])

    print("\n" + "═" * 56)
    print("  CONSTRUCTION EQUIPMENT DETECTOR – EVALUATION REPORT")
    print("═" * 56)
    print(f"  Model   : {metrics['model']}")
    print(f"  Dataset : {metrics['data_yaml']}")
    print(f"  Split   : {metrics['split']}")
    print(f"  IoU thr : {metrics['iou_threshold']}")
    print(f"  Conf thr: {metrics['conf_threshold']}")
    print("─" * 56)
    print(f"  mAP@0.50      : {summary['mAP50']:.4f}  ({summary['mAP50']*100:.1f}%)")
    print(f"  mAP@0.50:0.95 : {summary['mAP50_95']:.4f}  ({summary['mAP50_95']*100:.1f}%)")
    print(f"  Precision     : {summary['precision']:.4f}  ({summary['precision']*100:.1f}%)")
    print(f"  Recall        : {summary['recall']:.4f}  ({summary['recall']*100:.1f}%)")
    print("─" * 56)

    if per_class:
        print("  Per-Class Metrics:")
        header = f"  {'Class':<20} {'AP50':>8} {'AP50-95':>9} {'Prec':>8} {'Recall':>8}"
        print(header)
        print("  " + "─" * 58)
        for cls, vals in sorted(per_class.items()):
            ap50 = vals.get("ap50", float("nan"))
            ap = vals.get("ap50_95", float("nan"))
            prec = vals.get("precision", float("nan"))
            rec = vals.get("recall", float("nan"))
            print(f"  {cls:<20} {ap50:>8.4f} {ap:>9.4f} {prec:>8.4f} {rec:>8.4f}")
        print()

    if conf_matrix is not None and cm_names:
        mat = np.array(conf_matrix, dtype=float)
        # Normalise rows to [0,1] for readability (skip background row if present)
        n_classes = len(cm_names)
        if mat.shape[0] == n_classes + 1:
            mat = mat[:n_classes, :n_classes]  # drop background
        row_sums = mat.sum(axis=1, keepdims=True)
        norm_mat = np.divide(mat, row_sums, where=row_sums > 0, out=np.zeros_like(mat))
        display_names = cm_names[:n_classes]
        print_confusion_matrix(norm_mat, display_names, "Confusion Matrix (row-normalised)")

        # Per-class accuracy from confusion matrix
        accuracy = compute_per_class_accuracy(norm_mat, display_names)
        print("  Per-Class Accuracy (from confusion matrix):")
        for cls, acc in sorted(accuracy.items()):
            bar = "█" * int(acc * 20) if not np.isnan(acc) else ""
            val_str = f"{acc*100:.1f}%" if not np.isnan(acc) else "  n/a"
            print(f"  {cls:<20} {val_str:>6}  {bar}")

    print("═" * 56 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained construction equipment detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to model weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help=f"Path to data.yaml (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Image/video/directory for predict-only mode (no ground-truth needed)",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run inference without metrics; requires --source",
    )
    parser.add_argument("--iou",   type=float, default=0.5,  help="IoU threshold for mAP (default: 0.5)")
    parser.add_argument("--conf",  type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated prediction images",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "results.json"),
        help=f"Path to save JSON results (default: {RESULTS_DIR}/results.json)",
    )

    args = parser.parse_args()

    model_path = _resolve_model(args.model)

    # ── Predict-only mode ─────────────────────────────────────────────────────
    if args.predict_only:
        if not args.source:
            parser.error("--predict-only requires --source")
        run_predict(model_path, args.source, conf=args.conf, save_images=args.save_images)
        return

    # ── Validation mode ───────────────────────────────────────────────────────
    data_yaml = _resolve_data(args.data)
    if data_yaml is None:
        log.error(
            "No data.yaml found at '%s'. "
            "Run train_detector.py first or specify --data explicitly.",
            args.data,
        )
        sys.exit(1)

    metrics = run_validation(
        model_path=model_path,
        data_yaml=data_yaml,
        iou=args.iou,
        conf=args.conf,
        split=args.split,
    )

    print_report(metrics)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove numpy arrays for JSON serialisation
    metrics_serialisable = {
        k: v for k, v in metrics.items()
        if k not in ("confusion_matrix",)
    }
    metrics_serialisable["confusion_matrix"] = metrics.get("confusion_matrix")
    out_path.write_text(json.dumps(metrics_serialisable, indent=2, default=str))
    log.info("Results saved to: %s", out_path)


if __name__ == "__main__":
    main()
