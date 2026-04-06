from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class StopInterval:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class EquipmentValidation:
    equipment_id: str
    equipment_class: str
    tracked_sec: float = 0.0
    active_sec: float = 0.0
    idle_sec: float = 0.0
    utilization_percent: float = 0.0
    stop_intervals: List[StopInterval] = field(default_factory=list)

    @property
    def stop_count(self) -> int:
        return len(self.stop_intervals)

    @property
    def total_downtime_sec(self) -> float:
        return round(sum(s.duration_sec for s in self.stop_intervals), 3)


def summarize_timeline(csv_path: str | Path) -> Dict[str, EquipmentValidation]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Timeline CSV not found: {path}")

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["timestamp_sec"] = float(row["timestamp_sec"])
            rows.append(row)

    by_eq: Dict[str, List[dict]] = {}
    for row in rows:
        by_eq.setdefault(row["equipment_id"], []).append(row)

    report: Dict[str, EquipmentValidation] = {}
    for eq_id, eq_rows in by_eq.items():
        eq_rows = sorted(eq_rows, key=lambda r: r["timestamp_sec"])
        eq_class = eq_rows[-1].get("equipment_class", "unknown")
        result = EquipmentValidation(equipment_id=eq_id, equipment_class=eq_class)

        stop_open_at = None
        prev_ts = eq_rows[0]["timestamp_sec"]
        prev_state = eq_rows[0].get("state", "INACTIVE")

        for i, row in enumerate(eq_rows):
            ts = row["timestamp_sec"]
            state = row.get("state", "INACTIVE")
            dt = 0.0 if i == 0 else max(0.0, ts - prev_ts)
            result.tracked_sec += dt

            if state == "ACTIVE":
                result.active_sec += dt
            else:
                result.idle_sec += dt

            if prev_state == "ACTIVE" and state == "INACTIVE":
                stop_open_at = ts
            elif prev_state == "INACTIVE" and state == "ACTIVE" and stop_open_at is not None:
                result.stop_intervals.append(StopInterval(start_sec=stop_open_at, end_sec=ts))
                stop_open_at = None

            prev_ts, prev_state = ts, state

        if stop_open_at is not None:
            result.stop_intervals.append(StopInterval(start_sec=stop_open_at, end_sec=eq_rows[-1]["timestamp_sec"]))

        result.tracked_sec = round(result.tracked_sec, 3)
        result.active_sec = round(result.active_sec, 3)
        result.idle_sec = round(result.idle_sec, 3)
        result.utilization_percent = round((result.active_sec / result.tracked_sec * 100.0) if result.tracked_sec > 0 else 0.0, 2)
        report[eq_id] = result

    return report
