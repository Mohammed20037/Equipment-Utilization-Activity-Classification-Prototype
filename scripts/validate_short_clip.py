from pathlib import Path

import pandas as pd

from services.analytics_service.validation_report import summarize_timeline


def _print_counts_report(path: Path) -> None:
    if not path.exists():
        print("No validation_counts.csv found yet.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("validation_counts.csv is empty.")
        return

    print("Detection/tracking counts")
    print("-" * 80)
    print(
        "avg accepted={:.2f}, avg rejected={:.2f}, avg tracked={:.2f}".format(
            df["accepted_detections"].mean(),
            df["rejected_detections"].mean(),
            df["tracked_objects"].mean(),
        )
    )


def _print_transition_report(timeline_path: Path) -> None:
    if not timeline_path.exists():
        return
    df = pd.read_csv(timeline_path)
    if df.empty:
        return

    print("State transitions")
    print("-" * 80)
    for eq_id, group in df.sort_values("timestamp_sec").groupby("equipment_id"):
        prev_state = None
        transitions = []
        for _, row in group.iterrows():
            state = row["state"]
            if prev_state is not None and state != prev_state:
                transitions.append(f"{prev_state}->{state}@{row['timestamp_sec']:.2f}s")
            prev_state = state
        print(f"{eq_id}: {', '.join(transitions) if transitions else 'none'}")


def main() -> None:
    timeline_path = Path("data/processed/equipment_timeline.csv")
    counts_path = Path("data/processed/validation_counts.csv")

    report = summarize_timeline(timeline_path)

    if not report:
        print("No equipment rows found in timeline.")
        _print_counts_report(counts_path)
        return

    print("Short-clip validation summary")
    print("=" * 80)
    for eq_id, r in sorted(report.items()):
        print(
            f"{eq_id} ({r.equipment_class}): tracked={r.tracked_sec:.2f}s, "
            f"active={r.active_sec:.2f}s, idle={r.idle_sec:.2f}s, "
            f"stop_count={r.stop_count}, total_downtime={r.total_downtime_sec:.2f}s, "
            f"utilization={r.utilization_percent:.2f}%"
        )
        if r.stop_intervals:
            print("  stop_intervals:")
            for idx, interval in enumerate(r.stop_intervals, start=1):
                print(
                    f"    {idx}. start={interval.start_sec:.2f}s "
                    f"end={interval.end_sec:.2f}s duration={interval.duration_sec:.2f}s"
                )
        else:
            print("  stop_intervals: none")

    _print_counts_report(counts_path)
    _print_transition_report(timeline_path)


if __name__ == "__main__":
    main()
