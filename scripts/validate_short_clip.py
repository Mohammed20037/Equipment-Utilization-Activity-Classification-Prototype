from pathlib import Path

from services.analytics_service.validation_report import summarize_timeline


def main() -> None:
    timeline_path = Path("data/processed/equipment_timeline.csv")
    report = summarize_timeline(timeline_path)

    if not report:
        print("No equipment rows found in timeline.")
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


if __name__ == "__main__":
    main()
