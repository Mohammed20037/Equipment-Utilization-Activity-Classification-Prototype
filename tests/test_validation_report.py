from pathlib import Path

from services.analytics_service.validation_report import summarize_timeline


def test_validation_report_extracts_stop_intervals(tmp_path: Path):
    csv_path = tmp_path / "timeline.csv"
    csv_path.write_text(
        "frame_id,equipment_id,equipment_class,timestamp_sec,state,activity,current_stop_seconds,last_stop_seconds,total_downtime_seconds,utilization_percent\n"
        "1,EQ-001,excavator,0.0,ACTIVE,Digging,0,0,0,100\n"
        "2,EQ-001,excavator,1.0,INACTIVE,Waiting,1,0,1,50\n"
        "3,EQ-001,excavator,2.0,INACTIVE,Waiting,2,0,2,33\n"
        "4,EQ-001,excavator,3.0,ACTIVE,Swinging/Loading,0,2,2,50\n",
        encoding="utf-8",
    )

    report = summarize_timeline(csv_path)
    row = report["EQ-001"]
    assert row.stop_count == 1
    assert row.total_downtime_sec == 2.0
    assert row.stop_intervals[0].start_sec == 1.0
    assert row.stop_intervals[0].end_sec == 3.0
