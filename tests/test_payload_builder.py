from services.cv_service.payload_builder import PayloadBuilder


def test_payload_builder_tracks_stop_durations():
    builder = PayloadBuilder()

    e1 = builder.build(
        frame_id=1,
        timestamp="00:00:00.0",
        timestamp_sec=0.0,
        equipment_id="EQ-1",
        equipment_class="excavator",
        state="INACTIVE",
        activity="WAITING",
        motion_source="stationary",
        delta_t=1.0,
    )
    assert e1.time_analytics.current_stop_seconds == 1.0
    assert e1.time_analytics.stop_count == 0

    e2 = builder.build(
        frame_id=2,
        timestamp="00:00:01.0",
        timestamp_sec=1.0,
        equipment_id="EQ-1",
        equipment_class="excavator",
        state="ACTIVE",
        activity="DIGGING",
        motion_source="optical_flow_yolo",
        delta_t=1.0,
    )
    assert e2.time_analytics.current_stop_seconds == 0.0
    assert e2.time_analytics.last_stop_seconds == 1.0
    assert e2.time_analytics.stop_count == 1
