from shared.schemas import EquipmentEvent, TimeAnalyticsBlock, UtilizationBlock


def test_payload_schema_validates():
    payload = EquipmentEvent(
        frame_id=1,
        equipment_id="EX-001",
        equipment_class="excavator",
        timestamp="00:00:00.500",
        timestamp_sec=0.5,
        utilization=UtilizationBlock(
            current_state="ACTIVE",
            current_activity="DIGGING",
            motion_source="arm_only",
        ),
        time_analytics=TimeAnalyticsBlock(
            total_tracked_seconds=0.5,
            total_active_seconds=0.5,
            total_idle_seconds=0.0,
            utilization_percent=100.0,
        ),
    )
    assert payload.equipment_id == "EX-001"
