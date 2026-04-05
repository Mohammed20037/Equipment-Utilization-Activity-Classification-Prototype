from services.analytics_service.utilization_engine import UtilizationEngine


def test_utilization_accumulates():
    engine = UtilizationEngine()
    totals = engine.update("EQ-001", "ACTIVE", 1.0)
    totals = engine.update("EQ-001", "INACTIVE", 1.0)

    assert totals.total_tracked_seconds == 2.0
    assert totals.total_active_seconds == 1.0
    assert totals.total_idle_seconds == 1.0
    assert totals.utilization_percent == 50.0
