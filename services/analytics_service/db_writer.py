from sqlalchemy import text


class DbWriter:
    def __init__(self, engine):
        self.engine = engine

    def write_event_and_summary(self, event: dict):
        util = event["utilization"]
        analytics = event["time_analytics"]

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO frame_events (
                      frame_id, equipment_id, equipment_class, timestamp_sec,
                      current_state, current_activity, motion_source,
                      total_tracked_seconds, total_active_seconds, total_idle_seconds, utilization_percent
                    ) VALUES (
                      :frame_id, :equipment_id, :equipment_class, :timestamp_sec,
                      :current_state, :current_activity, :motion_source,
                      :total_tracked_seconds, :total_active_seconds, :total_idle_seconds, :utilization_percent
                    )
                    """
                ),
                {
                    "frame_id": event["frame_id"],
                    "equipment_id": event["equipment_id"],
                    "equipment_class": event["equipment_class"],
                    "timestamp_sec": event["timestamp"],
                    "current_state": util["current_state"],
                    "current_activity": util["current_activity"],
                    "motion_source": util["motion_source"],
                    "total_tracked_seconds": analytics["total_tracked_seconds"],
                    "total_active_seconds": analytics["total_active_seconds"],
                    "total_idle_seconds": analytics["total_idle_seconds"],
                    "utilization_percent": analytics["utilization_percent"],
                },
            )

            conn.execute(
                text(
                    """
                    INSERT INTO equipment_summary (
                      equipment_id, equipment_class, total_tracked_seconds, total_active_seconds,
                      total_idle_seconds, utilization_percent, last_activity, last_state
                    ) VALUES (
                      :equipment_id, :equipment_class, :total_tracked_seconds, :total_active_seconds,
                      :total_idle_seconds, :utilization_percent, :last_activity, :last_state
                    )
                    ON CONFLICT (equipment_id) DO UPDATE SET
                      equipment_class = EXCLUDED.equipment_class,
                      total_tracked_seconds = EXCLUDED.total_tracked_seconds,
                      total_active_seconds = EXCLUDED.total_active_seconds,
                      total_idle_seconds = EXCLUDED.total_idle_seconds,
                      utilization_percent = EXCLUDED.utilization_percent,
                      last_activity = EXCLUDED.last_activity,
                      last_state = EXCLUDED.last_state,
                      updated_at = NOW();
                    """
                ),
                {
                    "equipment_id": event["equipment_id"],
                    "equipment_class": event["equipment_class"],
                    "total_tracked_seconds": analytics["total_tracked_seconds"],
                    "total_active_seconds": analytics["total_active_seconds"],
                    "total_idle_seconds": analytics["total_idle_seconds"],
                    "utilization_percent": analytics["utilization_percent"],
                    "last_activity": util["current_activity"],
                    "last_state": util["current_state"],
                },
            )
