from pydantic import BaseModel


class UtilizationBlock(BaseModel):
    current_state: str
    current_activity: str
    motion_source: str


class TimeAnalyticsBlock(BaseModel):
    total_tracked_seconds: float
    total_active_seconds: float
    total_idle_seconds: float
    total_downtime_seconds: float
    current_stop_seconds: float
    last_stop_seconds: float
    stop_count: int
    utilization_percent: float


class EquipmentEvent(BaseModel):
    frame_id: int
    equipment_id: str
    equipment_class: str
    timestamp: str
    timestamp_sec: float
    utilization: UtilizationBlock
    time_analytics: TimeAnalyticsBlock
