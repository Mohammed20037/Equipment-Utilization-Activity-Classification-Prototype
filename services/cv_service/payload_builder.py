from dataclasses import dataclass
from typing import Dict

from shared.schemas import EquipmentEvent, TimeAnalyticsBlock, UtilizationBlock


@dataclass
class Totals:
    tracked: float = 0.0
    active: float = 0.0
    idle: float = 0.0
    current_stop: float = 0.0
    last_stop: float = 0.0
    stop_count: int = 0


class PayloadBuilder:
    def __init__(self):
        self.totals: Dict[str, Totals] = {}

    def build(
        self,
        frame_id: int,
        timestamp: str,
        timestamp_sec: float,
        equipment_id: str,
        equipment_class: str,
        state: str,
        activity: str,
        motion_source: str,
        delta_t: float,
    ) -> EquipmentEvent:
        totals = self.totals.setdefault(equipment_id, Totals())
        totals.tracked += delta_t

        if state == "ACTIVE":
            totals.active += delta_t
            if totals.current_stop > 0:
                totals.last_stop = totals.current_stop
                totals.stop_count += 1
                totals.current_stop = 0.0
        else:
            totals.idle += delta_t
            totals.current_stop += delta_t

        utilization = 100.0 * totals.active / totals.tracked if totals.tracked else 0.0

        return EquipmentEvent(
            frame_id=frame_id,
            equipment_id=equipment_id,
            equipment_class=equipment_class,
            timestamp=timestamp,
            timestamp_sec=timestamp_sec,
            utilization=UtilizationBlock(
                current_state=state,
                current_activity=activity,
                motion_source=motion_source,
            ),
            time_analytics=TimeAnalyticsBlock(
                total_tracked_seconds=round(totals.tracked, 3),
                total_active_seconds=round(totals.active, 3),
                total_idle_seconds=round(totals.idle, 3),
                total_downtime_seconds=round(totals.idle, 3),
                current_stop_seconds=round(totals.current_stop, 3),
                last_stop_seconds=round(totals.last_stop, 3),
                stop_count=totals.stop_count,
                utilization_percent=round(utilization, 2),
            ),
        )
