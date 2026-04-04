from dataclasses import dataclass
from typing import Dict

from shared.schemas import EquipmentEvent, TimeAnalyticsBlock, UtilizationBlock


@dataclass
class Totals:
    tracked: float = 0.0
    active: float = 0.0
    idle: float = 0.0


class PayloadBuilder:
    def __init__(self):
        self.totals: Dict[str, Totals] = {}

    def build(self, frame_id: int, timestamp: float, equipment_id: str, equipment_class: str, state: str, activity: str, motion_source: str, delta_t: float) -> EquipmentEvent:
        totals = self.totals.setdefault(equipment_id, Totals())
        totals.tracked += delta_t
        if state == "ACTIVE":
            totals.active += delta_t
        else:
            totals.idle += delta_t
        utilization = 100.0 * totals.active / totals.tracked if totals.tracked else 0.0

        return EquipmentEvent(
            frame_id=frame_id,
            equipment_id=equipment_id,
            equipment_class=equipment_class,
            timestamp=timestamp,
            utilization=UtilizationBlock(
                current_state=state,
                current_activity=activity,
                motion_source=motion_source,
            ),
            time_analytics=TimeAnalyticsBlock(
                total_tracked_seconds=round(totals.tracked, 3),
                total_active_seconds=round(totals.active, 3),
                total_idle_seconds=round(totals.idle, 3),
                utilization_percent=round(utilization, 2),
            ),
        )
