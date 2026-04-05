from dataclasses import dataclass
from typing import Dict


@dataclass
class UtilizationTotals:
    total_tracked_seconds: float = 0.0
    total_active_seconds: float = 0.0
    total_idle_seconds: float = 0.0

    @property
    def utilization_percent(self) -> float:
        if self.total_tracked_seconds <= 0:
            return 0.0
        return (self.total_active_seconds / self.total_tracked_seconds) * 100.0


class UtilizationEngine:
    def __init__(self):
        self.state: Dict[str, UtilizationTotals] = {}

    def update(self, equipment_id: str, state: str, delta_t: float) -> UtilizationTotals:
        totals = self.state.setdefault(equipment_id, UtilizationTotals())
        totals.total_tracked_seconds += delta_t
        if state == "ACTIVE":
            totals.total_active_seconds += delta_t
        else:
            totals.total_idle_seconds += delta_t
        return totals
