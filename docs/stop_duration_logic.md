# Stop Duration Logic Specification

## 1) Inputs
- Per-track debounced state (`ACTIVE` / `INACTIVE`).
- `delta_t` frame time.
- Track missing-frame count (`lost_frames`).

## 2) Debounce
- Raw states from motion analysis are not used directly.
- State transition is committed only after N consecutive candidate frames:
  - `ACTIVE_DEBOUNCE_FRAMES` (default 4)
  - `INACTIVE_DEBOUNCE_FRAMES` (default 6)

## 3) Missing detection handling
- If `lost_frames <= TRACK_MISSING_TOLERANCE_FRAMES`, pipeline holds the last confirmed state (`track_gap_hold`) and continues accumulation.
- If `lost_frames > tolerance`, the track is skipped to avoid synthetic motion/noise.

## 4) Stop interval semantics
- **Stop start**: first frame where confirmed state is `INACTIVE` after previously confirmed `ACTIVE`.
- **Stop continuation**: first accumulates into a pending stop; downtime is committed only after `MIN_STOP_SECONDS` is reached (micro-stops are ignored).
- **Stop end**: first confirmed `ACTIVE` frame after an open stop:
  - `last_stop_seconds = current_stop_seconds`
  - `stop_count += 1`
  - `current_stop_seconds = 0`

## 5) Totals
- `total_downtime_seconds` and `total_idle_seconds` increase whenever confirmed state is `INACTIVE`.
- `total_active_seconds` increases whenever confirmed state is `ACTIVE`.
- `utilization_percent = total_active_seconds / total_tracked_seconds * 100`.

