# Articulated Motion Handling (ACTIVE with Partial Machine Movement)

This project treats articulated movement as a first-class signal so a machine can be **ACTIVE** even when chassis motion is minimal.

## Signals
Inside each tracked equipment ROI, the optical-flow path computes:
- `full_body_score`: mean flow over the entire box.
- `articulated_score`: mean flow over upper-right articulated proxy region (arm/bucket side).
- `chassis_score`: mean flow over lower-left chassis proxy region.

## Productive-motion score
`productive_score = max(articulated_score, full_body_score - 0.55 * chassis_score)`

This favors arm/bucket activity and suppresses pure translation/chassis drift.

## ACTIVE/INACTIVE decision
1. If `productive_score >= PRODUCTIVE_MOTION_THRESHOLD` => `ACTIVE`.
2. Else if coarse flow thresholds are met (`full_body` or `articulated`) => weak `ACTIVE`.
3. Else => `INACTIVE`.

## Why this helps
For fixed-camera excavator clips, this catches real work cycles where the arm moves while tracks/chassis barely move.
