# Design Notes

## Articulated motion handling

ACTIVE/INACTIVE should not rely only on whole-bbox displacement. Excavator arm movement can indicate productive work even when tracks are stationary.

Prototype strategy:

1. Compute full-bbox center displacement.
2. Compute ROI-level motion score for articulated subregions (arm/bucket area).
3. Mark ACTIVE if either exceeds threshold.

## Activity mapping (rule-based, v1)

- arm motion high + bucket near dig zone -> DIGGING
- arm motion high + body rotation -> SWINGING_LOADING
- unload-like downward release event -> DUMPING
- low motion for N frames -> WAITING

## Trade-off

Rule-based logic is fast and explainable for interview scope. It can be swapped with a temporal model (e.g., TCN/LSTM/transformer) in v2.
