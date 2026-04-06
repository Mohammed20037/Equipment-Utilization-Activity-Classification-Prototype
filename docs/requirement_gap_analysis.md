# Requirement Coverage (Heavy-Equipment Utilization Objective)

| Requirement | Current implementation | Status | Evidence | Next fix |
|---|---|---|---|---|
| Restrict to relevant equipment only | Detector has configurable allowlist + normalized labels | **Full** | `TARGET_EQUIPMENT_CLASSES` filtering in detector | Tune allowlist per site camera |
| Exclude public-road/background traffic | Include/exclude ROI polygon filtering | **Full** | `ROI_INCLUDE_POLYGON` and `ROI_EXCLUDE_POLYGON` checks | Add optional ROI preview overlay |
| Stable ID continuity / Re-ID | Tracker uses centroid + IoU + HSV hist score + lost tolerance | **Partial** | Composite track matching in tracker | Add deep Re-ID embeddings if needed |
| ACTIVE/INACTIVE from productive motion | Optical flow computes articulated/chassis/productive scores; debounce confirms transitions | **Partial** | Productive threshold + debouncer in CV main | Calibrate thresholds per equipment profile |
| Activity labels (Digging, Swinging/Loading, Dumping, Waiting) | Equipment-specific heuristic classifier with required labels | **Partial** | Activity classifier rules | Replace with temporal model once labeled data is ready |
| Reliable stop interval and downtime | Debounced state drives stop accumulation; track-gap hold prevents flicker inflation | **Partial** | Debouncer + gap hold path + payload stop metrics | Add explicit stop start/end timestamps in schema |
| Validation outputs for demo | Per-equipment timeline CSV + validation report script | **Full** | `data/processed/equipment_timeline.csv` + `scripts/validate_short_clip.py` | Add auto summary report script |

