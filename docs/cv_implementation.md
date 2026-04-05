# Computer Vision Implementation Notes

## Did we use computer vision?
Yes. The prototype uses OpenCV-based CV methods in the `cv_service`:

1. **Foreground motion detection** (`MotionDetector`)
   - Background subtraction (MOG2)
   - Contour extraction and bounding box generation
2. **Object tracking** (`CentroidTracker`)
   - Track IDs across frames using centroid-distance matching
3. **Articulated motion handling** (`MotionAnalyzer`)
   - Full-body motion via bbox-center displacement
   - Arm-only proxy motion via ROI frame differencing in the upper-right bbox region
   - ACTIVE if full-body motion OR articulated-region motion crosses threshold
4. **Activity classification** (`ActivityClassifier`)
   - Rule-based labels: `DIGGING`, `SWINGING_LOADING`, `DUMPING`, `WAITING`

## Important limitation
The current version uses classical OpenCV CV (motion-based) and does **not** yet include a trained detector like YOLO/Mask R-CNN.

For interview positioning, describe this as:
- CV-complete **prototype** with articulated-motion logic implemented.
- Model-based detector/classifier integration is the next enhancement stage.
