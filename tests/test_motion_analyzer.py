import numpy as np

from services.cv_service.motion_analyzer import MotionAnalyzer


def test_motion_analyzer_initial_stationary():
    analyzer = MotionAnalyzer()
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    result = analyzer.analyze(frame, "EQ-001", (10, 10, 60, 60), None, "excavator")
    assert result.state == "INACTIVE"


def test_motion_analyzer_detects_masked_motion_activity():
    analyzer = MotionAnalyzer(full_body_threshold=1.0, articulated_threshold=1000, productive_threshold=0.2)
    frame_a = np.zeros((120, 120, 3), dtype=np.uint8)
    frame_b = np.zeros((120, 120, 3), dtype=np.uint8)
    frame_b[15:45, 35:55] = 255

    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[10:60, 10:60] = 1

    analyzer.analyze(frame_a, "EQ-001", (10, 10, 60, 60), mask, "excavator")
    result = analyzer.analyze(frame_b, "EQ-001", (10, 10, 60, 60), mask, "excavator")
    assert result.state == "ACTIVE"
    assert result.motion_source.startswith("masked_optical_flow")
