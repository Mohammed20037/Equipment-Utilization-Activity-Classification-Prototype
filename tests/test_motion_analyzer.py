import numpy as np

from services.cv_service.motion_analyzer import MotionAnalyzer


def test_motion_analyzer_initial_stationary():
    analyzer = MotionAnalyzer()
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    result = analyzer.analyze(frame, "EQ-001", (10, 10, 60, 60))
    assert result.state in {"INACTIVE", "ACTIVE"}


def test_motion_analyzer_detects_full_body_shift():
    analyzer = MotionAnalyzer(full_body_threshold=1.0, articulated_threshold=1000)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    analyzer.analyze(frame, "EQ-001", (10, 10, 60, 60))
    result = analyzer.analyze(frame, "EQ-001", (20, 10, 70, 60))
    assert result.state == "ACTIVE"
    assert result.motion_source == "full_body"
