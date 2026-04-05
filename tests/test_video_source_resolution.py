from pathlib import Path

from services.cv_service.main import resolve_video_source


def test_resolve_video_source_explicit_path(tmp_path: Path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"x")
    assert resolve_video_source(str(p)) == str(p)
