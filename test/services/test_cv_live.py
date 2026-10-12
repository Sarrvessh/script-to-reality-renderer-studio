"""Real OpenCLIP and YOLO checks against an existing local MP4. No paid API calls."""

from pathlib import Path

import pytest

from app.services import cv_engine

CLIP = next(Path("storage").rglob("*.mp4"), None)


@pytest.mark.skipif(CLIP is None, reason="no local mp4")
def test_openclip_and_yolo_score_an_existing_clip():
    frames = cv_engine.sample_frames(str(CLIP), count=4)
    similarity, backend = cv_engine.semantic_similarity(frames[0], "a moving outdoor scene")
    boxes = cv_engine.detect_yolo(frames[len(frames) // 2])
    tracked = cv_engine.track_objects(frames, cv_engine.detect_yolo)
    report = cv_engine.evaluate(frames, "a moving outdoor scene", failure_policy="fail")

    import torch
    _, _, _, clip_device = cv_engine.load_clip()
    yolo_device = str(next(cv_engine.load_yolo().model.parameters()).device)
    assert clip_device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert yolo_device.startswith(clip_device)

    assert backend == "open_clip"
    assert -1.0 <= similarity <= 1.0
    assert isinstance(boxes, list)
    assert tracked["detector"] == "custom"
    assert report["metrics"]["semantic_backend"] == "open_clip"
    assert report["metrics"]["detector"] == "yolov8n"
    assert isinstance(report["metrics"]["objects"], int)
    assert report["metrics"]["mean_flow"] >= 0
    assert report["metrics"]["blur"] >= 0
    assert "visual-lexicon" not in str(report)
    assert not report.get("capability_failure")
