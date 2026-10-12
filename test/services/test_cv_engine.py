import numpy as np
import pytest

from app.services import cv_engine


def test_motion_tracking_and_acceptance_on_a_moving_subject():
    frames = []
    for step in range(6):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (20, 90, 160)
        frame[::2, ::2] = (20, 70, 140)
        x = 4 + step * 6
        frame[12:36, x : x + 16] = (255, 255, 255)
        frame[12:36:2, x : x + 16:2] = (0, 0, 0)
        frames.append(frame)
    report = cv_engine.evaluate(
        frames,
        "a red boat moving on blue water",
        embed=lambda _frame, _text: 0.8,
        detect=lambda _frame: [{"x": 4, "y": 4, "w": 16, "h": 16, "label": "0"}],
    )
    assert report["metrics"]["mean_flow"] > 0.2
    assert report["metrics"]["tracks"] >= 1
    assert report["metrics"]["object_preservation"] >= 0
    assert report["accepted"]
    assert "not a guarantee" in report["disclaimer"]


def test_black_blur_and_duplicate_frames_are_rejected():
    black = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(4)]
    blank = cv_engine.evaluate(black, "a sunny harbor", embed=lambda _frame, _text: 0.9, detect=lambda _frame: [])
    assert not blank["accepted"]
    assert blank["metrics"]["black_ratio"] == 1

    import cv2
    sharp = np.zeros((48, 64, 3), dtype=np.uint8)
    sharp[10:40, 10:50] = (255, 255, 255)
    blurred = [cv2.GaussianBlur(sharp, (31, 31), 0) for _ in range(3)]
    soft = cv_engine.evaluate(blurred, "a bright square", embed=lambda _frame, _text: 0.9, detect=lambda _frame: [])
    assert soft["metrics"]["blur"] < cv_engine.blur_score(sharp)
    assert not soft["accepted"]


def test_reframe_crop_is_written_into_the_clip(tmp_path):
    import cv2
    source = tmp_path / "scene.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 48))
    assert writer.isOpened()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, 32:] = (255, 255, 255)
    for _ in range(4):
        writer.write(frame)
    writer.release()
    output = tmp_path / "scene-reframed.mp4"
    written = cv_engine.apply_reframe(str(source), {"x": 32, "y": 0, "w": 32, "h": 48}, str(output))
    assert written == str(output)
    capture = cv2.VideoCapture(written)
    ok, image = capture.read()
    capture.release()
    assert ok
    assert float(image.mean()) > 200


def test_yolo_parser_and_playable_clip(tmp_path):
    output = np.zeros((1, 6), dtype=np.float32)
    output[0, :4] = (0.5, 0.5, 0.25, 0.25)
    output[0, 4] = 0.9
    boxes = cv_engine.parse_yolo_output(output, 100, 80)
    assert boxes[0]["w"] == 25
    assert boxes[0]["confidence"] == pytest.approx(0.9)

    frame = np.full((48, 64, 3), 180, dtype=np.uint8)
    path = tmp_path / "clip.avi"
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 48))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(frame)
    writer.release()
    check = cv_engine.validate_playable(str(path))
    assert check["playable"]
    assert check["width"] == 64
    empty = cv_engine.validate_playable(str(tmp_path / "missing.avi"))
    assert not empty["playable"]
