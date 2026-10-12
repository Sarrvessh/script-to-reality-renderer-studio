"""Narration coverage uses original clips and does not repeat them."""

import wave

import cv2
import pytest

from app.models.schema import VideoAspect, VideoConcatMode
from app.services.video import combine_videos


def _clip(path, seconds):
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 48))
    assert writer.isOpened()
    image = np.full((48, 64, 3), 180, dtype=np.uint8)
    for _ in range(int(8 * seconds)):
        writer.write(image)
    writer.release()


def _wav(path, seconds):
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * int(8000 * seconds))


def test_short_ai_footage_fails_instead_of_looping(tmp_path):
    video = tmp_path / "scene.avi"
    audio = tmp_path / "narration.wav"
    _clip(video, 1)
    _wav(audio, 3)
    with pytest.raises(RuntimeError, match="were not repeated"):
        combine_videos(
            combined_video_path=str(tmp_path / "final.mp4"),
            video_paths=[str(video)],
            audio_file=str(audio),
            video_aspect=VideoAspect.portrait,
            video_concat_mode=VideoConcatMode.sequential,
            max_clip_duration=5,
            threads=1,
            allow_loop=False,
        )


def test_enough_ai_footage_composes_a_playable_file(tmp_path):
    from app.services.cv_engine import validate_playable

    video = tmp_path / "scene.avi"
    audio = tmp_path / "narration.wav"
    output = tmp_path / "final.mp4"
    _clip(video, 4)
    _wav(audio, 2)
    combine_videos(
        combined_video_path=str(output),
        video_paths=[str(video)],
        audio_file=str(audio),
        video_aspect=VideoAspect.portrait,
        video_concat_mode=VideoConcatMode.sequential,
        max_clip_duration=5,
        threads=1,
        allow_loop=False,
    )
    check = validate_playable(str(output))
    assert output.is_file()
    assert check["playable"]
