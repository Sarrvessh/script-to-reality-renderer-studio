import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy's built-in SlideIn is unstable for full-frame material in this processing chain,
    # and can look like "the transition was applied logically, but the picture barely changes".
    # Use an explicit black background plus a slide animation here so the transition is visible and predictable.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_start = max(clip.duration - t, 0)

    # SlideOut is also an explicit slide, so the end of the segment reliably slides out of frame.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(t, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Keep the original 20% zoom amount so a clip of about three seconds still has a clearly visible Ken Burns move.
# Zoom stability comes from the sub-pixel center sampling below, not from weakening the effect to hide source-encoding flicker.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Zoom with a sub-pixel center crop so there is no black edge and the scale stays stable.

    Do not convert the crop width and height to integers first. While the scale changes continuously, integer edges jump by uneven steps
    and the half-pixel sampling phase changes when the size flips between odd and even, which shows up as picture jitter. Pillow's EXTENT
    transform can take floating-point edges and sample at sub-pixel positions on a fixed output canvas. The left/right and top/bottom edges
    stay symmetric around the same floating-point center, so this fits a slow zoom across a whole video.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # At 1x zoom, return the original frame and skip a resample that would slightly blur the first frame for no reason.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # A continuous video zoom cares more about consistency between neighboring frames. BICUBIC/LANCZOS are sharper on a single frame,
        # but high-frequency texture crossing the sample grid tends to ring and flicker in brightness. BILINEAR is softer
        # and trades a little sharpness for a more stable look in motion.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Smoothly zoom from the original frame to 1.2x across the whole segment."""
    # t is kept for now so this matches the shared call signature of the other transition functions. The zoom has to cover the whole segment,
    # or the picture snaps to a stop when a short zoom ends, which is a poor fit for static or low-motion material.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """Smoothly zoom from 1.2x back to the original frame across the whole segment."""
    # Same as zoomin_transition: t only exists so this matches the shared transition call signature.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])
