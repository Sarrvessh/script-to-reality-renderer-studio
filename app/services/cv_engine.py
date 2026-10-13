"""Sampled-frame checks for generated scenes.

Scores are calibrated heuristics. They are not a guarantee that a scene is correct.
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Any, Callable

import numpy as np
from loguru import logger

DISCLAIMER = "Scores are measurements from the named model. They are not a guarantee of correctness."
# ViT-B-32 image-text cosine similarity. Matched frames usually land near 0.25 or higher.
DEFAULT_THRESHOLDS = {
    "blur": 12.0,
    "black": 12.0,
    "frozen_flow": 0.35,
    "discontinuity": 12.0,
    "duplicate_hamming": 4,
    "semantic": 0.24,
    "ssim_frozen": 0.992,
}
_model_lock = threading.Lock()
_clip_bundle = None
_yolo_model = None


class CVCapabilityError(RuntimeError):
    def __init__(self, component: str, detail: str) -> None:
        super().__init__(f"{component} unavailable: {detail}")
        self.component = component


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip():
    """Load ViT-B-32 once. The same weights are reused for every scene."""
    global _clip_bundle
    with _model_lock:
        if _clip_bundle is not None:
            return _clip_bundle
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", force_quick_gelu=True,
            )
            device = _device()
            model = model.to(device).eval()
            _clip_bundle = (model, preprocess, open_clip.get_tokenizer("ViT-B-32"), device)
            logger.info("cv clip loaded backend=open_clip model=ViT-B-32 device=%s", device)
            return _clip_bundle
        except Exception as exc:
            raise CVCapabilityError("open_clip", type(exc).__name__) from exc


def load_yolo():
    """Load YOLOv8n once and keep detections on a small input size."""
    global _yolo_model
    with _model_lock:
        if _yolo_model is not None:
            return _yolo_model
        try:
            from pathlib import Path
            from ultralytics import YOLO
            weights = Path(__file__).resolve().parents[2] / "storage" / "models" / "yolov8n.pt"
            weights.parent.mkdir(parents=True, exist_ok=True)
            downloaded = Path("yolov8n.pt")
            if not weights.exists() and downloaded.exists():
                downloaded.replace(weights)
            model = YOLO(str(weights if weights.exists() else "yolov8n.pt"))
            device = _device()
            model.to(device)
            _yolo_model = model
            logger.info("cv yolo loaded backend=yolov8n device=%s", device)
            return _yolo_model
        except Exception as exc:
            raise CVCapabilityError("yolov8n", type(exc).__name__) from exc


def _cv2():
    import cv2
    return cv2


def sample_frames(path: str, count: int = 8) -> list[np.ndarray]:
    cv2 = _cv2()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path} for frame sampling.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indexes = [0] if total <= 1 else [int(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
    frames = []
    for index in indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            frames.append(frame)
    capture.release()
    if not frames:
        logger.info("cv Frame sampling: failed — No readable frames in the clip.")
        raise RuntimeError(f"No readable frames in {path}.")
    logger.info(f"cv Frame sampling: passed — {len(frames)} frame(s) from the clip.")
    return frames


def validate_playable(path: str) -> dict[str, Any]:
    cv2 = _cv2()
    capture = cv2.VideoCapture(path)
    opened = capture.isOpened()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
    ok, frame = capture.read() if opened else (False, None)
    capture.release()
    playable = bool(ok and frame is not None and getattr(frame, "size", 0) and width > 0 and height > 0 and frames >= 1)
    return {
        "playable": playable,
        "frames": frames,
        "width": width,
        "height": height,
        "codec_opened": playable,
    }


def _gray(frame: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur_score(frame: np.ndarray) -> float:
    cv2 = _cv2()
    return float(cv2.Laplacian(_gray(frame), cv2.CV_64F).var())


def black_ratio(frames: list[np.ndarray], threshold: float) -> float:
    if not frames:
        return 1.0
    dark = sum(1 for frame in frames if float(frame.mean()) < threshold)
    return dark / len(frames)


def phash(frame: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    small = cv2.resize(_gray(frame), (32, 32)).astype(np.float32)
    block = cv2.dct(small)[:8, :8]
    return (block > np.median(block)).flatten()


def hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    cv2 = _cv2()
    a = cv2.resize(_gray(left), (64, 64)).astype(np.float64)
    b = cv2.resize(_gray(right), (64, 64)).astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2) + 1e-9))


def motion_metrics(frames: list[np.ndarray]) -> dict[str, float | str]:
    cv2 = _cv2()
    backend = "dis" if hasattr(cv2, "DISOpticalFlow_create") else "farneback"
    if len(frames) < 2:
        return {"mean_flow": 0.0, "max_jump": 0.0, "frozen": 1.0, "backend": backend}
    magnitudes = []
    previous = _gray(frames[0])
    for frame in frames[1:]:
        gray = _gray(frame)
        if backend == "dis":
            flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(previous, gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
        magnitudes.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
        previous = gray
    jumps = [abs(magnitudes[i] - magnitudes[i - 1]) for i in range(1, len(magnitudes))]
    mean_flow = float(np.mean(magnitudes))
    return {
        "mean_flow": mean_flow,
        "max_jump": float(max(jumps) if jumps else 0.0),
        "frozen": 1.0 if mean_flow < DEFAULT_THRESHOLDS["frozen_flow"] else 0.0,
        "backend": backend,
    }


def parse_yolo_output(output: np.ndarray, width: int, height: int, confidence: float = 0.25) -> list[dict[str, Any]]:
    """Read a YOLOv8 head: class scores sit after the 4 box values."""
    matrix = np.asarray(output)
    if matrix.ndim == 3:
        matrix = matrix[0]
    if matrix.shape[0] > 4 and matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    boxes = []
    for row in matrix:
        scores = row[4:]
        if scores.size == 0:
            continue
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if score < confidence:
            continue
        cx, cy, bw, bh = [float(value) for value in row[:4]]
        if max(cx, cy, bw, bh) <= 1.5:
            cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
        boxes.append({
            "x": int(max(0, cx - bw / 2)),
            "y": int(max(0, cy - bh / 2)),
            "w": int(max(1, bw)),
            "h": int(max(1, bh)),
            "label": str(class_id),
            "confidence": score,
        })
    return boxes


def track_objects(frames: list[np.ndarray], detect: Callable[[np.ndarray], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    finder = detect if detect is not None else detect_yolo
    next_id = 1
    active: list[dict[str, Any]] = []
    seen = 0
    matched = 0
    trails: dict[int, list[tuple[float, float]]] = {}
    for frame in frames:
        boxes = finder(frame)
        seen += len(boxes)
        used = set()
        refreshed = []
        for box in boxes:
            cx = box["x"] + box["w"] / 2
            cy = box["y"] + box["h"] / 2
            best_index = None
            best_distance = 80.0
            for index, previous in enumerate(active):
                if index in used:
                    continue
                distance = ((cx - previous["cx"]) ** 2 + (cy - previous["cy"]) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is None:
                identity = next_id
                next_id += 1
            else:
                used.add(best_index)
                identity = active[best_index]["id"]
                matched += 1
            refreshed.append({"id": identity, "cx": cx, "cy": cy, "box": box})
            trails.setdefault(identity, []).append((cx, cy))
        active = refreshed
    smooth = smooth_crops(trails, frames[0].shape[1], frames[0].shape[0]) if frames else []
    preservation = (matched / seen) if seen else 0.0
    return {
        "tracks": len(trails),
        "detections": seen,
        "preservation": float(preservation),
        "reframe": smooth[:1],
        "detector": "yolov8n" if detect is None else "custom",
    }


def smooth_crops(trails: dict[int, list[tuple[float, float]]], width: int, height: int) -> list[dict[str, int]]:
    if not trails or width < 2 or height < 2:
        return []
    identity = max(trails, key=lambda key: len(trails[key]))
    points = trails[identity]
    sx, sy = points[0]
    crops = []
    crop_w = max(2, int(width * 0.8))
    crop_h = max(2, int(height * 0.8))
    for cx, cy in points:
        sx = sx * 0.7 + cx * 0.3
        sy = sy * 0.7 + cy * 0.3
        x = int(min(max(0, sx - crop_w / 2), width - crop_w))
        y = int(min(max(0, sy - crop_h / 2), height - crop_h))
        crops.append({"x": x, "y": y, "w": crop_w, "h": crop_h})
    return crops


def _even(value: int) -> int:
    return int(value) - (int(value) % 2)


def apply_reframe(source: str, crop: dict[str, int], destination: str) -> str:
    """Crop a tracked subject and scale the clip back to its original size."""
    info = validate_playable(source)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if not info.get("playable") or not crop or width < 2 or height < 2:
        return source
    x = _even(max(0, int(crop.get("x") or 0)))
    y = _even(max(0, int(crop.get("y") or 0)))
    w = _even(min(width - x, int(crop.get("w") or 0)))
    h = _even(min(height - y, int(crop.get("h") or 0)))
    if w < 2 or h < 2 or (w >= width and h >= height):
        return source
    from app.utils.utils import get_ffmpeg_binary
    binary = get_ffmpeg_binary()
    if binary == "ffmpeg":
        from app.config import config
        configured = str(config.app.get("ffmpeg_path", "") or "")
        if configured and os.path.isfile(configured):
            binary = configured
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    command = [
        binary, "-y", "-i", source,
        "-vf", f"crop={w}:{h}:{x}:{y},scale={_even(width)}:{_even(height)}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", destination,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not validate_playable(destination).get("playable"):
        logger.warning("cv reframe failed path=%s", source)
        return source
    logger.info("cv reframe applied crop=%s,%s,%s,%s output=%s", x, y, w, h, destination)
    return destination


def detect_yolo(frame: np.ndarray) -> list[dict[str, Any]]:
    model = load_yolo()
    device = _device()
    result = model.predict(frame, imgsz=320, conf=0.25, max_det=10, verbose=False, device=device)[0]
    boxes = []
    height, width = frame.shape[:2]
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        boxes.append({
            "x": int(max(0, min(width - 1, x1))),
            "y": int(max(0, min(height - 1, y1))),
            "w": int(max(1, min(width, x2 - x1))),
            "h": int(max(1, min(height, y2 - y1))),
            "label": str(int(box.cls[0])) if box.cls is not None else "object",
            "confidence": float(box.conf[0]) if box.conf is not None else 0.0,
        })
    return boxes


def semantic_similarity(frame: np.ndarray, description: str, embed: Callable[[np.ndarray, str], float] | None = None) -> tuple[float, str]:
    if embed is not None:
        return float(embed(frame, description)), "injected"
    import torch
    from PIL import Image
    model, preprocess, tokenizer, device = load_clip()
    cv2 = _cv2()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = tokenizer([description or "a scene"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = float((image_features @ text_features.T)[0, 0])
    return score, "open_clip"


def _capability_report(exc: CVCapabilityError) -> dict[str, Any]:
    logger.warning(f"cv {exc.component}: failed — {exc}")
    return {
        "accepted": False,
        "capability_failure": True,
        "score": 0.0,
        "warnings": [str(exc)],
        "refinement": str(exc),
        "disclaimer": DISCLAIMER,
        "metrics": {"semantic_backend": "unavailable", "detector": "unavailable", "semantic_similarity": None},
        "components": [
            _component(exc.component, exc.component, "failed", str(exc)),
        ],
    }


def _component(check_id: str, name: str, status: str, detail: str) -> dict[str, Any]:
    """One showcase line for a named check. Status is passed, warning, or failed."""
    logger.info(f"cv {name}: {status} — {detail}")
    return {"id": check_id, "name": name, "status": status, "detail": detail}


def _component_logs(
    *,
    blur: float,
    darkness: float,
    duplicate_ratio: float,
    mean_ssim: float,
    motion: dict[str, Any],
    tracking: dict[str, Any],
    semantic: float | None,
    backend: str,
    limits: dict[str, float],
    frozen_reject: bool,
    duplicate_reject: bool,
) -> list[dict[str, Any]]:
    flow_name = "DIS optical flow" if motion.get("backend") == "dis" else "Farneback optical flow"
    yolo_status = "warning"
    if tracking.get("detector") == "unavailable":
        yolo_status = "failed"
        yolo_detail = "YOLOv8n was not available on this run."
    elif int(tracking.get("tracks") or 0) > 0:
        yolo_status = "passed"
        yolo_detail = (
            f"{int(tracking.get('detections') or 0)} detection(s), "
            f"{int(tracking.get('tracks') or 0)} track(s), "
            f"preservation {float(tracking.get('preservation') or 0):.2f}."
        )
    else:
        yolo_detail = "No recognizable object class. Recorded as a warning; the clip can still pass."
    if semantic is None:
        clip_status, clip_detail = "failed", f"OpenCLIP unavailable ({backend})."
    elif semantic < limits["semantic"]:
        clip_status = "failed"
        clip_detail = f"similarity {semantic:.3f}, below {limits['semantic']:.2f}."
    else:
        clip_status = "passed"
        clip_detail = f"similarity {semantic:.3f}, threshold {limits['semantic']:.2f}."
    frozen = bool(motion.get("frozen"))
    jump = float(motion.get("max_jump") or 0) > limits["discontinuity"]
    if frozen_reject:
        flow_status, ssim_status = "failed", "failed"
    elif jump:
        flow_status, ssim_status = "warning", "passed" if mean_ssim < limits["ssim_frozen"] else "warning"
    else:
        flow_status = "passed"
        ssim_status = "passed"
    return [
        _component(
            "blur",
            "Laplacian blur",
            "failed" if blur < limits["blur"] else "passed",
            f"median variance {blur:.1f}, need ≥ {limits['blur']:.0f}.",
        ),
        _component(
            "black",
            "Black-frame ratio",
            "failed" if darkness > 0.5 else "passed",
            f"{darkness:.0%} of samples were near black, reject above 50%.",
        ),
        _component(
            "phash",
            "DCT perceptual hash",
            "failed" if duplicate_reject else "passed",
            f"{duplicate_ratio:.0%} neighbouring samples matched, reject above 80%.",
        ),
        _component(
            "ssim",
            "SSIM",
            ssim_status,
            f"mean {mean_ssim:.3f}; a frozen shot is rejected at ≥ {limits['ssim_frozen']:.3f} with low flow.",
        ),
        _component(
            "flow",
            flow_name,
            flow_status,
            f"mean motion {float(motion.get('mean_flow') or 0):.3f}, jump {float(motion.get('max_jump') or 0):.2f} "
            f"(frozen below {limits['frozen_flow']:.2f}).",
        ),
        _component("yolo", "YOLOv8n", yolo_status, yolo_detail),
        _component("clip", "OpenCLIP ViT-B-32", clip_status, clip_detail),
    ]


def evaluate(
    frames: list[np.ndarray],
    description: str,
    *,
    thresholds: dict[str, float] | None = None,
    embed: Callable[[np.ndarray, str], float] | None = None,
    detect: Callable[[np.ndarray], list[dict[str, Any]]] | None = None,
    failure_policy: str = "fail",
) -> dict[str, Any]:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not frames:
        return {
            "accepted": False,
            "score": 0.0,
            "warnings": ["no frames"],
            "refinement": "Show a visible subject with motion and detail.",
            "metrics": {},
            "components": [_component("frames", "Frame sampling", "failed", "No readable frames in the clip.")],
            "disclaimer": DISCLAIMER,
        }
    blur_values = [blur_score(frame) for frame in frames]
    blur = float(np.median(blur_values))
    darkness = black_ratio(frames, limits["black"])
    hashes = [phash(frame) for frame in frames]
    duplicates = 0
    comparisons = 0
    similarities = []
    for index in range(1, len(frames)):
        comparisons += 1
        if hamming(hashes[index], hashes[index - 1]) <= limits["duplicate_hamming"]:
            duplicates += 1
        similarities.append(ssim(frames[index - 1], frames[index]))
    motion = motion_metrics(frames)
    degraded = []
    try:
        tracking = track_objects(frames, detect if detect is not None else detect_yolo)
    except CVCapabilityError as exc:
        if failure_policy != "degrade":
            return _capability_report(exc)
        degraded.append(str(exc))
        tracking = {"tracks": 0, "detections": 0, "preservation": 0.0, "reframe": [], "detector": "unavailable"}
    try:
        semantic, backend = semantic_similarity(frames[len(frames) // 2], description, embed)
    except CVCapabilityError as exc:
        if failure_policy != "degrade":
            return _capability_report(exc)
        degraded.append(str(exc))
        semantic, backend = None, "unavailable"
    warnings = list(degraded)
    reasons = []
    mean_ssim = float(np.mean(similarities)) if similarities else 1.0
    duplicate_ratio = (duplicates / comparisons) if comparisons else 0.0
    frozen_reject = bool(motion["frozen"]) and mean_ssim >= limits["ssim_frozen"]
    duplicate_reject = bool(comparisons) and duplicate_ratio > 0.8
    if darkness > 0.5:
        reasons.append("mostly blank or black")
    if blur < limits["blur"]:
        reasons.append("too blurry")
    if frozen_reject:
        reasons.append("little or no motion")
    if duplicate_reject:
        reasons.append("repeated frames")
    if float(motion["max_jump"]) > limits["discontinuity"]:
        warnings.append("motion jumps between samples")
    if semantic is not None and semantic < limits["semantic"]:
        reasons.append("weak match to the scene description")
    if tracking["detector"] != "unavailable" and tracking["tracks"] == 0:
        warnings.append("no recognizable object class")
    detector_name = "yolov8n" if detect is None and tracking["detector"] != "unavailable" else tracking["detector"]
    refinement = "Keep the same subject. " + (
        "Increase motion, contrast, and visible detail. Match this description: " + description
        if reasons else "Hold the subject steady and readable."
    )
    mapped = 0.0 if semantic is None else max(0.0, min(1.0, (float(semantic) + 1.0) / 2.0))
    quality = min(1.0, blur / 80.0) * (1 - darkness)
    score = float(max(0.0, min(1.0, 0.45 * mapped + 0.25 * quality + 0.2 * min(1.0, float(motion["mean_flow"])) + 0.1 * tracking["preservation"])))
    components = _component_logs(
        blur=blur,
        darkness=darkness,
        duplicate_ratio=duplicate_ratio,
        mean_ssim=mean_ssim,
        motion=motion,
        tracking=tracking,
        semantic=semantic,
        backend=backend,
        limits=limits,
        frozen_reject=frozen_reject,
        duplicate_reject=duplicate_reject,
    )
    report = {
        "accepted": not reasons,
        "score": score,
        "warnings": warnings + reasons,
        "refinement": refinement[:500],
        "disclaimer": DISCLAIMER,
        "components": components,
        "metrics": {
            "blur": blur,
            "black_ratio": darkness,
            "duplicate_ratio": duplicate_ratio,
            "mean_ssim": mean_ssim,
            "semantic_similarity": semantic,
            "semantic_backend": backend,
            "semantic_threshold": limits["semantic"],
            "mean_flow": motion["mean_flow"],
            "motion_jump": motion["max_jump"],
            "flow_backend": motion.get("backend"),
            "tracks": tracking["tracks"],
            "objects": tracking["detections"],
            "object_preservation": tracking["preservation"],
            "reframe": tracking["reframe"],
            "detector": detector_name,
        },
    }
    logger.info(
        f"cv summary: accepted={report['accepted']} score={score:.3f} "
        f"openclip={backend}:{('none' if semantic is None else f'{float(semantic):.3f}')} "
        f"yolo={detector_name} objects={tracking['detections']} "
        f"flow={float(motion['mean_flow']):.3f} blur={blur:.2f} black={darkness:.2f} "
        f"duplicate={duplicate_ratio:.2f} ssim={mean_ssim:.3f} reasons={','.join(reasons) or '-'}"
    )
    return report
