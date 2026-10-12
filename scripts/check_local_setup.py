"""Warn about local setup gaps without printing secret values."""

from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import config  # noqa: E402


def _nonempty(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    return bool(value)


def main() -> int:
    warnings: list[str] = []
    provider = str(config.app.get("llm_provider", "")).strip().lower()
    if provider == "openrouter" and not (
        _nonempty(config.app.get("openrouter_api_key", ""))
        or os.getenv("OPENROUTER_API_KEY", "").strip()
    ):
        warnings.append(
            "OpenRouter API key is not configured. Put it in .env as "
            "OPENROUTER_API_KEY, or in config.toml as openrouter_api_key. "
            "Create a key at https://openrouter.ai/settings/keys"
        )

    if str(config.app.get("video_source", "")).strip().lower() == "pexels" and not (
        _nonempty(config.app.get("pexels_api_keys"))
        or os.getenv("PEXELS_API_KEY", "").strip()
    ):
        warnings.append(
            "Pexels API key is not configured. Put it in .env as PEXELS_API_KEY, "
            "or in config.toml as pexels_api_keys. Create a free key at "
            "https://www.pexels.com/api/"
        )

    ffmpeg = shutil.which("ffmpeg") or os.environ.get("IMAGEIO_FFMPEG_EXE", "")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        warnings.append(
            "ffmpeg was not found on PATH. The bundled imageio binary may still "
            "work; a full build with ffprobe is in .tools\\ffmpeg\\bin when present."
        )
    if not ffprobe:
        warnings.append("ffprobe was not found on PATH.")

    paid_sources = {
        "wavespeed",
        "volcengine_seedance",
        "ofox",
        "metaso_minimax",
        "muapi",
        "loomloom",
        "openai_image",
    }
    source = str(config.app.get("video_source", "")).strip().lower()
    if source in paid_sources:
        warnings.append(
            f"video_source is '{source}', which can create paid video tasks. "
            "Use pexels, pixabay, coverr, or local for the free stock pipeline."
        )

    if warnings:
        print("***** Local setup warnings *****")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("***** Local setup check passed *****")
    print(f"***** LLM provider: {provider or 'unset'} *****")
    print(f"***** Video source: {source or 'unset'} *****")
    print(f"***** UI language: {config.ui.get('language', '') or 'unset'} *****")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
