# Script-to-Reality Renderer Studio architecture

Script-to-Reality Renderer Studio is a local, English-first studio for short narrated videos.

## What stays from the renderer

These parts are used, not rewritten:

- OpenRouter chat calls in `app/services/llm.py`
- Pexels search and download in `app/services/material.py`
- Edge TTS narration in `app/services/voice.py`
- Subtitle timing and burned-in captions in `app/services/video.py`
- FFmpeg and MoviePy assembly in `app/services/task.py`
- The task queue and `POST /api/v1/videos` path in `app/controllers/v1/video.py`

A studio render builds a `TaskVideoRequest` and hands it to that queue. Script text the user edited is passed through, so the renderer does not replace it with a second model draft.

## What the studio adds

- `app/services/frameforge.py` stores projects as JSON under `storage/frameforge/projects/`.
- `app/controllers/v1/frameforge.py` exposes project, storyboard, stock-search, settings, and library routes under `/api/v1/frameforge/`.
- The storyboard splits narration into ordered scenes and asks OpenRouter for one stock search term per scene, in the same order. If the model is unavailable, each scene falls back to words from its own narration and the project records that notice.
- Choosing a Pexels clip downloads that file into the existing local-materials directory and renders `video_source=local` in scene order. Scenes without a selection stay on ordered Pexels search.
- Render defaults that improve the current free pipeline: sequential assembly, a fade, sentence captions, a dark caption bar, and the bundled Be Vietnam Pro font when it is present.
- `frontend/` is a Vite, React, and TypeScript studio. It talks to the API through a local proxy, so the browser does not need a new CORS policy.
- Finished MP4 files already written under `storage/tasks/` appear in the library even after the in-memory task list is cleared.

## Request path

1. The studio saves the reel.
2. `POST /api/v1/frameforge/projects/{id}/render` validates the storyboard.
3. If any scene has a chosen clip, those files are downloaded first and the task uses local files in scene order. Otherwise the task keeps `video_source=pexels` and the scene terms.
4. The existing task manager runs voice, captions, footage, and the final encode.
5. The studio polls `GET /api/v1/tasks/{id}` and plays `/tasks/{id}/final-1.mp4` when the file exists.

Progress numbers come from the existing task record. The labels in the studio are a reading of those checkpoints, not a second timer.

## Settings and secrets

The settings screen reports whether OpenRouter, Pexels, and FFmpeg are available. It can store a model id and the defaults for a new reel. It does not accept or return API keys. Keys remain in `.env` or `config.toml`, both gitignored.

Saving a model id writes `openrouter_model_name` through the existing config saver.

## Run it locally

From the repository root, with the virtualenv already installed:

```text
.venv\Scripts\python.exe main.py
```

In another terminal:

```text
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. The original Streamlit interface can still be started with `webui.bat`, but Script-to-Reality Renderer Studio is the product surface.

The default generation path is OpenRouter for text, Pexels for stock footage, and Edge TTS for voice. Paid AI video providers are not enabled.

## Limits

- The API task list lives in memory unless Redis is turned on. Finished files on disk are the durable history.
- Encoding stays on CPU `libx264` when the installed NVIDIA driver cannot satisfy this FFmpeg build's NVENC requirement. Do not change the driver from this project.
- Stock search needs `PEXELS_API_KEY` or `pexels_api_keys`. Script writing needs `OPENROUTER_API_KEY` or `openrouter_api_key`.
