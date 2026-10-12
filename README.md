<div align="center">

# Script-to-Reality Renderer Studio 💸

### An All-in-One AI Short Video Generator

Provide a video <b>topic</b> or <b>keyword</b>, and Script-to-Reality Renderer Studio will generate the script, match footage, create subtitles and background music, and produce an HD short video.


</div>

## Screenshots 🖥️

<h4 align="center">WebUI</h4>

![](docs/webui-en.jpg)

<h4 align="center">API</h4>

![](docs/api.jpg)

---

## Features 🎯

### Creation Workflows

- [x] Use **AI Agent, WebUI, API, or CLI** workflows for quick creation or automated production
- [x] Go from a topic to script, voiceover, footage, subtitles, music, and editing automatically, while retaining control over every stage
- [x] Generate multiple output variants in batches, review task history, and import or export generation settings and API keys

### Scripts and Model Providers

- [x] Generate or rewrite **multilingual video scripts** with AI, or supply a complete custom script

### Video and Image Footage

- [x] Upload your own **local images and videos**, or get HD stock footage from [Pexels (free)](https://www.pexels.com/api/), [Pixabay (free)](https://pixabay.com/api/docs/), and [Coverr](https://coverr.co/developers?ctx=header_navigation)
- [x] Generate `768P` or `2K` source footage with [Metaso MiniMax H3](https://metaso.cn/minimax-h3/?s=MPT), with 4–15 second clips in `9:16`, `16:9`, or `1:1`
- [x] Create multiple AI video clips with [Shengsuan Cloud AI Video](https://www.shengsuanyun.com/?from=CH_XUQ4OTSK), then combine them with the project's voiceover, subtitle, and editing workflow
- [x] Use the native [Volcano Engine Ark Seedance](https://console.volcengine.com/ark/region:ark+cn-beijing/apikey) integration to generate cohesive visuals from individual script segments
- [x] Turn script keywords into original video footage with [WaveSpeed AI](https://wavespeed.ai)
- [x] Access Seedance, Wan, and other text-to-video models through [OFox](https://ofox.ai) with a single API key
- [x] Generate 3–12 second AI video materials through the asynchronous [MuAPI](https://muapi.ai) text-to-video API, with configurable endpoint, aspect ratio, resolution, and polling
- [x] Connect [OpenAI-compatible text-to-image](https://platform.openai.com/docs/guides/image-generation) services or custom image gateways and turn generated images into animated video clips
- [x] Adjust clip duration, frame fitting, and material order to suit different aspect ratios and storytelling styles

### Voiceover, Subtitles, and Background Music

- [x] Choose automatic voiceover, uploaded audio, or no voiceover, with voice samples and full narration previews
- [x] Use **Edge TTS (free, no API key required)**, Azure Speech, SiliconFlow, Google Gemini, Xiaomi MiMo, MiniMax, ElevenLabs, Chatterbox, Kokoro, Fish Audio, ModelBest VoxCPM, and other voice services
- [x] Generate subtitles and configure their font, position, color, size, outline, and background style
- [x] Use random, local, or AI-generated background music with independent volume control

### Output and Publishing

- [x] Export portrait `9:16 (1080×1920)`, landscape `16:9 (1920×1080)`, or square `1:1 (1080×1080)` videos
- [x] Publish completed videos directly to **TikTok, Instagram, and YouTube Shorts**

## Gallery 🎬

All examples below were generated with Script-to-Reality Renderer Studio.

### Portrait 9:16

<table width="100%">
<tr>
</tr>
<tr>
</tr>
</table>

### Landscape 16:9

<table width="100%">
<tr>
</tr>
<tr>
</tr>
</table>

## System Requirements 📦

- Recommended platforms: Windows 10+, macOS 11+, or a mainstream Linux distribution
- Local deployment requires Python 3.11 or later; Python 3.11 is recommended
- A GPU is not required, but it is recommended if you want faster local transcription, faster video processing, or smoother batch generation

| Item | Minimum      | Recommended  | Optimal    |
| ---- | ------------ | ------------ | ---------- |
| CPU  | 4 cores      | 6 to 8 cores | 8+ cores   |
| RAM  | 4 GB         | 8 GB         | 16+ GB     |
| GPU  | Not required | 4+ GB VRAM   | 8+ GB VRAM |

- If you mainly rely on cloud LLMs, cloud TTS, and online material sources, CPU and RAM matter more than GPU
- If you use `faster-whisper`, batch generation, or heavier local processing, a GPU will improve throughput noticeably

## Quick Start 🚀

### Recommended Paths

- If you do not want to install or configure the project manually: generate videos with an AI Agent
- Windows users: use the one-click package first for the fastest local trial
- macOS / Linux users: use `uv` for the primary local setup path
- If you want a more isolated runtime: use Docker deployment

### Generate Videos with an AI Agent

If your AI Agent can read Skill documents and operate a local terminal, send it the prompt below. The Agent will install and configure Script-to-Reality Renderer Studio, generate the video, and return the video file path. It will ask only for required API keys that are not already configured. This workflow currently supports macOS and Windows.

```text
Create a video with the topic "How AI is changing everyday life."
```

### Run in Google Colab

Want to try Script-to-Reality Renderer Studio without setting up a local environment? Run it directly in Google Colab!


### Windows

Download the latest Windows one-click package from GitHub Releases, then extract it directly.


> Download the `.7z` archive from the **Assets** section of that page. The
> auto-generated `Source code (zip)` / `Source code (tar.gz)` archives contain
> source code only: after extracting them you get `webui.bat` but no `start.bat`
> or `update.bat`.

After downloading, it is recommended to **double-click** `update.bat` first to update to the **latest code**, then double-click `start.bat` to launch

After launching, the browser will open automatically (if it opens blank, it is recommended to use **Chrome** or **Edge**)

### macOS / Linux

Use the local setup or Docker instructions below.

## Installation & Deployment 📥

### Prerequisites

- Local deployment requires Python 3.11 or later
- On Windows, avoid project paths containing non-ASCII characters, special characters, or spaces

#### ① Clone the Project

```shell
```

#### ② Complete the Initial Setup

On first launch, the project creates `config.toml` from `config.example.toml`, so you do not need to create the file manually. Before using cloud LLMs, online footage, or AI video services, add the corresponding API keys in the WebUI basic settings.

### Docker Deployment 🐳

#### ① Launch the Docker Container

If Docker is not installed, [download and install Docker Desktop](https://www.docker.com/products/docker-desktop/) first.
If you are using a Windows system, please refer to Microsoft's documentation:

1. [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
2. [Use Docker containers with WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers)

```shell
cd Script-to-Reality Renderer Studio
docker compose -f docker-compose.release.yml up
```

> If you need to build the image locally, you can still run `docker compose up`.
> Before the first start, copy `config.example.toml` to `config.toml` so it can be mounted into the containers.

#### ② Access the WebUI

Open your browser and visit http://127.0.0.1:8501

#### ③ Access the API Documentation

Open your browser and visit http://127.0.0.1:8080/docs or http://127.0.0.1:8080/redoc

> The API allows same-origin browser access by default. Set the `CORS_ALLOWED_ORIGINS` environment variable only when a separate browser frontend must call the API directly from another origin, for example `http://localhost:3000,https://frontend.example.com`. CORS does not affect curl, Postman, n8n, or other server-side clients.

### Manual Deployment 📦

#### ① Create a Python Virtual Environment

Use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. The project supports Python 3.11 or later; the example below uses Python 3.11.

```shell
cd Script-to-Reality Renderer Studio
uv python install 3.11
uv sync --frozen
```

If you are not using `uv` yet, you can still use `venv + pip`.

```shell
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `pyproject.toml` is now the primary dependency manifest.
- `uv.lock` pins the resolved environment, so `uv sync --frozen` is recommended by default.
- `requirements.txt` is kept only for legacy `pip`-based installation.

#### ② Launch the WebUI 🌐

Note that you need to execute the following commands in the `root directory` of the Script-to-Reality Renderer Studio project

###### Windows

```powershell
.\webui.bat
```

You can also run `webui.bat` in CMD.
`webui.bat` prefers the project `.venv` or bundled Python from the portable package. If no project Python is found but `uv` is installed, it automatically falls back to `uv run streamlit`.
To allow other devices on your LAN to access the WebUI, run `set MPT_WEBUI_HOST=0.0.0.0` before running `webui.bat`.

###### macOS or Linux

```shell
sh webui.sh
```

The script automatically uses the project virtual environment or `uv` and selects an available local port. To allow access from other devices on your LAN, run:

```shell
MPT_WEBUI_HOST=0.0.0.0 sh webui.sh
```

After launching, the browser will open automatically

#### ③ Launch the API Service 🚀

```shell
uv run python main.py
```

If you have already activated the virtual environment manually, you can still run:

```shell
python main.py
```

#### ④ Pure CLI Mode (No Browser) ⌨️

If you cannot use a browser or port forwarding, generate videos directly from the
command line. The simplest complete generation command is:

```shell
uv run python cli.py --video-subject "How AI is changing everyday life"
```

Subtitle style and voiceover options resolve in this order: **explicit CLI option >
saved `[ui]` value in `config.toml` > built-in default**. Other generation settings,
such as background music, video count, and paragraph count, are not inherited from
the WebUI. If the WebUI is set to use uploaded audio, pass `--custom-audio-file`
explicitly, since the uploaded path is not persisted.

For the complete command reference, parameter descriptions, and usage instructions,
run:

```shell
uv run python cli.py --help
```

To run several tasks sequentially, pass a UTF-8 JSON array or JSONL manifest. CLI
options act as defaults, and each object overrides fields from `VideoParams`:

```json
[
  { "video_subject": "How solar panels work" },
  { "video_subject": "How wind turbines work", "video_aspect": "16:9" }
]
```

```shell
uv run python cli.py --batch-file ./tasks.json --stop-at video
```

The manifest is resolved from the current working directory. Relative
`custom_audio_file` and local `video_materials[].url` values inside it are resolved
from the manifest's directory; file paths supplied as CLI defaults keep their normal
current-working-directory semantics. A manifest is limited to 100 tasks and 1 MiB.
All entries are validated before the first task starts, tasks continue after an
individual runtime failure, and the command prints one JSON summary when finished.
The summary contains `total`, `succeeded`, `failed`, and `tasks`; each task entry has
`index`, `task_id`, `status`, `result`, `failed_stage`, and `error`.

## Voiceover, Subtitles, and Background Music 🎙️

### Voice Synthesis


**Azure TTS V1** in the WebUI is powered by **Edge TTS** and is free to use without an API key. Script-to-Reality Renderer Studio also supports **Azure TTS V2**, **SiliconFlow TTS**, **Google Gemini TTS**, **Xiaomi MiMo TTS**, **MiniMax TTS**, **ElevenLabs TTS**, self-hosted **Chatterbox TTS**, self-hosted **Kokoro TTS**, **Fish Audio TTS**, [ModelBest VoxCPM TTS](https://platform.modelbest.cn/console/docs/api/audio), and a no-voice mode.

Select a provider and voice in the WebUI, then follow the on-screen instructions for any required credentials. Edge TTS does not require an API key; [Azure TTS V2](https://portal.azure.com/) and other cloud providers require credentials from their respective platforms. See the available Edge TTS voices in the [voice list](./docs/voice-list.txt).

ModelBest VoxCPM requires an API key and a model ID with the `speech_synthesis` capability. Its SSE response streams WAV audio, which Script-to-Reality Renderer Studio automatically converts to the MP3 used by the video pipeline. In addition to standard text-to-speech, the WebUI can clone speaker identity from optional reference audio. High-fidelity delivery reuses that clip with its exact transcript by default, or accepts a separate performance example, and sends `prompt_audio` plus `prompt_text` to continue its pacing, emotion, and pronunciation. The WebUI can use the local Whisper configuration to create an editable transcript draft. Uploads are limited to 20 MiB and each converted WAV to 5 MiB. Reference material is scoped to the current browser session and task and is not written to configuration, presets, task history, or logs.

### Subtitle Generation

Two subtitle generation modes are available:

- **edge**: Uses TTS timestamps, runs quickly without a GPU, and is the default mode.
- **whisper**: Uses local `faster-whisper` transcription when a more accurate subtitle timeline is needed. The model is downloaded on first use.

Set `subtitle_provider` in `config.toml` to switch modes. Whisper uses the approximately 3 GB `large-v3` model by default. To use the smaller and faster, approximately 1.6 GB `large-v3-turbo` model:

```toml
[app]
subtitle_provider = "whisper"

[whisper]
model_size = "large-v3-turbo"
```

> On first use, Whisper automatically downloads the model from Hugging Face. If the automatic download fails, download `whisper-large-v3` manually from [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3).

After extracting the model, place the entire directory in `.\Script-to-Reality Renderer Studio\models`. The final path should be `.\Script-to-Reality Renderer Studio\models\whisper-large-v3`:

```
Script-to-Reality Renderer Studio
  ├─models
  │   └─whisper-large-v3
  │          config.json
  │          model.bin
  │          preprocessor_config.json
  │          tokenizer.json
  │          vocabulary.json
```

### Background Music

Background music for videos is located in the project's `resource/songs` directory.

> The current project includes some default music from YouTube videos. If there are copyright issues, please delete
> them.

### Subtitle Fonts

Fonts for rendering video subtitles are located in the project's `resource/fonts` directory, and you can also add your
own fonts.

## Common Questions 🤔

<details>
<summary>RuntimeError: No ffmpeg exe could be found</summary>

Normally, ffmpeg will be automatically downloaded and detected.
However, if your environment has issues preventing automatic downloads, you may encounter the following error:

```
RuntimeError: No ffmpeg exe could be found.
Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.
```

In this case, download FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract it, and set `ffmpeg_path` to
the actual installation path.

```toml
[app]
# Please set according to your actual path, note that Windows path separators are \\
ffmpeg_path = "C:\\Users\\harry\\Downloads\\ffmpeg.exe"
```

</details>

<details>
<summary>OSError: [Errno 24] Too many open files</summary>

This issue is caused by the system's limit on the number of open files. You can solve it by modifying the system's file open limit.

Check the current limit:

```shell
ulimit -n
```

If it's too low, you can increase it, for example:

```shell
ulimit -n 10240
```

</details>

<details>
<summary>Whisper model download failed</summary>

```
LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and
outgoing traffic has been disabled.
To enable repo look-ups and downloads online, pass 'local_files_only=False' as input.
```

or

```
An error occurred while synchronizing the model Systran/faster-whisper-large-v3 from the Hugging Face Hub:
An error happened while trying to locate the files on the Hub and we cannot find the appropriate snapshot folder for the
specified revision on the local disk. Please check your internet connection and try again.
Trying to load the model directly from the local cache, if it exists.
```

Solution: [See how to download the model manually from Hugging Face](#subtitle-generation)

</details>

## Feedback & Suggestions 📢


## License 📝

Click to view the [`LICENSE`](LICENSE) file

### Editable local video projects

Use [revision-aware local projects](docs/video-projects.md) to replace prepared scene media, compare revisions and rebuild only affected stages. The opt-in CLI preserves verified artifacts and the last successful export; it makes no provider calls or automatic posts.
