# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React, TypeScript, and Tailwind CSS for the Script-to-Reality Renderer Studio interface. The existing Python FastAPI service remains the backend. The user specified this stack. Vite serves the interface in local Windows development and proxies API calls to the existing server.

## Users

The primary user is the project owner, working at a Windows development machine in Cursor, producing short English videos for a technical college demonstration. Assumption: reviewers will open the local app, create a project from a topic, and watch a finished MP4 without learning the original dashboard.

## Product Purpose

Script-to-Reality Renderer Studio turns a topic into a short narrated video. Success means the owner can write or generate an English script, arrange it as scenes, choose stock footage per scene, set an English voice and readable captions, render an MP4, and find that video again later.

## Positioning

The mechanism is a scene strip: each narration beat owns a search term and an optional chosen clip, and that order is what the renderer follows. Neighboring "topic in, video out" tools can hide that structure. Script-to-Reality Renderer Studio keeps it editable before any render starts.

## Brand

The product name is Script-to-Reality Renderer Studio.

## Capabilities

- Generate and edit an English narration script through the existing OpenRouter integration.
- Build a scene-by-scene storyboard and edit each scene's narration and search term.
- Search the configured free stock provider and select a clip for a scene.
- Choose an English Edge TTS voice, rate, caption style, aspect ratio, and transition.
- Render with the existing Edge TTS, subtitle, stock-download, and FFmpeg/MoviePy pipeline.
- Show real task progress, then preview and download the MP4.
- Keep projects and a library of finished renders on disk.

## Constraints

- Do not replace the working generation pipeline.
- Do not call paid AI video-generation providers unless the user explicitly configures one. The default source stays free stock footage.
- English is the default language. Existing multilingual generation remains available in the backend.
- API keys stay in `.env` or `config.toml` and are never shown in the interface or logs.
- Local Windows development. GPU diffusion models stay unloaded. Encoding stays on CPU when NVENC is incompatible with the installed driver.
- Decorative controls are not allowed. Every visible action calls a real service.

## Voice

Direct, specific, and calm. Name the action and the failure. Do not use hype, fake metrics, or claims about customers, pricing, or benchmark scores.

## Open decisions

- Whether a future public deployment should require the optional API key already supported by the backend.
