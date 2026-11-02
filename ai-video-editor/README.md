# AI Video Editor

A modular, code-driven video production engine for Reels, Shorts, TikToks, explainers, ads and faceless content. It turns a topic or a JSON
spec into a finished MP4: script, visuals, voice, captions, transitions, motion graphics, audio mix, QA. It also edits clips you upload.

It works **fully offline** (renders, styles, transitions, captions, audio, QA, batch). AI services (LLM, image/video generation, cloud
transcription) are **optional adapters**; if they are not configured the tool says so and falls back, it never fakes output.

```
TOPIC -> script -> assets (images/videos) -> voice -> transcription/captions -> render (styles + transitions + overlays)
      -> audio mix (enhance, music, ducking, sfx) -> mux -> QA -> optional post effects -> final MP4
```

## claude.ai web
Upload `ai-video-editor.skill` (Settings > Features > Skills) and just chat. Web-specific behaviour, limits and every command: [docs/WEB.md](docs/WEB.md). In short: Claude writes the script itself (no API key), AI image/video/cloud services are unreachable from the sandbox so your own media is used, run `scripts/web_setup.py` once per chat, keep each render <= 35 s.

## Quick start

```bash
bash install.sh --with-models          # Linux/macOS/Termux     (Windows: powershell -File install.ps1 -WithModels)
python scripts/env_check.py            # what works on this machine
python scripts/reel.py assets/example-spec.json --out reel.mp4
python scripts/pipeline.py --spec assets/example-spec.json --out reel.mp4 --workdir runs/demo     # all stages, logged
python -m pytest tests -q              # run the test suite
```

## Feature availability (read this before relying on a feature)

| # | Status | Features |
|---|---|---|
| 1 | **Fully implemented, no external model/API, tested locally** | Reel renderer (5 style packs), 17 transitions + presets, motion graphics (lower thirds, scene titles, animated stats/charts, glass panels, progress bar), image/video scene backgrounds with Ken-Burns, captions (ASS highlight / karaoke / plain, SRT), basic clip effects (spotlight, punch-zoom, color-grade, glass-widgets, split-chart, cinematic-zoom, face-crop), stabilised multi-face tracking (OpenCV Haar), audio pipeline (denoise, loudness, music, **sidechain ducking**, fades, SFX), QA framework (14 checks, JSON report, exit codes), batch queue (workers, retry, cancel, interrupt/resume), unified pipeline, asset manager, config/.env, secret redaction |
| 2 | **Implemented, needs a free model download (verified here)** | Natural voice: Kokoro-82M fp32 (`setup_voice.py`, 350 MB). Person matting: Robust Video Matting (`setup_models.py --matting`, 15 MB) -> bg-remove (transparent video), bg-replace, bg-blur, **text-behind-person**, smart-spotlight. Offline Whisper via sherpa-onnx (`--whisper tiny`): real **sentence-level** timestamps |
| 3 | **Implemented, needs an external API key; NOT verified against real providers** | AI script writer (OpenAI-compatible or Anthropic), image generation (OpenAI-compatible `/images/generations`), video-clip generation (config-driven async HTTP adapter), cloud transcription (`/audio/transcriptions`). Tested only against local mock servers |
| 4 | **Optional / not exercised here** | `faster-whisper` / `openai-whisper` for true word-level timestamps; hardware H.264 (NVENC/QSV/VideoToolbox; auto-detected by a real trial encode, CPU fallback); Windows, macOS and Termux (code is cross-platform, only Linux was executed) |
| 5 | **Not supported** | Urdu-script (Arabic) text rendering on screen (no shaping font bundled; use Roman Urdu); music generation (only a synthetic placeholder pad, bring your own music); word-level timestamps without engines from row 3/4; deep-learning face detection (Haar only); face recognition |

Honesty notes: Whisper `tiny` is weak on Hindi/Urdu (use `base`, or faster-whisper). For generated voiceovers the script is known, so captions use
**audio-aligned** timing (scene windows are exact; words are placed inside the voiced audio), not recognition. Word times from sherpa are interpolated inside
real sentence timestamps and flagged `approx`.

## Documentation
[docs/WEB.md](docs/WEB.md) · [docs/INSTALL.md](docs/INSTALL.md) · [docs/CONFIG.md](docs/CONFIG.md) · [docs/PROVIDERS.md](docs/PROVIDERS.md) · [docs/CLI.md](docs/CLI.md) · [docs/SPEC.md](docs/SPEC.md) · [docs/BATCH.md](docs/BATCH.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Layout
```
SKILL.md                 instructions for Claude (skill entry)       scripts/   all code
.env.example             placeholders only                          tests/     pytest suite (mocks + real renders)
assets/                  example spec, bundled OFL fonts            docs/ references/
```
Key scripts: `reel.py` render · `pipeline.py` all stages · `batch.py` queue · `effects.py` clip effects · `transcribe.py` · `scriptwriter.py` · `imagegen.py` · `videogen.py` · `audio.py` · `qa_check.py` · `env_check.py` · `setup_models.py`.
