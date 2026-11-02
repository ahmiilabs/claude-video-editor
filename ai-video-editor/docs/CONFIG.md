# Configuration

Precedence: real environment variables > `./.env` > `<skill>/.env` > `~/.config/ai-video-editor/.env`. Copy `.env.example` to `.env` (never commit it).
Placeholder values (`your-...-here`) are ignored. API keys are never printed: logs, reports, `run.json`, error messages and `env_check` all pass through `config.redact()`.

| Variable | Meaning |
|---|---|
| `AIVE_LLM_PROVIDER` `_BASE_URL` `_MODEL` `_API_KEY` | script writer. provider: `openai` (any OpenAI-compatible server incl. Ollama/LM Studio/vLLM/OpenRouter) \| `anthropic` \| `none`. `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are accepted as key fallbacks. Local base URLs need no key |
| `AIVE_IMAGE_PROVIDER` `_BASE_URL` `_MODEL` `_API_KEY` | image generation (`openai`). Extras: `AIVE_IMAGE_SIZES` (JSON aspect->size), `AIVE_IMAGE_EXTRA` (JSON merged into the request), `AIVE_IMAGE_REFERENCE_MODE=prompt\|edits` |
| `AIVE_VIDEO_PROVIDER=http` `_BASE_URL` `_MODEL` `_API_KEY` `AIVE_VIDEO_CONFIG` | video generation; the config JSON describes the provider's endpoints (docs/PROVIDERS.md) |
| `AIVE_TRANSCRIBE_ENGINE` | `auto` \| `faster-whisper` \| `openai-api` \| `sherpa` \| `openai-whisper`; `AIVE_WHISPER_MODEL` (tiny/base/...), `AIVE_TRANSCRIBE_API_KEY`, `AIVE_TRANSCRIBE_BASE_URL` |
| `AIVE_TIMEOUT` `AIVE_RETRIES` | HTTP timeout seconds, retries for 429/5xx/network errors |
| `AIVE_QUALITY` `AIVE_PRESET` | `draft` (720p) / `hd` (1080p); preset names reels/shorts/tiktok/youtube/square |
| `AIVE_WORKERS` | batch parallelism (each render is CPU-bound; match your CPU count) |
| `AIVE_GPU` | `auto` \| `on` \| `off` |
| `AIVE_MODEL_DIR` `AIVE_TMP` `AIVE_KEEP_TMP` `AIVE_TTS_CACHE` `AIVE_LOG_LEVEL` | model cache, temp dir, keep temp files for debugging, TTS cache, log level |
| `AIVE_FFMPEG` `AIVE_FFPROBE` | explicit binary paths |

Check without exposing secrets: `python scripts/config.py` (provider status and problems) or `python scripts/env_check.py`.
