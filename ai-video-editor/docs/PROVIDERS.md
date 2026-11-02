# AI providers (all optional)

**Status: adapters are implemented and unit-tested against local mock servers. They have NOT been verified against any real provider in this project.**
Test with a real key on a short job first. If a provider fails, the tool reports it; it never substitutes a fake result.

## Script writer (LLM)
```bash
export AIVE_LLM_PROVIDER=openai AIVE_LLM_BASE_URL=https://api.openai.com/v1 AIVE_LLM_MODEL=<model> AIVE_LLM_API_KEY=<key>
python scripts/scriptwriter.py --topic "Why is chai expensive on trains" --language roman-urdu --duration 30 --platform reels --tone energetic --out spec.json
```
`--language en|ur|roman-urdu|hi|multi`. Hindi/Urdu/Roman Urdu scripts include `voice_text` (Devanagari) for the Hindi TTS voice; on-screen text stays Roman.
Output is a renderable spec plus `script_meta` (hook, music suggestion, style anchor). Invalid model output is retried with feedback (3 attempts), then fails clearly.
**No LLM? Offline workflow:** `--prompt-only --out prompt.txt` -> paste into any chat model -> save its JSON -> `--import answer.json --out spec.json`.
Local models: point `AIVE_LLM_BASE_URL` at Ollama (`http://localhost:11434/v1`), no key needed.

## Image generation
`AIVE_IMAGE_PROVIDER=openai` (POST `{base}/images/generations`, `b64_json` or `url` responses). Scenes with `image_prompt` get images; `visual_style_anchor` is prefixed to every prompt for consistency; a stable seed is sent. Results are validated (decodes, >=256px), saved as `s03_image_<hash>.png`, recorded in `assets/manifest.json`; 429/5xx are retried, 4xx are not.
```bash
python scripts/imagegen.py spec.json --assets assets/ [--only 2,3] [--force]
```
Your own images: set `scene.image` to a file; it is validated and registered, never regenerated.

## Video clips (async)
Because vendors differ, describe the provider in `AIVE_VIDEO_CONFIG` (file path or inline JSON):
```json
{"submit": {"method": "POST", "path": "/generate", "body": {"model": "{model}", "prompt": "{prompt}", "duration": "{duration}", "aspect_ratio": "{aspect}", "image": "{image_b64}"}, "id_path": "id"},
 "status": {"path": "/jobs/{id}", "state_path": "status", "done": ["completed"], "failed": ["failed"], "result_url_path": "output.url"},
 "limits": {"durations": [4, 8], "aspects": ["9:16", "16:9"], "needs_image": false}, "poll_seconds": 5, "timeout_seconds": 600}
```
Durations are snapped to supported values and unsupported aspects fail before any request. Jobs are tracked in `assets/jobs.json`; downloads are validated and fitted to the reel canvas. If a job fails and the scene has an image, the renderer animates that image (real Ken-Burns). Standalone: `python scripts/videogen.py --kenburns photo.jpg --seconds 5 --out clip.mp4`.

## Transcription
Order for `auto`: faster-whisper (true word times) -> openai-api -> sherpa (offline, sentence-level) -> openai-whisper. Every transcript carries `granularity` (`word` or `segment`).
