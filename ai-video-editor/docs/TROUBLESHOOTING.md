# Troubleshooting

| Symptom | Fix |
|---|---|
| `'ffmpeg' was not found on PATH. Install it with: ...` | follow the printed command, or set `AIVE_FFMPEG` |
| Captions missing / `ass` filter error | ffmpeg without libass: install a full build (Windows: gyan.dev full) |
| Robotic voice | neural voice not installed: `python scripts/setup_voice.py`; best quality = your own recording (`voice.file`) |
| Hindi voice sounds like an English speaker | add Devanagari `voice_text` to each scene |
| `neural TTS produced NaN/inf` | the old int8 model is cached: delete `~/.cache/ai-video-editor/kokoro` and rerun `setup_voice.py` (fp32 is used now) |
| `person-segmentation model not installed` | `python scripts/setup_models.py --matting` |
| Matte effects slow | about 0.15-0.3 s/frame on one CPU core: trim the clip, `--max-width 480` |
| `no transcription engine available` | `python scripts/setup_models.py --whisper tiny` or install `faster-whisper`, or configure `openai-api` |
| Whisper text is wrong for Hindi/Urdu | `tiny` is weak there: use `--whisper base`, `faster-whisper`, or supply captions via `narration` (audio-aligned) |
| Words not exactly on the spoken word | engines other than faster-whisper/openai give sentence-level times; words are interpolated and flagged `approx` |
| Image/video provider errors | `python scripts/config.py` shows what is missing; failures are in `assets/manifest.json`; scenes fall back to style backgrounds / Ken-Burns |
| QA `black_frames` at a boundary | a transition produced dark frames: use another preset or shorter `duration` |
| QA `frozen_frames` warning | static scenes are normal; it fails only above 60% of the video |
| QA `captions` fails | captions did not burn in (libass) or text was off-screen; check `<out>.render.json` |
| Batch job stuck `running` after a crash | `batch.py resume <dir>` |
| Render too slow / memory errors | `--quality draft`, fewer scenes, `AIVE_WORKERS=1`; hd is downgraded automatically when RAM is low |
| Keep temp files for debugging | `AIVE_KEEP_TMP=1` or `reel.py --work dir` |
