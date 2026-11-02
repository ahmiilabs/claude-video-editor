# CLI cheat-sheet (every major feature)

```bash
# environment & models
python scripts/env_check.py [--install] [--strict]
python scripts/setup_voice.py ; python scripts/setup_models.py --matting --whisper tiny ; python scripts/setup_models.py --list

# script writer (see PROVIDERS.md for the offline workflow)
python scripts/scriptwriter.py --topic "..." --language roman-urdu --duration 30 --platform reels --out spec.json
# assets
python scripts/imagegen.py spec.json --assets assets/        # AI images for scenes with image_prompt
python scripts/videogen.py spec.json --assets assets/        # AI clips (async) for scenes with video_prompt
python scripts/videogen.py --kenburns photo.jpg --seconds 5 --aspect 9:16 --out clip.mp4

# render
python scripts/reel.py spec.json --preview                    # contact sheet first
python scripts/reel.py spec.json --out reel.mp4 [--quality hd] [--fps 30] [--work dir] [--no-captions] [--no-qa]
python scripts/pipeline.py --topic "..." --out reel.mp4 --workdir runs/x [--skip images,videos] [--only voice,render] [--strict] [--post-effects "--effect color-grade --look warm"]

# captions / transcription
python scripts/transcribe.py clip.mp4 --lang en --srt out.srt --ass out.ass --json out.json --style karaoke
python scripts/transcribe.py --list-engines

# clip effects (chain with commas)
python scripts/effects.py --list
python scripts/effects.py clip.mp4 --effect spotlight,color-grade --look cinematic --out out.mp4
python scripts/effects.py clip.mp4 --effect punch-zoom --every 2 --zoom 1.15
python scripts/effects.py clip.mp4 --effect glass-widgets --widgets "36|Followers;197K|Reach"
python scripts/effects.py clip.mp4 --effect split-chart --data "Jan:20,Feb:45,Mar:80" --title Growth
python scripts/effects.py clip.mp4 --effect captions --captions-from-audio --caption-style karaoke --caption-follow subject --srt out.srt
python scripts/effects.py clip.mp4 --effect cinematic-zoom --push 1.2
python scripts/effects.py wide.mp4 --effect face-crop --aspect 9:16               # face-crop must be first
python scripts/effects.py clip.mp4 --effect bg-replace --bg "gradient:#ff6b35:#1a1a2e"   # also blur:25 | color:#hex | image:p.jpg | video:p.mp4
python scripts/effects.py clip.mp4 --effect bg-blur --bg-sigma 20
python scripts/effects.py clip.mp4 --effect text-behind --title-text "FOCUS" --title-color "#FFD740"
python scripts/effects.py clip.mp4 --effect smart-spotlight --strength 0.75
python scripts/effects.py clip.mp4 --effect bg-remove --alpha-format mov --out person.mov     # mov (qtrle) | webm (VP9 alpha) | png

# audio
python scripts/audio.py enhance --in raw.wav --out clean.wav
python scripts/audio.py synth-music --kind calm --seconds 30 --out bed.wav ; python scripts/audio.py synth-sfx --kind whoosh --out whoosh.wav
python scripts/audio.py mix --voice v.wav --music bed.wav --sfx whoosh.wav@2.5 --seconds 20 --out mix.wav [--no-duck]
python scripts/audio.py analyse mix.wav

# QA (non-zero exit when a mandatory check fails)
python scripts/qa_check.py out.mp4 [--aspect 9:16] [--expect-res 720x1280] [--expect-fps 24] [--expect-duration 20] [--report qa.json]

# batch
python scripts/batch.py run specs/*.json --out batch_out --workers 2 --retries 1
python scripts/batch.py status|report|cancel|retry|resume batch_out

# tests
python -m pytest tests -q
```
