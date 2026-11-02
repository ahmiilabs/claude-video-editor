---
name: ai-video-editor
description: Make and edit short vertical videos (Reels, Shorts, TikToks, explainers, ads, faceless videos) entirely with code, no editing app. Use whenever the user wants to turn an idea, topic or script into a video ("reel banao", "video banao", "explainer", "kinetic text / comic / sketch / blueprint style video"), write a video script, add captions, subtitles, voiceover, music, transitions or motion graphics to a video, transcribe a video, remove or replace a background, put text behind a person, or apply effects to a clip they uploaded (spotlight, zoom, colour grade, glass stat widgets, chart split-screen). Also for "edit this video", "AI video editor", "/reel", "/effects". Built for claude.ai web (code execution) and also works in Claude Code and Cowork with Sonnet, Opus or Fable.
---

# AI Video Editor (claude.ai web first)

You are the script writer AND the operator: write the spec (JSON) yourself, run the scripts, check QA, deliver the MP4. No LLM/API key is needed. Reply in the user's language and register (Roman Urdu/Hinglish in -> Roman Urdu/Hinglish out). Keep tool output short (`| tail -15`).

```bash
S=/mnt/skills/user/ai-video-editor      # skill folder (read-only). If it does not exist: S=$(dirname $(dirname $(find / -name web_setup.py 2>/dev/null | head -1)))
```
Define `S` at the top of every bash call (shell state may not persist). Uploads: `/mnt/user-data/uploads/`. Outputs: write to `/mnt/user-data/outputs/`, then call `present_files`. Scratch files: `/tmp`. Never write inside `$S`.

## Step 0: once per conversation (containers start empty)
```bash
S=/mnt/skills/user/ai-video-editor; python3 $S/scripts/web_setup.py --voice          # pick only what the job needs, see table
python3 $S/scripts/env_check.py | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['features']);print(d['web_notes'])"
```
| Job needs | web_setup flags | time |
|---|---|---|
| silent reel / user's own audio / basic effects | (none) | ~30 s |
| natural voice | `--voice` (350 MB) | ~40 s |
| background replace/blur/remove, text-behind | `--matting` (15 MB) | ~10 s |
| transcribe a clip's speech | `--whisper tiny` (116 MB) | ~15 s |
| everything | `--all` | ~70 s |
Never promise a feature that `env_check` shows `false`. If setup reports `FAILED`, say so and use the fallback (own audio / espeak via `--espeak` / skip the effect).

## Web limits (design around them)
- **1 CPU, ~4 GB RAM, one command must finish in a few minutes.** Draft quality (720p) only. `reel.py` prints an estimate; aim for **<= 35 s of video per command**. Longer: render parts, then `python3 $S/scripts/join.py p1.mp4 p2.mp4 --out full.mp4`.
- Matte effects (~0.2 s/frame) and Whisper: work on a trimmed clip: add `--start 2 --duration 8` to `effects.py`.
- **Network**: only GitHub, PyPI and api.anthropic.com are reachable. OpenAI/Hugging Face/video-generation APIs are NOT, so `imagegen.py`, `videogen.py`, `scriptwriter.py` and cloud transcription do not work here (you are the writer). Visuals = the user's own images/videos (`scene.image`/`scene.video`, auto Ken-Burns) or the built-in style packs. Do not invent AI-generated media.
- Skills do not sync: the user must have uploaded this skill on claude.ai (Settings > Features/Capabilities, code execution on).

## Job A: reel from an idea
1. **Spec** (see `docs/SPEC.md`; start from `$S/assets/example-spec.json` or `demo-spec.json`). Hook in scene 1 (<8 words); one idea/scene; `text` 3-9 words; `narration` = spoken line; 1-2 `keywords`; last scene = call to action; 5-7 scenes for ~25-30 s. Styles: kinetic (hooks), clean (steps), blueprint (breakdowns), comic (humour), sketch (stories). Polish: `"transitions": {"preset": "dynamic"}`, `"audio": {"music": "synth:calm", "duck": true, "sfx_on_transitions": "whoosh"}`, `"progress_bar": {"enabled": true}`, `"captions": {"style": "karaoke"}`.
2. **Voice**: user's recording (`"voice": {"mode": "file", "file": "/mnt/user-data/uploads/x.m4a"}`, voice notes/ogg/opus work) > neural (`--voice`) > espeak > silent. Hindi/Urdu/Hinglish: `narration` Roman (captions) + `voice_text` **Devanagari** (spoken), else it sounds English-y. Say plainly that espeak is robotic.
3. `python3 $S/scripts/reel.py spec.json --preview` -> `view /tmp/reel_work/preview.png`; fix overflow.
4. `python3 $S/scripts/reel.py spec.json --out /mnt/user-data/outputs/reel.mp4 2>&1 | tail -20` (renders, captions, mixes audio, muxes, QA; writes `.render.json`/`.qa.json` beside it; remove those two helper files from outputs before presenting if you do not want to show them).
5. **Verify**: QA says PASSED; extract 1-2 frames (`ffmpeg -ss 3 -i out.mp4 -frames:v 1 /tmp/f.png`) and `view` them. Re-render after fixes; never report success on a failed QA.
6. `present_files` the MP4, then 2-3 lines: style, length, voice type, caption timing mode (`tts` = audio-aligned), anything that fell back.

## Job B: clip effects (uploaded video)
`python3 $S/scripts/effects.py --list`. Examples: `--effect spotlight,color-grade --look cinematic` · `--effect text-behind --title-text "FOCUS" --start 1 --duration 8` · `--effect bg-replace --bg "gradient:#ff6b35:#1a1a2e"` · `--effect captions --captions-from-audio --caption-style karaoke --caption-follow subject --srt /mnt/user-data/outputs/c.srt`. `face-crop` must be first in a chain; `bg-remove` must be alone (big `.mov`: prefer `--alpha-format webm`/`png` and keep clips short). Then `qa_check.py`, `view` a frame, `present_files`.

## Other jobs
Transcript/SRT of a clip: `transcribe.py clip.mp4 --lang en --srt out.srt` (needs `--whisper`; reports `granularity`). Several videos: one spec each, `batch.py run specs/*.json --out /tmp/batch --workers 1`. Full command list for users and Claude: `docs/WEB.md`; every flag: `docs/CLI.md`; fixes: `docs/TROUBLESHOOTING.md`.

## Honesty rules (do not break)
- Whisper (sherpa) gives real sentence times; word times are interpolated (`approx`). True word-level needs faster-whisper/API, not available on web. Whisper `tiny` is weak on Hindi/Urdu; for a generated voice the script is known, so captions are audio-aligned instead.
- Urdu script is not rendered on screen (use Roman Urdu). `synth:` music is a placeholder pad; ask for the user's music. No copyrighted music/logos/characters.
- Mock-tested only: LLM/image/video/cloud-transcription adapters. Never claim they ran here.
- Never print or ask for API keys on web; all output is redacted anyway. Keep important text out of the top 12% / bottom 18% (app UI).
- Feedback ("caption bada karo", "comic style", "scene 3 chhota") = edit the spec and re-render; voice audio is cached so visual tweaks are fast.
