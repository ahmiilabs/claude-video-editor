# claude.ai web: commands

On claude.ai you do not type CLI commands: you chat, and Claude runs them. Part 1 = what you can say. Part 2 = what Claude runs (also usable as-is in Claude Code / terminal; replace `S=` with the skill folder).

## Install the skill (once)
claude.ai > Settings > Features (Capabilities) > Skills > upload `ai-video-editor.skill` (a zip with the folder `ai-video-editor/` and `SKILL.md` inside). Needs code execution / file creation enabled and a plan that supports custom skills (Pro, Max, Team, Enterprise). Skills are per surface: upload again in Claude Code if you use it there. Start a new chat after uploading.

## Part 1: things to say in the chat
| Goal | Say (Roman Urdu / English both work) |
|---|---|
| Reel from an idea | `reel banao: train mein chai itni mehngi kyun? 30 sec, Roman Urdu, natural voice, comic style` |
| With my voice | (upload voice note) `meri awaaz pe reel banao: <topic/script>`, captions karaoke style |
| Captions on my video | (upload video) `is video pe karaoke captions lagao, speaker ke neeche follow karein` |
| Spotlight / zoom / grade | `is clip pe spotlight aur cinematic color grade lagao` |
| Text behind me | `is clip mein "FOCUS" text mere peeche lagao (pehle 8 sec)` |
| New background | `background gradient orange-dark kar do` · `background blur kar do` · `transparent video nikalo` |
| Stats / chart | `clip pe glass widgets lagao: 36 Followers, 197K Reach` · `neeche clip upar bar chart Jan 20, Feb 45, Mar 80` |
| Vertical crop | `is landscape video ko 9:16 mein face follow karke crop karo` |
| Transcript | `is video ka transcript aur SRT do` |
| My images as scenes | (upload images) `in images ko scenes bana ke reel banao, ken-burns motion` |
| Tweak | `caption bada karo` · `scene 3 chhota` · `sketch style me karo` · `music halka` · `transition cinematic` |
| Check | `QA report dikhao` |
| Long reel | `60 sec ki reel banao` (Claude renders parts and joins them) |

## Part 2: what Claude runs
```bash
S=/mnt/skills/user/ai-video-editor
# 0. setup (once per chat)            flags: --voice  --matting  --whisper tiny|base  --all  --espeak
python3 $S/scripts/web_setup.py --voice
python3 $S/scripts/env_check.py
# 1. reel
python3 $S/scripts/reel.py spec.json --preview
python3 $S/scripts/reel.py spec.json --out /mnt/user-data/outputs/reel.mp4 [--quality draft] [--fps 24] [--no-captions] [--no-qa]
python3 $S/scripts/pipeline.py --spec spec.json --out /mnt/user-data/outputs/reel.mp4 --workdir /tmp/run [--skip transcribe] [--post-effects "--effect color-grade --look warm"]
python3 $S/scripts/join.py part1.mp4 part2.mp4 --out /mnt/user-data/outputs/full.mp4
# 2. clip effects (use --start/--duration to keep within time limits)
python3 $S/scripts/effects.py --list
python3 $S/scripts/effects.py IN.mp4 --effect spotlight,color-grade --look cinematic --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect punch-zoom --every 2 --zoom 1.15 --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect glass-widgets --widgets "36|Followers;197K|Reach" --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect split-chart --data "Jan:20,Feb:45,Mar:80" --title Growth --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect cinematic-zoom --push 1.2 --out OUT.mp4
python3 $S/scripts/effects.py WIDE.mp4 --effect face-crop --aspect 9:16 --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect captions --captions-from-audio --caption-style karaoke --caption-follow subject --srt cap.srt --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect text-behind --title-text "FOCUS" --title-color "#FFD740" --start 1 --duration 8 --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect bg-replace --bg "gradient:#ff6b35:#1a1a2e" --start 1 --duration 8 --out OUT.mp4   # blur:25 | color:#hex | image:p.jpg | video:p.mp4
python3 $S/scripts/effects.py IN.mp4 --effect bg-blur --bg-sigma 20 --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect smart-spotlight --strength 0.75 --out OUT.mp4
python3 $S/scripts/effects.py IN.mp4 --effect bg-remove --alpha-format webm --start 1 --duration 6 --out person.webm      # mov | webm | png
# 3. transcription / audio
python3 $S/scripts/transcribe.py IN.mp4 --lang en --srt out.srt --ass out.ass --json out.json --style karaoke
python3 $S/scripts/audio.py enhance --in raw.wav --out clean.wav
python3 $S/scripts/audio.py mix --voice v.wav --music bed.wav --sfx whoosh.wav@2.5 --seconds 20 --out mix.wav
python3 $S/scripts/audio.py analyse mix.wav
# 4. QA and batch
python3 $S/scripts/qa_check.py OUT.mp4 --aspect 9:16 --report /tmp/qa.json
python3 $S/scripts/batch.py run /tmp/specs/*.json --out /tmp/batch --workers 1
python3 $S/scripts/batch.py status|report|cancel|retry|resume /tmp/batch
```
Not usable on web (blocked hosts): `scriptwriter.py`, `imagegen.py`, `videogen.py`, `--engine openai-api`. Claude writes the spec itself and uses your uploaded images/videos.

## Web limits in one line each
1 CPU / draft 720p only / keep each reel <= 35 s per run (longer = parts + `join.py`) / matte effects and Whisper on trimmed clips (`--start --duration`) / models re-download each new chat (`--all` ~70 s) / outputs must be in `/mnt/user-data/outputs/` / the skill folder is read-only.
