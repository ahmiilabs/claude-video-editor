# Clip effects (scripts/effects.py)  -  full list: `python scripts/effects.py --list`, examples: docs/CLI.md

Chain with commas. All run frame by frame (OpenCV + numpy), keep the source audio, and downscale clips wider than 720px for speed (`--max-width 0` keeps full size).

| effect | needs | key params |
|---|---|---|
| spotlight, punch-zoom, cinematic-zoom | opencv | `--strength`, `--every --zoom`, `--push`; follow the stabilised face tracker |
| color-grade | - | `--look warm|cool|cinematic|bw|vivid` |
| glass-widgets, split-chart | - | `--widgets "36|Followers;197K|Reach"`, `--data "Jan:20,Feb:45" --title` |
| captions | - | `--text` (estimated) or `--captions-from-audio` (real transcription), `--caption-style highlight|karaoke|plain`, `--caption-follow subject`, `--srt` |
| face-crop | opencv | `--aspect 9:16`; must be FIRST |
| bg-replace, bg-blur, text-behind, smart-spotlight | matting model | `--bg`, `--bg-sigma`, `--title-text/--title-color/--title-size/--title-y`, `--strength` |
| bg-remove | matting model | `--alpha-format mov|webm|png`; must be alone |

Matting = Robust Video Matting (ONNX, CPU). Install: `python scripts/setup_models.py --matting`. Tracking = OpenCV Haar (classic), stabilised with association, smoothing, dead-zone and hold; the matte also provides a head position when no face is found.
Without the model, matte effects stop with the install command; nothing is faked.
