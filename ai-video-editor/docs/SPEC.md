# JSON specification (backward compatible: old specs render exactly as before)

Minimal:
```json
{"style": "kinetic", "voice": {"mode": "auto", "lang": "en-us"}, "scenes": [{"text": "Hello world", "narration": "Hello world, welcome."}]}
```
Full example using the new features:
```json
{
  "title": "Demo", "style": "kinetic", "accent": "#FF6B35", "aspect": "9:16", "quality": "draft",
  "voice": {"mode": "auto", "lang": "hi", "voice": "hf_alpha", "speed": 1.0},
  "captions": {"enabled": true, "style": "karaoke", "timing": "auto", "position": "lower", "srt": true},
  "transitions": {"preset": "dynamic", "duration": 0.45},
  "audio": {"music": "synth:calm", "music_volume": 0.2, "duck": true, "sfx_on_transitions": "whoosh", "fade_out": 1.2, "target_lufs": -15, "enhance": "auto"},
  "progress_bar": {"enabled": true, "position": "top", "color": "#FFD740"},
  "scenes": [
    {"text": "Train mein chai itni mehngi kyun?", "narration": "Train mein chai itni mehngi kyun?", "voice_text": "ट्रेन में चाय इतनी महंगी क्यों?",
     "keywords": ["chai"], "image": "photo.jpg", "kenburns": "in", "overlays": [{"type": "scene_title", "text": "Chai", "at": 0.2, "duration": 2}]},
    {"text": "Costs", "narration": "...", "style": "blueprint", "boxes": ["Licence", "Rent"], "transition": {"type": "light_leak", "duration": 0.6},
     "video": "clip.mp4", "audio": {"file": "ding.wav", "at": 0.5, "volume": 0.8},
     "overlays": [{"type": "stat", "value": 2400000, "label": "Views"}, {"type": "chart", "kind": "bar", "data": {"A": 1, "B": 2}, "title": "Mix"},
                  {"type": "lower_third", "text": "Name", "subtext": "Title"}, {"type": "glass", "text": "Key point", "position": "bottom"}]}
  ]
}
```
**Fields.** Scene: `text`, `narration`, `voice_text`, `keywords`, `style` (kinetic|clean|blueprint|comic|sketch), `duration`, `image`, `video`, `kenburns` (in|out|left|right|up|down), `media_scrim` (0-0.9), `transition`, `overlays`, `audio`, `label`, `badge`, `boxes`, `sfx`, `accent`, plus generator hints `image_prompt`, `video_prompt`, `visual_description`.
**Transitions** (17): fade, zoom, slide_left/right/up/down, wipe_left/right/up/down, blur, iris, diagonal, light_leak, whip, glitch (+ none). Presets: none, smooth, clean, dynamic, cinematic, social. Duration is clamped to 40% of the shorter neighbouring scene so audio stays in sync. A spec without `transitions` keeps the legacy hard-cut-with-fade-in.
**Captions `timing`**: `auto` (TTS voice -> audio-aligned; user audio -> whisper if installed; else estimate) | `tts` | `whisper` | `estimate`. Sidecar `<out>.render.json` records which was used.
**Audio**: `music` is a path or `synth:calm|warm|tense|upbeat` (synthetic placeholder; use real music for production). Ducking is sidechain compression keyed on a level-normalised copy of the voice.
**Voice modes**: auto | neural | file | espeak | silent. Hindi voice reads Devanagari: put it in `voice_text`.
