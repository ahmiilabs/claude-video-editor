# Style packs

| style | look | best for | notes |
|---|---|---|---|
| `kinetic` | dark gradient, words pop in one by one, keywords in accent pills | hooks, punchy facts, quotes | Default. Keep text under 9 words. |
| `clean` | light dotted page, white card with thick outline, number badge, progress dots | step-by-step, tutorials, how-tos | Use `label` and `badge`. |
| `blueprint` | blue grid, mono typewriter text, boxes that draw in | breakdowns, finance, systems, "how it works" | Use `boxes` (max 5). |
| `comic` | halftone yellow, speech bubble, optional sticker | humour, storytelling, reactions | Use `sfx` for a punchline word. |
| `sketch` | paper texture, hand-drawn border, ink reveal, red underline | stories, emotional or explainer pieces | Underlines `keywords` after the text appears. |

Mixing: change `style` per scene. Good pattern: kinetic hook, 2-3 body scenes in one style, clean/kinetic CTA.

Add a style: write `def mystyle(t, dur, scene, ctx) -> PIL.Image` in `scripts/styles.py`, register it in `STYLES`. `ctx` has `W, H, index, total, accent, cache` (cache pre-rendered backgrounds there for speed). Fonts bundled in `assets/fonts` (Poppins, Anton, Bangers, Caveat, JetBrains Mono, all OFL).
