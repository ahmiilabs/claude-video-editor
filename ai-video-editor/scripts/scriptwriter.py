#!/usr/bin/env python3
"""AI script writer: topic -> hook, narration, scenes, visuals, on-screen text, transitions, music, image/video prompts -> a spec that reel.py renders.

  python scripts/scriptwriter.py --topic "Why is chai expensive on trains" --language roman-urdu --duration 30 --platform reels --out spec.json
  python scripts/scriptwriter.py --topic "..." --prompt-only --out prompt.txt        # offline/manual workflow: paste into any chat model
  python scripts/scriptwriter.py --import llm_answer.json --out spec.json              # validate + convert a manually obtained answer

LLM providers (config.py / .env): `openai` = any OpenAI-compatible /chat/completions server, `anthropic` = Messages API. No provider
configured -> a clear error with the offline workflow; nothing is invented. Languages: en, ur (Urdu), roman-urdu, hi, multi.
"""
import argparse, json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
from net import http_json, ProviderError
from common import get_logger

log = get_logger("aive.script")

class ScriptError(RuntimeError):
    pass

PLATFORMS = {  # name -> (aspect, max seconds, notes)
    "reels": ("9:16", 90, "Instagram Reels"), "instagram": ("9:16", 90, "Instagram Reels"), "tiktok": ("9:16", 180, "TikTok"), "shorts": ("9:16", 60, "YouTube Shorts"),
    "youtube": ("16:9", 600, "YouTube"), "facebook": ("1:1", 120, "Facebook feed"), "ad": ("1:1", 60, "Social ad"), "linkedin": ("1:1", 120, "LinkedIn"),
}
LANGS = {"en": ("English", "en-us", 2.6), "ur": ("Urdu", "hi", 2.3), "roman-urdu": ("Roman Urdu (Urdu written in Latin letters)", "hi", 2.3),
         "hi": ("Hindi", "hi", 2.3), "multi": ("mixed languages as requested", "en-us", 2.4)}
STYLE_BY_CATEGORY = {"education": "clean", "explainer": "clean", "tech": "blueprint", "finance": "blueprint", "business": "blueprint", "story": "sketch",
                     "comedy": "comic", "entertainment": "comic", "marketing": "kinetic", "motivation": "kinetic", "news": "kinetic", "product": "kinetic"}
TRANSITION_PRESET = {"energetic": "dynamic", "cinematic": "cinematic", "calm": "smooth", "professional": "clean", "playful": "social", "funny": "social"}
MUSIC_KIND = {"energetic": "upbeat", "cinematic": "tense", "calm": "calm", "professional": "calm", "playful": "upbeat", "funny": "upbeat", "warm": "warm"}

def build_prompt(topic, audience="general audience", language="en", duration=30, platform="reels", tone="energetic", category="education", scenes=None):
    lang_name = LANGS.get(language, LANGS["en"])[0]; wps = LANGS.get(language, LANGS["en"])[2]
    n = scenes or max(4, min(9, round(duration / 5))); words = int(duration * wps * 0.92)
    needs_deva = language in ("ur", "roman-urdu", "hi")
    system = ("You are a senior short-form video scriptwriter. You write tight, visual, hook-first scripts. "
              "Reply with ONE JSON object only (no markdown, no commentary).")
    schema = {
        "title": "string", "hook": "first spoken line, under 10 words, curiosity or surprise",
        "music_suggestion": "mood + genre description (do not name copyrighted songs)", "visual_style_anchor": "one sentence describing a consistent look for ALL generated images (palette, medium, lighting, character)",
        "scenes": [{
            "text": "on-screen text, 3-9 words", "narration": "spoken line shown as captions" + (", written in " + ("Latin letters (Roman)" if language in ("roman-urdu", "ur", "hi") else lang_name)),
            **({"voice_text": "the same spoken line written in Devanagari script for the text-to-speech voice"} if needs_deva else {}),
            "keywords": ["1-2 words from text to highlight"], "duration_estimate_seconds": "number", "visual_description": "what the viewer sees",
            "image_prompt": "self-contained prompt for an image generator, no on-screen text in the image", "video_prompt": "motion/camera description for a 3-5 second clip",
            "transition": "one of: fade, zoom, slide_left, wipe_left, blur, iris, light_leak, whip, glitch, none", "sound_effect": "optional: whoosh | pop | riser | ding | none"}],
    }
    user = (f"Write a {duration}-second video script.\nTopic: {topic}\nAudience: {audience}\nLanguage: {lang_name}\nPlatform: {PLATFORMS.get(platform, PLATFORMS['reels'])[2]}\n"
            f"Tone: {tone}\nCategory: {category}\n\nRules:\n- About {n} scenes; total spoken words about {words} (speech runs ~{wps} words/second).\n"
            "- Scene 1 is the hook. The last scene is a call to action.\n- One idea per scene. No filler. No claims you cannot support; if a number is uncertain, phrase it generally.\n"
            "- text is for the screen, narration is for the voice; they must not be identical sentences.\n"
            + ("- Write narration in Roman letters (Latin script) AND voice_text in Devanagari for each scene.\n" if needs_deva else "")
            + "\nJSON schema (follow exactly):\n" + json.dumps(schema, ensure_ascii=False, indent=2))
    return system, user

# ------------------------------------------------------------------ LLM adapters
class OpenAICompatLLM:
    def __init__(self, p): self.p = p
    def chat(self, system, user, temperature=0.8, max_tokens=3000):
        h = {"Authorization": f"Bearer {self.p.api_key}"} if self.p.api_key else {}
        body = {"model": self.p.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": temperature, "max_tokens": max_tokens}
        j = http_json("POST", self.p.base_url + "/chat/completions", h, body, self.p.timeout, self.p.retries)
        try: return j["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError): raise ProviderError("LLM response had no choices[0].message.content")

class AnthropicLLM:
    def __init__(self, p): self.p = p
    def chat(self, system, user, temperature=0.8, max_tokens=3000):
        h = {"x-api-key": self.p.api_key, "anthropic-version": "2023-06-01"}
        body = {"model": self.p.model, "system": system, "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": temperature}
        j = http_json("POST", self.p.base_url.rstrip("/") + "/messages", h, body, self.p.timeout, self.p.retries)
        try: return "".join(b.get("text", "") for b in j["content"] if b.get("type") == "text")
        except (KeyError, TypeError): raise ProviderError("Anthropic response had no content text")

def get_llm():
    p = config.get_provider("llm")
    if p.name in ("none", ""): raise ScriptError(OFFLINE_HELP)
    try: p.validate()
    except config.ConfigError as e: raise ScriptError(f"{e}\n\n{OFFLINE_HELP}")
    return {"openai": OpenAICompatLLM, "anthropic": AnthropicLLM}.get(p.name, lambda _: (_ for _ in ()).throw(ScriptError(f"unknown LLM provider '{p.name}' (use openai | anthropic | none)")))(p)

OFFLINE_HELP = ("No LLM is configured, so the script cannot be generated automatically. Offline/manual workflow:\n"
                "  1. python scripts/scriptwriter.py --topic \"...\" --prompt-only --out prompt.txt\n"
                "  2. Paste prompt.txt into any chat model and save its JSON answer as answer.json\n"
                "  3. python scripts/scriptwriter.py --import answer.json --out spec.json\n"
                "Or write spec.json by hand (see references/spec-format.md). To enable automatic writing set AIVE_LLM_PROVIDER / AIVE_LLM_MODEL / AIVE_LLM_API_KEY (see .env.example).")

# ------------------------------------------------------------------ parsing + validation
def extract_json(text):
    t = text.strip(); m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m: t = m.group(1).strip()
    try: return json.loads(t)
    except ValueError: pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try: return json.loads(t[a:b + 1])
        except ValueError as e: raise ScriptError(f"could not parse the model's JSON: {e}")
    raise ScriptError("the model reply contained no JSON object")

def to_spec(script, language="en", platform="reels", tone="energetic", category="education", duration=30, quality=None):
    """Validate a script object and convert it to a renderable spec. Raises ScriptError with every problem found."""
    problems = []
    scenes_in = script.get("scenes") if isinstance(script, dict) else None
    if not isinstance(scenes_in, list) or not scenes_in: raise ScriptError("script has no 'scenes' list")
    lang_name, tts_lang, wps = LANGS.get(language, LANGS["en"]); aspect, maxdur, _ = PLATFORMS.get(platform, PLATFORMS["reels"])
    style = STYLE_BY_CATEGORY.get(category, "kinetic"); scenes = []; total = 0.0
    for i, s in enumerate(scenes_in, 1):
        if not isinstance(s, dict): problems.append(f"scene {i} is not an object"); continue
        text = (s.get("text") or "").strip(); nar = (s.get("narration") or "").strip()
        if not (text or nar): problems.append(f"scene {i}: needs text or narration"); continue
        if language in ("ur", "roman-urdu", "hi") and nar and not (s.get("voice_text") or "").strip(): problems.append(f"scene {i}: voice_text (Devanagari) is required for language '{language}'")
        d = s.get("duration_estimate_seconds")
        try: d = float(d)
        except (TypeError, ValueError): d = max(1.6, len(nar.split()) / wps + 0.4)
        total += d
        sc = {"text": text or nar, "narration": nar or text, "keywords": [k for k in (s.get("keywords") or []) if isinstance(k, str)][:3]}
        if s.get("voice_text"): sc["voice_text"] = s["voice_text"].strip()
        for k in ("visual_description", "image_prompt", "video_prompt"):
            if s.get(k): sc[k] = str(s[k]).strip()
        tr = (s.get("transition") or "").strip()
        if tr and tr not in ("", "none"): sc["transition"] = tr
        if i == len(scenes_in): sc["label"] = "// your turn" if style == "clean" else sc.get("label", "")
        se = s.get("sound_effect")
        if se in ("whoosh", "pop", "riser", "ding"): sc["sfx_hint"] = se
        scenes.append(sc)
    if problems: raise ScriptError("script problems:\n  - " + "\n  - ".join(problems))
    if total > maxdur: problems.append(f"estimated {total:.0f}s exceeds {platform} limit {maxdur}s")
    if abs(total - duration) > max(6, duration * 0.35): log.warning("script is about %.0fs but %ds was requested", total, duration)
    mood = TRANSITION_PRESET.get(tone, "smooth")
    spec = {"title": script.get("title") or "Untitled", "style": style, "accent": "#FF6B35", "aspect": aspect,
            "voice": {"mode": "auto", "lang": tts_lang, "voice": "hf_alpha" if tts_lang == "hi" else "af_heart", "speed": 1.0},
            "captions": {"enabled": True, "style": "highlight", "timing": "auto"}, "transitions": {"preset": mood, "duration": 0.45},
            "audio": {"music": "synth:" + MUSIC_KIND.get(tone, "calm"), "music_volume": 0.14, "duck": True, "sfx_on_transitions": "whoosh", "sfx_volume": 0.4},
            "scenes": scenes, "script_meta": {"hook": script.get("hook"), "music_suggestion": script.get("music_suggestion"), "visual_style_anchor": script.get("visual_style_anchor"),
                                              "language": language, "platform": platform, "tone": tone, "category": category, "estimated_seconds": round(total, 1), "warnings": problems}}
    if quality: spec["quality"] = quality
    return spec

def generate(topic, language="en", duration=30, platform="reels", tone="energetic", category="education", audience="general audience", scenes=None, llm=None, attempts=3):
    llm = llm or get_llm(); system, user = build_prompt(topic, audience, language, duration, platform, tone, category, scenes); last = None
    for k in range(attempts):
        txt = llm.chat(system, user if k == 0 else user + f"\n\nYour previous answer was rejected: {last}\nReturn corrected JSON only.")
        try: return to_spec(extract_json(txt), language, platform, tone, category, duration)
        except ScriptError as e: last = str(e); log.warning("script attempt %d rejected: %s", k + 1, last.splitlines()[0])
    raise ScriptError(f"the model failed to produce a valid script after {attempts} attempts: {last}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic"); ap.add_argument("--audience", default="general audience"); ap.add_argument("--language", default="en", choices=list(LANGS)); ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--platform", default="reels", choices=list(PLATFORMS)); ap.add_argument("--tone", default="energetic"); ap.add_argument("--category", default="education"); ap.add_argument("--scenes", type=int)
    ap.add_argument("--out", default="spec.json"); ap.add_argument("--prompt-only", action="store_true"); ap.add_argument("--import", dest="imp")
    a = ap.parse_args()
    try:
        if a.imp:
            spec = to_spec(extract_json(Path(a.imp).read_text(encoding="utf-8")), a.language, a.platform, a.tone, a.category, a.duration)
        elif a.prompt_only:
            if not a.topic: raise ScriptError("--topic is required")
            s, u = build_prompt(a.topic, a.audience, a.language, a.duration, a.platform, a.tone, a.category, a.scenes)
            Path(a.out).write_text(f"SYSTEM:\n{s}\n\nUSER:\n{u}\n", encoding="utf-8"); print(f"prompt written to {a.out}"); return
        else:
            if not a.topic: raise ScriptError("--topic is required")
            spec = generate(a.topic, a.language, a.duration, a.platform, a.tone, a.category, a.audience, a.scenes)
    except (ScriptError, ProviderError) as e:
        print(f"SCRIPT ERROR: {config.redact(e)}", file=sys.stderr); sys.exit(1)
    Path(a.out).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"); print(f"spec written to {a.out}  ({len(spec['scenes'])} scenes, ~{spec['script_meta']['estimated_seconds']}s)")

if __name__ == "__main__":
    main()
