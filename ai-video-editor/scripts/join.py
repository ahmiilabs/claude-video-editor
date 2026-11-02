#!/usr/bin/env python3
"""Join several rendered videos into one (re-encodes to a common size/fps/audio so the parts always match).
Use it to build a long reel from parts that each fit in one claude.ai command (about 35 s each).

  python scripts/join.py part1.mp4 part2.mp4 part3.mp4 --out full.mp4 [--fade 0.0]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import run, ffprobe, pick_encoder

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+"); ap.add_argument("--out", required=True); a = ap.parse_args()
    for p in a.parts:
        if not os.path.exists(p): sys.exit(f"JOIN ERROR: file not found: {p}")
    v = [s for s in ffprobe(a.parts[0])["streams"] if s["codec_type"] == "video"][0]; w, h = int(v["width"]), int(v["height"]); num, den = v["r_frame_rate"].split("/"); fps = round(float(num) / float(den))
    cmd = ["ffmpeg", "-y", "-v", "error"]; flt = []
    for i, p in enumerate(a.parts):
        cmd += ["-i", p]; has_a = any(s["codec_type"] == "audio" for s in ffprobe(p)["streams"])
        flt.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v{i}]")
        flt.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[a{i}]" if has_a else f"anullsrc=r=44100:cl=stereo,atrim=0:{float(ffprobe(p)['format']['duration']):.3f}[a{i}]")
    flt.append("".join(f"[v{i}][a{i}]" for i in range(len(a.parts))) + f"concat=n={len(a.parts)}:v=1:a=1[v][a]")
    cmd += ["-filter_complex", ";".join(flt), "-map", "[v]", "-map", "[a]", *pick_encoder(), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", a.out]
    run(cmd); print(a.out)

if __name__ == "__main__":
    main()
