"""Side-by-side GIF of traced v3 episodes with the estimator's per-box readout and the choice overlaid."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = Path(sys.argv[1])           # trace dir
OUT = Path(sys.argv[2])         # output gif
SEEDS = [int(s) for s in sys.argv[3:]]
SCALE, HOLD, FPS = 1.5, 40, 15  # HOLD frames ~2.7 s of still frames at the end
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f_cap, f_txt, f_bold, f_tag = (ImageFont.truetype(BOLD, 15), ImageFont.truetype(FONT, 13),
                               ImageFont.truetype(BOLD, 13), ImageFont.truetype(BOLD, 12))
GREEN, RED = (40, 160, 70), (200, 50, 40)


def panel(img, x, r, final):
    W = int(img.shape[1] * SCALE)
    im = Image.fromarray(img).resize((W, W), Image.LANCZOS)
    cap = Image.new("RGB", (W, 26), "white")
    ImageDraw.Draw(cap).text((6, 5), f'"push the {r["goal"]}"', font=f_cap, fill="black")
    canvas = Image.new("RGB", (W, W + 26), "white")
    canvas.paste(cap, (0, 0))
    canvas.paste(im, (0, 26))
    dr = ImageDraw.Draw(canvas, "RGBA")
    chosen = x["chosen"]
    # number tags beside each box
    for i, (u, v) in enumerate(x["box_px"]):
        u, v = u * SCALE + 18, v * SCALE + 26 - 8
        col = (GREEN if chosen == r["truth"] else RED) if i == chosen else (30, 30, 30)
        dr.rounded_rectangle([u - 1, v - 1, u + 13, v + 14], 3, fill=col + (230,))
        dr.text((u + 3, v), str(i + 1), font=f_tag, fill="white")
    # estimator readout, bottom left
    lines = [("estimator", f_bold, "black")]
    for i, (probed, logm, sd) in enumerate(x["summary"]):
        if probed:
            lines.append((f"box {i + 1}: {math.exp(logm):.2f} kg ±{100 * (math.exp(sd) - 1):.0f}%", f_txt, "black"))
    if len(lines) == 1:
        lines.append(("probing…", f_txt, (90, 90, 90)))
    if chosen >= 0:
        ok = chosen == r["truth"]
        lines.append((f"→ chose box {chosen + 1} " + ("✓" if ok else "✗"), f_bold, GREEN if ok else RED))
    if final:
        lines.append(("true: " + " / ".join(f"{m:.2f}" for m in r["masses"]) + " kg", f_txt, (70, 70, 70)))
    h = 6 + 17 * len(lines)
    w = max(dr.textlength(t, font=f) for t, f, _ in lines) + 14
    y0 = canvas.height - h - 6
    dr.rounded_rectangle([6, y0, 6 + w, y0 + h], 5, fill=(255, 255, 255, 215))
    for k, (t, f, c) in enumerate(lines):
        dr.text((13, y0 + 4 + 17 * k), t, font=f, fill=c)
    return canvas


eps = []
for s in SEEDS:
    r = json.loads((S / f"seed{s}.json").read_text())
    eps.append((np.load(S / f"seed{s}_frames.npy"), r))
n = max(len(f) for f, _ in eps) + HOLD
tmp = S / "ovl_frames"
tmp.mkdir(exist_ok=True)
for p in tmp.glob("*.png"):
    p.unlink()
for k in range(n):
    panels = []
    for frames, r in eps:
        j = min(k, len(frames) - 1)
        panels.append(panel(frames[j], r["trace"][j], r, final=k >= len(frames) - 1))
    row = Image.new("RGB", (sum(p.width for p in panels) + 4 * (len(panels) - 1), panels[0].height), "white")
    xo = 0
    for p in panels:
        row.paste(p, (xo, 0))
        xo += p.width + 4
    row.save(tmp / f"{k:04d}.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "%04d.png"), "-filter_complex",
                "fps=10,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                str(OUT)], check=True)
print(OUT, OUT.stat().st_size // 1024, "KB,", n, "frames")
