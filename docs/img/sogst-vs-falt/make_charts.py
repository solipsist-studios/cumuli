#!/usr/bin/env python3
"""Static SVG charts for docs/sogst-vs-falt.md. Light and dark tokens inside each SVG
(prefers-color-scheme), categorical slot 1 = .sogst, slot 2 = .falt (validated pair).
Values are the measured numbers from the comparison; edit DATA and re-run after a re-measure.
usage: make_report_charts.py <out dir>"""
import os, sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

# (label, size MB, PSNR dB, LPIPS)
SOGST = [("1.5 MB/s target", 6.81, 39.395, 0.01395), ("3.0 MB/s target", 13.55, 39.448, 0.01303),
         ("recommended packing", 22.33, 39.362, 0.01291), ("full bake", 30.39, 39.389, 0.01288)]
FALT = [("rate target", 6.37, 37.138, 0.01454), ("end to end", 8.36, 38.138, 0.01506),
        ("quality-first", 12.83, 37.088, 0.01466)]
FIRST = [(".sogst 1.5 MB/s", 4.98, 6.81), (".falt rate target", 0.89, 6.37)]  # MB before first frame, total
TRAIN = [(".sogst", 73), (".falt", 326)]                                        # minutes to a finished stream

STYLE = """<style>
svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e3e2de;--s1:#2a78d6;--s2:#eb6834;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#363532;--s1:#3987e5;--s2:#d95926}}
.bg{fill:var(--surface)}.t{fill:var(--ink);font-size:15px;font-weight:600}.sub{fill:var(--ink2);font-size:12px}
.ax{fill:var(--ink2);font-size:11px}.lab{fill:var(--ink);font-size:11.5px}.g{stroke:var(--grid);stroke-width:1}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.ring{stroke:var(--surface);stroke-width:2}.ln1{stroke:var(--s1)}.ln2{stroke:var(--s2)}
</style>"""


def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-labelledby="t d"><title id="t">{title}</title><desc id="d">{desc}</desc>{STYLE}'
            f'<rect class="bg" width="{w}" height="{h}" rx="8"/>{body}</svg>')


def legend(x, y):
    return (f'<circle cx="{x}" cy="{y}" r="5" class="s1"/><text x="{x+10}" y="{y+4}" class="lab">.sogst</text>'
            f'<circle cx="{x+70}" cy="{y}" r="5" class="s2"/><text x="{x+80}" y="{y+4}" class="lab">.falt</text>')


def scatter(fname, key, ylabel, ylo, yhi, ystep, fmt, title, sub, better_low=False, offsets=None):
    W, H, L, R, T, B = 720, 400, 64, 24, 72, 52
    X = lambda v: L + v / 32 * (W - L - R)                                     # noqa: E731
    Y = lambda v: T + (yhi - v) / (yhi - ylo) * (H - T - B)                    # noqa: E731
    o = [f'<text x="{L}" y="28" class="t">{title}</text><text x="{L}" y="48" class="sub">{sub}</text>', legend(W - 170, 30)]
    v = ylo
    while v <= yhi + 1e-9:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g"/>'
                 f'<text x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end" class="ax">{fmt(v)}</text>')
        v += ystep
    for x in range(0, 33, 4):
        o.append(f'<text x="{X(x):.1f}" y="{H-B+18}" text-anchor="middle" class="ax">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-12}" text-anchor="middle" class="ax">file size, MB (4.77 s clip)</text>')
    o.append(f'<text x="16" y="{(T+H-B)/2}" transform="rotate(-90 16 {(T+H-B)/2})" text-anchor="middle" class="ax">{ylabel}</text>')
    for pts, cls, ln in ((SOGST, "s1", "ln1"), (FALT, "s2", "ln2")):
        for p in pts:
            dx, dy, anc = (offsets or {}).get(p[0], (8, -9, "start"))
            o.append(f'<circle cx="{X(p[1]):.1f}" cy="{Y(p[key]):.1f}" r="5" class="{cls} ring">'
                     f'<title>{p[0]}: {p[1]:.2f} MB, PSNR {p[2]:.3f} dB, LPIPS {p[3]:.5f}</title></circle>'
                     f'<text x="{X(p[1])+dx:.1f}" y="{Y(p[key])+dy:.1f}" text-anchor="{anc}" class="lab">{p[0]}</text>')
    open(os.path.join(out, fname), "w").write(svg(W, H, "".join(o), title, sub))


scatter("size-vs-psnr.svg", 2, "PSNR, dB", 36.5, 40.0, 0.5, lambda v: f"{v:.1f}",
        "File size against quality: PSNR", "Same capture and views for both formats. Up and to the left is better.",
        offsets={"end to end": (8, 16, "start"), "quality-first": (8, 16, "start"), "rate target": (-8, 16, "end"),
                 "recommended packing": (0, -11, "middle"), "full bake": (0, 19, "middle")})
scatter("size-vs-lpips.svg", 3, "LPIPS", 0.0125, 0.0155, 0.0005, lambda v: f"{v:.4f}",
        "File size against quality: LPIPS", "LPIPS is perceptual distance. Down and to the left is better.",
        better_low=True, offsets={"rate target": (-8, 16, "end"), "1.5 MB/s target": (8, 16, "start"),
                                  "recommended packing": (0, -11, "middle"), "full bake": (-8, 16, "end")})


def hbars(fname, rows, unit, title, sub, maxv, ticks, stacked=False):
    W, L, R, T = 720, 150, 70, 72
    bh, gap = 24, 22
    H = T + len(rows) * (bh + gap) + 40
    X = lambda v: L + v / maxv * (W - L - R)                                   # noqa: E731
    o = [f'<text x="24" y="28" class="t">{title}</text><text x="24" y="48" class="sub">{sub}</text>']
    for t in ticks:
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T-6}" y2="{H-34}" class="g"/>'
                 f'<text x="{X(t):.1f}" y="{H-18}" text-anchor="middle" class="ax">{t}{unit}</text>')
    for i, r in enumerate(rows):
        y = T + i * (bh + gap)
        cls = "s1" if r[0].startswith(".sogst") else "s2"
        o.append(f'<text x="{L-10}" y="{y+bh/2+4}" text-anchor="end" class="lab">{r[0]}</text>')
        if stacked:
            first, total = r[1], r[2]
            o.append(f'<rect x="{L}" y="{y}" width="{X(total)-L:.1f}" height="{bh}" rx="4" class="{cls}" fill-opacity="0.28"/>'
                     f'<rect x="{L}" y="{y}" width="{max(X(first)-L-2,4):.1f}" height="{bh}" rx="4" class="{cls}">'
                     f'<title>{r[0]}: {first:.2f} MB before the first frame, {total:.2f} MB in total</title></rect>'
                     f'<text x="{X(total)+8:.1f}" y="{y+bh/2+4}" class="lab">{first:.2f} of {total:.2f} MB ({100*first/total:.0f}%)</text>')
        else:
            o.append(f'<rect x="{L}" y="{y}" width="{X(r[1])-L:.1f}" height="{bh}" rx="4" class="{cls}"><title>{r[0]}: {r[1]} min</title></rect>'
                     f'<text x="{X(r[1])+8:.1f}" y="{y+bh/2+4}" class="lab">{r[1] // 60} h {r[1] % 60:02d} min</text>')
    open(os.path.join(out, fname), "w").write(svg(W, H, "".join(o), title, sub))


hbars("first-frame.svg", FIRST, "", "Bytes needed before the first frame",
      "Solid: downloaded before anything can draw. Pale: the rest of the file.", 8, [0, 2, 4, 6, 8], stacked=True)
hbars("training-time.svg", TRAIN, "", "Training time to a finished stream",
      "Same capture, same GPU (RTX 5090). Excludes dataset preparation, which both share.", 360, [0, 60, 120, 180, 240, 300, 360])
print("wrote", sorted(os.listdir(out)))
