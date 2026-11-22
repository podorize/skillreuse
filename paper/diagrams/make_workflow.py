"""Workflow diagram of the study, one band per section of the paper (3.1 to 3.8).

Run: uv run python paper/diagrams/make_workflow.py paper/diagrams
(writes workflow.png, workflow.svg and workflow.pdf)

Coordinates are in pixels of a 1000-pixel-wide drawing (10 in at 100 dpi); y grows downwards.
Placed at 15 cm of width in the paper, 15 px of text is about 6.4 pt.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1])
matplotlib.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"  # keep the text editable in the SVG
PX = 0.72  # points per pixel at 100 dpi

INK, LINE = "#1d2127", "#4a5260"
BAND, WHITE, EXTERNAL, HOLDOUT = "#eceff3", "#ffffff", "#fff2cc", "#dae8fc"
EXTERNAL_EDGE, HOLDOUT_EDGE, BLUE_LINE = "#c9a93f", "#5b7fb5", "#2f5f9e"

BAND_H, BOX_H, PAD = 52, 40, 6
GAPS = [26, 26, 40, 26, 40]
band_y = [6]
for gap in GAPS:
    band_y.append(band_y[-1] + BAND_H + gap)
W, H = 1000, band_y[-1] + BAND_H + 6

# each band: label, boxes (text, kind); kind is "" (white), "external" or "holdout"
BANDS = [
    (
        "3.1 Data\nconstruction",
        [
            ("GitSkills dataset", "external"),
            ("Distinct skills\n(content hash)", ""),
            ("Aggregator\nfiltering", ""),
            ("Random sample\n(150k skills)", ""),
        ],
    ),
    (
        "3.2 Copy\nmeasurement",
        [
            ("GitHub API", "external"),
            ("Arrival of each\ncopy (dating)", ""),
            ("T0, origin,\nand author", ""),
            ("Days to first\ncopy by others", ""),
        ],
    ),
    (
        "3.3–3.4 Attributes\nand split",
        [
            ("Text attributes (27)", ""),
            ("Origin repository\nattributes (control)", ""),
            ("Split by author\n(≤ 50 skills per author)", ""),
        ],
    ),
    (
        "3.5 Exploration\nand preprocessing",
        [
            ("Window W\nand target", ""),
            ("Exploratory\ndata analysis", ""),
            ("Preprocessing\nrules and pipeline", ""),
        ],
    ),
    (
        "3.6 Model\ndevelopment",
        [
            ("Spot-check of\n7 algorithms", ""),
            ("Nested tuning\nand selection", ""),
            ("Final models\nand threshold", ""),
        ],
    ),
    (
        "3.7–3.8 Evaluation\nand interpretation",
        [
            ("Holdout\nevaluation", "holdout"),
            ("Robustness\nanalyses", "holdout"),
            ("Interpretation\n(permutation, SHAP)", "holdout"),
        ],
    ),
]

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
problems = []

X0, X1 = 196, 946  # where the boxes of a band start and end
boxes = []  # boxes[band][i] = (x, y, w)

for b, (label, items) in enumerate(BANDS):
    y = band_y[b]
    ax.add_patch(
        FancyBboxPatch(
            (6, y), 966, BAND_H, boxstyle="round,pad=0,rounding_size=8", fc=BAND, ec="none"
        )
    )
    t = ax.text(
        16,
        y + BAND_H / 2,
        label,
        ha="left",
        va="center",
        fontsize=15 * PX,
        fontweight="bold",
        color=INK,
        linespacing=1.25,
    )
    if t.get_window_extent(renderer).width > 176:
        problems.append(f"band label too wide: {label!r}")
    n = len(items)
    w = 165 if n == 4 else 225
    gap = (X1 - X0 - n * w) / (n - 1)
    row = []
    for i, (text, kind) in enumerate(items):
        x = X0 + i * (w + gap)
        fc, ec = {
            "": (WHITE, LINE),
            "external": (EXTERNAL, EXTERNAL_EDGE),
            "holdout": (HOLDOUT, HOLDOUT_EDGE),
        }[kind]
        ax.add_patch(
            FancyBboxPatch(
                (x, y + PAD), w, BOX_H, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec=ec, lw=1.3
            )
        )
        t = ax.text(
            x + w / 2,
            y + PAD + BOX_H / 2,
            text,
            ha="center",
            va="center",
            fontsize=15 * PX,
            color=INK,
            linespacing=1.15,
        )
        if t.get_window_extent(renderer).width > w - 12:
            problems.append(f"text too wide for its box: {text!r}")
        row.append((x, y + PAD, w))
        if i:  # arrow from the previous box of the band
            px, py, pw = row[i - 1]
            ax.annotate(
                "",
                xy=(x, y + PAD + BOX_H / 2),
                xytext=(px + pw, y + PAD + BOX_H / 2),
                arrowprops=dict(
                    arrowstyle="-|>", color=LINE, lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=11
                ),
            )
    boxes.append(row)


def path(points, color, head=True, lw=1.3):
    """A line through the points; the last segment ends with an arrowhead."""
    xs, ys = zip(*points[:-1], strict=True)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt")
    ax.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops=dict(
            arrowstyle="-|>" if head else "-",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=11,
        ),
    )


def centre(b, i):
    x, y, w = boxes[b][i]
    return x + w / 2


def top(b, i):
    return boxes[b][i][1]


def bottom(b, i):
    return boxes[b][i][1] + BOX_H


# between bands: down from the last box of a band, left along the gap, down into the next band
def elbow(b, start_x, target_i, run_offset, color=LINE):
    """Down from band b (from start_x, by default the centre of its last box), along the gap,
    and into box target_i of the next band."""
    sx = centre(b, len(boxes[b]) - 1) if start_x is None else start_x
    y_from, y_to = bottom(b, 0), top(b + 1, target_i)
    run = y_from + run_offset
    path(
        [(sx, y_from), (sx, run), (centre(b + 1, target_i), run), (centre(b + 1, target_i), y_to)],
        color,
    )
    return run


elbow(0, None, 1, GAPS[0] / 2)  # sample -> dating (the API box feeds the dating box from its left)
elbow(1, None, 0, GAPS[1] / 2)  # dated copies -> attributes
dev_x = 760  # the development set leaves the split box a bit to the left of its centre
run_cd = elbow(2, dev_x, 0, 30)
ax.text(
    centre(3, 0) + 12,
    bottom(2, 0) + 17,
    "Development set (70% of authors)",
    ha="left",
    va="baseline",
    fontsize=13 * PX,
    color=INK,
)
elbow(3, None, 0, GAPS[3] / 2)
# final models -> holdout evaluation, joined by the holdout coming down the right side
run_ef = bottom(4, 0) + 30
GUTTER = 985
split_x, split_y, split_w = boxes[2][2]
path(
    [
        (split_x + split_w, split_y + BOX_H / 2),
        (GUTTER, split_y + BOX_H / 2),
        (GUTTER, run_ef),
        (centre(5, 0), run_ef),
        (centre(5, 0), top(5, 0)),
    ],
    BLUE_LINE,
)
ax.plot([centre(4, 2), centre(4, 2)], [bottom(4, 0), run_ef], color=LINE, lw=1.3)
ax.text(
    GUTTER - 10,
    bottom(2, 0) + 17,
    "Holdout (30% of authors)",
    ha="right",
    va="baseline",
    fontsize=13 * PX,
    color=BLUE_LINE,
)
ax.text(
    centre(5, 0) + 12,
    bottom(4, 0) + 17,
    "Final models evaluated once on the holdout",
    ha="left",
    va="baseline",
    fontsize=13 * PX,
    color=BLUE_LINE,
)

# the dating box also receives the API: the arrow API -> dating is the in-band arrow already drawn
fig.savefig(OUT / "workflow.png", dpi=300)
fig.savefig(OUT / "workflow.svg")
fig.savefig(OUT / "workflow.pdf")
print("size", W, "x", H, "px;", f"{H / W * 15:.1f} cm tall at 15 cm wide")
print("problems:", problems or "none")
