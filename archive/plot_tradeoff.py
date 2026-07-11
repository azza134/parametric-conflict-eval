import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from config import MODELS as CONFIG_MODELS, SYSTEM_INSTRUCTIONS

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

INSTRUCTION_COLOR = {
    "SOURCE_EXCLUSIVE": "#2a78d6",
    "FLAG_INVITING": "#e34948",
    "WEAK_GROUNDING": "#008300",
    "SOURCE_EXCLUSIVE_FLAG_INVITING": "#8a4fbf",
}
MODEL_MARKER = {
    "gpt-4o-mini": "o",
    "gpt-5.4-nano": "s",
    "claude-sonnet-5": "^",
}
MODELS = [m for m, _ in CONFIG_MODELS]
INSTRUCTIONS = [i for i, _ in SYSTEM_INSTRUCTIONS]

flag = defaultdict(list)
with open("caveat_curve.csv") as f:
    for row in csv.DictReader(f):
        if int(row["severity"]) >= 1:
            flag[(row["model"], row["instruction"])].append(float(row["questioned_rate"]))

faith = defaultdict(list)
with open("abstention_curve.csv") as f:
    for row in csv.DictReader(f):
        faith[(row["model"], row["instruction"])].append(1 - float(row["ungrounded_rate"]))

points = {}
for model in MODELS:
    for instr in INSTRUCTIONS:
        x = sum(flag[(model, instr)]) / len(flag[(model, instr)])
        y = sum(faith[(model, instr)]) / len(faith[(model, instr)])
        points[(model, instr)] = (x, y)

fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.14)
ticks = [0, 0.25, 0.5, 0.75, 1.0]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks], color=MUTED_INK, fontsize=9)
ax.set_yticklabels([f"{t:g}" for t in ticks], color=MUTED_INK, fontsize=9)
ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color(BASELINE)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(length=0)

for (model, instr), (x, y) in points.items():
    ax.plot(
        x, y,
        marker=MODEL_MARKER[model],
        markersize=11,
        markerfacecolor=INSTRUCTION_COLOR[instr],
        markeredgecolor=SURFACE,
        markeredgewidth=1.6,
        linestyle="none",
        zorder=3,
    )

# direct labels: only the claude-sonnet-5 points (the series the section is about)
label_offsets = {
    "SOURCE_EXCLUSIVE": (10, 16),
    "FLAG_INVITING": (10, -14),
    "WEAK_GROUNDING": (10, 6),
    "SOURCE_EXCLUSIVE_FLAG_INVITING": (12, 10),
}
for instr in INSTRUCTIONS:
    x, y = points[("claude-sonnet-5", instr)]
    dx, dy = label_offsets[instr]
    ax.annotate(
        instr.replace("_", " ").title(),
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        color=SECONDARY_INK,
        zorder=4,
    )

# the three SOURCE_EXCLUSIVE points land on the exact same coordinate (0, 1) for
# all three models -- note it rather than fake a jitter that isn't in the data
ax.annotate(
    "all 3 models overlap here",
    points[("gpt-4o-mini", "SOURCE_EXCLUSIVE")],
    xytext=(10, -18),
    textcoords="offset points",
    fontsize=7.5,
    color=MUTED_INK,
    style="italic",
)

ax.annotate(
    "better →",
    (0.60, 0.06),
    fontsize=9,
    color=MUTED_INK,
    style="italic",
)
ax.annotate(
    "better ↑",
    (0.86, 0.55),
    fontsize=9,
    color=MUTED_INK,
    style="italic",
    rotation=90,
)

ax.set_xlabel("error-flagging rate (avg. S1–S5)", color=SECONDARY_INK, fontsize=10)
ax.set_ylabel("faithful-abstention rate (avg. P1–P5)", color=SECONDARY_INK, fontsize=10)
ax.set_title(
    "The trade-off: flag the wrong value vs. withhold the absent one",
    color=PRIMARY_INK, fontsize=12, fontweight="bold", loc="left", pad=14,
)

# legend 1: instruction -> color (patches)
from matplotlib.lines import Line2D
color_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markersize=9,
           markerfacecolor=INSTRUCTION_COLOR[i], markeredgecolor=SURFACE, markeredgewidth=1,
           label=i.replace("_", " ").title())
    for i in INSTRUCTIONS
]
legend1 = ax.legend(
    handles=color_handles, title="instruction", loc="upper left",
    bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9, title_fontsize=9,
    labelcolor=SECONDARY_INK,
)
legend1.get_title().set_color(PRIMARY_INK)
ax.add_artist(legend1)

# legend 2: model -> marker shape
shape_handles = [
    Line2D([0], [0], marker=MODEL_MARKER[m], linestyle="none", markersize=9,
           markerfacecolor=MUTED_INK, markeredgecolor=SURFACE, markeredgewidth=1,
           label=m)
    for m in MODELS
]
legend2 = ax.legend(
    handles=shape_handles, title="model", loc="upper left",
    bbox_to_anchor=(1.02, 0.55), frameon=False, fontsize=9, title_fontsize=9,
    labelcolor=SECONDARY_INK,
)
legend2.get_title().set_color(PRIMARY_INK)

fig.text(
    0.02, 0.005,
    "n=48/cell caveat, n=16/cell abstention. Flagging averaged over S1-S5 (excludes the S0 false-positive check); "
    "faithful averaged over P1-P5. Full grids: caveat_curve.csv / abstention_curve.csv.",
    fontsize=7, color=MUTED_INK,
)

fig.tight_layout(rect=[0, 0.02, 0.8, 1])
fig.savefig("tradeoff_scatter.png", facecolor=SURFACE, bbox_inches="tight")
print("wrote tradeoff_scatter.png")
