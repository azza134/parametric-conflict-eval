import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import MODELS, SYSTEM_INSTRUCTIONS
from harness import CAVEAT_RESULTS, SEVERITIES

OUT = "figures/flagging_by_severity.png"
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
INSTRUCTION_STYLE = {
    "SOURCE_EXCLUSIVE": ("#2a78d6", "o", "SE"),
    "FLAG_INVITING": ("#e34948", "^", "FI"),
    "WEAK_GROUNDING": ("#008300", "s", "WG"),
    "SOURCE_EXCLUSIVE_FLAG_INVITING": ("#4a3aa7", "D", "SE+FI"),
    "SELECTIVE_AUDIT": ("#eb6834", "v", "AUDIT"),
}


def flag_rates():
    counts = defaultdict(lambda: [0, 0])
    with open(CAVEAT_RESULTS, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["model"], r["instruction"], r["severity"])
            counts[key][1] += 1
            counts[key][0] += r["stance"] == "questioned"
    return {k: x / n for k, (x, n) in counts.items()}


def main():
    rates = flag_rates()
    models = [m for m, _ in MODELS]
    fig, axes = plt.subplots(1, len(models), figsize=(2.9 * len(models), 3.4),
                             sharey=True, facecolor=SURFACE)
    for ax, model in zip(axes, models):
        ax.set_facecolor(SURFACE)
        for iname, _ in SYSTEM_INSTRUCTIONS:
            color, marker, label = INSTRUCTION_STYLE[iname]
            ys = [rates.get((model, iname, s)) for s in SEVERITIES]
            ax.plot(SEVERITIES, ys, color=color, marker=marker, label=label,
                    linewidth=2, markersize=5.5, clip_on=False)
        ax.set_title(model, fontsize=10, color=PRIMARY_INK, pad=8)
        ax.set_xticks(SEVERITIES)
        ax.set_xticklabels([f"S{s}" for s in SEVERITIES], fontsize=8, color=MUTED_INK)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED_INK, length=0)
        ax.tick_params(axis="x", colors=MUTED_INK, length=0)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
    axes[0].set_ylabel("error-flagging rate", fontsize=9, color=SECONDARY_INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=SECONDARY_INK, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Error-flagging rate vs perturbation severity (S0 = unperturbed control)",
                 fontsize=12, color=PRIMARY_INK, y=1.12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"{OUT}: {len(rates)} cells plotted")


if __name__ == "__main__":
    main()
