import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import MODELS, SYSTEM_INSTRUCTIONS
from harness import (CAVEAT_RESULTS, SEVERITIES, threshold_estimates,
                     OPUS_FI_PROBE_RESULTS, OPUS_PROBE_MODEL)

OUT = "figures/flagging_by_severity.png"
OUT_THRESHOLDS = "figures/detection_thresholds.png"
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


def threshold_data():
    cav = [r for r in (json.loads(l) for l in open(CAVEAT_RESULTS, encoding="utf-8"))
           if not r.get("truncated")]
    try:
        opus = [r for r in (json.loads(l) for l in open(OPUS_FI_PROBE_RESULTS, encoding="utf-8"))
                if not r.get("truncated")]
    except FileNotFoundError:
        opus = []
    models = [m for m, _ in MODELS] + ([OPUS_PROBE_MODEL[0]] if opus else [])
    return threshold_estimates(cav + opus, models), models


def plot_thresholds():
    estimates, _ = threshold_data()
    row_order = ["gpt-4o-mini", "gpt-5.4-nano", "gpt-5.6-terra",
                 "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"]
    offsets = {name: off for (name, _), off in zip(SYSTEM_INSTRUCTIONS, (-0.30, -0.15, 0.0, 0.15, 0.30))}
    fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for t in estimates:
        y = row_order.index(t["model"]) + offsets[t["instruction"]]
        color, marker, _ = INSTRUCTION_STYLE[t["instruction"]]
        if t["est"] is None:
            ax.plot([t["max_ratio"]], [y], marker=">", markersize=6, color=color,
                    markerfacecolor="none", markeredgewidth=1.6, clip_on=False)
        else:
            ax.plot([t["lo"], t["hi"]], [y, y], color=color, linewidth=2,
                    solid_capstyle="round", clip_on=False)
            ax.plot([t["est"]], [y], marker=marker, markersize=6.5, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(1.5, 70000)
    decades = [10, 100, 1000, 10000]
    ax.set_xticks(decades)
    ax.set_xticklabels(["x10", "x100", "x1,000", "x10,000"], fontsize=8, color=MUTED_INK)
    ax.set_xticks([], minor=True)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=9, color=SECONDARY_INK)
    ax.set_ylim(len(row_order) - 0.5, -0.5)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.axhline(2.5, color=GRIDLINE, linewidth=0.8)
    ax.set_xlabel("perturbation multiplier at 50% flagging (ratio50, log scale)",
                  fontsize=9, color=SECONDARY_INK)
    handles = [plt.Line2D([], [], color=INSTRUCTION_STYLE[n][0], marker=INSTRUCTION_STYLE[n][1],
                          linewidth=2, markersize=6, label=INSTRUCTION_STYLE[n][2])
               for n, _ in SYSTEM_INSTRUCTIONS]
    handles.append(plt.Line2D([], [], color=MUTED_INK, marker=">", linestyle="none",
                              markerfacecolor="none", markersize=6, markeredgewidth=1.6,
                              label="no crossing in observed range"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=8.5, labelcolor=SECONDARY_INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("How big must the error be? Dots: fitted ratio50 with 95% cluster bootstrap; "
                 "arrows: flagging never reaches 50%",
                 fontsize=11, color=PRIMARY_INK, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT_THRESHOLDS, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"{OUT_THRESHOLDS}: {len(estimates)} cells plotted")


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
    plot_thresholds()


if __name__ == "__main__":
    main()
