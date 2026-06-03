"""Plot IF-Eval performance over training steps for two reward types."""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Data ────────────────────────────────────────────────────────────────────
base = {
    "strict_prompt": 0.8281, "strict_inst": 0.8813,
    "loose_prompt":  0.8558, "loose_inst":  0.9017,
}

lexical = {
    "steps":         [0,      120,    240,    360],
    "strict_prompt": [base["strict_prompt"], 0.8373, 0.8226, 0.7967],
    "strict_inst":   [base["strict_inst"],   0.8861, 0.8753, 0.8609],
    "loose_prompt":  [base["loose_prompt"],  0.8725, 0.8484, 0.8318],
    "loose_inst":    [base["loose_inst"],    0.9113, 0.8933, 0.8861],
}

no_bias = {
    "steps":         [0,      120,    240,    480],
    "strict_prompt": [base["strict_prompt"], 0.8226, 0.8540, 0.8540],
    "strict_inst":   [base["strict_inst"],   0.8789, 0.8993, 0.9005],
    "loose_prompt":  [base["loose_prompt"],  0.8558, 0.8854, 0.8817],
    "loose_inst":    [base["loose_inst"],    0.9029, 0.9209, 0.9197],
}

metrics = [
    ("strict_prompt", "Strict — Prompt Accuracy"),
    ("strict_inst",   "Strict — Instruction Accuracy"),
    ("loose_prompt",  "Loose  — Prompt Accuracy"),
    ("loose_inst",    "Loose  — Instruction Accuracy"),
]

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
fig.suptitle(
    "IF-Eval Performance over Training Steps\n(Qwen3-4B, step 0 = base model)",
    fontsize=14, fontweight="bold",
)

COLORS  = {"lexical_bias": "#e05c5c", "no_bias": "#4c8be0"}
MARKERS = {"lexical_bias": "o",       "no_bias": "s"}
LABELS  = {"lexical_bias": "Lexical-bias reward", "no_bias": "No-bias reward"}

for ax, (key, title) in zip(axes.flatten(), metrics):
    for tag, data in [("lexical_bias", lexical), ("no_bias", no_bias)]:
        ax.plot(
            data["steps"], data[key],
            color=COLORS[tag], marker=MARKERS[tag],
            linewidth=2, markersize=7,
            label=LABELS[tag],
        )
        for x, y in zip(data["steps"], data[key]):
            ax.annotate(
                f"{y:.3f}", (x, y),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.5, color=COLORS[tag],
            )

    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_xticks(sorted(set(lexical["steps"]) | set(no_bias["steps"])))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5)

    all_vals = lexical[key] + no_bias[key]
    margin = (max(all_vals) - min(all_vals)) * 0.35 + 0.003
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

out_path = "/data/wangxk/hackingRubricsRL/figures/ifeval_bias_comparison.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("Saved →", out_path)
