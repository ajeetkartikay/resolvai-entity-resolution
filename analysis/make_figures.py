"""
Figures for the ResolvAI analysis summary (analysis/README.md).

Every number below was measured during the project (data profiling, blocking evaluation,
held-out validation and error analysis), so the figures can be regenerated without the dataset:
    python analysis/make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# palette (validated: categorical slots 1-2, neutral for de-emphasis)
BLUE, ORANGE = "#2a78d6", "#eb6834"
MUTED = "#b8b7b1"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


def finish(ax, fig, name, note=None):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if note:
        fig.text(0.01, 0.01, note, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)


def hbar(ax, labels, values, colors, fmt, xmax=None):
    y = range(len(labels))[::-1]
    bars = ax.barh(list(y), values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    top = xmax or max(values) * 1.15
    ax.set_xlim(0, top)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + top * 0.01, b.get_y() + b.get_height() / 2, fmt(v),
                va="center", fontsize=10, color=INK)
    return bars


# 1. Scale of the data ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.6))
labels = ["Reference (Source 1)", "Source 2", "Source 3"]
train = [2_206_821, 5_034_616, 5_285_603]
test = [1_732_544, 4_887_273, 5_082_316]
x = range(len(labels)); w = 0.38
b1 = ax.bar([i - w / 2 for i in x], train, w, color=BLUE, label="Train", edgecolor=SURFACE, linewidth=2)
b2 = ax.bar([i + w / 2 for i in x], test, w, color=ORANGE, label="Test", edgecolor=SURFACE, linewidth=2)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f"{b.get_height()/1e6:.1f}M",
                ha="center", va="bottom", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.tick_params(axis="x", length=0)
ax.set_ylabel("Records"); ax.set_ylim(0, 6.2e6)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
ax.legend(frameon=False, loc="upper left")
ax.set_title("24.2M records across three sources")
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_data_scale.png"), dpi=160); plt.close(fig)

# 2. Matches per reference entity ----------------------------------------------------------
counts = [123_247, 119_157, 375_212, 530_841, 484_115, 321_957, 164_868, 63_968, 18_680, 4_205, 534, 37]
total = sum(counts)
fig, ax = plt.subplots(figsize=(8, 3.6))
colors = [ORANGE] + [BLUE] * (len(counts) - 1)
bars = ax.bar(range(len(counts)), counts, color=colors, width=0.75, edgecolor=SURFACE, linewidth=2)
for i in (0, 3):
    ax.text(i, counts[i] * 1.02, f"{counts[i]/total:.1%}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(len(counts))); ax.tick_params(axis="x", length=0)
ax.set_xlabel("Matching records per reference business")
ax.set_ylabel("Businesses")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
ax.set_title("Most businesses have 2-5 matches; 5.6% have none (singletons)")
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
fig.text(0.01, 0.01, "Orange = singletons: predicting 'no match' correctly scores a full 1.0 for these.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "02_matches_per_entity.png"), dpi=160); plt.close(fig)

# 3. What true matches have in common (profiling of 183K true pairs) -------------------------
fig, ax = plt.subplots(figsize=(8, 3.4))
hbar(ax, ["Same country", "Share a name OR address token", "Share an address token",
          "Share a name token", "Share a postcode"],
     [100.0, 100.0, 95.6, 86.2, 4.8], [BLUE, BLUE, BLUE, BLUE, MUTED], lambda v: f"{v:.1f}%", xmax=112)
ax.set_xlabel("% of true matching pairs")
ax.set_title("Profiling true matches decided the blocking design")
finish(ax, fig, "03_match_signals.png",
       "Country + name/address tokens recover every true pair; postcodes are almost never shared, so they were dropped as a key.")

# 4. Sampling bias -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 2.8))
hbar(ax, ["10% sample (validation)", "Full scale (validation)", "Full scale (hidden test)"],
     [0.923, 0.718, 0.641], [MUTED, BLUE, BLUE], lambda v: f"{v:.3f}", xmax=1.08)
ax.set_xlabel("F0.5 of the same similarity rule")
ax.set_title("A 10% sample overstated accuracy by 0.2 F0.5")
finish(ax, fig, "04_sampling_bias.png",
       "Sampling removes most look-alike competitors, so all later tuning was done at full data density.")

# 5. Score progression ---------------------------------------------------------------------
stages = ["TF-IDF similarity rule", "LightGBM stage 1", "+ competition context", "+ sibling agreement",
          "+ multilingual cross-encoder", "+ learned stacker"]
scores = [0.718, 0.932, 0.937, 0.943, 0.967, 0.970]
fig, ax = plt.subplots(figsize=(8, 3.6))
hbar(ax, stages, scores, [MUTED] + [BLUE] * 5, lambda v: f"{v:.3f}", xmax=1.08)
ax.set_xlabel("Held-out F0.5 (full data density)")
ax.set_title("Each stage's contribution to accuracy")
finish(ax, fig, "05_score_progression.png", "Hidden test set: 0.641 (rule) → 0.917 → 0.948 → 0.951 (final).")

# 6. Error analysis (three small charts, one shared scale each) ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.25, 1.1, 0.8]})
hbar(axes[0], ["Belongs to\nno business", "Owner not in\nsampled data", "Wrong competing\nowner"],
     [65.6, 27.4, 7.1], [BLUE, MUTED, BLUE], lambda v: f"{v:.1f}%", xmax=85)
axes[0].set_title("Wrong merges: what went wrong", fontsize=12)
axes[0].set_xlabel("% of 54K wrong merges")
hbar(axes[1], ["Found", "Candidate,\nrejected by model", "Never a\ncandidate"],
     [87.7, 8.1, 4.2], [BLUE, ORANGE, ORANGE], lambda v: f"{v:.1f}%", xmax=110)
axes[1].set_title("True matches: where they were lost", fontsize=12)
axes[1].set_xlabel("% of 3.24M true matches")
hbar(axes[2], ["US", "India"], [0.950, 0.916], [BLUE, BLUE], lambda v: f"{v:.3f}", xmax=1.2)
axes[2].set_title("Accuracy by country", fontsize=12)
axes[2].set_xlabel("Held-out F0.5")
axes[2].set_xticks([0, 0.5, 1.0])
for a in axes:
    a.grid(axis="x", color=GRID, linewidth=0.8); a.set_axisbelow(True)
fig.text(0.01, 0.01, "Error analysis of the LightGBM matcher on held-out businesses (before the cross-encoder).", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "06_error_analysis.png"), dpi=160); plt.close(fig)

# 7. Blocking recall per channel ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 2.9))
hbar(ax, ["Name (char 3-grams)", "Address (word tokens)", "Name + address (char n-grams)", "Union of all three"],
     [74.1, 92.3, 97.1, 98.8], [MUTED, MUTED, MUTED, BLUE], lambda v: f"{v:.1f}%", xmax=112)
ax.set_xlabel("% of true pairs retrieved (about 75 candidates per business)")
ax.set_title("Three blocking channels together keep 98.8%")
finish(ax, fig, "07_blocking_recall.png")

print("figures written to", OUT)
