"""
plot_map_results.py  —  Visualization of GT-based mAP results (Step 3)

Reads ONLY data/tasks_graphs_and_tables/map_summary.csv (produced by
src/evaluate_map_gt.py) and writes NEW png files into
data/tasks_graphs_and_tables/plots/ with a 'map_' prefix.
No existing team file is read, imported, or modified.

Outputs:
  1) map_curve_{distortion}.png  (x4)
     mAP50-95 vs. SNR (dB) — clean baseline (dashed) vs. distorted vs.
     enhanced. This is the professor's "accuracy per intensity
     (measure as SNR)" chart.
  2) map_per_class_clean.png
     Per-class mAP50-95 bar chart on clean images (like the professor's
     per-class example slide).
  3) map_per_class_drop.png
     Which classes break first: per-class mAP on clean vs. under
     moderate distortion (level 2), averaged over the 4 distortions.

Run from the project root:
    python src/plot_map_results.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save files, no GUI window needed
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "tasks_graphs_and_tables" / "map_summary.csv"
PLOTS_DIR = PROJECT_ROOT / "data" / "tasks_graphs_and_tables" / "plots"

DISTORTIONS = ["gaussian_noise", "salt_pepper", "low_light", "motion_blur"]
NICE_NAME = {
    "gaussian_noise": "Gaussian noise",
    "salt_pepper": "Salt & pepper noise",
    "low_light": "Low light",
    "motion_blur": "Motion blur",
}

COLOR_DISTORTED = "#d62728"   # red
COLOR_ENHANCED = "#2ca02c"    # green
COLOR_CLEAN = "#7f7f7f"       # gray


def plot_snr_curves(df):
    """One figure per distortion: mAP50-95 vs SNR(dB), distorted vs enhanced,
    with the clean baseline as a dashed reference line.

    X-axis note: we use the SNR of the DISTORTED images as the severity axis
    for both curves (the enhancement is applied to those same images, so the
    input intensity is identical); this matches 'accuracy per intensity'."""
    overall = df[df["class_name"] == "all"]
    clean_map = overall.loc[overall["condition"] == "clean", "map50_95"].iloc[0]

    for dist in DISTORTIONS:
        d_rows = overall[(overall["condition"] == "distorted")
                         & (overall["distortion_type"] == dist)].sort_values("level")
        e_rows = overall[(overall["condition"] == "enhanced")
                         & (overall["distortion_type"] == dist)].sort_values("level")

        x = d_rows["mean_snr_db"].astype(float).values      # severity axis
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.axhline(clean_map, color=COLOR_CLEAN, linestyle="--", linewidth=1.5,
                   label=f"Clean baseline ({clean_map:.3f})")
        ax.plot(x, d_rows["map50_95"].values, "o-", color=COLOR_DISTORTED,
                linewidth=2, label="Distorted")
        ax.plot(x, e_rows["map50_95"].values, "s-", color=COLOR_ENHANCED,
                linewidth=2, label="Enhanced")

        for xi, lvl in zip(x, d_rows["level"].values):       # annotate levels
            ax.annotate(f"L{lvl}", (xi, -0.02), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=8, color="gray")

        ax.invert_xaxis()   # left = high SNR (mild) -> right = low SNR (severe)
        ax.set_xlabel("SNR (dB)  \u2190 more severe distortion")
        ax.set_ylabel("mAP50-95 (vs. GT)")
        ax.set_ylim(0, max(0.65, clean_map + 0.05))
        ax.set_title(f"Object detection robustness \u2014 {NICE_NAME[dist]}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out = PLOTS_DIR / f"map_curve_{dist}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"   \U0001F4C8 {out.name}")


def plot_per_class_clean(df):
    """Per-class mAP50-95 on clean images, sorted (professor's example style)."""
    rows = df[(df["condition"] == "clean") & (df["class_name"] != "all")]
    rows = rows.sort_values("map50_95", ascending=False)
    mean_val = rows["map50_95"].mean()

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(rows)), 4.5))
    ax.bar(rows["class_name"], rows["map50_95"], color="#1f77b4")
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.2,
               label=f"mean mAP = {mean_val:.3f}")
    ax.set_ylabel("mAP50-95 (vs. GT)")
    ax.set_title("Per-class mAP on clean images (30-image sample)")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    ax.legend()
    fig.tight_layout()

    out = PLOTS_DIR / "map_per_class_clean.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   \U0001F4C8 {out.name}")


def plot_per_class_drop(df):
    """Which classes break first: clean vs. moderate distortion (level 2,
    averaged over the 4 distortion types)."""
    clean = df[(df["condition"] == "clean") & (df["class_name"] != "all")]
    clean = clean.set_index("class_name")["map50_95"]

    l2 = df[(df["condition"] == "distorted") & (df["level"] == 2)
            & (df["class_name"] != "all")]
    l2_mean = l2.groupby("class_name")["map50_95"].mean()

    both = pd.DataFrame({"clean": clean, "distorted_l2": l2_mean}).dropna()
    both = both.sort_values("clean", ascending=False)

    x = range(len(both))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(both)), 4.5))
    ax.bar([i - width / 2 for i in x], both["clean"], width,
           label="Clean", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], both["distorted_l2"], width,
           label="Distorted (level 2, avg of 4 distortions)", color=COLOR_DISTORTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(both.index, rotation=75, fontsize=8)
    ax.set_ylabel("mAP50-95 (vs. GT)")
    ax.set_title("Per-class sensitivity: which classes break first?")
    ax.legend()
    fig.tight_layout()

    out = PLOTS_DIR / "map_per_class_drop.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   \U0001F4C8 {out.name}")


def main():
    print("\U0001F680 Plotting GT-based mAP results")
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run 'python src/evaluate_map_gt.py' first.")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    plot_snr_curves(df)
    plot_per_class_clean(df)
    plot_per_class_drop(df)
    print(f"\u2705 Done! Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()