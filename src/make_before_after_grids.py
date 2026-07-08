"""
make_before_after_grids.py  —  Before/after visualization grids

Builds the consolidated before/after figures required by the project spec
("Input/Output processing steps: Image with annotation, before/after"):

  1) grid_distortions.png — one sample image: clean + 4 distortions x 4
     severity levels, each tile labeled with its SNR (dB).
  2) grid_enhancement.png — per distortion (level 3): clean -> distorted ->
     enhanced side by side.
  3) grid_annotated.png — the same 3-way comparison with YOLO detections
     drawn on each tile. Detections are generated live (a few predictions,
     ~2s) so the figure has no dependency on pipeline output folders.

Reads only existing images; writes only 3 new PNGs into
data/tasks_graphs_and_tables/plots/. Nothing else is touched.

Run from the project root:
    python src/make_before_after_grids.py
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distortions import calculate_snr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "datasets" / "coco128" / "images" / "train2017"
DISTORTED_DIR = PROJECT_ROOT / "data" / "distorted_images"
ENHANCED_DIR = PROJECT_ROOT / "data" / "enhanced_images"
PLOTS_DIR = PROJECT_ROOT / "data" / "tasks_graphs_and_tables" / "plots"

SAMPLE = "000000000009.jpg"
DISTORTIONS = ["gaussian_noise", "salt_pepper", "low_light", "motion_blur"]
NICE = {"gaussian_noise": "Gaussian noise", "salt_pepper": "Salt & pepper",
        "low_light": "Low light", "motion_blur": "Motion blur"}
LEVELS = [1, 2, 3, 4]
SHOW_LEVEL = 3


def rgb(path):
    img = cv2.imread(str(path))
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tile(ax, img, title=""):
    ax.axis("off")
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
        return
    ax.imshow(img)
    if title:
        ax.set_title(title, fontsize=9)


def row_label(ax, text):
    ax.axis("on")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel(text)


def grid_distortions():
    clean_bgr = cv2.imread(str(CLEAN_DIR / SAMPLE))
    clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(len(DISTORTIONS), len(LEVELS) + 1,
                             figsize=(3 * (len(LEVELS) + 1), 2.6 * len(DISTORTIONS)))
    for r, dist in enumerate(DISTORTIONS):
        tile(axes[r][0], clean, "Clean" if r == 0 else "")
        row_label(axes[r][0], NICE[dist])
        for c, lvl in enumerate(LEVELS, start=1):
            img_bgr = cv2.imread(str(DISTORTED_DIR / f"{dist}_l{lvl}" / SAMPLE))
            img = None if img_bgr is None else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            snr = calculate_snr(clean_bgr, img_bgr) if img_bgr is not None else float("nan")
            tile(axes[r][c], img, f"L{lvl}  (SNR {snr:.1f} dB)")
    fig.suptitle(f"Distortion severity overview — sample {SAMPLE}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS_DIR / "grid_distortions.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"   \U0001F5BC {out.name}")


def grid_enhancement():
    clean = rgb(CLEAN_DIR / SAMPLE)
    fig, axes = plt.subplots(len(DISTORTIONS), 3, figsize=(9, 2.6 * len(DISTORTIONS)))
    for r, dist in enumerate(DISTORTIONS):
        d = rgb(DISTORTED_DIR / f"{dist}_l{SHOW_LEVEL}" / SAMPLE)
        e = rgb(ENHANCED_DIR / f"{dist}_l{SHOW_LEVEL}" / SAMPLE)
        tile(axes[r][0], clean, "Clean" if r == 0 else "")
        row_label(axes[r][0], NICE[dist])
        tile(axes[r][1], d, f"Distorted (L{SHOW_LEVEL})" if r == 0 else "")
        tile(axes[r][2], e, "Enhanced" if r == 0 else "")
    fig.suptitle(f"Enhancement before/after (level {SHOW_LEVEL}) — sample {SAMPLE}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS_DIR / "grid_enhancement.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"   \U0001F5BC {out.name}")


def grid_annotated():
    """Run YOLO live on each tile and draw its detections (correct colors,
    no dependency on pipeline output folders)."""
    model = YOLO("yolov8n.pt")

    def detect_rgb(path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        res = model.predict(img, conf=0.25, verbose=False)[0]
        return cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)   # plot() returns BGR

    clean_annot = detect_rgb(CLEAN_DIR / SAMPLE)
    fig, axes = plt.subplots(len(DISTORTIONS), 3, figsize=(9, 2.6 * len(DISTORTIONS)))
    for r, dist in enumerate(DISTORTIONS):
        d = detect_rgb(DISTORTED_DIR / f"{dist}_l{SHOW_LEVEL}" / SAMPLE)
        e = detect_rgb(ENHANCED_DIR / f"{dist}_l{SHOW_LEVEL}" / SAMPLE)
        tile(axes[r][0], clean_annot, "Clean + detections" if r == 0 else "")
        row_label(axes[r][0], NICE[dist])
        tile(axes[r][1], d, f"Distorted (L{SHOW_LEVEL})" if r == 0 else "")
        tile(axes[r][2], e, "Enhanced" if r == 0 else "")
    fig.suptitle(f"YOLO detections: clean vs. distorted vs. enhanced "
                 f"(level {SHOW_LEVEL}) — sample {SAMPLE}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS_DIR / "grid_annotated.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"   \U0001F5BC {out.name}")


def main():
    print("\U0001F680 Building before/after grids")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_distortions()
    grid_enhancement()
    grid_annotated()
    print(f"\u2705 Done! Grids saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()