"""
evaluate_map_gt.py  —  GT-based mAP evaluation (per class, per SNR)

Closes the course requirement: "Measure performance: per class, per SNR"
and the Project Outcomes items "Performance on distorted images (per
distortion/class)" / "Performance on enhanced images (per distortion/class)",
all measured against Ground Truth.

What it does:
  For every condition (clean, and each distortion x level, for both the
  distorted and the enhanced versions):
    1. Builds a small temporary YOLO validation dataset that pairs the
       condition's images with the ORIGINAL COCO128 labels (valid because
       our distortions never move objects).
    2. Runs ultralytics model.val() -> real mAP50 / mAP50-95, overall and
       per class, against GT.
    3. Computes the mean SNR (dB) of the condition's images vs. the clean
       originals, so accuracy can be plotted against SNR.
  All results go to ONE NEW file:
       data/tasks_graphs_and_tables/map_summary.csv
  Nothing in the existing pipeline (metadata_summary_base.csv, plots,
  other scripts) is read from or written to, except read-only image files.

Run from the project root:
    python src/evaluate_map_gt.py
"""

import os
import shutil
import csv
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from distortions import calculate_snr

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_IMAGES_DIR = PROJECT_ROOT / "datasets" / "coco128" / "images" / "train2017"
CLEAN_LABELS_DIR = PROJECT_ROOT / "datasets" / "coco128" / "labels" / "train2017"

DISTORTED_DIR = PROJECT_ROOT / "data" / "distorted_images"
ENHANCED_DIR = PROJECT_ROOT / "data" / "enhanced_images"

TMP_DIR = PROJECT_ROOT / "data" / "map_eval_tmp"        # rebuilt per condition
OUTPUT_CSV = PROJECT_ROOT / "data" / "tasks_graphs_and_tables" / "map_summary.csv"

DISTORTIONS = ["gaussian_noise", "salt_pepper", "low_light", "motion_blur"]
LEVELS = [1, 2, 3, 4]

MODEL_PATH = "yolov8n.pt"   # same pretrained detection model as the baseline


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def build_tmp_dataset(image_dir: Path, class_names: dict) -> Path:
    """Create data/map_eval_tmp/{images,labels}/val from image_dir + original
    COCO labels, and return the path to a dataset yaml for model.val()."""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    img_out = TMP_DIR / "images" / "val"
    lbl_out = TMP_DIR / "labels" / "val"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    for img_path in sorted(image_dir.glob("*.jpg")):
        shutil.copy2(img_path, img_out / img_path.name)
        label_file = CLEAN_LABELS_DIR / (img_path.stem + ".txt")
        if label_file.exists():          # missing label = background image, OK
            shutil.copy2(label_file, lbl_out / label_file.name)

    yaml_path = TMP_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(
             {"path": str(TMP_DIR),
             "train": "images/val",
             "val": "images/val",
             "names": class_names},
            f,
        )
    return yaml_path


def mean_snr_db(image_dir: Path):
    """Mean SNR (dB) of all images in image_dir vs. their clean originals."""
    snrs = []
    for img_path in sorted(image_dir.glob("*.jpg")):
        clean = cv2.imread(str(CLEAN_IMAGES_DIR / img_path.name))
        other = cv2.imread(str(img_path))
        if clean is None or other is None:
            continue
        snr = calculate_snr(clean, other)
        if np.isfinite(snr):
            snrs.append(snr)
    return round(float(np.mean(snrs)), 2) if snrs else None


def evaluate_condition(model, image_dir: Path, condition, distortion, level, rows):
    """Run GT validation on one condition folder and append result rows."""
    if not image_dir.exists():
        print(f"   ⚠️  Skipping (folder not found): {image_dir}")
        return

    print(f"🔍 Evaluating: {condition} | {distortion} | level {level}")
    yaml_path = build_tmp_dataset(image_dir, model.names)
    snr = None if condition == "clean" else mean_snr_db(image_dir)

    metrics = model.val(
        data=str(yaml_path),
        batch=1,
        verbose=False,
        plots=False,
        project=str(TMP_DIR / "runs"),   # keep val artifacts out of runs/
        name="val",
        exist_ok=True,
    )

    common = {
        "condition": condition,
        "distortion_type": distortion,
        "level": level,
        "mean_snr_db": snr,
    }

    # Overall row
    rows.append({**common,
                 "class_name": "all",
                 "map50": round(float(metrics.box.map50), 4),
                 "map50_95": round(float(metrics.box.map), 4)})

    # Per-class rows (only classes that appear in this condition's GT)
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        rows.append({**common,
                     "class_name": model.names[int(cls_idx)],
                     "map50": round(float(metrics.box.ap50[i]), 4),
                     "map50_95": round(float(metrics.box.ap[i]), 4)})


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("🚀 GT-based mAP evaluation (per class, per SNR)")
    model = YOLO(MODEL_PATH)
    rows = []

    # 1) Clean baseline — but only on the SAME 30 images used in the pipeline,
    #    so all comparisons are apples-to-apples. We take the image names from
    #    any distorted folder (they mirror the originals).
    sample_ref = DISTORTED_DIR / f"{DISTORTIONS[0]}_l1"
    sample_names = [p.name for p in sorted(sample_ref.glob("*.jpg"))]
    clean_subset = TMP_DIR.parent / "map_eval_clean_subset"
    if clean_subset.exists():
        shutil.rmtree(clean_subset)
    clean_subset.mkdir(parents=True)
    for name in sample_names:
        shutil.copy2(CLEAN_IMAGES_DIR / name, clean_subset / name)
    evaluate_condition(model, clean_subset, "clean", "none", 0, rows)

    # 2) Distorted and Enhanced, every distortion x level
    for base_dir, condition in [(DISTORTED_DIR, "distorted"),
                                (ENHANCED_DIR, "enhanced")]:
        for dist in DISTORTIONS:
            for level in LEVELS:
                evaluate_condition(model, base_dir / f"{dist}_l{level}",
                                   condition, dist, level, rows)

    # 3) Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "distortion_type", "level", "mean_snr_db",
                  "class_name", "map50", "map50_95"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Cleanup temp folders
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    shutil.rmtree(clean_subset, ignore_errors=True)

    print("✅ Done!")
    print(f"   📄 Results: {OUTPUT_CSV}")
    print(f"   📊 Total rows: {len(rows)} "
          f"(33 conditions: 1 clean + 16 distorted + 16 enhanced)")


if __name__ == "__main__":
    main()