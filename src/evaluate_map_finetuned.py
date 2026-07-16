"""
evaluate_map_finetuned.py  —  GT-based mAP: pretrained vs. fine-tuned

Completes the required project outcome "Performance of fine-tuned models
(on distorted images)" with a trustworthy metric (mAP vs. GT), replacing the
counting-based evaluation shown to be unreliable.

Leakage-proof design:
  The recovered fine-tuned model (best.pt) was trained on distorted versions
  of the pipeline's 30 images, with an exact split that is not recoverable.
  To guarantee a fair test, this script evaluates on FRESH COCO128 images
  the pipeline never used (by default images #31-40 in sorted order),
  generating their distorted versions on the fly with the project's own
  distortion functions. Both models are evaluated on exactly the same images.

Conditions: clean + 4 distortions x 4 levels  ->  17 conditions x 2 models.
Output: data/tasks_graphs_and_tables/map_summary_finetuned.csv
(new file; nothing existing is modified; temp folders are cleaned up).

Run from the project root, with best.pt at
runs/detect/finetune_distorted/weights/best.pt :
    python src/evaluate_map_finetuned.py
"""

import shutil
import csv
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distortions import (calculate_snr, apply_gaussian_noise,
                         apply_salt_and_pepper_noise, apply_low_light,
                         apply_motion_blur)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_IMAGES_DIR = PROJECT_ROOT / "datasets" / "coco128" / "images" / "train2017"
CLEAN_LABELS_DIR = PROJECT_ROOT / "datasets" / "coco128" / "labels" / "train2017"
TMP_DIR = PROJECT_ROOT / "data" / "map_eval_ft_tmp"
OUTPUT_CSV = PROJECT_ROOT / "data" / "tasks_graphs_and_tables" / "map_summary_finetuned.csv"

PRETRAINED = PROJECT_ROOT / "yolov8n.pt"
FINETUNED = PROJECT_ROOT / "runs" / "detect" / "finetune_distorted" / "weights" / "best.pt"

# The pipeline used the first 30 sorted images; we take the next 10 —
# guaranteed unseen by the fine-tuned model regardless of its training split.
SKIP_FIRST = 30
N_EVAL = 10

DISTORTION_FUNCS = {
    "gaussian_noise": apply_gaussian_noise,
    "salt_pepper": apply_salt_and_pepper_noise,
    "low_light": apply_low_light,
    "motion_blur": apply_motion_blur,
}
LEVELS = [1, 2, 3, 4]


def fresh_image_names():
    all_names = sorted(p.name for p in CLEAN_IMAGES_DIR.glob("*.jpg"))
    names = all_names[SKIP_FIRST:SKIP_FIRST + N_EVAL]
    if len(names) < N_EVAL:
        raise RuntimeError("Not enough images in coco128 beyond the first 30.")
    return names


def build_condition(names, distortion, level):
    """Create TMP_DIR/{images,labels}/val for one condition, generating the
    distorted images on the fly. Returns (yaml_path, mean_snr_db)."""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    img_out = TMP_DIR / "images" / "val"
    lbl_out = TMP_DIR / "labels" / "val"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    snrs = []
    for name in names:
        clean = cv2.imread(str(CLEAN_IMAGES_DIR / name))
        if distortion == "none":
            out_img = clean
        else:
            out_img = DISTORTION_FUNCS[distortion](clean, level)
            s = calculate_snr(clean, out_img)
            if np.isfinite(s):
                snrs.append(s)
        cv2.imwrite(str(img_out / name), out_img)
        label = CLEAN_LABELS_DIR / (Path(name).stem + ".txt")
        if label.exists():
            shutil.copy2(label, lbl_out / label.name)

    yaml_path = TMP_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"path": str(TMP_DIR), "train": "images/val",
                        "val": "images/val", "names": YOLO(str(PRETRAINED)).names}, f)
    snr = round(float(np.mean(snrs)), 2) if snrs else None
    return yaml_path, snr


def main():
    if not FINETUNED.exists():
        raise FileNotFoundError(
            f"{FINETUNED} not found. Place best.pt at that path first.")
    print("\U0001F680 GT-based mAP: pretrained vs. fine-tuned "
          f"(on {N_EVAL} fresh images the model never saw)")

    names = fresh_image_names()
    print(f"   Eval images (#{SKIP_FIRST + 1}-#{SKIP_FIRST + N_EVAL}): "
          + ", ".join(n[-8:] for n in names))

    conditions = [("clean", "none", 0)] + [
        ("distorted", d, l) for d in DISTORTION_FUNCS for l in LEVELS]

    models = [("pretrained", YOLO(str(PRETRAINED))),
              ("fine-tuned", YOLO(str(FINETUNED)))]

    rows = []
    for condition, distortion, level in conditions:
        yaml_path, snr = build_condition(names, distortion, level)
        for label, model in models:
            print(f"\U0001F50D {label:11s} | {condition:9s} | {distortion:15s} | L{level}")
            m = model.val(data=str(yaml_path), batch=1, verbose=False,
                          plots=False, project=str(TMP_DIR / "runs"),
                          name="val", exist_ok=True)
            rows.append({
                "model": label, "condition": condition,
                "distortion_type": distortion, "level": level,
                "mean_snr_db": snr,
                "map50": round(float(m.box.map50), 4),
                "map50_95": round(float(m.box.map), 4),
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "condition", "distortion_type",
                                          "level", "mean_snr_db", "map50", "map50_95"])
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print(f"\u2705 Done! {len(rows)} rows -> {OUTPUT_CSV}")
    print("   Report note: evaluated on 10 COCO128 images outside the "
          "pipeline's 30 — unseen by the fine-tuned model by construction.")


if __name__ == "__main__":
    main()