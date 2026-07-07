"""
Week 2 - Enhancements (creation step): applies the classical enhancement matched to
each distortion type on the already-generated distorted images and saves the result
to data/enhanced_images/ (mirrors the data/distorted_images/ layout). No CSV writing,
no YOLO - see evaluate_enhancements.py for the task-evaluation step that consumes
this output.
"""
import cv2
import pandas as pd
from pathlib import Path

from enhancements import ENHANCEMENT_FOR_DISTORTION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / 'data' / 'tasks_graphs_and_tables' / 'metadata_summary_base.csv'
ENHANCED_IMAGES_DIR = PROJECT_ROOT / 'data' / 'enhanced_images'


def main():
    print(f"Loading metadata from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    combos = (
        df[df['distortion_type'] != 'clean']
        [['image_name', 'distortion_type', 'level', 'distorted_image_path']]
        .drop_duplicates(subset=['image_name', 'distortion_type', 'level'])
        .reset_index(drop=True)
    )
    print(f"Found {len(combos)} distorted images to enhance.")

    for _, r in combos[['distortion_type', 'level']].drop_duplicates().iterrows():
        subdir = f"{r['distortion_type']}_l{int(r['level'])}"
        (ENHANCED_IMAGES_DIR / subdir).mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, row in combos.iterrows():
        img_name = row['image_name']
        distortion_type = row['distortion_type']
        level = int(row['level'])

        distorted_img = cv2.imread(row['distorted_image_path'])
        if distorted_img is None:
            print(f"  [WARN] Could not load {row['distorted_image_path']}, skipping.")
            continue

        enhanced_img = ENHANCEMENT_FOR_DISTORTION[distortion_type](distorted_img)

        out_path = ENHANCED_IMAGES_DIR / f"{distortion_type}_l{level}" / img_name
        cv2.imwrite(str(out_path), enhanced_img)
        saved += 1

        if (idx + 1) % 50 == 0 or idx + 1 == len(combos):
            print(f"[{idx + 1}/{len(combos)}] enhanced and saved.")

    print(f"\nDone! Saved {saved} enhanced images under {ENHANCED_IMAGES_DIR}")


if __name__ == '__main__':
    main()
