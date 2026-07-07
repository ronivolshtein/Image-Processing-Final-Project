"""
Week 2 - Enhancements (evaluation step): loads the enhanced images produced by
apply_enhancements.py, runs all 4 tasks on them, saves the annotated task images,
and appends 'Enhanced' rows to the shared metadata_summary_base.csv (same
read -> append -> save pattern evaluate_finetuned.py uses for its Fine-Tuned rows).

Run apply_enhancements.py first.
"""
import os
import cv2
import pandas as pd
from pathlib import Path

from distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur
)
from enhancements import ENHANCEMENT_FOR_DISTORTION
from run_classical_experiments import evaluate_optical_flow, evaluate_template_matching
from run_dl_experiments import evaluate_object_detection, evaluate_segment_instances
from yolo_tasks import YoloTasks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / 'data' / 'tasks_graphs_and_tables' / 'metadata_summary_base.csv'
ENHANCED_IMAGES_DIR = PROJECT_ROOT / 'data' / 'enhanced_images'
ENHANCED_TASK_IMAGES_DIR = PROJECT_ROOT / 'data' / 'tasks_applied_on_enhanced'

TASK_NAMES = ['optical_flow', 'template_matching', 'segment_instances', 'object_detection']

# Only needed to regenerate optical flow's synthetic second frame - it was never
# saved to disk by anyone in this project (see make_template_and_frame2 below).
DISTORTION_FUNCS = {
    'gaussian_noise': apply_gaussian_noise,
    'salt_pepper': apply_salt_and_pepper_noise,
    'low_light': apply_low_light,
    'motion_blur': apply_motion_blur
}


def resolve_dataset_dir():
    matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
    roni_path = "/home/roni/datasets/coco128/images/train2017"
    if os.path.exists(matan_path):
        return matan_path
    if os.path.exists(roni_path):
        return roni_path
    return "datasets/coco128/images/train2017"


def make_template_and_frame2(clean_img):
    """Mirrors Week1BulkRunner._create_template_and_frame2 so results stay comparable."""
    h, w = clean_img.shape[:2]
    cy, cx = h // 2, w // 2
    template = clean_img[max(0, cy - 50):min(h, cy + 50), max(0, cx - 50):min(w, cx + 50)]
    frame2 = apply_motion_blur(clean_img, level=3)
    return template, frame2


def run_all_tasks(enhanced_img, enhanced_frame2, template, yolo_tasks, distortion_type, level):
    results = {}
    img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

    try:
        results['optical_flow'] = evaluate_optical_flow(enhanced_img, enhanced_frame2)
    except Exception as e:
        print(f"  [WARN] optical_flow failed for {distortion_type}_l{level}: {e}")
        results['optical_flow'] = {'metrics': {'tracked_points': 0}, 'visualized_image': enhanced_img.copy()}

    try:
        results['template_matching'] = evaluate_template_matching(enhanced_img, template)
    except Exception as e:
        print(f"  [WARN] template_matching failed for {distortion_type}_l{level}: {e}")
        results['template_matching'] = {'metrics': {'matching_score': 0.0, 'location': '(0,0)'}, 'visualized_image': enhanced_img.copy()}

    try:
        results['segment_instances'] = evaluate_segment_instances(img_rgb, yolo_tasks.seg_model)
    except Exception as e:
        print(f"  [WARN] segment_instances failed for {distortion_type}_l{level}: {e}")
        results['segment_instances'] = {'metrics': {'segmented_instances': 0, 'avg_confidence': 0.0}, 'visualized_image': enhanced_img.copy()}

    try:
        results['object_detection'] = evaluate_object_detection(img_rgb, yolo_tasks.det_model)
    except Exception as e:
        print(f"  [WARN] object_detection failed for {distortion_type}_l{level}: {e}")
        results['object_detection'] = {'metrics': {'detected_objects': 0, 'avg_confidence': 0.0}, 'visualized_image': enhanced_img.copy()}

    return results


def main():
    print(f"Loading metadata from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    if 'model_type' not in df.columns:
        df['model_type'] = 'Baseline'

    dataset_dir = resolve_dataset_dir()
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return

    combos = (
        df[df['distortion_type'] != 'clean']
        [['image_name', 'distortion_type', 'level']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"Found {len(combos)} (image, distortion, level) combinations to evaluate.")

    for _, r in combos[['distortion_type', 'level']].drop_duplicates().iterrows():
        subdir = f"{r['distortion_type']}_l{int(r['level'])}"
        for task_name in TASK_NAMES:
            (ENHANCED_TASK_IMAGES_DIR / task_name / subdir).mkdir(parents=True, exist_ok=True)

    yolo_tasks = YoloTasks(
        det_model_path=str(PROJECT_ROOT / "yolov8n.pt"),
        seg_model_path=str(PROJECT_ROOT / "yolov8n-seg.pt")
    )

    new_records = []
    clean_cache = {}
    missing_enhanced = 0

    for idx, combo in combos.iterrows():
        img_name = combo['image_name']
        distortion_type = combo['distortion_type']
        level = int(combo['level'])

        print(f"[{idx + 1}/{len(combos)}] Evaluating {img_name} | {distortion_type} L{level}")

        enhanced_img_path = ENHANCED_IMAGES_DIR / f"{distortion_type}_l{level}" / img_name
        enhanced_img = cv2.imread(str(enhanced_img_path))
        if enhanced_img is None:
            print(f"  [WARN] Enhanced image not found at {enhanced_img_path} - run apply_enhancements.py first. Skipping.")
            missing_enhanced += 1
            continue

        if img_name not in clean_cache:
            clean_path = os.path.join(dataset_dir, img_name)
            clean_img = cv2.imread(clean_path)
            if clean_img is None:
                print(f"  [WARN] Could not load clean image {clean_path}, skipping.")
                continue
            clean_cache[img_name] = make_template_and_frame2(clean_img)
        template, frame2 = clean_cache[img_name]

        distorted_row = df[
            (df['image_name'] == img_name) &
            (df['distortion_type'] == distortion_type) &
            (df['level'] == level)
        ].iloc[0]

        snr_before = float(distorted_row['snr_distorted_db'])

        # Optical flow's second frame is regenerated here and enhanced with the same
        # function used on the main image, so both tracked frames stay consistent.
        distorted_frame2 = DISTORTION_FUNCS[distortion_type](frame2, level)
        enhanced_frame2 = ENHANCEMENT_FOR_DISTORTION[distortion_type](distorted_frame2)

        task_results = run_all_tasks(enhanced_img, enhanced_frame2, template, yolo_tasks, distortion_type, level)

        for task_name, task_data in task_results.items():
            out_subdir = ENHANCED_TASK_IMAGES_DIR / task_name / f"{distortion_type}_l{level}"
            task_image_path = out_subdir / img_name
            cv2.imwrite(str(task_image_path), task_data['visualized_image'])

            for metric_name, metric_value in task_data['metrics'].items():
                try:
                    final_value = float(metric_value)
                except (TypeError, ValueError):
                    final_value = str(metric_value)

                new_records.append({
                    'image_name': img_name,
                    'distortion_type': distortion_type,
                    'level': level,
                    'snr_distorted_db': snr_before,
                    'task_name': task_name,
                    'metric_name': metric_name,
                    'metric_value': final_value,
                    'task_image_path': str(task_image_path).replace('\\', '/'),
                    'original_image_path': str(distorted_row['original_image_path']),
                    'distorted_image_path': str(distorted_row['distorted_image_path']),
                    'model_type': 'Enhanced'
                })

    if missing_enhanced:
        print(f"\n[WARN] Skipped {missing_enhanced} combos with no enhanced image on disk.")

    if not new_records:
        print("No new rows generated - nothing to append.")
        return

    new_df = pd.DataFrame(new_records)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(CSV_PATH, index=False)

    print(f"\nSuccess! Appended {len(new_df)} 'Enhanced' rows to {CSV_PATH}.")
    print(f"Enhanced task visualizations saved under {ENHANCED_TASK_IMAGES_DIR}")


if __name__ == '__main__':
    main()
