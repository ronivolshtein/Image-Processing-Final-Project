import os
import cv2
import pandas as pd
from src.yolo_tasks import YoloTasks

from src.distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur,
    calculate_snr
)

def run_heavy_dataset_pipeline():
    print("\n🚀 STARTING BULK DATASET PROCESSING PIPELINE (DL) 🚀")

    # אתחול מודלי ה-DL
    yolo = YoloTasks(det_model_path="yolov8n.pt", seg_model_path="yolov8n-seg.pt")

    # ניהול נתיבים דינמי
    matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
    roni_path = "/home/roni/datasets/coco128/images/train2017"

    if os.path.exists(matan_path):
        dataset_dir = matan_path
    elif os.path.exists(roni_path):
        dataset_dir = roni_path
    else:
        dataset_dir = "datasets/coco128/images/train2017"

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Directory not found at {dataset_dir}")
        return

    # לקיחת 30 התמונות הראשונות
    all_images = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_subset = all_images[:30] 
    
    distortion_funcs = {
        "gaussian_noise": apply_gaussian_noise,
        "salt_pepper": apply_salt_and_pepper_noise,
        "low_light": apply_low_light,
        "motion_blur": apply_motion_blur
    }

    pipeline_records = []

    # הלולאה המרכזית על כל ה-Dataset
    for idx, img_name in enumerate(image_subset, 1):
        img_path = os.path.join(dataset_dir, img_name)
        clean_img = cv2.imread(img_path)

        if clean_img is None:
            continue

        print(f"🔄 [{idx}/30] Processing: {img_name}")

        # Baseline: תמונה נקייה
        clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        det_clean_res = yolo.detect_objects(clean_rgb)
        seg_clean_res = yolo.segment_instances(clean_rgb)

        clean_det_count = len(det_clean_res[0].boxes) if det_clean_res else 0
        clean_seg_count = len(seg_clean_res[0].masks) if seg_clean_res and seg_clean_res[0].masks is not None else 0

        pipeline_records.append({
            "image_name": img_name,
            "distortion_type": "clean",
            "level": 0,
            "snr_db": float("inf"),
            "detected_objects": clean_det_count,
            "segmented_instances": clean_seg_count
        })

        # לולאת העיוותים והרצות ה-DL
        for dist_name, dist_func in distortion_funcs.items():
            for level in range(1, 5):
                distorted_bgr = dist_func(clean_img, level)
                snr_val = calculate_snr(clean_img, distorted_bgr)
                distorted_rgb = cv2.cvtColor(distorted_bgr, cv2.COLOR_BGR2RGB)

                det_res = yolo.detect_objects(distorted_rgb)
                seg_res = yolo.segment_instances(distorted_rgb)

                det_count = len(det_res[0].boxes) if det_res else 0
                seg_count = len(seg_res[0].masks) if seg_res and seg_res[0].masks is not None else 0

                pipeline_records.append({
                    "image_name": img_name,
                    "distortion_type": dist_name,
                    "level": level,
                    "snr_db": snr_val,
                    "detected_objects": det_count,
                    "segmented_instances": seg_count
                })

    # שמירת ה-CSV המסכם
    os.makedirs("data/output", exist_ok=True)
    df = pd.DataFrame(pipeline_records)
    df.to_csv("data/output/dataset_dl_results.csv", index=False)
    print("\n✅ PIPELINE DONE! Comprehensive CSV saved to: data/output/dataset_dl_results.csv")

if __name__ == "__main__":
    run_heavy_dataset_pipeline()