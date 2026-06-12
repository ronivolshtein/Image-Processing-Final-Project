import os
import cv2
import csv
from yolo_tasks import YoloTasks
# ייבוא פונקציות העיוות הרשמיות שלך מהמודול distortions.py
from distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur,
    calculate_snr
)

class DLExperimentRunner:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
        # 1. טעינת מודלי ה-DL (YOLO Object Detection & Segmentation)
        self.yolo = YoloTasks(det_model_path="yolov8n.pt", seg_model_path="yolov8n-seg.pt")

        # 2. טעינת התמונה הבודדת לניסוי - בדיוק כמו בקוד של רוני
        self.img_path = os.path.join(base_dir, "000000000009.jpg")
        self.img = cv2.imread(self.img_path)

        if self.img is None:
            raise ValueError(f"Image not found at {self.img_path}")

        # מיפוי פונקציות העיוות שלך עם השמות המעודכנים
        self.distortion_funcs = {
            "gaussian_noise": apply_gaussian_noise,
            "salt_pepper": apply_salt_and_pepper_noise,
            "low_light": apply_low_light,
            "motion_blur": apply_motion_blur
        }

    def run_yolo_experiments(self, output_csv="data/output/dl_results.csv"):
        results = []

        # --- BASELINE: הרצה על תמונה נקייה (Clean RGB) ---
        clean_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        det_clean_res = self.yolo.detect_objects(clean_rgb)
        seg_clean_res = self.yolo.segment_instances(clean_rgb)
        
        clean_det_count = len(det_clean_res[0].boxes) if det_clean_res else 0
        clean_seg_count = len(seg_clean_res[0].masks) if seg_clean_res and seg_clean_res[0].masks is not None else 0
        
        results.append(["clean", 0, float("inf"), clean_det_count, clean_seg_count])
        print(f"[DL] clean -> Detected: {clean_det_count}, Segmented: {clean_seg_count}")

        # --- DISTORTIONS: הרצה על 4 עיוותים X 4 רמות עוצמה ---
        for name, func in self.distortion_funcs.items():
            for level in range(1, 5):
                # הפעלת העיוות בזיכרון (BGR)
                distorted_bgr = func(self.img, level)
                
                # חישוב מדד ה-SNR המדויק שלך
                snr_val = calculate_snr(self.img, distorted_bgr)
                
                # המרה ל-RGB עבור מודל ה-YOLO
                distorted_rgb = cv2.cvtColor(distorted_bgr, cv2.COLOR_BGR2RGB)
                
                # הרצת משימות ה-DL
                det_res = self.yolo.detect_objects(distorted_rgb)
                seg_res = self.yolo.segment_instances(distorted_rgb)
                
                # ספירת התוצאות מהמודלים
                det_count = len(det_res[0].boxes) if det_res else 0
                seg_count = len(seg_res[0].masks) if seg_res and seg_res[0].masks is not None else 0
                
                results.append([name, level, snr_val, det_count, seg_count])
                print(f"[DL] {name} (Level {level}) -> SNR: {snr_val:.2f} dB | Det: {det_count}, Seg: {seg_count}")

        # שמירת התוצאות ל-CSV
        self._save_csv(output_csv, results, ["distortion", "level", "snr", "detected_objects", "segmented_instances"])

    def _save_csv(self, path, data, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)


def run_all_dl():
    # ניהול נתיבים דינמי שתואם למחשב שלך ושל רוני
    matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
    roni_path = "/home/roni/datasets/coco128/images/train2017"

    if os.path.exists(matan_path):
        BASE_DIR = matan_path
    elif os.path.exists(roni_path):
        BASE_DIR = roni_path
    else:
        BASE_DIR = "datasets/coco128/images/train2017"

    runner = DLExperimentRunner(BASE_DIR)
    print("\n--- Running YOLO Deep Learning Experiments (Single Image) ---")
    runner.run_yolo_experiments()


if __name__ == "__main__":
    run_all_dl()