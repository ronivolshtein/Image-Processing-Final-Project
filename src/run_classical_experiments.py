import cv2
import os
import csv
import numpy as np
from classical_tasks import ClassicalTasks
# אינטגרציה מלאה עם קובץ העיוותים המעודכן שלך
from distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur,
    calculate_snr
)

class ClassicalExperimentRunner:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ct = ClassicalTasks()

        # טוענים תמונה אחת לניסויים
        self.img_path = os.path.join(base_dir, "000000000009.jpg")
        self.img = cv2.imread(self.img_path)    

        if self.img is None:
            raise ValueError(f"Image not found at {self.img_path}")

        # לוקחים patch קטן מהתמונה כ-template
        self.template = self.img[100:200, 100:200]

        # שימוש בלעדי ב-4 העיוותים הרשמיים שלך (כל עיוות ירוץ ב-4 רמות)
        self.distortion_funcs = {
            "gaussian_noise": apply_gaussian_noise,
            "salt_pepper": apply_salt_and_pepper_noise,
            "low_light": apply_low_light,
            "motion_blur": apply_motion_blur
        }

    def _save_csv(self, path, data, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    # -------------------------
    # TEMPLATE MATCHING
    # -------------------------
    def run_template_matching(self, output_csv="data/output/template_results.csv"):
        results = []
        vis_dir = "data/output_images"
        os.makedirs(vis_dir, exist_ok=True)

        # --- BASELINE: הרצה על התמונה הנקייה המקורית ---
        result_map, best_loc, score = self.ct.template_match(self.img, self.template)
        
        # שמירת פלט ויזואלי נקי לבאזליין
        clean_vis = self.img.copy()
        h_t, w_t = self.template.shape[:2]
        cv2.rectangle(clean_vis, best_loc, (best_loc[0] + w_t, best_loc[1] + h_t), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(vis_dir, "task_3_clean_baseline_template.jpg"), clean_vis)
        
        # הדפסה מקורית של רוני לתמונה הנקייה
        print(f"[TEMPLATE] clean -> score: {score:.4f}, loc: {best_loc}")
        # הוספת SNR לתמונה נקייה (אינסוף או 0, נגדיר 100 כבנצ'מרק או נחשב ישירות מולה)
        results.append(["clean", float(score), best_loc, 100.0])

        # --- EXPERIMENTS: ריצה על 4 העיוותים ב-4 הרמות שלך ---
        for name, func in self.distortion_funcs.items():
            for level in range(1, 5):
                distorted = func(self.img, level)

                # 📊 חישוב ה-SNR עבור התמונה המעוותת הנוכחית
                snr_value = calculate_snr(self.img, distorted)

                # קריאה לפונקציה של רוני עם התמונה המעוותת שלך
                result_map, best_loc, score = self.ct.template_match(
                    distorted, self.template
                )

                # 📸 יצירת התמונה הוויזואלית: ציור מלבן ירוק במיקום שנמצא
                matched_vis = distorted.copy()
                cv2.rectangle(matched_vis, best_loc, (best_loc[0] + w_t, best_loc[1] + h_t), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir, f"task_3_{name}_l{level}_template.jpg"), matched_vis)

                # שמירה והדפסה לפי הפורמט המקורי של רוני + עמודת SNR
                results.append([f"{name}_l{level}", float(score), best_loc, float(snr_value)])
                print(f"[TEMPLATE] {name}_l{level} -> score: {score:.4f}, loc: {best_loc} (SNR: {snr_value:.2f}dB)")

        # עדכון ה-Header של ה-CSV שיכלול את ה-SNR
        self._save_csv(output_csv, results, ["distortion", "score", "location", "snr"])

    # -------------------------
    # OPTICAL FLOW
    # -------------------------
    def run_optical_flow(self, output_csv="data/output/optical_flow_results.csv"):
        results = []
        vis_dir = "data/output_images"
        os.makedirs(vis_dir, exist_ok=True)

        # פריים 2 המקורי של רוני שמדמה תנועה
        img2 = apply_motion_blur(self.img, level=3)

        # --- BASELINE: הרצה על התמונות הנקיות ---
        prev_pts, next_pts, status = self.ct.optical_flow(self.img, img2)
        
        clean_vis = self.img.copy()
        if prev_pts is not None and next_pts is not None and status is not None:
            good_prev = prev_pts[status == 1]
            good_next = next_pts[status == 1]
            for pt_prev, pt_next in zip(good_prev, good_next):
                x1, y1 = map(int, pt_prev)
                x2, y2 = map(int, pt_next)
                cv2.arrowedLine(clean_vis, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.3)
        cv2.imwrite(os.path.join(vis_dir, "task_4_clean_baseline_optical_flow.jpg"), clean_vis)
        
        good_points = int(status.sum()) if status is not None else 0
        print(f"[FLOW] clean -> tracked points: {good_points}")
        results.append(["clean", good_points, 100.0])

        # --- EXPERIMENTS: ריצה על 4 העיוותים ב-4 הרמות שלך ---
        for name, func in self.distortion_funcs.items():
            for level in range(1, 5):
                # הפעלת העיוות שלך על שני הפריימים בהתאמה
                distorted1 = func(self.img, level)
                distorted2 = func(img2, level)

                # 📊 חישוב ה-SNR של הפריימים המעוותים מול המקור הנקי
                snr_value = calculate_snr(self.img, distorted1)

                # קריאה לפונקציה של רוני
                prev_pts, next_pts, status = self.ct.optical_flow(
                    distorted1, distorted2
                )

                # 📸 יצירת התמונה הוויזואלית: ציור וקטורי התנועה (חיצים אדומים) על גבי הפריים המעוות הראשון
                flow_vis = distorted1.copy()
                if prev_pts is not None and next_pts is not None and status is not None:
                    good_prev = prev_pts[status == 1]
                    good_next = next_pts[status == 1]
                    for pt_prev, pt_next in zip(good_prev, good_next):
                        x1, y1 = map(int, pt_prev)
                        x2, y2 = map(int, pt_next)
                        cv2.arrowedLine(flow_vis, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.3)
                
                cv2.imwrite(os.path.join(vis_dir, f"task_4_{name}_l{level}_optical_flow.jpg"), flow_vis)

                # שמירה והדפסה לפי הפורמט המקורי של רוני + עמודת SNR
                good_points = int(status.sum()) if status is not None else 0
                results.append([f"{name}_l{level}", good_points, float(snr_value)])
                print(f"[FLOW] {name}_l{level} -> tracked points: {good_points} (SNR: {snr_value:.2f}dB)")

        # עדכון ה-Header של ה-CSV שיכלול את ה-SNR
        self._save_csv(output_csv, results, ["distortion", "tracked_points", "snr"])


def run_all_classical():
    # ניהול נתיבים דינמי ובטוח עבור מתן (Windows) ורוני (Linux)
    matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
    roni_path = "/home/roni/datasets/coco128/images/train2017"

    if os.path.exists(matan_path):
        BASE_DIR = matan_path
    elif os.path.exists(roni_path):
        BASE_DIR = roni_path
    else:
        BASE_DIR = "datasets/coco128/images/train2017"

    runner = ClassicalExperimentRunner(BASE_DIR)

    print("\n--- TEMPLATE MATCHING EXPERIMENTS ---")
    runner.run_template_matching()

    print("\n--- OPTICAL FLOW EXPERIMENTS ---")
    runner.run_optical_flow()


if __name__ == "__main__":
    run_all_classical()