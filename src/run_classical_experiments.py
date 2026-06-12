import cv2
import os
import csv
import numpy as np
from classical_tasks import ClassicalTasks
# Matching your exact function names from distortions.py
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

        # 4 עיוותים + clean
        self.distortion_funcs = {
            "gaussian_noise": apply_gaussian_noise,
            "salt_pepper": apply_salt_and_pepper_noise,
            "low_light": apply_low_light,
            "motion_blur": apply_motion_blur
        }

    # # -------------------------
    # # פונקציות עיוות
    # # -------------------------
    # def add_gaussian_noise(self, img):
    #     noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    #     return cv2.add(img, noise)

    # def add_salt_pepper(self, img, amount=0.02):
    #     out = img.copy()
    #     h, w, c = out.shape
    #     num_pixels = int(amount * h * w)
    #     coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
    #     out[coords[0], coords[1]] = 255
    #     coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
    #     out[coords[0], coords[1]] = 0
    #     return out

    # def motion_blur(self, img):
    #     kernel = np.zeros((15, 15))
    #     kernel[7, :] = np.ones(15)
    #     kernel = kernel / 15
    #     return cv2.filter2D(img, -1, kernel)

    # -----------------------------------------------------------------
    # TEMPLATE MATCHING EXPERIMENT (Iterating over 4 levels + SNR)
    # -----------------------------------------------------------------
    def run_template_matching(self, output_csv="data/output/template_results.csv"):
        results = []

        # --- BASELINE: Run on clean image first ---
        _, _, clean_score = self.ct.template_match(self.img, self.template)
        results.append(["clean", 0, float("inf"), clean_score])
        print(f"[TEMPLATE] clean -> score: {clean_score:.4f}")

        # --- DISTORTIONS: Run on 4 types x 4 intensity levels ---
        for name, func in self.distortion_funcs.items():
            for level in range(1, 5):
                # Apply your exact distortion level
                distorted = func(self.img, level)
                
                # Compute exact scientific SNR
                snr_val = calculate_snr(self.img, distorted)

                # Run classical template matching on the distorted image
                _, _, score = self.ct.template_match(distorted, self.template)

                results.append([name, level, snr_val, float(score)])
                print(f"[TEMPLATE] {name} (Level {level}) -> SNR: {snr_val:.2f} dB, score: {score:.4f}")

        self._save_csv(output_csv, results, ["distortion", "level", "snr", "score"])
    
    # -----------------------------------------------------------------
    # OPTICAL FLOW EXPERIMENT (Iterating over 4 levels + SNR)
    # -----------------------------------------------------------------
    def run_optical_flow(self, output_csv="data/output/optical_flow_results.csv"):
        results = []

        # Generate Frame 2 by applying a standard light movement simulation (using your Motion Blur level 1)
        img2 = apply_motion_blur(self.img, level=1)

        # --- BASELINE: Run on clean frames first ---
        _, _, status = self.ct.optical_flow(self.img, img2)
        clean_tracked = int(status.sum()) if status is not None else 0
        results.append(["clean", 0, float("inf"), clean_tracked])
        print(f"[FLOW] clean -> tracked points: {clean_tracked}")

        # --- DISTORTIONS: Run on 4 types x 4 intensity levels ---
        for name, func in self.distortion_funcs.items():
            for level in range(1, 5):
                # Apply same distortion level symmetrically to both consecutive video frames
                distorted1 = func(self.img, level)
                distorted2 = func(img2, level)

                # Compute SNR based on Frame 1
                snr_val = calculate_snr(self.img, distorted1)

                # Run classical Lucas-Kanade optical flow
                _, _, status = self.ct.optical_flow(distorted1, distorted2)

                good_points = int(status.sum()) if status is not None else 0
                results.append([name, level, snr_val, good_points])
                print(f"[FLOW] {name} (Level {level}) -> SNR: {snr_val:.2f} dB, tracked points: {good_points}")

        self._save_csv(output_csv, results, ["distortion", "level", "snr", "tracked_points"])
    
    # -----------------------------------------------------------------
    def _save_csv(self, path, data, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

def run_all_classical():
    # Dynamic path handling for both Matan (Windows) and Roni (Linux)
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