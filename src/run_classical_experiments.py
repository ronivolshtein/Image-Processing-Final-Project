import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classical_tasks import ClassicalTasks
from distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur,
    calculate_snr
)

class ClassicalExperimentRunner:
    def __init__(self, base_dir, distortion_images_dir, distortion_grids_dir, graphs_tables_dir, task_images_dir):
        self.base_dir = base_dir
        self.distortion_images_dir = distortion_images_dir
        self.distortion_grids_dir = distortion_grids_dir
        self.graphs_tables_dir = graphs_tables_dir
        self.task_images_dir = task_images_dir
        self.ct = ClassicalTasks()

        # Load one representative image for experiments
        self.img_path = os.path.join(base_dir, "000000000009.jpg")
        self.img = cv2.imread(self.img_path)    

        if self.img is None:
            raise ValueError(f"Image not found at {self.img_path}")

        # Crop a small patch to use as a template for Task 3
        self.template = self.img[100:200, 100:200]

        # Official distortions
        self.distortion_funcs = {
            "gaussian_noise": apply_gaussian_noise,
            "salt_pepper": apply_salt_and_pepper_noise,
            "low_light": apply_low_light,
            "motion_blur": apply_motion_blur
        }

    def save_before_after_grid(self, clean, distorted, distortion_name, level):
        """
        Saves side-by-side distortion grids inside distortion_before_after
        """
        grid = np.hstack((clean, distorted))
        cv2.putText(grid, "Original (GT)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(grid, f"Distorted ({distortion_name} L{level})", (clean.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        path = os.path.join(self.distortion_grids_dir, f"{distortion_name}_l{level}_before_after.jpg")
        cv2.imwrite(path, grid)

    def run_template_matching(self):
        print("\n--- Running Template Matching Degradation Experiments ---")
        results = []
        degradation_data = []

        # Baseline: Run on original clean image
        _, best_loc, score = self.ct.template_match(self.img, self.template)
        clean_vis = self.img.copy()
        h_t, w_t = self.template.shape[:2]
        cv2.rectangle(clean_vis, best_loc, (best_loc[0] + w_t, best_loc[1] + h_t), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.task_images_dir, "task_3_clean_baseline_template.jpg"), clean_vis)
        
        results.append(["clean", 0, 100.0, float(score), str(best_loc)])

        # Experiments: Run across 4 distortions and 4 levels
        for name, func in self.distortion_funcs.items():
            for level in range(1, 5):
                distorted = func(self.img, level)
                snr_value = calculate_snr(self.img, distorted)

                # Save raw distorted images and grids to respective folders
                cv2.imwrite(os.path.join(self.distortion_images_dir, f"{name}_l{level}.jpg"), distorted)
                self.save_before_after_grid(self.img, distorted, name, level)

                # Execute template matching
                _, best_loc, score = self.ct.template_match(distorted, self.template)

                # Draw bounding box on the distorted image (Saved to task_images)
                matched_vis = distorted.copy()
                cv2.rectangle(matched_vis, best_loc, (best_loc[0] + w_t, best_loc[1] + h_t), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(self.task_images_dir, f"annotated_task_3_{name}_l{level}.jpg"), matched_vis)

                results.append([name, level, float(snr_value), float(score), str(best_loc)])
                degradation_data.append({"Distortion": name, "Level": level, "Score": float(score)})

        # Save Performance Table into tasks_graphs_and_tables
        df = pd.DataFrame(results, columns=["Distortion", "Level", "SNR", "Matching_Score", "Location"])
        df.to_csv(os.path.join(self.graphs_tables_dir, "classical_template_matching_performance.csv"), index=False)
        
        # Plot Degradation Chart inside tasks_graphs_and_tables
        self.plot_classical_degradation(degradation_data, "Template Matching Score", "template_matching_degradation.png")

    def run_optical_flow(self):
        print("\n--- Running Optical Flow Degradation Experiments ---")
        results = []
        degradation_data = []

        img2 = apply_motion_blur(self.img, level=3)

        # Baseline: Run on clean frames
        prev_pts, next_pts, status = self.ct.optical_flow(self.img, img2)
        clean_vis = self.img.copy()
        if prev_pts is not None and next_pts is not None and status is not None:
            good_prev = prev_pts[status == 1]
            good_next = next_pts[status == 1]
            for pt_prev, pt_next in zip(good_prev, good_next):
                x1, y1 = map(int, pt_prev)
                x2, y2 = map(int, pt_next)
                cv2.arrowedLine(clean_vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
        cv2.imwrite(os.path.join(self.task_images_dir, "task_4_clean_baseline_optical_flow.jpg"), clean_vis)
        
        baseline_points = int(status.sum()) if status is not None else 0
        results.append(["clean", 0, 100.0, baseline_points])

        for name, func in self.distortion_funcs.items():
            for level in range(1, 5):
                distorted1 = func(self.img, level)
                distorted2 = func(img2, level)
                snr_value = calculate_snr(self.img, distorted1)

                # Execute optical flow tracking
                prev_pts, next_pts, status = self.ct.optical_flow(distorted1, distorted2)

                # Draw tracking arrows (Saved to task_images)
                flow_vis = distorted1.copy()
                tracked_points = 0
                if prev_pts is not None and next_pts is not None and status is not None:
                    tracked_points = int(status.sum())
                    good_prev = prev_pts[status == 1]
                    good_next = next_pts[status == 1]
                    for pt_prev, pt_next in zip(good_prev, good_next):
                        x1, y1 = map(int, pt_prev)
                        x2, y2 = map(int, pt_next)
                        cv2.arrowedLine(flow_vis, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.3)
                
                cv2.imwrite(os.path.join(self.task_images_dir, f"annotated_task_4_{name}_l{level}.jpg"), flow_vis)
                
                results.append([name, level, float(snr_value), tracked_points])
                degradation_data.append({"Distortion": name, "Level": level, "Score": tracked_points})

        # Save Performance Table into tasks_graphs_and_tables
        df = pd.DataFrame(results, columns=["Distortion", "Level", "SNR", "Tracked_Points"])
        df.to_csv(os.path.join(self.graphs_tables_dir, "classical_optical_flow_performance.csv"), index=False)
        
        # Plot Degradation Chart inside tasks_graphs_and_tables
        self.plot_classical_degradation(degradation_data, "Tracked Keypoints Count", "optical_flow_degradation.png")

    def plot_classical_degradation(self, data, y_label, filename):
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 5))
        for distortion_name in df['Distortion'].unique():
            subset = df[df['Distortion'] == distortion_name]
            plt.plot(subset['Level'], subset['Score'], marker='s', linewidth=2, label=distortion_name)
            
        plt.title(f"Classical CV Performance Degradation: {y_label}")
        plt.xlabel("Distortion Intensity Level")
        plt.ylabel(y_label)
        plt.xticks([1, 2, 3, 4])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        # FIXED: self.self removed here
        plt.savefig(os.path.join(self.graphs_tables_dir, filename))
        plt.close()

def run_all_classical():
    matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
    roni_path = "/home/roni/datasets/coco128/images/train2017"
    base_dir = matan_path if os.path.exists(matan_path) else (roni_path if os.path.exists(roni_path) else "datasets/coco128/images/train2017")

    distortion_images_dir = "data/distortions_output/distortion_images"
    distortion_grids_dir = "data/distortions_output/distortion_before_after"
    graphs_tables_dir = "data/tasks_output/tasks_graphs_and_tables"
    task_images_dir = "data/tasks_output/task_images"

    os.makedirs(distortion_images_dir, exist_ok=True)
    os.makedirs(distortion_grids_dir, exist_ok=True)
    os.makedirs(graphs_tables_dir, exist_ok=True)
    os.makedirs(task_images_dir, exist_ok=True)

    runner = ClassicalExperimentRunner(base_dir, distortion_images_dir, distortion_grids_dir, graphs_tables_dir, task_images_dir)
    runner.run_template_matching()
    runner.run_optical_flow()
    print("\n✅ Classical Experiments Complete! Folders updated perfectly based on formatting requirements.")

if __name__ == "__main__":
    run_all_classical()