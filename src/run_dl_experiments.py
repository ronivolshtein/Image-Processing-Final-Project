import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur
)

# ================================================================
# PURE TASK EVALUATION FUNCTIONS (No Disk I/O)
# ================================================================

def evaluate_object_detection(img_rgb, model=None):
    """
    Pure object detection evaluation function.
    
    Args:
        img_rgb: Image in RGB format (NumPy array)
        model: Optional YOLO model. If None, loads yolov8n.pt
    
    Returns:
        dict: {
            'metrics': {'detected_objects': int, 'avg_confidence': float},
            'visualized_image': NumPy BGR array with bounding boxes
        }
    """
    if model is None:
        model = YOLO("yolov8n.pt")
    
    results = model.predict(img_rgb, imgsz=320, verbose=False)
    res = results[0]
    
    # Convert annotated image from RGB to BGR for consistency
    annotated_img = res.plot()
    
    boxes = res.boxes
    num_objects = len(boxes)
    avg_conf = np.mean([float(b.conf[0]) for b in boxes]) if num_objects > 0 else 0.0
    
    return {
        'metrics': {'detected_objects': num_objects, 'avg_confidence': avg_conf},
        'visualized_image': annotated_img
    }


def evaluate_segment_instances(img_rgb, model=None):
    """
    Pure instance segmentation evaluation function.
    
    Args:
        img_rgb: Image in RGB format (NumPy array)
        model: Optional YOLO model. If None, loads yolov8n-seg.pt
    
    Returns:
        dict: {
            'metrics': {'segmented_instances': int, 'avg_confidence': float},
            'visualized_image': NumPy BGR array with masks and boxes
        }
    """
    if model is None:
        model = YOLO("yolov8n-seg.pt")
    
    results = model.predict(img_rgb, imgsz=320, verbose=False)
    res = results[0]
    
    # Convert annotated image from RGB to BGR for consistency
    annotated_img = res.plot()
    
    boxes = res.boxes
    masks = res.masks
    num_instances = len(boxes)
    avg_conf = np.mean([float(b.conf[0]) for b in boxes]) if num_instances > 0 else 0.0
    
    return {
        'metrics': {'segmented_instances': num_instances, 'avg_confidence': avg_conf},
        'visualized_image': annotated_img
    }


class DeepLearningExperimentRunner:
    def __init__(self, base_dir, distortion_images_dir, distortion_grids_dir, graphs_tables_dir, task_images_dir):
        self.base_dir = base_dir
        self.distortion_images_dir = distortion_images_dir
        self.distortion_grids_dir = distortion_grids_dir
        self.graphs_tables_dir = graphs_tables_dir
        self.task_images_dir = task_images_dir
        
        self.det_model = YOLO("yolov8n.pt")
        
        self.img_name = "000000000009.jpg"
        self.img_path = os.path.join(base_dir, self.img_name)
        self.clean_img = cv2.imread(self.img_path)
        
        if self.clean_img is None:
            raise FileNotFoundError(f"Source image not found at {self.img_path}")
            
        self.distortions = {
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

    def extract_per_class_performance(self, yolo_results):
        boxes = yolo_results.boxes
        if boxes is None or len(boxes) == 0:
            return {}
            
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_results.names[cls_id]
            conf = float(box.conf[0])
            
            if cls_name not in class_counts:
                class_counts[cls_name] = []
            class_counts[cls_name].append(conf)
            
        return {cls: np.mean(confs) for cls, confs in class_counts.items()}

    def run_experiments(self):
        """Legacy method kept for backward compatibility. Deprecated - use pure functions."""
        print("🚀 Running Week 1 Deep Learning Experiments...")
        
        # Part 1: Baseline Per-Class Performance (Saved to tasks_graphs_and_tables as CSV)
        clean_img_rgb = cv2.cvtColor(self.clean_img, cv2.COLOR_BGR2RGB)
        res_clean_det = self.det_model.predict(clean_img_rgb, imgsz=320, verbose=False)[0]
        gt_per_class = self.extract_per_class_performance(res_clean_det)
        
        df_gt = pd.DataFrame(list(gt_per_class.items()), columns=['Class', 'Baseline Confidence (mAP Proxy)'])
        df_gt.to_csv(os.path.join(self.graphs_tables_dir, "baseline_per_class_performance.csv"), index=False)

        degradation_data = []

        for name, func in self.distortions.items():
            for level in range(1, 5):
                # Apply and save raw distorted images and grids to respective directories
                distorted = func(self.clean_img, level)
                cv2.imwrite(os.path.join(self.distortion_images_dir, f"{name}_l{level}.jpg"), distorted)
                self.save_before_after_grid(self.clean_img, distorted, name, level)
                
                # Run Model on Distorted Data via pure function
                distorted_rgb = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
                result = evaluate_object_detection(distorted_rgb, self.det_model)
                cv2.imwrite(os.path.join(self.task_images_dir, f"annotated_{name}_l{level}.jpg"), result['visualized_image'])
                
                degradation_data.append({"Distortion": name, "Level": level, "Score": result['metrics']['avg_confidence']})
                
        # Plot Degradation Chart inside tasks_graphs_and_tables
        self.plot_degradation(degradation_data)

    def plot_degradation(self, data):
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        for distortion_name in df['Distortion'].unique():
            subset = df[df['Distortion'] == distortion_name]
            plt.plot(subset['Level'], subset['Score'], marker='o', linewidth=2, label=distortion_name)
            
        plt.title("YOLOv8 Performance Degradation Analysis")
        plt.xlabel("Distortion Intensity Level")
        plt.ylabel("Detection Confidence Score (mAP Proxy)")
        plt.xticks([1, 2, 3, 4])
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_tables_dir, "deep_learning_degradation_chart.png"))
        plt.close()

def main():
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
    
    runner = DeepLearningExperimentRunner(base_dir, distortion_images_dir, distortion_grids_dir, graphs_tables_dir, task_images_dir)
    runner.run_experiments()
    print("\n✅ Deep Learning Experiments Complete! Folders updated perfectly based on formatting requirements.")

if __name__ == "__main__":
    main()