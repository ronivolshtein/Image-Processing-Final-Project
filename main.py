import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from yolo_tasks import YoloTasks
from run_classical_experiments import run_all_classical
# import cv2
# import os
# from src.classical_tasks import ClassicalTasks


def main():
    print("Starting the vision project...")

    # --- YOLO part ---

    # 1. Initialize YOLO models (This will automatically download yolov8n.pt and yolov8n-seg.pt)
    print("Loading models...")
    yolo = YoloTasks(det_model_path="yolov8n.pt",
                     seg_model_path="yolov8n-seg.pt")

    # 2. Evaluate baseline to trigger COCO128 dataset download and get clean metrics
    print("\n--- Running Detection Baseline ---")
    # For Detection: Using the regular coco128 dataset
    det_metrics = yolo.evaluate_baseline_mAP(
        data_yaml="coco128.yaml", task="detect")
    print(f"Detection mAP50-95: {det_metrics.box.map:.4f}")

    print("\n--- Running Segmentation Baseline ---")
    # For Segmentation: Using the coco128-seg dataset which contains masks
    seg_metrics = yolo.evaluate_baseline_mAP(
        data_yaml="coco128-seg.yaml", task="segment")
    print(f"Segmentation mAP50-95: {seg_metrics.seg.map:.4f}")

    print("\nDone! Data and models are loaded.")
    # END



    # --- CLASSICAL TASKS part ---    
    print("\n==============================\n")

    run_classical_classical = run_all_classical
    run_classical_classical()

    print("\nDONE: All experiments finished.")
    # END

if __name__ == "__main__":
    main()
