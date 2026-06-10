from src.yolo_tasks import YoloTasks

def main():
    print("Starting the vision project...")

    # 1. Initialize YOLO models (This will automatically download yolov8n.pt and yolov8n-seg.pt)
    print("Loading models...")
    yolo = YoloTasks(det_model_path="yolov8n.pt", seg_model_path="yolov8n-seg.pt")

    # 2. Evaluate baseline to trigger COCO128 dataset download and get clean metrics
    print("\n--- Running Detection Baseline ---")
    # For Detection: Using the regular coco128 dataset
    det_metrics = yolo.evaluate_baseline_mAP(data_yaml="coco128.yaml", task="detect")
    print(f"Detection mAP50-95: {det_metrics.box.map:.4f}")

    print("\n--- Running Segmentation Baseline ---")
    # For Segmentation: Using the coco128-seg dataset which contains masks
    seg_metrics = yolo.evaluate_baseline_mAP(data_yaml="coco128-seg.yaml", task="segment")
    print(f"Segmentation mAP50-95: {seg_metrics.seg.map:.4f}")

    print("\nDone! Data and models are loaded.")

if __name__ == "__main__":
    main()