import numpy as np
from ultralytics import YOLO

class YoloTasks:
    def __init__(self, det_model_path="yolov8n.pt", seg_model_path="yolov8n-seg.pt"):
        """
        Loads the YOLO models for object detection and instance segmentation.
        The model weights will be downloaded automatically on the first run.
        """
        # Load pre-trained models
        self.det_model = YOLO(det_model_path)
        self.seg_model = YOLO(seg_model_path)

    def detect_objects(self, img_array, conf_threshold=0.25):
        """
        Runs object detection on a clean or distorted image.
        img_array: Image in NumPy array format (RGB)
        """
        # imgsz=320 is used to keep inference fast on weak hardware
        results = self.det_model.predict(img_array, imgsz=320, conf=conf_threshold, verbose=False)
        return results # Returns a single Results object containing boxes, classes, etc.

    def segment_instances(self, img_array, conf_threshold=0.25):
        """
        Runs instance segmentation on a clean or distorted image.
        img_array: Image in NumPy array format (RGB)
        """
        results = self.seg_model.predict(img_array, imgsz=320, conf=conf_threshold, verbose=False)
        return results # Returns a single Results object containing masks, boxes, etc.

    def evaluate_baseline_mAP(self, data_yaml="coco128.yaml", task="detect"):
        """
        Helper function to run full validation on the entire dataset 
        in order to get the baseline metrics (mAP).
        """
        print(f"Evaluating {task} baseline mAP...")
        if task == "detect":
            metrics = self.det_model.val(data=data_yaml, imgsz=320, batch=1)
        elif task == "segment":
            metrics = self.seg_model.val(data=data_yaml, imgsz=320, batch=1)
        return metrics