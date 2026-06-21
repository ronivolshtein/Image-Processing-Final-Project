"""
REFACTORING SUMMARY & VALIDATION GUIDE
=====================================

This document outlines the refactored architecture for Week 1 of the Computer Vision project.

## ARCHITECTURE OVERVIEW

### STEP 1: Pure Task Evaluation Functions (No Disk I/O)

#### File: run_classical_experiments.py
Two new pure functions exposed:

1. evaluate_optical_flow(prev_img, next_img) -> dict
   - Input: Two BGR image arrays
   - Output: {
       'metrics': {'tracked_points': int},
       'visualized_image': BGR array with flow arrows
     }
   - No disk I/O inside function

2. evaluate_template_matching(img, template) -> dict
   - Input: BGR image and template array
   - Output: {
       'metrics': {'matching_score': float, 'location': str},
       'visualized_image': BGR array with bounding box
     }
   - No disk I/O inside function

#### File: run_dl_experiments.py
Two new pure functions exposed:

1. evaluate_object_detection(img_rgb, model=None) -> dict
   - Input: RGB image array, optional YOLO model
   - Output: {
       'metrics': {'detected_objects': int, 'avg_confidence': float},
       'visualized_image': BGR array with bounding boxes
     }
   - No disk I/O inside function

2. evaluate_segment_instances(img_rgb, model=None) -> dict
   - Input: RGB image array, optional YOLO model
   - Output: {
       'metrics': {'segmented_instances': int, 'avg_confidence': float},
       'visualized_image': BGR array with masks and boxes
     }
   - No disk I/O inside function

### STEP 2: Centralized Bulk Runner (run_30_pic_dataset.py)

#### Class: Week1BulkRunner

Orchestrates the complete pipeline with:
- Dynamic path resolution (Matan/Roni/relative)
- Pre-created directory structure for minimal I/O
- Baseline evaluation on 30 clean images
- Application of 4 distortions × 4 levels
- Task execution and metric collection
- Centralized CSV output

#### Output Structure:

```
data/
├── distorted_images/
│   ├── gaussian_noise_l1/  (30 images)
│   ├── gaussian_noise_l2/  (30 images)
│   ├── gaussian_noise_l3/  (30 images)
│   ├── gaussian_noise_l4/  (30 images)
│   ├── salt_pepper_l1/     (30 images)
│   ├── salt_pepper_l2/     (30 images)
│   ├── salt_pepper_l3/     (30 images)
│   ├── salt_pepper_l4/     (30 images)
│   ├── low_light_l1/       (30 images)
│   ├── low_light_l2/       (30 images)
│   ├── low_light_l3/       (30 images)
│   ├── low_light_l4/       (30 images)
│   ├── motion_blur_l1/     (30 images)
│   ├── motion_blur_l2/     (30 images)
│   ├── motion_blur_l3/     (30 images)
│   └── motion_blur_l4/     (30 images)
│
├── tasks_applied_on_distorted/
│   ├── optical_flow/
│   │   ├── clean_0/        (30 images)
│   │   ├── gaussian_noise_l1/  (30 images)
│   │   ├── ... (16 distortion levels × 30 images each)
│   │   └── motion_blur_l4/     (30 images)
│   │
│   ├── template_matching/
│   │   ├── clean_0/        (30 images)
│   │   ├── ... (17 folders × 30 images each)
│   │   └── motion_blur_l4/
│   │
│   ├── segment_instances/
│   │   ├── clean_0/        (30 images)
│   │   ├── ... (17 folders × 30 images each)
│   │   └── motion_blur_l4/
│   │
│   └── object_detection/
│       ├── clean_0/        (30 images)
│       ├── ... (17 folders × 30 images each)
│       └── motion_blur_l4/
│
└── tasks_graphs_and_tables/
    └── metadata_summary_base.csv  (2,040 rows)
```

#### CSV Format: metadata_summary_base.csv

Columns (10 total):
- image_name: Original image filename
- distortion_type: 'clean', 'gaussian_noise', 'salt_pepper', 'low_light', 'motion_blur'
- level: 0 for clean, 1-4 for distorted
- snr_distorted_db: Signal-to-Noise Ratio in dB (inf for clean)
- task_name: 'optical_flow', 'template_matching', 'segment_instances', 'object_detection'
- metric_name: Specific metric for that task
- metric_value: Numeric value of the metric
- task_image_path: Path to visualized task output (forward slashes)
- original_image_path: Path to original image (forward slashes)
- distorted_image_path: Path to distorted image (forward slashes)

Row Structure:
- Each row represents ONE metric from ONE task on ONE image state
- One image state (clean or distorted) with 4 tasks × N metrics = multiple rows
- 30 images × 4 tasks × 1 metric (clean) = 120 rows (optical_flow + template_matching + 2 DL)
- 30 images × 4 distortions × 4 levels × 4 tasks × metrics = 1,920 rows (distorted)
- Total: 2,040 rows

#### Metrics per Task:

1. optical_flow: tracked_points (int)
2. template_matching: matching_score (float), location (str)
3. segment_instances: segmented_instances (int), avg_confidence (float)
4. object_detection: detected_objects (int), avg_confidence (float)

Note: Metrics with multiple values (e.g., avg_confidence from object_detection) create 
one row per metric.

### STEP 3: Backward Compatibility

Legacy methods in ClassicalExperimentRunner and DeepLearningExperimentRunner:
- run_template_matching()
- run_optical_flow()
- run_experiments()

These still work but are marked as deprecated. They internally use the pure functions.

## USAGE

### Running the bulk pipeline:

```python
from src.run_30_pic_dataset import Week1BulkRunner

runner = Week1BulkRunner(num_images=30)
df_results = runner.run()
```

### Using pure functions directly:

```python
from src.run_classical_experiments import evaluate_optical_flow
from src.run_dl_experiments import evaluate_object_detection
import cv2

# Load images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Run task
result = evaluate_optical_flow(img1, img2)
metrics = result['metrics']  # {'tracked_points': 123}
vis_img = result['visualized_image']  # BGR array with arrows
```

## VALIDATION CHECKLIST

□ Pure functions in run_classical_experiments.py:
  □ evaluate_optical_flow returns correct format
  □ evaluate_template_matching returns correct format
  □ No cv2.imwrite or plt.savefig inside functions

□ Pure functions in run_dl_experiments.py:
  □ evaluate_object_detection returns correct format
  □ evaluate_segment_instances returns correct format
  □ No cv2.imwrite or plt.savefig inside functions

□ Directory structure created correctly:
  □ data/distorted_images/{distortion}_l{level}/ exists for all 16 combinations
  □ data/tasks_applied_on_distorted/{task}/{distortion}_l{level}/ exists
  □ data/tasks_graphs_and_tables/ exists

□ CSV output validation:
  □ Total rows = 2,040 (or close to it)
  □ All 10 columns present
  □ image_name column populated correctly
  □ distortion_type values correct
  □ level values: 0 for clean, 1-4 for distorted
  □ snr_distorted_db: inf for clean, numeric for distorted
  □ task_name values: optical_flow, template_matching, segment_instances, object_detection
  □ metric_name values correct per task
  □ All paths use forward slashes

□ Image outputs saved:
  □ Distorted images in correct folders (480 total: 30 images × 16 levels)
  □ Task visualizations in correct folders (2,040 total: 30 × 68 states × 4 tasks)
  □ Clean baseline visualizations saved (120 total: 30 × 4 tasks)

□ No errors during execution:
  □ All YOLO models load correctly
  □ Template creation works for all images
  □ Distortion functions handle all image sizes
  □ CSV writes without errors

## ESTIMATED EXECUTION TIME

- Model loading: ~10-15 seconds
- Image I/O: Varies by disk speed
- Task execution per image: ~2-5 seconds
- CSV writing: ~5 seconds
- Total for 30 images: ~3-10 minutes (depending on hardware)

## TROUBLESHOOTING

1. Dataset path not found:
   - Check if Matan's path exists: C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017
   - Check if Roni's path exists: /home/roni/datasets/coco128/images/train2017
   - Create relative path: datasets/coco128/images/train2017

2. Import errors in run_30_pic_dataset.py:
   - Ensure src/ folder is in Python path
   - Change imports to relative paths if needed

3. YOLO model not loading:
   - Check internet connection (models download on first run)
   - Verify disk space for model files (~200MB combined)

4. Memory errors on low-end hardware:
   - Reduce num_images parameter
   - Process in batches
   - Reduce image size before processing
"""
