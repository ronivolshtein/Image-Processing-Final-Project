# Week 1 Setup & Usage Guide

## ✅ Refactoring Complete!

Your computer vision pipeline has been refactored into a clean, modular architecture. Here's what was done and how to use it.

---

## 📋 What Changed?

### BEFORE (Mixed Concerns)
```python
# File I/O mixed with task logic
def run_template_matching(self):
    # ... task logic ...
    cv2.imwrite(path, image)  # Disk I/O inside task
    plt.savefig(path)         # More disk I/O
```

### AFTER (Pure Functions + Bulk Runner)
```python
# Pure function - no disk I/O
def evaluate_template_matching(img, template):
    return {
        'metrics': {'matching_score': 0.95},
        'visualized_image': vis_img
    }

# Bulk runner handles all I/O
runner = Week1BulkRunner(num_images=30)
runner.run()  # Orchestrates everything
```

---

## 🏗️ Architecture Overview

### Layer 1: Pure Task Functions (No Disk I/O)

#### `src/run_classical_experiments.py`
```python
# Task 1: Optical Flow Tracking
evaluate_optical_flow(prev_img, next_img) 
→ {'metrics': {'tracked_points': 123}, 'visualized_image': array}

# Task 2: Template Matching
evaluate_template_matching(img, template)
→ {'metrics': {'matching_score': 0.95, 'location': '(100,200)'}, 'visualized_image': array}
```

#### `src/run_dl_experiments.py`
```python
# Task 3: Instance Segmentation (YOLO)
evaluate_segment_instances(img_rgb, model)
→ {'metrics': {'segmented_instances': 5, 'avg_confidence': 0.87}, 'visualized_image': array}

# Task 4: Object Detection (YOLO)
evaluate_object_detection(img_rgb, model)
→ {'metrics': {'detected_objects': 8, 'avg_confidence': 0.92}, 'visualized_image': array}
```

### Layer 2: Bulk Runner (Orchestrates Everything)

#### `src/run_30_pic_dataset.py`
```python
from run_30_pic_dataset import Week1BulkRunner

runner = Week1BulkRunner(num_images=30)
df_results = runner.run()
```

**What it does:**
1. Detects dataset path (Matan/Roni/relative)
2. Loads first 30 images
3. Pre-creates all output directories
4. For each of 30 images:
   - Runs 4 tasks on clean image → saves baseline
   - Applies 4 distortions × 4 levels (16 total)
   - For each distorted image: runs 4 tasks
   - Saves all visualizations to structured folders
5. Outputs single CSV with 2,040 rows

---

## 📁 Output Structure

```
project_root/
│
├── data/
│   ├── distorted_images/                    (480 raw distorted images)
│   │   ├── gaussian_noise_l1/ → 30 images
│   │   ├── gaussian_noise_l2/ → 30 images
│   │   ├── gaussian_noise_l3/ → 30 images
│   │   ├── gaussian_noise_l4/ → 30 images
│   │   ├── salt_pepper_l1/ → 30 images
│   │   ├── salt_pepper_l2/ → 30 images
│   │   ├── salt_pepper_l3/ → 30 images
│   │   ├── salt_pepper_l4/ → 30 images
│   │   ├── low_light_l1/ → 30 images
│   │   ├── low_light_l2/ → 30 images
│   │   ├── low_light_l3/ → 30 images
│   │   ├── low_light_l4/ → 30 images
│   │   ├── motion_blur_l1/ → 30 images
│   │   ├── motion_blur_l2/ → 30 images
│   │   ├── motion_blur_l3/ → 30 images
│   │   └── motion_blur_l4/ → 30 images
│   │
│   ├── tasks_applied_on_distorted/          (2,040 task visualization images)
│   │   ├── optical_flow/
│   │   │   ├── clean_0/ → 30 images
│   │   │   ├── gaussian_noise_l1/ → 30 images
│   │   │   ├── ... (16 distortion combos)
│   │   │   └── motion_blur_l4/ → 30 images
│   │   │
│   │   ├── template_matching/
│   │   │   └── ... (same structure: 17 × 30 = 510 images)
│   │   │
│   │   ├── segment_instances/
│   │   │   └── ... (same structure: 17 × 30 = 510 images)
│   │   │
│   │   └── object_detection/
│   │       └── ... (same structure: 17 × 30 = 510 images)
│   │
│   └── tasks_graphs_and_tables/
│       └── metadata_summary_base.csv         (2,040 rows)
│
└── src/
    ├── run_30_pic_dataset.py               ← RUN THIS
    ├── run_classical_experiments.py        (pure functions inside)
    ├── run_dl_experiments.py               (pure functions inside)
    ├── yolo_tasks.py
    ├── classical_tasks.py
    ├── distortions.py
    └── metrics.py
```

---

## 🚀 Quick Start

### Step 1: Ensure your dataset exists
```
Choose one:
1. C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017 (Matan)
2. /home/roni/datasets/coco128/images/train2017 (Roni)
3. datasets/coco128/images/train2017 (relative path)
```

### Step 2: Run the pipeline
```bash
cd src/
python run_30_pic_dataset.py
```

### Step 3: Validate output (optional but recommended)
```bash
cd ..
python validate_pipeline.py
```

---

## 📊 CSV Output Format

### File: `data/tasks_graphs_and_tables/metadata_summary_base.csv`

**Columns (10):**
1. `image_name` - Original filename (e.g., "000000000009.jpg")
2. `distortion_type` - Type of distortion applied
3. `level` - Intensity level (0=clean, 1-4=distorted)
4. `snr_distorted_db` - Signal-to-Noise Ratio in dB
5. `task_name` - Which task was executed
6. `metric_name` - Specific metric from that task
7. `metric_value` - Numeric value
8. `task_image_path` - Where task visualization was saved
9. `original_image_path` - Path to clean image
10. `distorted_image_path` - Path to distorted image

**Row Structure:**
- One row per metric per task per image state
- 30 images × 4 tasks × (1 clean + 16 distorted) = 2,040 rows minimum

**Example Rows:**
```
image_name,distortion_type,level,snr_distorted_db,task_name,metric_name,metric_value,task_image_path,...
000000000009.jpg,clean,0,inf,optical_flow,tracked_points,145,...
000000000009.jpg,clean,0,inf,template_matching,matching_score,0.95,...
000000000009.jpg,gaussian_noise,1,15.32,optical_flow,tracked_points,89,...
000000000009.jpg,gaussian_noise,1,15.32,template_matching,matching_score,0.72,...
```

---

## 🔧 Using Pure Functions Directly

You don't need to use the bulk runner. You can import and use the pure functions directly:

### Example 1: Optical Flow
```python
from src.run_classical_experiments import evaluate_optical_flow
import cv2

frame1 = cv2.imread('frame1.jpg')
frame2 = cv2.imread('frame2.jpg')

result = evaluate_optical_flow(frame1, frame2)
print(result['metrics'])  # {'tracked_points': 145}
cv2.imshow('Flow', result['visualized_image'])
```

### Example 2: Object Detection
```python
from src.run_dl_experiments import evaluate_object_detection
from src.yolo_tasks import YoloTasks
import cv2

yolo = YoloTasks()
img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = evaluate_object_detection(img_rgb, yolo.det_model)
print(result['metrics'])  # {'detected_objects': 8, 'avg_confidence': 0.92}
cv2.imshow('Detections', result['visualized_image'])
```

### Example 3: Template Matching
```python
from src.run_classical_experiments import evaluate_template_matching
import cv2

img = cv2.imread('image.jpg')
template = img[100:200, 100:200]  # Extract a patch

result = evaluate_template_matching(img, template)
print(result['metrics'])  # {'matching_score': 0.95, 'location': '(100,200)'}
cv2.imshow('Match', result['visualized_image'])
```

### Example 4: Instance Segmentation
```python
from src.run_dl_experiments import evaluate_segment_instances
from src.yolo_tasks import YoloTasks
import cv2

yolo = YoloTasks()
img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = evaluate_segment_instances(img_rgb, yolo.seg_model)
print(result['metrics'])  # {'segmented_instances': 5, 'avg_confidence': 0.87}
cv2.imshow('Segments', result['visualized_image'])
```

---

## 📈 Expected Results

### Dataset Statistics (30 images, 4 tasks)
```
Clean baseline rows:           30 images × 4 tasks × ~1 metric = 120 rows
Distorted images:             30 × 4 distortions × 4 levels = 480 images
Distorted task rows:          480 distorted × 4 tasks × ~1 metric = 1,920 rows
─────────────────────────────────────────────────────────────────────
TOTAL CSV ROWS:               2,040 rows
```

### Disk Usage Estimates
- Original 30 images: ~1-2 MB
- 480 distorted images: ~2-4 MB
- 2,040 task visualizations: ~50-100 MB
- CSV file: ~500 KB
- **Total: ~60-110 MB**

### Execution Time Estimates
- Model loading: ~10-15 seconds
- Image processing: ~1-3 seconds per image
- Task execution: ~1-2 seconds per task
- **Total for 30 images: ~5-15 minutes** (depending on hardware)

---

## 🐛 Troubleshooting

### Error: "Dataset directory not found"
**Solution:** Check one of these paths exists:
- Windows: `C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017`
- Linux: `/home/roni/datasets/coco128/images/train2017`
- Relative: `datasets/coco128/images/train2017`

### Error: "ModuleNotFoundError" for imports
**Solution:** Ensure you're running from the `src/` directory or add parent to path:
```python
import sys
sys.path.insert(0, '..')
```

### Error: "CUDA out of memory" or "Killed"
**Solution:** Reduce batch size or num_images:
```python
runner = Week1BulkRunner(num_images=10)  # Process fewer images
```

### Error: "YOLO models not downloading"
**Solution:** Check internet and run manually first:
```python
from ultralytics import YOLO
YOLO("yolov8n.pt")  # Downloads if not cached
YOLO("yolov8n-seg.pt")
```

### CSV shows wrong column order or missing data
**Solution:** Validate output:
```bash
python validate_pipeline.py
```

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `run_30_pic_dataset.py` | **Main runner** - Execute this to process 30 images |
| `run_classical_experiments.py` | Contains `evaluate_optical_flow()`, `evaluate_template_matching()` |
| `run_dl_experiments.py` | Contains `evaluate_object_detection()`, `evaluate_segment_instances()` |
| `yolo_tasks.py` | YOLO model wrapper (detect_objects, segment_instances) |
| `classical_tasks.py` | Classical CV methods (template_match, optical_flow) |
| `distortions.py` | Distortion functions (gaussian_noise, salt_pepper, low_light, motion_blur) |
| `validate_pipeline.py` | **Validates output** - Run this after pipeline completes |
| `REFACTORING_GUIDE.md` | Detailed architecture documentation |

---

## ✨ Key Features

✅ **Pure functions** - Testable, reusable, no side effects  
✅ **Modular design** - Task logic separated from I/O  
✅ **Efficient I/O** - Pre-created directories minimize disk overhead  
✅ **Structured output** - Organized directories and normalized CSV  
✅ **Cross-platform** - Forward slashes in paths for compatibility  
✅ **Backward compatible** - Legacy methods still work  
✅ **Error handling** - Graceful fallback on task failures  
✅ **Extensible** - Easy to add new tasks or distortions  

---

## 🎯 Next Steps

### Week 2 Extension
Once Week 1 data is ready, you can extend with:
- Additional enhancement techniques
- More sophisticated task runners
- Results analysis and visualization
- Performance comparison plots

### Custom Usage
To use on different data:
```python
runner = Week1BulkRunner(num_images=50)  # Process 50 instead of 30
df = runner.run()
```

---

## 📝 Notes

- All paths in CSV use forward slashes (`/`) for cross-platform compatibility
- SNR values for clean images are stored as `inf` (infinity)
- Task visualizations are saved as the task sees them (arrows for optical flow, boxes for detection, etc.)
- Metrics are flattened: each metric from a task creates one row

---

## ❓ Questions?

Refer to `REFACTORING_GUIDE.md` for more detailed information about the architecture and validation.

**Last Updated:** June 21, 2026  
**Version:** 1.0 - Week 1 Complete
