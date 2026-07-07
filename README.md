# Image Processing Final Project

Studies how 4 classical distortions (Gaussian noise, salt & pepper, low light, motion blur), each at 4 severity levels, degrade 4 computer vision tasks (object detection, instance segmentation, template matching, optical flow) — and compares two recovery strategies: fine-tuning YOLO on distorted data vs. classical image enhancement.

## 1. Environment Setup

```bash
# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO weights + coco128 dataset if missing, and sanity-check the base models
python main.py
```

**Dataset path:** the scripts look for `coco128/images/train2017` in this order: a hardcoded Windows path, a hardcoded Linux path, then `datasets/coco128/images/train2017` relative to the project root. If none of these exist on your machine, download [coco128](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip) and place it under `datasets/` in the project root, or edit `resolve_dataset_dir()`/`_resolve_dataset_path()` in the scripts below to add your own path.

## 2. Pipeline — run in this order

Each stage reads from and appends to one shared file: **`data/tasks_graphs_and_tables/metadata_summary_base.csv`**. Every task's result is one row, and a `model_type` column (`Baseline` / `Fine-Tuned` / `Enhanced`) marks which stage produced it — this is what lets every plotting script do simple filters instead of joining multiple files.

### Week 1 — Baseline & Distortions
```bash
python src/run_30_pic_dataset.py
```
Takes the first 30 images from `coco128/train2017`, applies all 4 distortions × 4 severity levels, runs all 4 tasks on both the clean and distorted versions of each image, and writes the results as `model_type='Baseline'` rows in `metadata_summary_base.csv`. Distorted images are saved to `data/distorted_images/`, annotated task outputs to `data/tasks_applied_on_distorted/`.

Optional sanity checks / charts:
```bash
python validate_pipeline.py       # validates the CSV structure and output folders
python src/generate_plots.py      # per-task degradation charts (metric vs. level, metric vs. SNR)
```

### Week 2, Track A — YOLO Fine-Tuning
```bash
python src/prepare_yolo_dataset.py   # splits distorted images into train/test
python src/train_yolo.py             # short YOLOv8 fine-tuning run on distorted images
python src/evaluate_finetuned.py     # runs the fine-tuned model on the same distorted images (object detection only), appends 'Fine-Tuned' rows
python src/plot_finetune_results.py  # Baseline vs. Fine-Tuned bar + line charts
```

### Week 2, Track B — Classical Enhancement
```bash
python src/apply_enhancements.py      # applies the matched enhancement to every distorted image -> data/enhanced_images/
python src/evaluate_enhancements.py   # runs all 4 tasks on the enhanced images, appends 'Enhanced' rows
python src/plot_enhancement_results.py  # Baseline vs. Enhanced bar + line charts, per task
```
Which enhancement is applied to which distortion is defined once, in `src/enhancements.py`'s `ENHANCEMENT_FOR_DISTORTION` map: Gaussian noise → Gaussian filter, salt & pepper → median filter, low light → CLAHE, motion blur → sharpening.

## 3. Data & Output Layout

```
data/
├── distorted_images/              (Week 1 - 480 raw distorted images: 30 × 4 distortions × 4 levels)
├── enhanced_images/                (Week 2 - same 480 images after the matched classical enhancement)
├── tasks_applied_on_distorted/      (annotated task outputs on clean + distorted images)
│   └── {task_name}/{distortion_type}_l{level}/{image_name}.jpg
├── tasks_applied_on_enhanced/       (annotated task outputs on enhanced images, same layout)
│   └── {task_name}/{distortion_type}_l{level}/{image_name}.jpg
└── tasks_graphs_and_tables/
    ├── metadata_summary_base.csv    (the central results table - see schema below)
    └── plots/                       (all generated comparison charts)
```

`{task_name}` is one of `object_detection`, `segment_instances`, `template_matching`, `optical_flow`. `{distortion_type}` is one of `gaussian_noise`, `salt_pepper`, `low_light`, `motion_blur`; `{level}` is `1`-`4`.

## 4. CSV Schema — `metadata_summary_base.csv`

One row per (image, condition, task, metric, model_type):

| Column | Meaning |
|---|---|
| `image_name` | Original filename, e.g. `000000000009.jpg` |
| `distortion_type` | `clean`, `gaussian_noise`, `salt_pepper`, `low_light`, `motion_blur` |
| `level` | `0` for clean, `1`-`4` for distorted (severity) |
| `snr_distorted_db` | Signal-to-Noise Ratio vs. the clean image (`inf` for clean) |
| `task_name` | `object_detection`, `segment_instances`, `template_matching`, `optical_flow` |
| `metric_name` | The specific metric for that task (see below) |
| `metric_value` | The metric's value |
| `task_image_path` | Path to the annotated task-output image |
| `original_image_path` | Path to the clean source image |
| `distorted_image_path` | Path to the distorted image (same as original for clean rows) |
| `model_type` | `Baseline`, `Fine-Tuned`, or `Enhanced` — which stage produced this row |

Metrics per task: `object_detection` → `detected_objects`, `avg_confidence`; `segment_instances` → `segmented_instances`, `avg_confidence`; `template_matching` → `matching_score`, `location`; `optical_flow` → `tracked_points`.

Note: `Fine-Tuned` only covers `object_detection` (that's the model that was retrained); `Enhanced` and `Baseline` cover all 4 tasks.

## 5. File Reference

| File | Purpose |
|---|---|
| `distortions.py` | The 4 distortion functions + SNR calculation |
| `classical_tasks.py` | Optical flow and template matching implementations |
| `yolo_tasks.py` | YOLO model wrapper for detection/segmentation |
| `run_classical_experiments.py` | `evaluate_optical_flow()`, `evaluate_template_matching()` — pure, reusable functions |
| `run_dl_experiments.py` | `evaluate_object_detection()`, `evaluate_segment_instances()` — pure, reusable functions |
| `run_30_pic_dataset.py` | **Week 1 main runner** — generates the base distorted dataset + CSV |
| `generate_plots.py` | Week 1 degradation charts |
| `prepare_yolo_dataset.py`, `train_yolo.py` | Week 2 fine-tuning: data prep + training |
| `evaluate_finetuned.py`, `plot_finetune_results.py` | Week 2 fine-tuning: evaluation + charts |
| `enhancements.py` | Classical enhancement functions + the distortion→enhancement map |
| `apply_enhancements.py` | Week 2 enhancement: applies enhancements, saves images |
| `evaluate_enhancements.py` | Week 2 enhancement: runs the 4 tasks, appends CSV rows |
| `plot_enhancement_results.py` | Week 2 enhancement: charts |
| `validate_pipeline.py` | Validates `run_30_pic_dataset.py`'s output |

## 6. Troubleshooting

- **"Dataset directory not found"** — see the dataset path note in Section 1.
- **`ModuleNotFoundError` for local imports** — run scripts from the project root (`python src/script_name.py`), not from inside `src/`.
- **YOLO models not downloading** — check your internet connection; `python main.py` downloads `yolov8n.pt`/`yolov8n-seg.pt` on first run.
- **Enhanced image not found (in `evaluate_enhancements.py`)** — run `apply_enhancements.py` first; `evaluate_enhancements.py` only evaluates images that already exist in `data/enhanced_images/`.
