"""
QUICK REFERENCE: Common Tasks & Code Patterns
==============================================
"""

# ============================================
# PATTERN 1: Run the full Week 1 pipeline
# ============================================
from src.run_30_pic_dataset import Week1BulkRunner

runner = Week1BulkRunner(num_images=30)
df_results = runner.run()

# Access results
print(f"Total rows: {len(df_results)}")
print(df_results.head())
print(df_results.info())


# ============================================
# PATTERN 2: Use pure functions directly
# ============================================
from src.run_classical_experiments import evaluate_optical_flow, evaluate_template_matching
from src.run_dl_experiments import evaluate_object_detection, evaluate_segment_instances
from src.yolo_tasks import YoloTasks
import cv2

# Load images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
template = img1[50:150, 50:150]

# Initialize YOLO
yolo = YoloTasks()

# Task 1: Optical Flow
of_result = evaluate_optical_flow(img1, img2)
print(f"Tracked points: {of_result['metrics']['tracked_points']}")

# Task 2: Template Matching
tm_result = evaluate_template_matching(img1, template)
print(f"Match score: {tm_result['metrics']['matching_score']}")

# Task 3: Segmentation
seg_result = evaluate_segment_instances(img_rgb, yolo.seg_model)
print(f"Segmented instances: {seg_result['metrics']['segmented_instances']}")

# Task 4: Detection
det_result = evaluate_object_detection(img_rgb, yolo.det_model)
print(f"Detected objects: {det_result['metrics']['detected_objects']}")


# ============================================
# PATTERN 3: Process custom dataset
# ============================================
from src.run_30_pic_dataset import Week1BulkRunner

# Process 10 images instead of 30
runner = Week1BulkRunner(num_images=10)
df = runner.run()

# Process 50 images
runner = Week1BulkRunner(num_images=50)
df = runner.run()


# ============================================
# PATTERN 4: Access and analyze CSV results
# ============================================
import pandas as pd

# Load the output CSV
df = pd.read_csv('data/tasks_graphs_and_tables/metadata_summary_base.csv')

# Filter by task
optical_flow_data = df[df['task_name'] == 'optical_flow']
print(f"Optical flow entries: {len(optical_flow_data)}")

# Filter by distortion
gaussian_data = df[df['distortion_type'] == 'gaussian_noise']
print(f"Gaussian noise entries: {len(gaussian_data)}")

# Filter by level
level_3_data = df[df['level'] == 3]
print(f"Level 3 entries: {len(level_3_data)}")

# Get metrics by distortion type
for distortion in df['distortion_type'].unique():
    subset = df[df['distortion_type'] == distortion]
    print(f"{distortion}: {len(subset)} rows")

# Calculate average metrics
avg_confidence = df[df['metric_name'] == 'avg_confidence']['metric_value'].mean()
print(f"Average confidence: {avg_confidence:.2f}")

# Group by image
for img_name in df['image_name'].unique()[:3]:  # First 3 images
    img_data = df[df['image_name'] == img_name]
    print(f"{img_name}: {len(img_data)} rows")


# ============================================
# PATTERN 5: Visualize a specific task result
# ============================================
import cv2
import os

# Load a task visualization image
task_img_path = 'data/tasks_applied_on_distorted/optical_flow/gaussian_noise_l2/image.jpg'

if os.path.exists(task_img_path):
    vis_img = cv2.imread(task_img_path)
    cv2.imshow('Task Visualization', vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================
# PATTERN 6: Validate pipeline output
# ============================================
from validate_pipeline import PipelineValidator

validator = PipelineValidator(data_dir='data')
success = validator.run_all_validations()

if success:
    print("✅ All validations passed!")
else:
    print("❌ Some validations failed. Check the report above.")


# ============================================
# PATTERN 7: Extract metrics for specific image/task
# ============================================
import pandas as pd

df = pd.read_csv('data/tasks_graphs_and_tables/metadata_summary_base.csv')

# Get all metrics for one image
image_name = '000000000009.jpg'
image_metrics = df[df['image_name'] == image_name]

# Get metrics for one image + one task
image_task = image_metrics[image_metrics['task_name'] == 'optical_flow']
print(image_task[['distortion_type', 'level', 'metric_name', 'metric_value']])

# Get metrics for one image + one distortion + all tasks
gaussian_metrics = image_metrics[image_metrics['distortion_type'] == 'gaussian_noise']
print(gaussian_metrics[['task_name', 'level', 'metric_name', 'metric_value']])


# ============================================
# PATTERN 8: Compare metrics across distortions
# ============================================
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/tasks_graphs_and_tables/metadata_summary_base.csv')

# Filter for one task and one metric
task_data = df[(df['task_name'] == 'object_detection') & 
               (df['metric_name'] == 'detected_objects')]

# Pivot to see degradation
pivot = task_data.pivot_table(
    values='metric_value',
    index='distortion_type',
    columns='level',
    aggfunc='mean'
)

print(pivot)

# Plot degradation
pivot.plot(kind='line', marker='o')
plt.title('Object Detection Degradation by Distortion')
plt.xlabel('Distortion Type')
plt.ylabel('Average Detected Objects')
plt.legend(title='Intensity Level')
plt.show()


# ============================================
# PATTERN 9: Calculate average SNR per distortion
# ============================================
import pandas as pd

df = pd.read_csv('data/tasks_graphs_and_tables/metadata_summary_base.csv')

# Remove clean entries (SNR = inf)
distorted = df[df['snr_distorted_db'] != float('inf')]

# Group by distortion type
snr_by_distortion = distorted.groupby('distortion_type')['snr_distorted_db'].agg(['mean', 'std', 'min', 'max'])
print(snr_by_distortion)

# Group by distortion + level
snr_by_distortion_level = distorted.groupby(['distortion_type', 'level'])['snr_distorted_db'].mean()
print(snr_by_distortion_level)


# ============================================
# PATTERN 10: Track metric degradation across levels
# ============================================
import pandas as pd

df = pd.read_csv('data/tasks_graphs_and_tables/metadata_summary_base.csv')

# Track how a specific metric degrades with distortion intensity
metric_filter = (df['task_name'] == 'object_detection') & \
                (df['metric_name'] == 'detected_objects') & \
                (df['distortion_type'] == 'gaussian_noise')

degradation = df[metric_filter].groupby('level')['metric_value'].agg(['mean', 'std', 'count'])
print(degradation)

# Calculate degradation rate
clean_baseline = df[(df['distortion_type'] == 'clean') & 
                   (df['task_name'] == 'object_detection') & 
                   (df['metric_name'] == 'detected_objects')]['metric_value'].mean()

for level in [1, 2, 3, 4]:
    level_data = df[metric_filter & (df['level'] == level)]
    avg_val = level_data['metric_value'].mean()
    degradation_pct = ((clean_baseline - avg_val) / clean_baseline) * 100
    print(f"Level {level}: {avg_val:.2f} ({degradation_pct:.1f}% degradation)")


# ============================================
# PATTERN 11: Generate per-image report
# ============================================
import pandas as pd

df = pd.read_csv('data/tasks_graphs_and_tables/metadata_summary_base.csv')

# Generate summary for each image
for image_name in df['image_name'].unique()[:5]:  # First 5 images
    print(f"\n{'='*60}")
    print(f"IMAGE: {image_name}")
    print(f"{'='*60}")
    
    img_data = df[df['image_name'] == image_name]
    
    # Baseline metrics
    clean = img_data[img_data['distortion_type'] == 'clean']
    print("\nBaseline (Clean):")
    print(clean[['task_name', 'metric_name', 'metric_value']])
    
    # Distortion summary
    distorted = img_data[img_data['distortion_type'] != 'clean']
    print("\nDistortion Summary:")
    dist_summary = distorted.groupby('distortion_type')['snr_distorted_db'].mean()
    print(dist_summary)


# ============================================
# PATTERN 12: Export filtered results to new CSV
# ============================================
import pandas as pd

df = pd.read_csv('data/tasks_graphs_and_tables/metadata_summary_base.csv')

# Export only optical flow results
optical_flow = df[df['task_name'] == 'optical_flow']
optical_flow.to_csv('optical_flow_results.csv', index=False)

# Export only distorted image results
distorted = df[df['distortion_type'] != 'clean']
distorted.to_csv('distorted_results.csv', index=False)

# Export high-SNR results (strong distortion)
strong_distortion = df[df['snr_distorted_db'] < 10]  # SNR < 10 dB
strong_distortion.to_csv('strong_distortion_results.csv', index=False)
