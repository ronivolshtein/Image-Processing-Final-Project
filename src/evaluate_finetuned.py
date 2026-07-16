import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import os

# ==========================================
# 1. Configuration & Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / 'data' / 'tasks_graphs_and_tables' / 'metadata_summary_base.csv'


def resolve_finetuned_model_path():
    candidates = [
        PROJECT_ROOT / 'runs' / 'detect' / 'finetune_distorted' / 'weights' / 'best.pt',
        PROJECT_ROOT / 'runs' / 'detect' / 'runs' / 'detect' / 'finetune_distorted' / 'weights' / 'best.pt',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for match in sorted((PROJECT_ROOT / 'runs').rglob('best.pt')):
        if match.is_file():
            return match

    return candidates[0]


def validate_finetuned_rows(new_df, expected_conditions):
    """Warn about incomplete, duplicate, or invalid fine-tuned metrics."""
    condition_cols = ['image_name', 'distortion_type', 'level']
    key_cols = condition_cols + ['metric_name']

    duplicate_mask = new_df.duplicated(subset=key_cols, keep=False)
    if duplicate_mask.any():
        print(f"WARNING: Found {duplicate_mask.sum()} duplicate Fine-Tuned rows "
              "for the same image/condition/metric.")

    required_metrics = {'detected_objects', 'avg_confidence'}
    metrics_per_condition = new_df.groupby(condition_cols)['metric_name'].agg(set)
    complete_conditions = int(metrics_per_condition.apply(
        lambda metrics: metrics == required_metrics
    ).sum())
    if complete_conditions != expected_conditions or len(metrics_per_condition) != expected_conditions:
        print(f"WARNING: Expected two metrics for each of {expected_conditions} "
              f"processed conditions, but only {complete_conditions} are complete.")

    confidence = pd.to_numeric(
        new_df.loc[new_df['metric_name'] == 'avg_confidence', 'metric_value'],
        errors='coerce'
    )
    invalid_confidence = confidence.isna() | ~confidence.between(0.0, 1.0)
    if invalid_confidence.any():
        print(f"WARNING: Found {invalid_confidence.sum()} invalid avg_confidence values "
              "(expected a number in [0, 1]).")

    counts = pd.to_numeric(
        new_df.loc[new_df['metric_name'] == 'detected_objects', 'metric_value'],
        errors='coerce'
    )
    invalid_counts = counts.isna() | (counts < 0) | (counts % 1 != 0)
    if invalid_counts.any():
        print(f"WARNING: Found {invalid_counts.sum()} invalid detected_objects values "
              "(expected a non-negative integer).")


def main():
    finetuned_model_path = resolve_finetuned_model_path()
    print(f"🚀 Loading fine-tuned model from {finetuned_model_path}...")
    try:
        model = YOLO(str(finetuned_model_path))
    except Exception as e:
        print(f"❌ Error: Could not load the fine-tuned model. Check the path. Details: {e}")
        return

    print(f"📊 Loading baseline CSV from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("❌ Error: metadata_summary_base.csv not found!")
        return

    # Add a 'model_type' column if it doesn't exist to separate baseline from fine-tuned
    if 'model_type' not in df.columns:
        df['model_type'] = 'Baseline'

    # Use one metadata row per image/condition so inference is performed once,
    # independently of how many metric rows exist in the baseline CSV.
    condition_cols = ['image_name', 'distortion_type', 'level']
    base_det_df = df[
        (df['task_name'] == 'object_detection') &
        (df['model_type'] == 'Baseline')
    ].drop_duplicates(subset=condition_cols).copy()
    
    if base_det_df.empty:
        print("⚠️ No baseline object detection rows found to evaluate.")
        return

    # Identify the column containing the image path
    path_col = 'distorted_image_path' if 'distorted_image_path' in df.columns else 'image_path'
    if path_col not in df.columns:
        print(f"❌ Error: Could not find image path column. Available columns: {list(df.columns)}")
        return

    new_rows = []
    processed_conditions = 0
    print(f"🔍 Evaluating {len(base_det_df)} distorted images with the Fine-Tuned model...")

    for index, row in base_det_df.iterrows():
        img_path = str(row[path_col])
        
        if not os.path.exists(img_path):
            continue

        # Run inference using the newly trained model
        result = model.predict(img_path, conf=0.25, verbose=False)[0]

        detected_objects = len(result.boxes)
        if detected_objects > 0:
            avg_confidence = float(result.boxes.conf.mean().item())
        else:
            avg_confidence = 0.0

        # Create both metrics explicitly while retaining all existing metadata.
        count_row = row.copy()
        count_row['model_type'] = 'Fine-Tuned'
        count_row['metric_name'] = 'detected_objects'
        count_row['metric_value'] = detected_objects

        confidence_row = row.copy()
        confidence_row['model_type'] = 'Fine-Tuned'
        confidence_row['metric_name'] = 'avg_confidence'
        confidence_row['metric_value'] = avg_confidence

        new_rows.extend([count_row, confidence_row])
        processed_conditions += 1

    if new_rows:
        # Create a DataFrame for the new results and append to the original
        new_df = pd.DataFrame(new_rows)
        validate_finetuned_rows(new_df, expected_conditions=processed_conditions)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        
        # Save back to CSV
        updated_df.to_csv(CSV_PATH, index=False)
        print(f"✅ Success! Added {len(new_rows)} fine-tuned evaluation rows to {CSV_PATH}.")
    else:
        print("⚠️ No new results generated.")

if __name__ == "__main__":
    main()
