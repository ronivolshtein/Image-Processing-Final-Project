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

    # Filter only object detection tasks that were run with the baseline model
    base_det_df = df[(df['task_name'] == 'object_detection') & (df['model_type'] == 'Baseline')]
    
    if base_det_df.empty:
        print("⚠️ No baseline object detection rows found to evaluate.")
        return

    # Identify the column containing the image path
    path_col = 'distorted_image_path' if 'distorted_image_path' in df.columns else 'image_path'
    if path_col not in df.columns:
        print(f"❌ Error: Could not find image path column. Available columns: {list(df.columns)}")
        return

    new_rows = []
    print(f"🔍 Evaluating {len(base_det_df)} distorted images with the Fine-Tuned model...")

    for index, row in base_det_df.iterrows():
        img_path = str(row[path_col])
        
        if not os.path.exists(img_path):
            continue

        # Run inference using the newly trained model
        results = model.predict(img_path, conf=0.25, verbose=False)[0]
        
        # Count the number of detected objects
        detected_objects = len(results.boxes)

        # Copy the baseline row and update it for the fine-tuned results
        new_row = row.copy()
        new_row['model_type'] = 'Fine-Tuned'
        new_row['metric_value'] = detected_objects
        new_rows.append(new_row)

    if new_rows:
        # Create a DataFrame for the new results and append to the original
        new_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        
        # Save back to CSV
        updated_df.to_csv(CSV_PATH, index=False)
        print(f"✅ Success! Added {len(new_rows)} fine-tuned evaluation rows to {CSV_PATH}.")
    else:
        print("⚠️ No new results generated.")

if __name__ == "__main__":
    main()