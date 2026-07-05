import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path('data/tasks_graphs_and_tables/metadata_summary_base.csv')
    out_dir = Path('data/tasks_graphs_and_tables/plots')
    
    print(f"📊 Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("❌ Error: CSV not found!")
        return

    # Filter only YOLO Object Detection
    df_yolo = df[df['task_name'] == 'object_detection'].copy()

    if 'model_type' not in df_yolo.columns:
        print("❌ Error: 'model_type' column missing. The evaluation script didn't finish properly.")
        return

    # 🛠️ THE FIX: Smartly detect the exact name of the 'level' column
    level_col = None
    for col in ['distortion_level', 'level', 'source_level', 'severity']:
        if col in df_yolo.columns:
            level_col = col
            break
            
    if not level_col:
        print(f"❌ Error: Could not find the level column. Available columns are: {list(df_yolo.columns)}")
        return

    # Remove the 'clean' images (Level 0) for the direct comparison on distorted images
    df_distorted = df_yolo[df_yolo[level_col] > 0].copy()

    # Convert metric_value to numeric so matplotlib understands it's a number, not text!
    df_distorted['metric_value'] = pd.to_numeric(df_distorted['metric_value'], errors='coerce')

    # Set up the visual style
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Overall Baseline vs Fine-Tuned per Distortion Type (Bar Chart) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_distorted, x='distortion_type', y='metric_value', hue='model_type', errorbar=None, palette='Set2')
    plt.title('YOLO Object Detection: Baseline vs. Fine-Tuned (Averaged across all noise levels)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Detected Objects', fontsize=12)
    plt.xlabel('Distortion Type', fontsize=12)
    plt.legend(title='Model Version')
    plt.tight_layout()
    bar_path = out_dir / 'finetune_recovery_bar.png'
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # --- Plot 2: Detailed Line Plot by Level ---
    g = sns.FacetGrid(df_distorted, col="distortion_type", col_wrap=2, height=4, aspect=1.5)
    g.map_dataframe(sns.lineplot, x=level_col, y="metric_value", hue="model_type", marker="o", palette='Set2', linewidth=2)
    g.add_legend(title='Model Version')
    g.set_axis_labels("Distortion Level (1=Mild, 4=Severe)", "Detected Objects")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Degradation vs. Recovery per Distortion Level', fontsize=16, fontweight='bold')
    
    line_path = out_dir / 'finetune_recovery_lines.png'
    plt.savefig(line_path, dpi=300)
    plt.close()

    print(f"✅ Success! Comparative plots generated:")
    print(f"   1. {bar_path}")
    print(f"   2. {line_path}")

if __name__ == "__main__":
    main()