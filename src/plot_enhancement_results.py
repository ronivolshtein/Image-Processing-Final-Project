"""
Week 2 - Enhancements: Baseline vs. Enhanced comparison plots, one pair per task,
mirroring plot_finetune_results.py's bar/line approach but across all 4 tasks
instead of just object_detection.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path('data/tasks_graphs_and_tables/metadata_summary_base.csv')
OUTPUT_DIR = Path('data/tasks_graphs_and_tables/plots')

# The metric that best represents "recovered performance" for each task
# (avoids averaging mismatched metrics like detected_objects + avg_confidence together,
# and skips non-numeric metrics like template_matching's 'location').
PRIMARY_METRIC = {
    'object_detection': 'detected_objects',
    'segment_instances': 'segmented_instances',
    'template_matching': 'matching_score',
    'optical_flow': 'tracked_points',
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {CSV_PATH}...")

    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("Error: CSV not found!")
        return

    if 'model_type' not in df.columns or 'Enhanced' not in df['model_type'].unique():
        print("Error: No 'Enhanced' rows found. Run apply_enhancements.py then evaluate_enhancements.py first.")
        return

    sns.set_theme(style="whitegrid")

    for task_name, metric_name in PRIMARY_METRIC.items():
        task_df = df[
            (df['task_name'] == task_name) &
            (df['metric_name'] == metric_name) &
            (df['model_type'].isin(['Baseline', 'Enhanced']))
        ].copy()

        if task_df.empty:
            print(f"[SKIP] No rows for {task_name}/{metric_name}")
            continue

        task_df['metric_value'] = pd.to_numeric(task_df['metric_value'], errors='coerce')
        # Only distorted images went through enhancement - clean (level 0) has nothing to recover.
        task_df = task_df[task_df['level'] > 0]

        metric_label = metric_name.replace('_', ' ').title()

        # --- Bar chart: recovery per distortion type, averaged across levels ---
        plt.figure(figsize=(10, 6))
        sns.barplot(data=task_df, x='distortion_type', y='metric_value', hue='model_type', errorbar=None, palette='Set2')
        plt.title(f'{task_name.replace("_", " ").title()}: Baseline vs. Enhanced (Averaged across noise levels)', fontsize=14, fontweight='bold')
        plt.ylabel(metric_label, fontsize=12)
        plt.xlabel('Distortion Type', fontsize=12)
        plt.legend(title='Processing')
        plt.tight_layout()
        bar_path = OUTPUT_DIR / f'{task_name}_enhancement_recovery_bar.png'
        plt.savefig(bar_path, dpi=300)
        plt.close()

        # --- Line chart: recovery per level, faceted by distortion type ---
        g = sns.FacetGrid(task_df, col='distortion_type', col_wrap=2, height=4, aspect=1.5)
        g.map_dataframe(sns.lineplot, x='level', y='metric_value', hue='model_type', marker='o', palette='Set2', linewidth=2)
        g.add_legend(title='Processing')
        g.set_axis_labels('Distortion Level (1=Mild, 4=Severe)', metric_label)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f'{task_name.replace("_", " ").title()}: Degradation vs. Enhancement Recovery', fontsize=16, fontweight='bold')
        line_path = OUTPUT_DIR / f'{task_name}_enhancement_recovery_lines.png'
        plt.savefig(line_path, dpi=300)
        plt.close()

        print(f"Saved: {bar_path.name}, {line_path.name}")

    print("Done.")


if __name__ == '__main__':
    main()
