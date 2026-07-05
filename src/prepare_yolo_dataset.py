print("🚀 Script started! Loading libraries...")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ==========================================
# 1. Configuration & Paths
# ==========================================
CSV_PATH = Path('data/tasks_graphs_and_tables/metadata_summary_base.csv')
OUTPUT_DIR = Path('data/tasks_graphs_and_tables/plots')

def main():
    # Create the output directory for the plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📊 Loading data from {CSV_PATH}...")
    
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("❌ Error: CSV file not found. Make sure you've run the base pipeline first.")
        return

    # Set visualization style
    sns.set_theme(style="whitegrid")
    
    tasks = df['task_name'].unique()
    print(f"🔍 Found {len(tasks)} tasks: {list(tasks)}")

    # ==========================================
    # 2. Generate Plots per Task
    # ==========================================
    for task in tasks:
        task_df = df[df['task_name'] == task]
        
        # Determine the metric name for the Y-axis (e.g., mAP50, Error, etc.)
        metric_name = task_df['metric_name'].iloc[0]
        
        # ---------------------------------------------------------
        # Plot 1: Metric vs. Distortion Level
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=task_df, x='level', y='metric_value', hue='distortion_type', marker='o', linewidth=2)
        
        plt.title(f'Performance Degradation: {task.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.xlabel('Distortion Level (0 = Clean, 4 = Severe)', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks([0, 1, 2, 3, 4])
        plt.legend(title='Distortion Type')
        
        level_plot_path = OUTPUT_DIR / f'{task}_vs_level.png'
        plt.savefig(level_plot_path, bbox_inches='tight')
        plt.close()
        
        # ---------------------------------------------------------
        # Plot 2: Metric vs. SNR (Signal to Noise Ratio)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        
        snr_df = task_df.replace([float('inf'), -float('inf')], pd.NA).dropna(subset=['snr_distorted_db'])
        sns.scatterplot(data=snr_df, x='snr_distorted_db', y='metric_value', hue='distortion_type', s=60, alpha=0.7)
        
        plt.title(f'Performance vs SNR: {task.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.xlabel('SNR (dB) - Lower means more noise', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        
        plt.gca().invert_xaxis() 
        plt.legend(title='Distortion Type')
        
        snr_plot_path = OUTPUT_DIR / f'{task}_vs_snr.png'
        plt.savefig(snr_plot_path, bbox_inches='tight')
        plt.close()

    print(f"✅ Successfully generated {len(tasks) * 2} plots!")
    print(f"📂 Check the folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()