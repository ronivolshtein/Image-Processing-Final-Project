# 🌟 Image Processing Final Project - Setup & Execution

A quick guide to setting up the environment and running the visual experiments for.

## 🛠️ 1. Environment Setup
Before running the project for the first time, ensure that your virtual environment (`venv`) is activated and all required packages are installed. 

Run the following commands from the project's root directory:

```bash
# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Run the validation script to download resources (models & datasets)
python main.py
Note: Running main.py will automatically download the YOLOv8 weights and the official coco128 dataset if they are missing.

# Step A: Deep Learning (DL) Experiments
# This command initializes and cleans the output directory, applies 4 types of distortions across 4 intensity levels, and executes # Tasks 1 and 2 (YOLOv8 Object Detection & Instance Segmentation):
python src/run_DL_experiments.py

# Step B: Classical Computer Vision Experiments
# This command runs the classical CV algorithms on the exact same distorted images, adding Tasks 3 and 4 (Template Matching with 
# SNR calculation, and Lucas-Kanade Optical Flow with motion arrows) to the output directory:

python src/run_classical_experiments.py

# 📂 3. Verifying the Results
# Once both runs are complete, open the following directory in VS Code:
#📁 data/output_images/
# Inside, you will find the step-by-step visual workflow of Week 1, organized from task_1 to task_4. Additionally, the statistical # CSV reports will be saved in the data/output/ directory.