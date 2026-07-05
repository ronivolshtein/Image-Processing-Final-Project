"""
Week 1 Bulk Runner: Baseline & Distortions Dataset
=====================================================
This script processes the first 30 clean images from the dataset and applies 
4 distortions at 4 intensity levels each, running all 4 tasks on each image state.

Architecture:
- Pure task evaluation functions (returns metrics + visualized image)
- Centralized directory pre-creation for minimal disk I/O
- Single output CSV with all metrics in normalized format

Output:
- metadata_summary_base.csv: 2,040 rows (120 clean baseline + 1,920 distorted)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Import pure task evaluation functions
from run_classical_experiments import evaluate_optical_flow, evaluate_template_matching
from run_dl_experiments import evaluate_object_detection, evaluate_segment_instances
from yolo_tasks import YoloTasks
from distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur,
    calculate_snr
)


class Week1BulkRunner:
    """Orchestrates baseline and distortion experiments for 30 images."""
    
    def __init__(self, num_images=30):
        """
        Initialize the bulk runner.
        
        Args:
            num_images: Number of images to process from the dataset (default: 30)
        """
        self.num_images = num_images
        self.task_names = ['optical_flow', 'template_matching', 'segment_instances', 'object_detection']
        self.distortion_types = ['gaussian_noise', 'salt_pepper', 'low_light', 'motion_blur']
        self.distortion_levels = [1, 2, 3, 4]
        
        # Initialize models
        self.yolo_tasks = YoloTasks(det_model_path="yolov8n.pt", seg_model_path="yolov8n-seg.pt")
        
        # Resolve dataset path dynamically
        self.dataset_dir = self._resolve_dataset_path()
        if not self.dataset_dir or not os.path.exists(self.dataset_dir):
            raise ValueError(f"❌ Dataset directory not found")
        
        # Setup output directories
        self.output_dirs = self._setup_output_directories()
        print(f"✅ Output directories initialized")
        
        # All metrics collected during processing
        self.all_records = []
    
    def _resolve_dataset_path(self):
        """Detect if matan_path or roni_path exists, otherwise use relative path."""
        matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
        roni_path = "/home/roni/datasets/coco128/images/train2017"
        
        if os.path.exists(matan_path):
            print(f"📁 Using Matan's dataset path")
            return matan_path
        elif os.path.exists(roni_path):
            print(f"📁 Using Roni's dataset path")
            return roni_path
        else:
            # Try relative path
            rel_path = "datasets/coco128/images/train2017"
            if os.path.exists(rel_path):
                print(f"📁 Using relative dataset path")
                return rel_path
            return None
    
    def _setup_output_directories(self):
        """
        Pre-create all output directory structure to minimize I/O inside loops.
        Returns a dict with all directory paths.
        """
        base_output = "data"
        dirs = {}
        
        # Create base directories
        dirs['distorted_images'] = os.path.join(base_output, "distorted_images")
        dirs['tasks_applied'] = os.path.join(base_output, "tasks_applied_on_distorted")
        dirs['metadata_tables'] = os.path.join(base_output, "tasks_graphs_and_tables")
        
        # Pre-create directory structure for distorted images
        # Structure: data/distorted_images/{distortion_type}_l{level}/
        os.makedirs(dirs['distorted_images'], exist_ok=True)
        for dist_type in self.distortion_types:
            for level in self.distortion_levels:
                dist_dir = os.path.join(dirs['distorted_images'], f"{dist_type}_l{level}")
                os.makedirs(dist_dir, exist_ok=True)
        
        # Pre-create directory structure for task outputs
        # Structure: data/tasks_applied_on_distorted/{task_name}/{distortion_type}_l{level}/
        os.makedirs(dirs['tasks_applied'], exist_ok=True)
        for task_name in self.task_names:
            task_dir = os.path.join(dirs['tasks_applied'], task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.makedirs(os.path.join(task_dir, "clean_0"), exist_ok=True)
            for dist_type in self.distortion_types:
                for level in self.distortion_levels:
                    dist_task_dir = os.path.join(task_dir, f"{dist_type}_l{level}")
                    os.makedirs(dist_task_dir, exist_ok=True)
        
        # Create metadata tables directory
        os.makedirs(dirs['metadata_tables'], exist_ok=True)
        
        return dirs

    def _get_task_output_subdir(self, distortion_type, level):
        """Return the folder name used for a task output."""
        if distortion_type == "clean" and level == 0:
            return "clean_0"
        return f"{distortion_type}_l{level}"
    
    def _normalize_path(self, path):
        """Convert all path slashes to forward slashes for cross-platform compatibility."""
        return path.replace("\\", "/")
    
    def _get_image_files(self):
        """Get the first N image files from the dataset."""
        all_images = sorted([
            f for f in os.listdir(self.dataset_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        return all_images[:self.num_images]
    
    def _create_template_and_frame2(self, img):
        """Create template and second frame for classical tasks."""
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        # Crop 100x100 template from center
        template = img[
            max(0, cy - 50):min(h, cy + 50), 
            max(0, cx - 50):min(w, cx + 50)
        ]
        # Create second frame using motion blur for optical flow
        frame2 = apply_motion_blur(img, level=3)
        return template, frame2
    
    def _run_all_tasks(self, img, template, frame2, distortion_type="clean", level=0):
        """
        Execute all 4 tasks on an image and return metrics + visualizations.
        
        Returns:
            dict: {task_name: {'metrics': dict, 'visualized_image': array}}
        """
        results = {}
        
        # Convert BGR to RGB for YOLO models
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Task 1: Optical Flow
        try:
            of_result = evaluate_optical_flow(img, frame2)
            results['optical_flow'] = of_result
        except Exception as e:
            print(f"⚠️  Optical flow failed for {distortion_type}_l{level}: {e}")
            results['optical_flow'] = {
                'metrics': {'tracked_points': 0},
                'visualized_image': img.copy()
            }
        
        # Task 2: Template Matching
        try:
            tm_result = evaluate_template_matching(img, template)
            results['template_matching'] = tm_result
        except Exception as e:
            print(f"⚠️  Template matching failed for {distortion_type}_l{level}: {e}")
            results['template_matching'] = {
                'metrics': {'matching_score': 0.0, 'location': '(0,0)'},
                'visualized_image': img.copy()
            }
        
        # Task 3: Instance Segmentation
        try:
            seg_result = evaluate_segment_instances(img_rgb, self.yolo_tasks.seg_model)
            results['segment_instances'] = seg_result
        except Exception as e:
            print(f"⚠️  Segmentation failed for {distortion_type}_l{level}: {e}")
            results['segment_instances'] = {
                'metrics': {'segmented_instances': 0, 'avg_confidence': 0.0},
                'visualized_image': img.copy()
            }
        
        # Task 4: Object Detection
        try:
            det_result = evaluate_object_detection(img_rgb, self.yolo_tasks.det_model)
            results['object_detection'] = det_result
        except Exception as e:
            print(f"⚠️  Object detection failed for {distortion_type}_l{level}: {e}")
            results['object_detection'] = {
                'metrics': {'detected_objects': 0, 'avg_confidence': 0.0},
                'visualized_image': img.copy()
            }
        
        return results
    
    def _save_task_visualizations(self, img_name, distortion_type, level, task_results):
        """Save all task visualization images to structured directories."""
        for task_name, task_data in task_results.items():
            vis_img = task_data['visualized_image']
            task_dir = os.path.join(
                self.output_dirs['tasks_applied'],
                task_name,
                self._get_task_output_subdir(distortion_type, level)
            )
            output_path = os.path.join(task_dir, img_name)
            cv2.imwrite(output_path, vis_img)
    
    def _append_metrics_to_records(self, img_name, orig_img_path, distorted_img_path,
                                   distortion_type, level, snr_db, task_results):
        """
        Append metrics for all tasks to records list.
        Each task metric creates one row in the CSV.
        """
        for task_name, task_data in task_results.items():
            metrics = task_data['metrics']
            
            # Flatten metrics: each metric becomes a separate row
            for metric_name, metric_value in metrics.items():
                task_image_path = os.path.join(
                    self.output_dirs['tasks_applied'],
                    task_name,
                    self._get_task_output_subdir(distortion_type, level),
                    img_name
                )
                try:
                    # tries to convert to a number (for objects, confidence, IoU etc.)
                    final_metric_value = float(metric_value)
                except (ValueError, TypeError):
                    # if this fails (like a tuple or string of the location), saves as a clean string
                    final_metric_value = str(metric_value)
                # ---------------------------------------------------

                record = {
                    'image_name': img_name,
                    'distortion_type': distortion_type,
                    'level': level,
                    'snr_distorted_db': snr_db if distortion_type != 'clean' else float('inf'),
                    'task_name': task_name,
                    'metric_name': metric_name,
                    'metric_value': final_metric_value,  # שימוש בערך הבטוח
                    'task_image_path': self._normalize_path(task_image_path),
                    'original_image_path': self._normalize_path(orig_img_path),
                    'distorted_image_path': self._normalize_path(distorted_img_path) if distortion_type != 'clean' else self._normalize_path(orig_img_path)
                }
                self.all_records.append(record)
    
    def run(self):
        """Execute the full Week 1 pipeline."""
        print(f"\n🚀 Starting Week 1 Bulk Pipeline: Processing {self.num_images} images")
        print(f"   - Tasks per image: {len(self.task_names)} (optical_flow, template_matching, segment_instances, object_detection)")
        print(f"   - Distortions per image: {len(self.distortion_types)} × {len(self.distortion_levels)} levels")
        print(f"   - Expected total rows: {self.num_images * 4} (clean) + {self.num_images * 16 * 4} (distorted) = {self.num_images * 4 + self.num_images * 16 * 4}")
        
        image_files = self._get_image_files()
        print(f"📋 Found {len(image_files)} images, processing first {self.num_images}\n")
        
        # Distortion functions
        distortion_funcs = {
            'gaussian_noise': apply_gaussian_noise,
            'salt_pepper': apply_salt_and_pepper_noise,
            'low_light': apply_low_light,
            'motion_blur': apply_motion_blur
        }
        
        # ========================================
        # MAIN LOOP: Process each image
        # ========================================
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(self.dataset_dir, img_file)
            clean_img = cv2.imread(img_path)
            
            if clean_img is None:
                print(f"⚠️  [{idx}/{len(image_files)}] Skipping {img_file} (failed to load)")
                continue
            
            print(f"🔄 [{idx}/{len(image_files)}] Processing: {img_file}")
            
            # Prepare inputs for classical tasks
            template, frame2 = self._create_template_and_frame2(clean_img)
            
            # ==================================================
            # BASELINE: Run all tasks on clean image
            # ==================================================
            clean_results = self._run_all_tasks(clean_img, template, frame2, distortion_type="clean", level=0)
            self._save_task_visualizations(img_file, "clean", 0, clean_results)
            
            # Normalize original path
            orig_img_path = os.path.join(self.dataset_dir, img_file)
            self._append_metrics_to_records(
                img_file, orig_img_path, orig_img_path,
                distortion_type="clean", level=0, snr_db=float('inf'),
                task_results=clean_results
            )
            
            # ==================================================
            # DISTORTIONS: Apply each distortion and run tasks
            # ==================================================
            for dist_type in self.distortion_types:
                dist_func = distortion_funcs[dist_type]
                
                for level in self.distortion_levels:
                    # Apply distortion
                    distorted_img = dist_func(clean_img, level)
                    snr_val = calculate_snr(clean_img, distorted_img)
                    
                    # Create second distorted frame for optical flow
                    distorted_frame2 = dist_func(frame2, level)
                    
                    # Save raw distorted image
                    distorted_img_dir = os.path.join(
                        self.output_dirs['distorted_images'],
                        f"{dist_type}_l{level}"
                    )
                    distorted_img_path = os.path.join(distorted_img_dir, img_file)
                    cv2.imwrite(distorted_img_path, distorted_img)
                    
                    # Run all tasks on distorted image
                    distorted_results = self._run_all_tasks(
                        distorted_img, template, distorted_frame2,
                        distortion_type=dist_type, level=level
                    )
                    
                    # Save task visualizations
                    self._save_task_visualizations(img_file, dist_type, level, distorted_results)
                    
                    # Append metrics to records
                    self._append_metrics_to_records(
                        img_file, orig_img_path, distorted_img_path,
                        distortion_type=dist_type, level=level, snr_db=snr_val,
                        task_results=distorted_results
                    )
        
        # ==================================================
        # FINALIZE: Save comprehensive CSV
        # ==================================================
        print(f"\n📊 Finalizing results...")
        df = pd.DataFrame(self.all_records)
        
        output_csv = os.path.join(self.output_dirs['metadata_tables'], 'metadata_summary_base.csv')
        df.to_csv(output_csv, index=False)
        
        total_rows = len(df)
        print(f"\n✅ Pipeline Complete!")
        print(f"   📁 Output CSV: {output_csv}")
        print(f"   📊 Total rows: {total_rows} (Expected: {self.num_images * 4 + self.num_images * 16 * 4})")
        print(f"   📁 Distorted images: {self.output_dirs['distorted_images']}")
        print(f"   📁 Task visualizations: {self.output_dirs['tasks_applied']}")
        
        return df


def main():
    """Entry point for the bulk runner."""
    try:
        runner = Week1BulkRunner(num_images=30)
        df_results = runner.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()