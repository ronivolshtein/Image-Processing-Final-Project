"""
Validation Script for Week 1 Refactored Pipeline
==================================================

Run this script after executing run_30_pic_dataset.py to validate the output.
"""

import os
import pandas as pd
from pathlib import Path
import sys


class PipelineValidator:
    """Validates the output structure and CSV format."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.base_output = os.path.join(data_dir, "tasks_graphs_and_tables")
        self.csv_path = os.path.join(self.base_output, "metadata_summary_base.csv")
        self.distorted_dir = os.path.join(data_dir, "distorted_images")
        self.tasks_dir = os.path.join(data_dir, "tasks_applied_on_distorted")
        self.errors = []
        self.warnings = []
        self.info = []
    
    def validate_csv_exists(self):
        """Check if CSV file exists."""
        if not os.path.exists(self.csv_path):
            self.errors.append(f"❌ CSV file not found at {self.csv_path}")
            return False
        self.info.append(f"✅ CSV file exists: {self.csv_path}")
        return True
    
    def validate_csv_structure(self):
        """Validate CSV columns and format."""
        try:
            df = pd.read_csv(self.csv_path)
            
            required_columns = {
                'image_name', 'distortion_type', 'level', 'snr_distorted_db',
                'task_name', 'metric_name', 'metric_value', 'task_image_path',
                'original_image_path', 'distorted_image_path'
            }
            
            csv_columns = set(df.columns)
            
            if not required_columns.issubset(csv_columns):
                missing = required_columns - csv_columns
                self.errors.append(f"❌ Missing columns: {missing}")
                return False
            
            extra = csv_columns - required_columns
            if extra:
                self.warnings.append(f"⚠️  Extra columns (not required): {extra}")
            
            self.info.append(f"✅ All required columns present: {required_columns}")
            return True
        except Exception as e:
            self.errors.append(f"❌ Error reading CSV: {e}")
            return False
    
    def validate_csv_rows(self):
        """Validate CSV row count and content."""
        try:
            df = pd.read_csv(self.csv_path)
            total_rows = len(df)
            
            # Expected: 30 images × 4 tasks × (1 clean + 16 distorted levels) × metrics
            # Worst case (1 metric per task): 30 × 4 × 17 = 2,040
            # Each task may have multiple metrics, so actual could be higher
            expected_min = 30 * 4 * 17  # 2,040
            
            if total_rows < expected_min:
                self.warnings.append(f"⚠️  Total rows ({total_rows}) < expected minimum ({expected_min})")
            else:
                self.info.append(f"✅ Total rows ({total_rows}) >= expected minimum ({expected_min})")
            
            # Validate distortion_type values
            valid_distortions = {'clean', 'gaussian_noise', 'salt_pepper', 'low_light', 'motion_blur'}
            invalid_dist = set(df['distortion_type'].unique()) - valid_distortions
            if invalid_dist:
                self.errors.append(f"❌ Invalid distortion_type values: {invalid_dist}")
                return False
            self.info.append(f"✅ Valid distortion_type values: {valid_distortions}")
            
            # Validate level values
            valid_levels = {0, 1, 2, 3, 4}
            invalid_levels = set(df['level'].unique()) - valid_levels
            if invalid_levels:
                self.errors.append(f"❌ Invalid level values: {invalid_levels}")
                return False
            self.info.append(f"✅ Valid level values: {valid_levels}")
            
            # Validate task_name values
            valid_tasks = {'optical_flow', 'template_matching', 'segment_instances', 'object_detection'}
            invalid_tasks = set(df['task_name'].unique()) - valid_tasks
            if invalid_tasks:
                self.errors.append(f"❌ Invalid task_name values: {invalid_tasks}")
                return False
            self.info.append(f"✅ Valid task_name values: {valid_tasks}")
            
            # Check for path slashes (should be forward slashes)
            path_cols = ['task_image_path', 'original_image_path', 'distorted_image_path']
            for col in path_cols:
                if df[col].str.contains('\\').any():
                    self.warnings.append(f"⚠️  Column '{col}' contains backslashes (should be forward slashes)")
            
            return True
        except Exception as e:
            self.errors.append(f"❌ Error validating CSV content: {e}")
            return False
    
    def validate_directory_structure(self):
        """Validate output directory structure."""
        distortion_types = ['gaussian_noise', 'salt_pepper', 'low_light', 'motion_blur']
        levels = [1, 2, 3, 4]
        task_names = ['optical_flow', 'template_matching', 'segment_instances', 'object_detection']
        
        # Check distorted images
        if not os.path.exists(self.distorted_dir):
            self.errors.append(f"❌ Distorted images directory not found: {self.distorted_dir}")
            return False
        
        for dist_type in distortion_types:
            for level in levels:
                dist_path = os.path.join(self.distorted_dir, f"{dist_type}_l{level}")
                if not os.path.exists(dist_path):
                    self.errors.append(f"❌ Missing distorted directory: {dist_path}")
                    return False
        
        self.info.append(f"✅ All distorted image directories exist (4 types × 4 levels = 16 dirs)")
        
        # Check task output directories
        if not os.path.exists(self.tasks_dir):
            self.errors.append(f"❌ Task output directory not found: {self.tasks_dir}")
            return False
        
        for task in task_names:
            task_base = os.path.join(self.tasks_dir, task)
            if not os.path.exists(task_base):
                self.errors.append(f"❌ Missing task directory: {task_base}")
                return False
            
            for dist_type in distortion_types:
                for level in levels:
                    task_dist_path = os.path.join(task_base, f"{dist_type}_l{level}")
                    if not os.path.exists(task_dist_path):
                        self.errors.append(f"❌ Missing task-distortion directory: {task_dist_path}")
                        return False
            
            # Check for clean baseline
            clean_path = os.path.join(task_base, "clean_0")
            if not os.path.exists(clean_path):
                self.errors.append(f"❌ Missing clean baseline directory: {clean_path}")
                return False
        
        self.info.append(f"✅ All task output directories exist (4 tasks × 17 distortion combos = 68 subdirs)")
        
        return True
    
    def validate_image_counts(self):
        """Validate that expected number of images are saved."""
        try:
            # Count distorted images
            distortion_types = ['gaussian_noise', 'salt_pepper', 'low_light', 'motion_blur']
            levels = [1, 2, 3, 4]
            
            total_distorted_files = 0
            for dist_type in distortion_types:
                for level in levels:
                    dist_path = os.path.join(self.distorted_dir, f"{dist_type}_l{level}")
                    if os.path.exists(dist_path):
                        num_files = len([f for f in os.listdir(dist_path) if f.endswith(('.jpg', '.png'))])
                        total_distorted_files += num_files
            
            expected_distorted = 30 * 16  # 30 images × 16 distortion levels
            if total_distorted_files >= expected_distorted * 0.8:  # Allow 20% tolerance
                self.info.append(f"✅ Distorted images count OK: {total_distorted_files} (expected ~{expected_distorted})")
            else:
                self.warnings.append(f"⚠️  Distorted images count low: {total_distorted_files} (expected ~{expected_distorted})")
            
            return True
        except Exception as e:
            self.warnings.append(f"⚠️  Could not validate image counts: {e}")
            return True
    
    def run_all_validations(self):
        """Run all validations and report results."""
        print("\n" + "="*70)
        print("WEEK 1 PIPELINE VALIDATION REPORT")
        print("="*70 + "\n")
        
        validations = [
            ("CSV File Existence", self.validate_csv_exists),
            ("CSV Structure", self.validate_csv_structure),
            ("CSV Content", self.validate_csv_rows),
            ("Directory Structure", self.validate_directory_structure),
            ("Image Counts", self.validate_image_counts),
        ]
        
        passed = 0
        failed = 0
        
        for name, validation_func in validations:
            print(f"🔍 Validating {name}...", end=" ")
            try:
                result = validation_func()
                if result:
                    print("✅ PASS")
                    passed += 1
                else:
                    print("❌ FAIL")
                    failed += 1
            except Exception as e:
                print(f"❌ ERROR: {e}")
                failed += 1
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70 + "\n")
        
        if self.info:
            print("ℹ️  Information:")
            for msg in self.info:
                print(f"   {msg}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for msg in self.warnings:
                print(f"   {msg}")
        
        if self.errors:
            print("\n❌ Errors:")
            for msg in self.errors:
                print(f"   {msg}")
        
        print("\n" + "="*70)
        print(f"Results: {passed} passed, {failed} failed")
        print("="*70 + "\n")
        
        return failed == 0


def main():
    validator = PipelineValidator()
    success = validator.run_all_validations()
    
    if success:
        print("✅ All validations passed! Pipeline output is correct.\n")
        return 0
    else:
        print("❌ Some validations failed. Please review the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
