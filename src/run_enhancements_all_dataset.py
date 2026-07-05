import os
import cv2
import pandas as pd
from distortions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_light,
    apply_motion_blur,
    calculate_snr
)
from enhancements import (
    gaussian_filter,
    median_filter,
    clahe_correction,
    sharpen_image
)

def run_bulk_enhancements():
    print("\n[START] STARTING BULK IMAGE ENHANCEMENT PIPELINE (30 IMAGES)")

    # Dynamic path resolution
    matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
    roni_path = "/home/roni/datasets/coco128/images/train2017"

    if os.path.exists(matan_path):
        dataset_dir = matan_path
    elif os.path.exists(roni_path):
        dataset_dir = roni_path
    else:
        dataset_dir = "datasets/coco128/images/train2017"
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found at {dataset_dir}")
        return

    # Select first 30 images
    all_images = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_subset = all_images[:30]

    # Create directories for outputs
    distorted_base_dir = "data/enhancement_output/distorted_images"
    enhanced_base_dir = "data/enhancement_output/enhanced_images"
    csv_output_dir = "data/enhancement_output/enhancement_output_all_dataset"

    os.makedirs(distorted_base_dir, exist_ok=True)
    os.makedirs(enhanced_base_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)

    distortion_funcs = {
        "gaussian_noise": apply_gaussian_noise,
        "salt_pepper": apply_salt_and_pepper_noise,
        "low_light": apply_low_light,
        "motion_blur": apply_motion_blur
    }

    records = []

    for idx, img_name in enumerate(image_subset, 1):
        img_path = os.path.join(dataset_dir, img_name)
        clean_img = cv2.imread(img_path)

        if clean_img is None:
            continue

        print(f"[{idx}/30] Processing: {img_name}")

        for dist_name, dist_func in distortion_funcs.items():
            for level in range(1, 5):
                # 1. Apply Distortion
                distorted = dist_func(clean_img, level)
                snr_distorted = calculate_snr(clean_img, distorted)

                # 2. Apply corresponding enhancement mapping
                if dist_name == "gaussian_noise":
                    ksize = 3 if level <= 2 else 5
                    enhanced = gaussian_filter(distorted, ksize=ksize)
                    method = f"Gaussian Filter (k={ksize})"
                elif dist_name == "salt_pepper":
                    ksize = 3 if level <= 2 else 5
                    enhanced = median_filter(distorted, ksize=ksize)
                    method = f"Median Filter (k={ksize})"
                elif dist_name == "low_light":
                    enhanced = clahe_correction(distorted, clip_limit=2.0)
                    method = "CLAHE Correction"
                elif dist_name == "motion_blur":
                    enhanced = sharpen_image(distorted)
                    method = "Sharpening Kernel (3x3)"

                snr_enhanced = calculate_snr(clean_img, enhanced)

                # 3. Save images to structured folder directories
                dist_subfolder = os.path.join(distorted_base_dir, f"{dist_name}_l{level}")
                enh_subfolder = os.path.join(enhanced_base_dir, f"{dist_name}_l{level}")
                
                os.makedirs(dist_subfolder, exist_ok=True)
                os.makedirs(enh_subfolder, exist_ok=True)

                dist_path = os.path.join(dist_subfolder, img_name)
                enh_path = os.path.join(enh_subfolder, img_name)

                cv2.imwrite(dist_path, distorted)
                cv2.imwrite(enh_path, enhanced)

                # 4. Record details (converting all paths to forward slashes for cross-platform compatibility)
                records.append({
                    "image_name": img_name,
                    "distortion_type": dist_name,
                    "level": level,
                    "snr_distorted_db": snr_distorted,
                    "snr_enhanced_db": snr_enhanced,
                    "enhancement_method": method,
                    "original_image_path": img_path.replace("\\", "/"),
                    "distorted_image_path": dist_path.replace("\\", "/"),
                    "enhanced_image_path": enh_path.replace("\\", "/"),
                })

    # Save to CSV
    csv_file_path = os.path.join(csv_output_dir, "metadata_summary.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_file_path, index=False)

    print(f"\n[SUCCESS] BULK PIPELINE DONE! Metadata CSV successfully saved to: {csv_file_path}")

if __name__ == "__main__":
    run_bulk_enhancements()
