import os
import cv2
import numpy as np
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
    gamma_correction,
    clahe_correction,
    sharpen_image
)

def create_labeled_image(img, label_text, color=(0, 255, 0)):
    """Creates a copy of the image and draws a text label overlay."""
    vis = img.copy()
    cv2.putText(vis, label_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return vis

def run_single_image_demo():
    print("\n[START] STARTING SINGLE-IMAGE ENHANCEMENT DEMO")

    # Dynamic path resolution
    matan_path = r"C:\Users\compu\Documents\cv_project\datasets\coco128\images\train2017"
    roni_path = "/home/roni/datasets/coco128/images/train2017"

    if os.path.exists(matan_path):
        dataset_dir = matan_path
    elif os.path.exists(roni_path):
        dataset_dir = roni_path
    else:
        dataset_dir = "datasets/coco128/images/train2017"

    img_name = "000000000009.jpg"
    img_path = os.path.join(dataset_dir, img_name)
    clean_img = cv2.imread(img_path)
    if clean_img is None:
        print(f"[ERROR] Could not read sample image at {img_path}")
        return

    output_dir = "data/enhancement_output/enhancement_images"
    os.makedirs(output_dir, exist_ok=True)

    # Labeled original clean image
    orig_lbl = create_labeled_image(clean_img, "Original (GT)", (0, 255, 0))

    # 1. Gaussian Noise (Level 2) -> Gaussian Filter
    print("Processing: Gaussian Noise...")
    dist_gauss = apply_gaussian_noise(clean_img, level=2)
    snr_dist_gauss = calculate_snr(clean_img, dist_gauss)
    
    enh_gauss = gaussian_filter(dist_gauss, ksize=5)
    snr_enh_gauss = calculate_snr(clean_img, enh_gauss)

    dist_lbl = create_labeled_image(dist_gauss, f"Distorted (SNR: {snr_dist_gauss:.2f}dB)", (0, 0, 255))
    enh_lbl = create_labeled_image(enh_gauss, f"Gaussian Filter (SNR: {snr_enh_gauss:.2f}dB)", (255, 255, 0))
    
    grid_gauss = np.hstack((orig_lbl, dist_lbl, enh_lbl))
    cv2.imwrite(os.path.join(output_dir, "gaussian_noise_grid.jpg"), grid_gauss)

    # 2. Salt & Pepper (Level 2) -> Median Filter
    print("Processing: Salt & Pepper Noise...")
    dist_sp = apply_salt_and_pepper_noise(clean_img, level=2)
    snr_dist_sp = calculate_snr(clean_img, dist_sp)

    enh_sp = median_filter(dist_sp, ksize=5)
    snr_enh_sp = calculate_snr(clean_img, enh_sp)

    dist_lbl = create_labeled_image(dist_sp, f"Distorted (SNR: {snr_dist_sp:.2f}dB)", (0, 0, 255))
    enh_lbl = create_labeled_image(enh_sp, f"Median Filter (SNR: {snr_enh_sp:.2f}dB)", (255, 255, 0))
    
    grid_sp = np.hstack((orig_lbl, dist_lbl, enh_lbl))
    cv2.imwrite(os.path.join(output_dir, "salt_pepper_grid.jpg"), grid_sp)

    # 3. Low Light (Level 2) -> Gamma & CLAHE correction
    print("Processing: Low Light Contrast...")
    dist_light = apply_low_light(clean_img, level=2)
    snr_dist_light = calculate_snr(clean_img, dist_light)

    enh_gamma = gamma_correction(dist_light, gamma=1.5)
    snr_enh_gamma = calculate_snr(clean_img, enh_gamma)

    enh_clahe = clahe_correction(dist_light, clip_limit=2.0)
    snr_enh_clahe = calculate_snr(clean_img, enh_clahe)

    dist_lbl = create_labeled_image(dist_light, f"Distorted (SNR: {snr_dist_light:.2f}dB)", (0, 0, 255))
    gamma_lbl = create_labeled_image(enh_gamma, f"Gamma 1.5 (SNR: {snr_enh_gamma:.2f}dB)", (255, 255, 0))
    clahe_lbl = create_labeled_image(enh_clahe, f"CLAHE (SNR: {snr_enh_clahe:.2f}dB)", (255, 0, 255))

    grid_light = np.hstack((orig_lbl, dist_lbl, gamma_lbl, clahe_lbl))
    cv2.imwrite(os.path.join(output_dir, "low_light_grid.jpg"), grid_light)

    # 4. Motion Blur (Level 2) -> Sharpening filter
    print("Processing: Motion Blur...")
    dist_blur = apply_motion_blur(clean_img, level=2)
    snr_dist_blur = calculate_snr(clean_img, dist_blur)

    enh_sharp = sharpen_image(dist_blur)
    snr_enh_sharp = calculate_snr(clean_img, enh_sharp)

    dist_lbl = create_labeled_image(dist_blur, f"Distorted (SNR: {snr_dist_blur:.2f}dB)", (0, 0, 255))
    enh_lbl = create_labeled_image(enh_sharp, f"Sharpen Filter (SNR: {snr_enh_sharp:.2f}dB)", (255, 255, 0))

    grid_blur = np.hstack((orig_lbl, dist_lbl, enh_lbl))
    cv2.imwrite(os.path.join(output_dir, "motion_blur_grid.jpg"), grid_blur)

    print(f"[SUCCESS] Demo results successfully saved in: {output_dir}")

if __name__ == "__main__":
    run_single_image_demo()
