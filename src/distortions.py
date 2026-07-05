"""
Distortion Module for Image Processing Project (COCO128 Dataset)
================================================================
This module contains 4 distortion functions and an SNR calculation helper.
All functions are designed to work directly with OpenCV/NumPy images (BGR, uint8).

Available Distortions & Levels:
-------------------------------
1. Gaussian Noise       - Levels 1 to 4 (Sigma: 15, 30, 50, 75)
2. Salt & Pepper Noise  - Levels 1 to 4 (Density: 2%, 5%, 15%, 30%)
3. Low Light            - Levels 1 to 4 (Combined Linear Scaling & Gamma)
4. Motion Blur          - Levels 1 to 4 (Kernel Sizes: 5x5, 11x11, 21x21, 35x35)

Usage Example:
--------------
    import cv2
    from distortions import apply_gaussian_noise, calculate_snr

    # Load clean image
    clean_img = cv2.imread("path_to_image.jpg")

    # Apply distortion (Level 2)
    distorted_img = apply_gaussian_noise(clean_img, level=2)

    # Calculate SNR
    snr_value = calculate_snr(clean_img, distorted_img)
    print(f"SNR: {snr_value:.2f} dB")
"""

import cv2
import numpy as np


# ==========================================
# Helper Function: SNR Calculation
# ==========================================
def calculate_snr(clean_img, distorted_img):
    """Calculates the Signal-to-Noise Ratio (SNR) in dB between clean and distorted images."""
    clean_f = clean_img.astype(np.float64)
    distorted_f = distorted_img.astype(np.float64)

    # Calculate signal power (clean image)
    signal_power = np.sum(clean_f**2)

    # Calculate noise power (squared difference)
    noise_img = clean_f - distorted_f
    noise_power = np.sum(noise_img**2)

    # Prevent division by zero if images are completely identical
    if noise_power == 0:
        return float("inf")

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# ==========================================
# 1. Gaussian Noise Distortion
# ==========================================
def apply_gaussian_noise(image, level):
    """Applies Gaussian noise based on 4 intensity levels."""
    # Standard deviation (Sigma) levels from light to heavy
    levels_sigma = {1: 15, 2: 30, 3: 50, 4: 75}
    sigma = levels_sigma.get(level, 15)

    row, col, ch = image.shape
    mean = 0
    # Generate Gaussian distribution noise
    gauss = np.random.normal(mean, sigma, (row, col, ch))

    # Add noise and clip values to maintain valid pixel range (0-255)
    noisy_image = image.astype(np.float64) + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


# ==========================================
# 2. Salt & Pepper Noise Distortion
# ==========================================
def apply_salt_and_pepper_noise(image, level):
    """Applies Salt & Pepper noise based on 4 intensity levels."""
    # Pixel noise density levels
    levels_density = {1: 0.02, 2: 0.05, 3: 0.15, 4: 0.30}
    prob = levels_density.get(level, 0.02)

    noisy_image = image.copy()

    # Create a random matrix to determine noise placement
    random_matrix = np.random.rand(*image.shape[:2])

    # Apply Salt (white pixels) - half of total noise probability
    noisy_image[random_matrix < (prob / 2)] = 255

    # Apply Pepper (black pixels) - second half of noise probability
    noisy_image[
        (random_matrix >= (prob / 2)) & (random_matrix < prob)
    ] = 0

    return noisy_image


# ==========================================
# 3. Low-Light Distortion
# ==========================================
def apply_low_light(image, level):
    """Applies low-light and low contrast using linear scaling and gamma reduction."""
    # Combination of scaling (max brightness) and gamma (mid-tone reduction)
    levels_params = {
        1: {"scale": 0.7, "gamma": 0.8},
        2: {"scale": 0.5, "gamma": 0.6},
        3: {"scale": 0.3, "gamma": 0.4},
        4: {"scale": 0.15, "gamma": 0.3},
    }
    params = levels_params.get(level, levels_params[1])

    # Normalize to 0-1 range for gamma correction
    img_normalized = image.astype(np.float64) / 255.0

    # Apply gamma adjustment and compress contrast range
    low_light_img = (img_normalized ** params["gamma"]) * params["scale"]

    # Scale back to original 0-255 range
    low_light_img = (low_light_img * 255).astype(np.uint8)

    return low_light_img


# ==========================================
# 4. Motion Blur Distortion
# ==========================================
def apply_motion_blur(image, level):
    """Applies a diagonal motion blur using custom convolution kernels."""
    # Kernel size determines blur distance (larger kernel = heavier blur)
    levels_kernel_size = {1: 5, 2: 11, 3: 21, 4: 35}
    size = levels_kernel_size.get(level, 5)

    # Create a linear diagonal kernel (simulates 45-degree movement)
    kernel = np.zeros((size, size))
    np.fill_diagonal(kernel, 1)
    kernel = kernel / size  # Normalize kernel to preserve image brightness

    # Apply classical 2D convolution filtering
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image