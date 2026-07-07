import cv2
import numpy as np

def gaussian_filter(image, ksize=5):
    """
    Applies Gaussian filtering to smooth Gaussian noise.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def median_filter(image, ksize=5):
    """
    Applies Median filtering to remove Salt & Pepper noise.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)

def gamma_correction(image, gamma=1.5):
    """
    Applies Gamma correction using a Look-Up Table (LUT) for efficiency.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def clahe_correction(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies CLAHE on the L-channel after converting BGR to LAB color space,
    then converts back to BGR.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    """
    Applies a standard 3x3 sharpening kernel via filter2D.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Single source of truth for which enhancement recovers which distortion, shared by
# apply_enhancements.py and evaluate_enhancements.py so both stages always agree.
ENHANCEMENT_FOR_DISTORTION = {
    'gaussian_noise': lambda img: gaussian_filter(img, ksize=5),
    'salt_pepper': lambda img: median_filter(img, ksize=5),
    'low_light': lambda img: clahe_correction(img, clip_limit=2.0),
    'motion_blur': lambda img: sharpen_image(img)
}
