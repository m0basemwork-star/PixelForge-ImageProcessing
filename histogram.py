import cv2
import numpy as np

def equalize(image):
    """Histogram Equalization for grayscale image"""
    temp_image = np.copy(image)
    hist = cv2.calcHist([temp_image], [0], None, [256], [0, 256])
    pdf = hist / np.prod(temp_image.shape)
    cdf = np.cumsum(pdf)
    equalized = np.round(cdf * 255).astype(np.uint8)
    equalized_img = equalized[temp_image]
    return equalized_img

def streching(image):
    """Histogram Stretching for grayscale image"""
    temp_image = np.copy(image).astype(np.float32)
    low = np.min(temp_image)
    high = np.max(temp_image)
    temp_image = np.clip((255 * ((temp_image - low) / (high - low))), 0, 255).astype(np.uint8)
    return temp_image