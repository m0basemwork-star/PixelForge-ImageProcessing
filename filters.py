
import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, generic_filter

def _validate_kernel_size(kernel_size):
    if kernel_size <= 0:
        raise ValueError("Kernel size must be greater than zero")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    return kernel_size

def _to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def average_filter(image, kernel_size):
    kernel_size = _validate_kernel_size(kernel_size)
    return cv2.blur(image, (kernel_size, kernel_size))

def laplacian_filter(image, kernel_size):
    
    kernel_size = _validate_kernel_size(kernel_size)
    gray = _to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    laplacian_abs = np.absolute(laplacian)
    return np.uint8(laplacian_abs)


def maximum_filter_op(image, kernel_size):
    kernel_size = _validate_kernel_size(kernel_size)
    return maximum_filter(image, size=kernel_size).astype(np.uint8)


def minimum_filter_op(image, kernel_size):
    kernel_size = _validate_kernel_size(kernel_size)
    return minimum_filter(image, size=kernel_size).astype(np.uint8)


def median_filter(image, kernel_size):
    kernel_size = _validate_kernel_size(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def _mode_function(window):
    unique, counts = np.unique(window, return_counts=True)
    return unique[np.argmax(counts)]


def mode_filter(image, kernel_size):
    kernel_size = _validate_kernel_size(kernel_size)
    footprint = np.ones((kernel_size, kernel_size), dtype=bool)
    return generic_filter(image, _mode_function, footprint=footprint).astype(np.uint8)


def gaussian_filter(image, kernel_size):
    kernel_size = _validate_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)