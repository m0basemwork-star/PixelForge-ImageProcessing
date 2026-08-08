import cv2
import numpy as np


def _to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def sobel_edge(image, direction="both"):
    """
    Sobel Edge Detection.
    direction: 'x', 'y', or 'both'
    """
    gray = _to_grayscale(image)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    if direction == "x":
        result = np.abs(sobel_x)
    elif direction == "y":
        result = np.abs(sobel_y)
    else:
        result = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def prewitt_edge(image, direction="both"):
    """
    Prewitt Edge Detection.
    direction: 'x', 'y', or 'both'
    """
    gray = _to_grayscale(image)
    kernel_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float32)

    prewitt_x = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
    prewitt_y = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)

    if direction == "x":
        result = np.abs(prewitt_x)
    elif direction == "y":
        result = np.abs(prewitt_y)
    else:
        result = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def roberts_edge(image):
    """Roberts Cross Edge Detection."""
    gray = _to_grayscale(image)
    kernel_x = np.array([[1,  0],
                          [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0,  1],
                          [-1, 0]], dtype=np.float32)

    roberts_x = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
    roberts_y = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
    result = np.sqrt(roberts_x ** 2 + roberts_y ** 2)
    return np.clip(result, 0, 255).astype(np.uint8)


def canny_edge(image, threshold1=100, threshold2=200):
    """
    Canny Edge Detection.
    threshold1: lower hysteresis threshold
    threshold2: upper hysteresis threshold
    """
    gray = _to_grayscale(image)
    return cv2.Canny(gray, threshold1, threshold2)


def laplacian_of_gaussian(image, kernel_size=5, sigma=1.0):
    """Laplacian of Gaussian (LoG) Edge Detection."""
    gray = _to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.clip(np.abs(log), 0, 255).astype(np.uint8)
