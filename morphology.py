import cv2
import numpy as np


def _get_kernel(kernel_size, shape="rect"):
    shapes = {
        "rect":    cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross":   cv2.MORPH_CROSS,
    }
    morph_shape = shapes.get(shape, cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (kernel_size, kernel_size))


def erosion(image, kernel_size=3, shape="rect", iterations=1):
    """Morphological Erosion — shrinks bright regions."""
    kernel = _get_kernel(kernel_size, shape)
    return cv2.erode(image, kernel, iterations=iterations)


def dilation(image, kernel_size=3, shape="rect", iterations=1):
    """Morphological Dilation — expands bright regions."""
    kernel = _get_kernel(kernel_size, shape)
    return cv2.dilate(image, kernel, iterations=iterations)


def opening(image, kernel_size=3, shape="rect"):
    """
    Morphological Opening (erosion then dilation).
    Removes small bright noise objects.
    """
    kernel = _get_kernel(kernel_size, shape)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image, kernel_size=3, shape="rect"):
    """
    Morphological Closing (dilation then erosion).
    Fills small dark holes inside bright regions.
    """
    kernel = _get_kernel(kernel_size, shape)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def morphological_gradient(image, kernel_size=3, shape="rect"):
    """
    Morphological Gradient (dilation - erosion).
    Highlights edges/boundaries of objects.
    """
    kernel = _get_kernel(kernel_size, shape)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def top_hat(image, kernel_size=9, shape="rect"):
    """
    Top Hat Transform (image - opening).
    Extracts small bright details from a dark background.
    """
    kernel = _get_kernel(kernel_size, shape)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def black_hat(image, kernel_size=9, shape="rect"):
    """
    Black Hat Transform (closing - image).
    Extracts small dark details from a bright background.
    """
    kernel = _get_kernel(kernel_size, shape)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def skeletonize(image):
    """
    Morphological Skeletonization.
    Reduces binary objects to their topological skeleton.
    Expects a binary (grayscale) image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = binary.copy()

    while True:
        eroded = cv2.erode(temp, kernel)
        dilated = cv2.dilate(eroded, kernel)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton
