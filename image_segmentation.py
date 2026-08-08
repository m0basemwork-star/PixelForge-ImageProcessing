import cv2
import numpy as np


def _to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def global_threshold(image, threshold=127):
    """
    Global (Manual) Thresholding.
    Pixels above threshold → 255, else → 0.
    """
    gray = _to_grayscale(image)
    _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return result


def otsu_threshold(image):
    """
    Otsu's Automatic Thresholding.
    Automatically finds the optimal threshold value.
    """
    gray = _to_grayscale(image)
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def adaptive_threshold(image, block_size=11, C=2, method="mean"):
    """
    Adaptive Thresholding.
    block_size: size of neighbourhood area (must be odd)
    C: constant subtracted from the mean
    method: 'mean' or 'gaussian'
    """
    gray = _to_grayscale(image)
    if block_size % 2 == 0:
        block_size += 1

    adaptive_method = (cv2.ADAPTIVE_THRESH_MEAN_C
                       if method == "mean"
                       else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    return cv2.adaptiveThreshold(gray, 255, adaptive_method,
                                  cv2.THRESH_BINARY, block_size, C)


def kmeans_segmentation(image, k=3, attempts=10):
    """
    K-Means Clustering Segmentation.
    k: number of clusters/segments
    """
    pixel_vals = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, attempts,
                                     cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    return result.reshape(image.shape)


def region_growing(image, seed_point, threshold=15):
    """
    Region Growing Segmentation.
    seed_point: (row, col) tuple — starting point
    threshold: max intensity difference from seed to be included
    """
    gray = _to_grayscale(image)
    rows, cols = gray.shape
    visited = np.zeros((rows, cols), dtype=bool)
    segmented = np.zeros((rows, cols), dtype=np.uint8)

    r, c = seed_point
    seed_val = int(gray[r, c])
    stack = [(r, c)]
    visited[r, c] = True

    while stack:
        cr, cc = stack.pop()
        segmented[cr, cc] = 255
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                if abs(int(gray[nr, nc]) - seed_val) <= threshold:
                    stack.append((nr, nc))
                    visited[nr, nc] = True

    return segmented


def watershed_segmentation(image):
    """
    Watershed Segmentation using distance transform markers.
    Returns the segmented image with colored regions.
    """
    gray = _to_grayscale(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background and foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = (image.copy() if len(image.shape) == 3
                 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    cv2.watershed(img_color, markers)
    img_color[markers == -1] = [0, 0, 255]  # boundary in red
    return img_color
