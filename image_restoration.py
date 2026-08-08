import cv2
import numpy as np


# ──────────────────────────────────────────────
# Noise Addition (for demonstration / testing)
# ──────────────────────────────────────────────

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image, density=0.05):
    """Add Salt & Pepper noise to an image."""
    noisy = image.copy()
    total_pixels = image.size
    num_salt = int(total_pixels * density / 2)
    num_pepper = int(total_pixels * density / 2)

    # Salt (white)
    coords = [np.random.randint(0, d, num_salt) for d in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper (black)
    coords = [np.random.randint(0, d, num_pepper) for d in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy


def add_speckle_noise(image, intensity=0.1):
    """Add Speckle (multiplicative) noise to an image."""
    noise = np.random.randn(*image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + image.astype(np.float32) * noise * intensity
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Noise Removal / Restoration Filters
# ──────────────────────────────────────────────

def mean_filter(image, kernel_size=3):
    """Mean (Average) Filter — reduces Gaussian noise."""
    return cv2.blur(image, (kernel_size, kernel_size))


def median_filter(image, kernel_size=3):
    """Median Filter — best for Salt & Pepper noise."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def gaussian_filter(image, kernel_size=5, sigma=0):
    """Gaussian Filter — smooth Gaussian noise with blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral Filter — edge-preserving noise removal.
    d: diameter of pixel neighbourhood
    sigma_color: filter sigma in color space
    sigma_space: filter sigma in coordinate space
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def wiener_filter(image, kernel_size=5, noise_var=None):
    """
    Simplified Wiener Filter in frequency domain.
    Effective for Gaussian blur + noise restoration.
    kernel_size: assumed blur kernel size
    noise_var: estimated noise variance (auto-estimated if None)
    """
    gray = image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_float = gray.astype(np.float32) / 255.0
    img_fft = np.fft.fft2(img_float)

    # Assume a simple average (box) PSF
    psf = np.zeros_like(img_float)
    half = kernel_size // 2
    psf[:kernel_size, :kernel_size] = 1.0 / (kernel_size ** 2)
    psf_fft = np.fft.fft2(psf)

    if noise_var is None:
        noise_var = np.var(img_float) * 0.01  # rough estimate

    psf_conj = np.conj(psf_fft)
    wiener = psf_conj / (np.abs(psf_fft) ** 2 + noise_var)
    restored_fft = img_fft * wiener
    restored = np.abs(np.fft.ifft2(restored_fft))
    restored = np.clip(restored * 255, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)

    return restored


def non_local_means_filter(image, h=10, template_window=7, search_window=21):
    """
    Non-Local Means Denoising — preserves textures.
    h: filter strength (higher = more smoothing)
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h,
                                                template_window, search_window)
    return cv2.fastNlMeansDenoising(image, None, h,
                                     template_window, search_window)


# ──────────────────────────────────────────────
# Sharpening / Enhancement
# ──────────────────────────────────────────────

def unsharp_masking(image, kernel_size=5, sigma=1.0, amount=1.5):
    """
    Unsharp Masking — enhances edges by adding high-frequency detail.
    amount: strength of sharpening (1.0 = moderate, 2.0 = strong)
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
