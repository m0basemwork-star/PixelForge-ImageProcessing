import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_result(original, result, title="Result"):
    def to_display(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
        return img, "gray"

    orig_disp, orig_cmap = to_display(original)
    res_disp,  res_cmap  = to_display(result)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_disp, cmap=orig_cmap)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(res_disp, cmap=res_cmap)
    axes[1].set_title(title, fontsize=13)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def _match_size(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2


def add_images(img1, img2, alpha=0.5, beta=0.5):
    img1, img2 = _match_size(img1, img2)
    return cv2.addWeighted(img1, alpha, img2, beta, 0)


def add_brightness(img, value=50):
    value = int(np.clip(value, -255, 255))
    offset = np.ones_like(img, dtype=np.uint8) * abs(value)
    if value >= 0:
        return cv2.add(img, offset)
    else:
        return cv2.subtract(img, offset)


def subtract_images(img1, img2):
    img1, img2 = _match_size(img1, img2)
    return cv2.subtract(img1, img2)


def subtract_brightness(img, value=50):
    value = int(np.clip(value, 0, 255))
    offset = np.ones_like(img, dtype=np.uint8) * value
    return cv2.subtract(img, offset)


def divide_images(img1, img2):
    img1, img2 = _match_size(img1, img2)
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    result_f = np.where(img2_f != 0, img1_f / img2_f, 0.0)
    max_val = result_f.max()
    if max_val > 0:
        result_f = (result_f / max_val) * 255.0
    return np.clip(result_f, 0, 255).astype(np.uint8)


def divide_scalar(img, value=2.0):
    if value <= 0:
        raise ValueError("قيمة القسمة لازم تكون أكبر من صفر!")
    result = img.astype(np.float32) / float(value)
    return np.clip(result, 0, 255).astype(np.uint8)


def complement(img):
    return cv2.bitwise_not(img)


def apply_point_operation(img, operation, value=None, img2=None, show=True):
    if img is None:
        return None

    result = None
    title  = ""

    if operation == "add":
        alpha  = float(value) if value is not None else 0.5
        beta   = 1.0 - alpha
        result = add_images(img, img2, alpha=alpha, beta=beta)
        title  = f"Addition (α={alpha:.1f} · β={beta:.1f})"

    elif operation == "add_brightness":
        v      = int(value) if value is not None else 50
        result = add_brightness(img, v)
        sign   = "+" if v >= 0 else ""
        title  = f"Brightness {sign}{v}"

    elif operation == "subtract":
        result = subtract_images(img, img2)
        title  = "Subtraction (img1 − img2)"

    elif operation == "sub_brightness":
        v      = int(value) if value is not None else 50
        result = subtract_brightness(img, v)
        title  = f"Subtract Brightness −{v}"

    elif operation == "divide":
        result = divide_images(img, img2)
        title  = "Division (img1 ÷ img2)"

    elif operation == "divide_scalar":
        v      = float(value) if value is not None else 2.0
        result = divide_scalar(img, v)
        title  = f"Division ÷ {v}"

    elif operation == "complement":
        result = complement(img)
        title  = "Complement (Negative)"

    else:
        print(f"[point_ops] عملية غير معروفة: '{operation}'")
        return img

    if show and result is not None:
        show_result(img, result, title=title)

    return result
