# ⬡ PixelForge — Image Processing Toolkit

A complete desktop image processing application built with Python, OpenCV, and Tkinter.

🔗 **Interactive web version available:** [PixelForge-Web](https://github.com/m0basemwork-star/PixelForge-Web)

## 👥 Team & Modules

| Member | Module(s) |
|---|---|
| أحمد باسم (Leader) | `main_gui.py` + `edge_detection.py` + `image_segmentation.py` |
| محمد باسم | `point_ops.py` — Add, Subtract, Divide, Complement |
| علي حسام علي | `color_operation.py` + `histogram.py` |
| السيد عصام | `filters.py` — Linear & Non-linear Filters |
| زياد محمد أبو العنين | `image_restoration.py` + `morphology.py` |

## 📁 File Structure

```
PixelForge/
├── main_gui.py            # Main application window (Tkinter GUI)
├── point_ops.py           # Point operations (add, subtract, divide, complement)
├── histogram.py           # Histogram equalization & stretching
├── color_operation.py     # Color channel operations
├── filters.py             # Neighborhood filters (avg, gaussian, median, etc.)
├── edge_detection.py      # Edge detection (Sobel, Prewitt, Roberts, Canny, LoG)
├── image_segmentation.py  # Segmentation (threshold, K-means, watershed, etc.)
├── morphology.py          # Mathematical morphology (erosion, dilation, etc.)
├── image_restoration.py   # Noise addition & removal, Wiener, NLM, bilateral
└── requirements.txt       # Python dependencies
```

## 🚀 Run

```bash
pip install -r requirements.txt
python main_gui.py
```

## ✨ Features

- **Point Operations**: Brightness ±, image add/subtract/divide, complement
- **Histogram**: Equalization, Stretching, live histogram plot
- **Color Operations**: Per-channel lighting, channel swap, channel elimination
- **Filters**: Average, Gaussian, Median, Maximum, Minimum, Mode, Laplacian
- **Edge Detection**: Sobel (X/Y/both), Prewitt, Roberts, Canny, LoG
- **Segmentation**: Global/Otsu/Adaptive threshold, K-Means, Watershed
- **Morphology**: Erosion, Dilation, Opening, Closing, Gradient, Top/Black Hat, Skeletonize
- **Restoration**: Add noise (Gaussian/S&P/Speckle), Mean/Median/Gaussian/Bilateral/Wiener/NLM/Unsharp Mask
