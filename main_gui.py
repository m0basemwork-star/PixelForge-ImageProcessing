"""
PixelForge — Image Processing Toolkit
Main GUI  (Tkinter + Matplotlib)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Local modules ────────────────────────────────────────────
from point_ops       import apply_point_operation
from histogram       import equalize, streching
from color_operation import (changing_the_image_lighting_color,
                             swapping_image_channels,
                             eliminating_color_channels)
from filters         import (average_filter, laplacian_filter,
                             maximum_filter_op, minimum_filter_op,
                             median_filter, mode_filter, gaussian_filter)
from edge_detection  import (sobel_edge, prewitt_edge, roberts_edge,
                             canny_edge, laplacian_of_gaussian)
from image_segmentation import (global_threshold, otsu_threshold,
                                adaptive_threshold, kmeans_segmentation,
                                watershed_segmentation)
from morphology      import (erosion, dilation, opening, closing,
                             morphological_gradient, top_hat, black_hat,
                             skeletonize)
from image_restoration import (add_gaussian_noise, add_salt_and_pepper_noise,
                               add_speckle_noise, mean_filter, median_filter as med_filter,
                               gaussian_filter as gauss_filter, bilateral_filter,
                               wiener_filter, non_local_means_filter, unsharp_masking)


# ════════════════════════════════════════════════════════════
#  Colour palette
# ════════════════════════════════════════════════════════════
BG       = "#0f0f14"
PANEL    = "#1a1a24"
ACCENT   = "#6c63ff"
ACCENT2  = "#ff6584"
TEXT     = "#e8e8f0"
SUBTEXT  = "#8888aa"
BORDER   = "#2a2a3a"
BTN_BG   = "#252535"
BTN_HOV  = "#353550"


# ════════════════════════════════════════════════════════════
#  Helper — numpy BGR → Tkinter PhotoImage
# ════════════════════════════════════════════════════════════
def _np_to_tk(img, max_w=520, max_h=420):
    if img is None:
        return None
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


# ════════════════════════════════════════════════════════════
#  Collapsible sidebar section
# ════════════════════════════════════════════════════════════
class Section(tk.Frame):
    def __init__(self, parent, title, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._open = tk.BooleanVar(value=False)

        hdr = tk.Frame(self, bg=BORDER, cursor="hand2")
        hdr.pack(fill="x")
        self._arrow = tk.Label(hdr, text="▶", bg=BORDER, fg=ACCENT,
                               font=("Consolas", 9))
        self._arrow.pack(side="left", padx=(8, 4), pady=5)
        tk.Label(hdr, text=title, bg=BORDER, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side="left", pady=5)
        hdr.bind("<Button-1>", self._toggle)
        self._arrow.bind("<Button-1>", self._toggle)

        self.body = tk.Frame(self, bg=PANEL)

    def _toggle(self, _=None):
        if self._open.get():
            self.body.pack_forget()
            self._arrow.config(text="▶")
            self._open.set(False)
        else:
            self.body.pack(fill="x", padx=8, pady=(4, 8))
            self._arrow.config(text="▼")
            self._open.set(True)


# ════════════════════════════════════════════════════════════
#  Styled helpers
# ════════════════════════════════════════════════════════════
def _lbl(parent, text, size=9, color=SUBTEXT, **kw):
    return tk.Label(parent, text=text, bg=PANEL, fg=color,
                    font=("Segoe UI", size), **kw)

def _entry(parent, default="", width=7):
    e = tk.Entry(parent, bg=BTN_BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", width=width, font=("Consolas", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT)
    e.insert(0, str(default))
    return e

def _btn(parent, text, cmd, color=ACCENT, fg=TEXT):
    b = tk.Button(parent, text=text, command=cmd, bg=color, fg=fg,
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  activebackground=ACCENT2, activeforeground=TEXT,
                  cursor="hand2", padx=8, pady=4)
    b.bind("<Enter>", lambda e: b.config(bg=BTN_HOV))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b

def _combo(parent, values, default=0, width=14):
    cb = ttk.Combobox(parent, values=values, state="readonly",
                      width=width, font=("Segoe UI", 9))
    cb.current(default)
    return cb

def _row(parent, **kw):
    return tk.Frame(parent, bg=PANEL, **kw)


# ════════════════════════════════════════════════════════════
#  Main application
# ════════════════════════════════════════════════════════════
class PixelForge(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PixelForge  —  Image Processing Toolkit")
        self.configure(bg=BG)
        self.geometry("1280x780")
        self.minsize(1100, 680)

        self.original_img = None   # BGR numpy array
        self.current_img  = None   # BGR numpy array (after ops)
        self.second_img   = None   # for two-image operations
        self._tk_orig     = None
        self._tk_curr     = None

        self._build_ui()

    # ── UI layout ────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        bar = tk.Frame(self, bg=ACCENT, height=50)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Label(bar, text="⬡  PixelForge", bg=ACCENT, fg="white",
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=18)
        tk.Label(bar, text="Image Processing Toolkit",
                 bg=ACCENT, fg="#ddd",
                 font=("Segoe UI", 9)).pack(side="left")

        # Main panes
        pane = tk.PanedWindow(self, orient="horizontal",
                              bg=BG, sashwidth=5, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=0, pady=0)

        # LEFT: sidebar
        sidebar = tk.Frame(pane, bg=PANEL, width=310)
        sidebar.pack_propagate(False)
        pane.add(sidebar, minsize=260)

        # RIGHT: canvas area
        canvas_area = tk.Frame(pane, bg=BG)
        pane.add(canvas_area, minsize=600)

        self._build_sidebar(sidebar)
        self._build_canvas_area(canvas_area)

    # ── Sidebar ──────────────────────────────────────────────
    def _build_sidebar(self, parent):
        # Scroll
        outer = tk.Frame(parent, bg=PANEL)
        outer.pack(fill="both", expand=True)
        scr = tk.Scrollbar(outer, bg=PANEL, troughcolor=BG,
                           relief="flat", width=8)
        scr.pack(side="right", fill="y")
        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0,
                           yscrollcommand=scr.set)
        canvas.pack(fill="both", expand=True)
        scr.config(command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: (
            canvas.configure(scrollregion=canvas.bbox("all")),
            canvas.itemconfig(win, width=canvas.winfo_width())))
        canvas.bind("<Configure>", lambda e:
            canvas.itemconfig(win, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        # ── File ──────────────────────────────────────────
        sec0 = Section(inner, "📁  File")
        sec0.pack(fill="x", pady=1)
        r = _row(sec0.body); r.pack(fill="x", pady=2)
        _btn(r, "Open Image",  self._open_image ).pack(side="left", padx=2)
        _btn(r, "Open 2nd",   self._open_second ).pack(side="left", padx=2)
        r2 = _row(sec0.body); r2.pack(fill="x", pady=2)
        _btn(r2, "Reset",     self._reset,  color="#555" ).pack(side="left", padx=2)
        _btn(r2, "Save",      self._save,   color="#2a7a4f").pack(side="left", padx=2)
        _btn(r2, "Histogram", self._show_histogram, color="#444").pack(side="left", padx=2)
        sec0._toggle()

        # ── Point Operations ──────────────────────────────
        sec1 = Section(inner, "🔢  Point Operations")
        sec1.pack(fill="x", pady=1)
        b = sec1.body

        # brightness
        r = _row(b); r.pack(fill="x", pady=3)
        _lbl(r, "Brightness Δ").pack(side="left", padx=4)
        self.e_bright = _entry(r, 50); self.e_bright.pack(side="left")
        _btn(r, "Apply", lambda: self._point("add_brightness",
             value=self.e_bright.get())).pack(side="left", padx=4)

        # subtract brightness
        r = _row(b); r.pack(fill="x", pady=3)
        _lbl(r, "Sub. Bright Δ").pack(side="left", padx=4)
        self.e_sbright = _entry(r, 30); self.e_sbright.pack(side="left")
        _btn(r, "Apply", lambda: self._point("sub_brightness",
             value=self.e_sbright.get())).pack(side="left", padx=4)

        # add images
        r = _row(b); r.pack(fill="x", pady=3)
        _lbl(r, "Add imgs α").pack(side="left", padx=4)
        self.e_alpha = _entry(r, 0.5); self.e_alpha.pack(side="left")
        _btn(r, "Apply", lambda: self._point("add",
             value=self.e_alpha.get())).pack(side="left", padx=4)

        # subtract images
        r = _row(b); r.pack(fill="x", pady=3)
        _btn(r, "Subtract Images",
             lambda: self._point("subtract")).pack(padx=4)

        # divide images
        r = _row(b); r.pack(fill="x", pady=3)
        _btn(r, "Divide Images",
             lambda: self._point("divide")).pack(padx=4)

        # divide scalar
        r = _row(b); r.pack(fill="x", pady=3)
        _lbl(r, "Divide ÷").pack(side="left", padx=4)
        self.e_div = _entry(r, 2.0); self.e_div.pack(side="left")
        _btn(r, "Apply", lambda: self._point("divide_scalar",
             value=self.e_div.get())).pack(side="left", padx=4)

        # complement
        r = _row(b); r.pack(fill="x", pady=3)
        _btn(r, "Complement (Negative)",
             lambda: self._point("complement")).pack(padx=4)

        # ── Histogram ────────────────────────────────────
        sec2 = Section(inner, "📊  Histogram")
        sec2.pack(fill="x", pady=1)
        b2 = sec2.body
        r = _row(b2); r.pack(fill="x", pady=3)
        _btn(r, "Equalize",  self._hist_eq ).pack(side="left", padx=4)
        _btn(r, "Stretching", self._hist_str).pack(side="left", padx=4)

        # ── Color Operations ──────────────────────────────
        sec3 = Section(inner, "🎨  Color Operations")
        sec3.pack(fill="x", pady=1)
        b3 = sec3.body
        # channel names
        ch_names = ["Blue (0)", "Green (1)", "Red (2)"]

        r = _row(b3); r.pack(fill="x", pady=3)
        _lbl(r, "Channel").pack(side="left", padx=4)
        self.cb_color_ch = _combo(r, ch_names)
        self.cb_color_ch.pack(side="left")
        _lbl(r, "Δ").pack(side="left", padx=4)
        self.e_color_c = _entry(r, 50, width=5)
        self.e_color_c.pack(side="left")
        _btn(r, "Light/Color", self._color_light).pack(side="left", padx=4)

        r = _row(b3); r.pack(fill="x", pady=3)
        _lbl(r, "Ch1").pack(side="left", padx=4)
        self.cb_swap1 = _combo(r, ch_names, width=9); self.cb_swap1.pack(side="left")
        _lbl(r, "Ch2").pack(side="left", padx=4)
        self.cb_swap2 = _combo(r, ch_names, default=1, width=9)
        self.cb_swap2.pack(side="left")
        _btn(r, "Swap", self._color_swap).pack(side="left", padx=4)

        r = _row(b3); r.pack(fill="x", pady=3)
        _lbl(r, "Eliminate Ch").pack(side="left", padx=4)
        self.cb_elim = _combo(r, ch_names); self.cb_elim.pack(side="left")
        _btn(r, "Eliminate", self._color_elim).pack(side="left", padx=4)

        # ── Filters ──────────────────────────────────────
        sec4 = Section(inner, "🌀  Filters")
        sec4.pack(fill="x", pady=1)
        b4 = sec4.body
        r = _row(b4); r.pack(fill="x", pady=3)
        _lbl(r, "Kernel").pack(side="left", padx=4)
        self.e_ker = _entry(r, 3, width=4); self.e_ker.pack(side="left")
        filter_ops = ["Average", "Gaussian", "Median", "Maximum",
                      "Minimum", "Mode", "Laplacian"]
        self.cb_filter = _combo(b4, filter_ops, width=20)
        self.cb_filter.pack(pady=3, padx=4)
        _btn(b4, "Apply Filter", self._apply_filter).pack(pady=3, padx=4)

        # ── Edge Detection ────────────────────────────────
        sec5 = Section(inner, "🔍  Edge Detection")
        sec5.pack(fill="x", pady=1)
        b5 = sec5.body
        edge_ops = ["Sobel (Both)", "Sobel X", "Sobel Y",
                    "Prewitt (Both)", "Prewitt X", "Prewitt Y",
                    "Roberts", "Canny", "LoG"]
        self.cb_edge = _combo(b5, edge_ops, width=20)
        self.cb_edge.pack(pady=3, padx=4)

        r = _row(b5); r.pack(fill="x", pady=2)
        _lbl(r, "Canny T1").pack(side="left", padx=4)
        self.e_canny1 = _entry(r, 100, width=5); self.e_canny1.pack(side="left")
        _lbl(r, "T2").pack(side="left", padx=4)
        self.e_canny2 = _entry(r, 200, width=5); self.e_canny2.pack(side="left")

        _btn(b5, "Detect Edges", self._apply_edge).pack(pady=3, padx=4)

        # ── Segmentation ──────────────────────────────────
        sec6 = Section(inner, "✂️  Segmentation")
        sec6.pack(fill="x", pady=1)
        b6 = sec6.body
        seg_ops = ["Global Threshold", "Otsu", "Adaptive (Mean)",
                   "Adaptive (Gaussian)", "K-Means", "Watershed"]
        self.cb_seg = _combo(b6, seg_ops, width=22)
        self.cb_seg.pack(pady=3, padx=4)

        r = _row(b6); r.pack(fill="x", pady=2)
        _lbl(r, "Threshold").pack(side="left", padx=4)
        self.e_thresh = _entry(r, 127, width=5); self.e_thresh.pack(side="left")
        _lbl(r, "K").pack(side="left", padx=4)
        self.e_k = _entry(r, 3, width=3); self.e_k.pack(side="left")

        _btn(b6, "Segment", self._apply_seg).pack(pady=3, padx=4)

        # ── Morphology ────────────────────────────────────
        sec7 = Section(inner, "🔬  Morphology")
        sec7.pack(fill="x", pady=1)
        b7 = sec7.body
        morph_ops = ["Erosion", "Dilation", "Opening", "Closing",
                     "Gradient", "Top Hat", "Black Hat", "Skeletonize"]
        self.cb_morph = _combo(b7, morph_ops, width=20)
        self.cb_morph.pack(pady=3, padx=4)

        r = _row(b7); r.pack(fill="x", pady=2)
        _lbl(r, "Kernel").pack(side="left", padx=4)
        self.e_morph_k = _entry(r, 5, width=4); self.e_morph_k.pack(side="left")
        shapes = ["rect", "ellipse", "cross"]
        self.cb_morph_shape = _combo(r, shapes, width=8)
        self.cb_morph_shape.pack(side="left", padx=4)
        _btn(b7, "Apply", self._apply_morph).pack(pady=3, padx=4)

        # ── Restoration ───────────────────────────────────
        sec8 = Section(inner, "🛠️  Restoration")
        sec8.pack(fill="x", pady=1)
        b8 = sec8.body

        _lbl(b8, "— Add Noise —", color=ACCENT).pack(pady=(4, 0))
        r = _row(b8); r.pack(fill="x", pady=3)
        noise_types = ["Gaussian", "Salt & Pepper", "Speckle"]
        self.cb_noise = _combo(r, noise_types, width=14)
        self.cb_noise.pack(side="left", padx=4)
        _btn(r, "Add Noise", self._add_noise).pack(side="left", padx=2)

        _lbl(b8, "— Remove Noise —", color=ACCENT).pack(pady=(4, 0))
        restore_ops = ["Mean", "Median", "Gaussian", "Bilateral",
                       "Wiener", "Non-Local Means", "Unsharp Mask"]
        self.cb_restore = _combo(b8, restore_ops, width=22)
        self.cb_restore.pack(pady=3, padx=4)
        _btn(b8, "Apply Restoration", self._apply_restore).pack(pady=3, padx=4)

    # ── Canvas area ──────────────────────────────────────────
    def _build_canvas_area(self, parent):
        # Status bar
        self.status_var = tk.StringVar(value="Open an image to begin.")
        sb = tk.Label(parent, textvariable=self.status_var,
                      bg=BORDER, fg=SUBTEXT,
                      font=("Consolas", 9), anchor="w", padx=10)
        sb.pack(fill="x", side="bottom", ipady=4)

        # Images row
        imgs = tk.Frame(parent, bg=BG)
        imgs.pack(fill="both", expand=True, padx=10, pady=10)

        lf = tk.LabelFrame(imgs, text=" Original ", bg=BG, fg=SUBTEXT,
                           font=("Segoe UI", 9), bd=1,
                           highlightbackground=BORDER)
        lf.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.lbl_orig = tk.Label(lf, bg=BG, fg=SUBTEXT,
                                 text="No image loaded")
        self.lbl_orig.pack(fill="both", expand=True)

        rf = tk.LabelFrame(imgs, text=" Result ", bg=BG, fg=ACCENT,
                           font=("Segoe UI", 9, "bold"), bd=1,
                           highlightbackground=ACCENT)
        rf.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.lbl_result = tk.Label(rf, bg=BG, fg=SUBTEXT,
                                   text="Result will appear here")
        self.lbl_result.pack(fill="both", expand=True)

    # ── File ops ─────────────────────────────────────────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        self.original_img = img
        self.current_img  = img.copy()
        self._refresh_display()
        self._status(f"Loaded: {path.split('/')[-1]}  |  {img.shape[1]}×{img.shape[0]}")

    def _open_second(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read second image.")
            return
        self.second_img = img
        self._status(f"Second image loaded: {path.split('/')[-1]}")

    def _reset(self):
        if self.original_img is not None:
            self.current_img = self.original_img.copy()
            self._refresh_display()
            self._status("Reset to original.")

    def _save(self):
        if self.current_img is None:
            messagebox.showwarning("No image", "Nothing to save."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if path:
            cv2.imwrite(path, self.current_img)
            self._status(f"Saved to {path}")

    def _show_histogram(self):
        if self.current_img is None:
            return
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="#111")
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Histogram", color=TEXT, fontsize=12)
        ax.tick_params(colors=SUBTEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        if len(self.current_img.shape) == 3:
            colors = [("#4fc3f7", 0), ("#81c784", 1), ("#e57373", 2)]
            for col, ch in colors:
                hist = cv2.calcHist([self.current_img], [ch], None, [256], [0, 256])
                ax.plot(hist, color=col, linewidth=1.2)
            ax.legend(["Blue", "Green", "Red"], facecolor="#222", labelcolor=TEXT)
        else:
            hist = cv2.calcHist([self.current_img], [0], None, [256], [0, 256])
            ax.fill_between(range(256), hist.flatten(), color=ACCENT, alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ── Display helpers ───────────────────────────────────────
    def _refresh_display(self):
        tk_orig = _np_to_tk(self.original_img)
        tk_curr = _np_to_tk(self.current_img)
        self._tk_orig = tk_orig
        self._tk_curr = tk_curr
        if tk_orig:
            self.lbl_orig.config(image=tk_orig, text="")
        if tk_curr:
            self.lbl_result.config(image=tk_curr, text="")

    def _update_result(self, result):
        if result is None:
            return
        self.current_img = result
        tk_curr = _np_to_tk(result)
        self._tk_curr = tk_curr
        self.lbl_result.config(image=tk_curr, text="")

    def _status(self, msg):
        self.status_var.set(msg)

    def _guard(self):
        if self.current_img is None:
            messagebox.showwarning("No image", "Please open an image first.")
            return False
        return True

    # ── Point Operations ──────────────────────────────────────
    def _point(self, op, value=None):
        if not self._guard(): return
        try:
            result = apply_point_operation(
                self.current_img, op,
                value=float(value) if value else None,
                img2=self.second_img,
                show=False)
            self._update_result(result)
            self._status(f"Point op: {op}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # ── Histogram ────────────────────────────────────────────
    def _hist_eq(self):
        if not self._guard(): return
        gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        result = equalize(gray)
        self._update_result(result)
        self._status("Histogram Equalization applied.")

    def _hist_str(self):
        if not self._guard(): return
        gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        result = streching(gray)
        self._update_result(result)
        self._status("Histogram Stretching applied.")

    # ── Color Operations ──────────────────────────────────────
    def _color_light(self):
        if not self._guard(): return
        ch = self.cb_color_ch.current()
        C  = int(self.e_color_c.get())
        result = changing_the_image_lighting_color(self.current_img, ch, C)
        self._update_result(result)
        self._status(f"Channel {ch} adjusted by {C}")

    def _color_swap(self):
        if not self._guard(): return
        ch1, ch2 = self.cb_swap1.current(), self.cb_swap2.current()
        result = swapping_image_channels(self.current_img, ch1, ch2)
        self._update_result(result)
        self._status(f"Swapped channels {ch1} ↔ {ch2}")

    def _color_elim(self):
        if not self._guard(): return
        ch = self.cb_elim.current()
        result = eliminating_color_channels(self.current_img, ch)
        self._update_result(result)
        self._status(f"Eliminated channel {ch}")

    # ── Filters ──────────────────────────────────────────────
    def _apply_filter(self):
        if not self._guard(): return
        k   = int(self.e_ker.get())
        sel = self.cb_filter.get()
        fns = {
            "Average":   lambda i: average_filter(i, k),
            "Gaussian":  lambda i: gaussian_filter(i, k),
            "Median":    lambda i: median_filter(i, k),
            "Maximum":   lambda i: maximum_filter_op(i, k),
            "Minimum":   lambda i: minimum_filter_op(i, k),
            "Mode":      lambda i: mode_filter(i, k),
            "Laplacian": lambda i: laplacian_filter(i, k if k % 2 == 1 and k >= 1 else 3),
        }
        try:
            result = fns[sel](self.current_img)
            self._update_result(result)
            self._status(f"Filter: {sel}  kernel={k}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # ── Edge Detection ────────────────────────────────────────
    def _apply_edge(self):
        if not self._guard(): return
        sel = self.cb_edge.get()
        t1  = int(self.e_canny1.get())
        t2  = int(self.e_canny2.get())
        ops = {
            "Sobel (Both)":    lambda i: sobel_edge(i, "both"),
            "Sobel X":         lambda i: sobel_edge(i, "x"),
            "Sobel Y":         lambda i: sobel_edge(i, "y"),
            "Prewitt (Both)":  lambda i: prewitt_edge(i, "both"),
            "Prewitt X":       lambda i: prewitt_edge(i, "x"),
            "Prewitt Y":       lambda i: prewitt_edge(i, "y"),
            "Roberts":         lambda i: roberts_edge(i),
            "Canny":           lambda i: canny_edge(i, t1, t2),
            "LoG":             lambda i: laplacian_of_gaussian(i),
        }
        try:
            result = ops[sel](self.current_img)
            self._update_result(result)
            self._status(f"Edge: {sel}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # ── Segmentation ─────────────────────────────────────────
    def _apply_seg(self):
        if not self._guard(): return
        sel = self.cb_seg.get()
        th  = int(self.e_thresh.get())
        k   = int(self.e_k.get())
        ops = {
            "Global Threshold":    lambda i: global_threshold(i, th),
            "Otsu":                lambda i: otsu_threshold(i),
            "Adaptive (Mean)":     lambda i: adaptive_threshold(i, method="mean"),
            "Adaptive (Gaussian)": lambda i: adaptive_threshold(i, method="gaussian"),
            "K-Means":             lambda i: kmeans_segmentation(i, k),
            "Watershed":           lambda i: watershed_segmentation(i),
        }
        try:
            result = ops[sel](self.current_img)
            self._update_result(result)
            self._status(f"Segmentation: {sel}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # ── Morphology ────────────────────────────────────────────
    def _apply_morph(self):
        if not self._guard(): return
        sel   = self.cb_morph.get()
        k     = int(self.e_morph_k.get())
        shape = self.cb_morph_shape.get()
        ops = {
            "Erosion":    lambda i: erosion(i, k, shape),
            "Dilation":   lambda i: dilation(i, k, shape),
            "Opening":    lambda i: opening(i, k, shape),
            "Closing":    lambda i: closing(i, k, shape),
            "Gradient":   lambda i: morphological_gradient(i, k, shape),
            "Top Hat":    lambda i: top_hat(i, k, shape),
            "Black Hat":  lambda i: black_hat(i, k, shape),
            "Skeletonize": lambda i: skeletonize(i),
        }
        try:
            result = ops[sel](self.current_img)
            self._update_result(result)
            self._status(f"Morphology: {sel}  k={k}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # ── Restoration ───────────────────────────────────────────
    def _add_noise(self):
        if not self._guard(): return
        sel = self.cb_noise.get()
        ops = {
            "Gaussian":        lambda i: add_gaussian_noise(i),
            "Salt & Pepper":   lambda i: add_salt_and_pepper_noise(i),
            "Speckle":         lambda i: add_speckle_noise(i),
        }
        result = ops[sel](self.current_img)
        self._update_result(result)
        self._status(f"Noise added: {sel}")

    def _apply_restore(self):
        if not self._guard(): return
        sel = self.cb_restore.get()
        ops = {
            "Mean":            lambda i: mean_filter(i),
            "Median":          lambda i: med_filter(i),
            "Gaussian":        lambda i: gauss_filter(i),
            "Bilateral":       lambda i: bilateral_filter(i),
            "Wiener":          lambda i: wiener_filter(i),
            "Non-Local Means": lambda i: non_local_means_filter(i),
            "Unsharp Mask":    lambda i: unsharp_masking(i),
        }
        try:
            result = ops[sel](self.current_img)
            self._update_result(result)
            self._status(f"Restoration: {sel}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = PixelForge()
    app.mainloop()
