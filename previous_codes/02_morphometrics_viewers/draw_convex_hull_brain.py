# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : draw_convex_hull_brain.py
# Purpose       :
#   Interactive viewer: brain contour + convex hull on each scan (first prototype of the geometry pipeline).
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class InteractiveHullViewer:
    def __init__(self, base_dir, target_class="glioma"):
        self.folder_path = os.path.join(base_dir, target_class)
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder path does not exist: {self.folder_path}")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_extensions)]

        if not self.image_files:
            raise ValueError(f"No valid images found in {self.folder_path}")

        self.current_idx = 0
        self.total_images = len(self.image_files)

        # Setup GUI Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.15)  # Leave room at the bottom for buttons

        # Button Axes [left, bottom, width, height]
        ax_prev = plt.axes([0.25, 0.03, 0.2, 0.06])
        ax_next = plt.axes([0.55, 0.03, 0.2, 0.06])

        self.btn_prev = Button(ax_prev, '◄ Previous')
        self.btn_next = Button(ax_next, 'Next ►')

        # Connect click events
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

        # Connect keyboard arrow shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.im_display = None
        self.update_plot()
        plt.show()

    def process_hull(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # 1. Background Thresholding
        _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)

        # 2. Morphological Cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_OPEN, kernel)

        # 3. Contour & Hull Extraction
        contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)

            # Green: Outer Skull Contour | Red: Convex Hull
            cv2.drawContours(img_rgb, [largest_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(img_rgb, [hull], -1, (255, 0, 0), 2)

        return img_rgb

    def update_plot(self):
        img_name = self.image_files[self.current_idx]
        full_path = os.path.join(self.folder_path, img_name)

        img_rgb = self.process_hull(full_path)

        if img_rgb is None:
            print(f"[Error] Failed to load {img_name}")
            return

        if self.im_display is None:
            self.im_display = self.ax.imshow(img_rgb)
        else:
            self.im_display.set_data(img_rgb)

        self.ax.set_title(
            f"Image [{self.current_idx + 1}/{self.total_images}]: {img_name}\n"
            f"Green: Skull Contour | Red: Convex Hull",
            fontsize=11
        )
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def next_image(self, event=None):
        self.current_idx = (self.current_idx + 1) % self.total_images
        self.update_plot()

    def prev_image(self, event=None):
        self.current_idx = (self.current_idx - 1) % self.total_images
        self.update_plot()

    def on_key(self, event):
        if event.key in ['right', 'down', ' ']:
            self.next_image()
        elif event.key in ['left', 'up']:
            self.prev_image()


if __name__ == "__main__":
    BASE_DIR = r"D:\Datasets\Medical\brisc2025\classification_task\train"
    TARGET_CLASS = "glioma"  # Options: 'glioma', 'meningioma', 'pituitary', 'no_tumor'

    # Launch interactive GUI viewer (Use 'Next'/'Previous' buttons or Left/Right arrow keys)
    viewer = InteractiveHullViewer(BASE_DIR, target_class=TARGET_CLASS)