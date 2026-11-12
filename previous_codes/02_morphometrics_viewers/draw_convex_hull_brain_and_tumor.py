# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : draw_convex_hull_brain_and_tumor.py
# Purpose       :
#   Interactive viewer: brain hull and tumor contour/hull overlaid on each scan.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class InteractiveHullAndTumorViewer:
    def __init__(self, base_dir, seg_dir, target_class="glioma"):
        self.folder_path = os.path.join(base_dir, target_class)
        self.seg_dir = seg_dir
        self.target_class = target_class

        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Classification folder does not exist: {self.folder_path}")

        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(self.valid_extensions)]

        if not self.image_files:
            raise ValueError(f"No valid images found in {self.folder_path}")

        self.current_idx = 0
        self.total_images = len(self.image_files)

        # Setup GUI Figure
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        plt.subplots_adjust(bottom=0.15)  # Leave room at bottom for UI buttons

        # Button Axes [left, bottom, width, height]
        ax_prev = plt.axes([0.25, 0.03, 0.2, 0.06])
        ax_next = plt.axes([0.55, 0.03, 0.2, 0.06])

        self.btn_prev = Button(ax_prev, '◄ Previous')
        self.btn_next = Button(ax_next, 'Next ►')

        # Connect GUI events
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.im_display = None
        self.update_plot()
        plt.show()

    def find_mask_path(self, base_name_no_ext):
        """Locates the matching segmentation mask file from the segmentation folder."""
        search_pattern = os.path.join(self.seg_dir, "**", f"{base_name_no_ext}.*")
        mask_matches = [f for f in glob.glob(search_pattern, recursive=True) if
                        f.lower().endswith(self.valid_extensions)]

        if not mask_matches:
            search_pattern_alt = os.path.join(self.seg_dir, "**", f"*{base_name_no_ext}*")
            mask_matches = [f for f in glob.glob(search_pattern_alt, recursive=True) if
                            f.lower().endswith(self.valid_extensions)]

        for match in mask_matches:
            if "mask" in match.lower() or len(mask_matches) == 1 or "mask" in self.seg_dir.lower():
                return match
        return mask_matches[0] if mask_matches else None

    def process_anatomical_and_tumor_hulls(self, image_path, img_name):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # -----------------------------------------------------------------
        # 1. BRAIN / SKULL CONVEX HULL EXTRACTION
        # -----------------------------------------------------------------
        _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_OPEN, kernel)

        brain_contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if brain_contours:
            largest_brain_contour = max(brain_contours, key=cv2.contourArea)
            brain_hull = cv2.convexHull(largest_brain_contour)

            # Draw Brain Boundary (Green) & Brain Hull (Red)
            cv2.drawContours(img_rgb, [largest_brain_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(img_rgb, [brain_hull], -1, (255, 0, 0), 2)

        # -----------------------------------------------------------------
        # 2. TUMOR HIGHLIGHT & TUMOR CONVEX HULL EXTRACTION
        # -----------------------------------------------------------------
        if self.target_class != "no_tumor":
            base_name_no_ext = os.path.splitext(img_name)[0]
            mask_path = self.find_mask_path(base_name_no_ext)

            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Align mask dimensions if needed
                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    tumor_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if tumor_contours:
                        # Semi-transparent overlay to highlight tumor area (Yellow fill)
                        overlay = img_rgb.copy()
                        cv2.drawContours(overlay, tumor_contours, -1, (255, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.35, img_rgb, 0.65, 0, img_rgb)

                        # Draw Tumor Boundary (Yellow outline)
                        cv2.drawContours(img_rgb, tumor_contours, -1, (255, 255, 0), 2)

                        # Compute & Draw Tumor Convex Hull (Cyan)
                        all_tumor_pts = np.vstack(tumor_contours)
                        tumor_hull = cv2.convexHull(all_tumor_pts)
                        cv2.drawContours(img_rgb, [tumor_hull], -1, (0, 255, 255), 2)

        return img_rgb

    def update_plot(self):
        img_name = self.image_files[self.current_idx]
        full_path = os.path.join(self.folder_path, img_name)

        img_rgb = self.process_anatomical_and_tumor_hulls(full_path, img_name)

        if img_rgb is None:
            print(f"[Error] Failed to load or process {img_name}")
            return

        if self.im_display is None:
            self.im_display = self.ax.imshow(img_rgb)
        else:
            self.im_display.set_data(img_rgb)

        self.ax.set_title(
            f"Image [{self.current_idx + 1}/{self.total_images}]: {img_name}\n"
            f"Green: Skull Boundary | Red: Skull Hull\n"
            f"Yellow: Tumor Highlight | Cyan: Tumor Hull",
            fontsize=10
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
    BASE_CLASS_DIR = r"D:\Datasets\Medical\brisc2025\classification_task\train"
    SEG_TASK_DIR = r"D:\Datasets\Medical\brisc2025\segmentation_task\train"

    # Options: 'glioma', 'meningioma', 'pituitary', 'no_tumor'
    TARGET_CLASS = "glioma"

    # Launch GUI (Use 'Next'/'Previous' buttons or Left/Right Arrow keys)
    viewer = InteractiveHullAndTumorViewer(
        base_dir=BASE_CLASS_DIR,
        seg_dir=SEG_TASK_DIR,
        target_class=TARGET_CLASS
    )