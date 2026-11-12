# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : draw_convex_hull_brain_and_tumor_information.py
# Purpose       :
#   Same as above plus on-screen numeric measurements (areas, centroids, distances).
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

        # Setup GUI Figure: Split into Image Display (Left) and Metrics Card (Right)
        self.fig, (self.ax_img, self.ax_text) = plt.subplots(
            1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [2.2, 1]}
        )
        plt.subplots_adjust(bottom=0.15, right=0.95, left=0.05, wspace=0.15)

        self.ax_text.axis('off')

        # Button Axes [left, bottom, width, height]
        ax_prev = plt.axes([0.25, 0.03, 0.15, 0.05])
        ax_next = plt.axes([0.55, 0.03, 0.15, 0.05])

        self.btn_prev = Button(ax_prev, '◄ Previous')
        self.btn_next = Button(ax_next, 'Next ►')

        # Connect GUI events
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.im_display = None
        self.text_display = None

        self.update_plot()
        plt.show()

    def parse_view_plane(self, filename):
        """Extracts acquisition view plane from standard dataset filename conventions."""
        fn_lower = filename.lower()
        if "_ax_" in fn_lower or "_ax" in fn_lower or "axial" in fn_lower:
            return "Axial"
        elif "_cor_" in fn_lower or "_cor" in fn_lower or "coronal" in fn_lower:
            return "Coronal"
        elif "_sag_" in fn_lower or "_sag" in fn_lower or "sagittal" in fn_lower:
            return "Sagittal"
        return "Unknown"

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
            return None, {}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        metrics = {
            "plane": self.parse_view_plane(img_name),
            "brain_area": 0.0,
            "tumor_area": 0.0,
            "area_ratio": 0.0,
            "tumor_solidity": 0.0,
            "dist_to_skull_px": 0.0,
            "rel_centroid_offset": (0.0, 0.0),
            "tumor_found": False
        }

        # -----------------------------------------------------------------
        # 1. BRAIN / SKULL CONVEX HULL EXTRACTION
        # -----------------------------------------------------------------
        _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_OPEN, kernel)

        brain_contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        brain_centroid = None
        brain_w, brain_h = img.shape[1], img.shape[0]

        if brain_contours:
            largest_brain_contour = max(brain_contours, key=cv2.contourArea)
            brain_hull = cv2.convexHull(largest_brain_contour)
            metrics["brain_area"] = float(cv2.contourArea(largest_brain_contour))

            # Brain Bounding Box and Centroid
            bx, by, bw, bh = cv2.boundingRect(brain_hull)
            brain_w, brain_h = bw, bh
            M_b = cv2.moments(brain_hull)
            if M_b["m00"] != 0:
                brain_centroid = (int(M_b["m10"] / M_b["m00"]), int(M_b["m01"] / M_b["m00"]))
                # Red cross for Brain Hull Centroid
                cv2.drawMarker(img_rgb, brain_centroid, (255, 0, 0), cv2.MARKER_CROSS, 15, 2)

            # Draw Brain Boundary (Green) & Brain Hull (Red)
            cv2.drawContours(img_rgb, [largest_brain_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(img_rgb, [brain_hull], -1, (255, 0, 0), 2)

        # -----------------------------------------------------------------
        # 2. TUMOR HIGHLIGHT & METRICS EXTRACTION
        # -----------------------------------------------------------------
        if self.target_class != "no_tumor":
            base_name_no_ext = os.path.splitext(img_name)[0]
            mask_path = self.find_mask_path(base_name_no_ext)

            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    tumor_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if tumor_contours:
                        metrics["tumor_found"] = True

                        # Semi-transparent overlay (Yellow fill)
                        overlay = img_rgb.copy()
                        cv2.drawContours(overlay, tumor_contours, -1, (255, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.35, img_rgb, 0.65, 0, img_rgb)

                        # Draw Tumor Boundary (Yellow outline)
                        cv2.drawContours(img_rgb, tumor_contours, -1, (255, 255, 0), 2)

                        # Compute Tumor Hull & Solidity
                        all_tumor_pts = np.vstack(tumor_contours)
                        tumor_hull = cv2.convexHull(all_tumor_pts)
                        cv2.drawContours(img_rgb, [tumor_hull], -1, (0, 255, 255), 2)

                        tumor_area = float(sum(cv2.contourArea(c) for c in tumor_contours))
                        tumor_hull_area = float(cv2.contourArea(tumor_hull))

                        metrics["tumor_area"] = tumor_area
                        if metrics["brain_area"] > 0:
                            metrics["area_ratio"] = tumor_area / metrics["brain_area"]
                        if tumor_hull_area > 0:
                            metrics["tumor_solidity"] = tumor_area / tumor_hull_area

                        # Tumor Centroid
                        M_t = cv2.moments(all_tumor_pts)
                        if M_t["m00"] != 0:
                            t_cx = int(M_t["m10"] / M_t["m00"])
                            t_cy = int(M_t["m01"] / M_t["m00"])
                            tumor_centroid = (t_cx, t_cy)

                            # Cyan cross for Tumor Centroid
                            cv2.drawMarker(img_rgb, tumor_centroid, (0, 255, 255), cv2.MARKER_CROSS, 15, 2)

                            # Normalized Centroid Offset from Brain Centroid
                            if brain_centroid is not None:
                                dx = (t_cx - brain_centroid[0]) / float(brain_w)
                                dy = (t_cy - brain_centroid[1]) / float(brain_h)
                                metrics["rel_centroid_offset"] = (dx, dy)

                            # Distance to Skull Boundary (negative if inside, positive if outside, 0 on edge)
                            if brain_contours:
                                dist = cv2.pointPolygonTest(largest_brain_contour, (float(t_cx), float(t_cy)), True)
                                metrics["dist_to_skull_px"] = abs(dist)

        return img_rgb, metrics

    def update_plot(self):
        img_name = self.image_files[self.current_idx]
        full_path = os.path.join(self.folder_path, img_name)

        img_rgb, metrics = self.process_anatomical_and_tumor_hulls(full_path, img_name)

        if img_rgb is None:
            print(f"[Error] Failed to load or process {img_name}")
            return

        # 1. Update Image Plot
        if self.im_display is None:
            self.im_display = self.ax_img.imshow(img_rgb)
        else:
            self.im_display.set_data(img_rgb)

        self.ax_img.set_title(
            f"Image [{self.current_idx + 1}/{self.total_images}]: {img_name}",
            fontsize=10
        )
        self.ax_img.axis('off')

        # 2. Update Metrics Side Card
        self.ax_text.clear()
        self.ax_text.axis('off')

        panel_text = (
            f"=== FILE METADATA ===\n"
            f"Class Label: {self.target_class.upper()}\n"
            f"View Plane:  {metrics.get('plane', 'Unknown')}\n\n"
            f"=== SPATIAL METRICS ===\n"
            f"Brain Area:   {int(metrics['brain_area']):,} px²\n"
            f"Tumor Area:   {int(metrics['tumor_area']):,} px²\n"
            f"Area Ratio:   {metrics['area_ratio'] * 100:.2f}%\n"
            f"Dist to Skull:{metrics['dist_to_skull_px']:.1f} px\n\n"
            f"=== MORPHOMETRICS ===\n"
            f"Solidity:     {metrics['tumor_solidity']:.3f}\n"
            f"Norm Offset (Δx, Δy):\n"
            f"  ({metrics['rel_centroid_offset'][0]:+.3f}, {metrics['rel_centroid_offset'][1]:+.3f})\n\n"
            f"=== LEGEND ===\n"
            f"• Green: Skull Boundary\n"
            f"• Red: Brain Hull & Centroid (+)\n"
            f"• Yellow: Tumor Boundary & Fill\n"
            f"• Cyan: Tumor Hull & Centroid (+)"
        )

        self.ax_text.text(
            0.05, 0.95, panel_text,
            transform=self.ax_text.transAxes,
            fontsize=10,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', alpha=0.9, edgecolor='#ccc')
        )

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

    # Launch GUI
    viewer = InteractiveHullAndTumorViewer(
        base_dir=BASE_CLASS_DIR,
        seg_dir=SEG_TASK_DIR,
        target_class=TARGET_CLASS
    )