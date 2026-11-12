# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : draw_convex_normalize.py
# Purpose       :
#   Viewer that normalises sagittal left/right orientation before computing the metrics.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class InteractiveDirectionNormalizedViewer:
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

        # Setup 3-Panel GUI: Left (Original) | Center (Direction-Normalized) | Right (Info & Metrics)
        self.fig, (self.ax_orig, self.ax_norm, self.ax_text) = plt.subplots(
            1, 3, figsize=(18, 7.5), gridspec_kw={'width_ratios': [1.3, 1.3, 0.9]}
        )
        plt.subplots_adjust(bottom=0.14, left=0.04, right=0.96, top=0.92, wspace=0.14)

        self.ax_text.axis('off')

        # Navigation Buttons
        ax_prev = plt.axes([0.30, 0.03, 0.12, 0.05])
        ax_next = plt.axes([0.58, 0.03, 0.12, 0.05])

        self.btn_prev = Button(ax_prev, '◄ Previous')
        self.btn_next = Button(ax_next, 'Next ►')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_plot()
        plt.show()

    def parse_view_plane(self, filename):
        """
        Parses acquisition plane from filename tokens:
        - Coronal: 'co', 'cor', 'coronal'
        - Sagittal: 'sa', 'sag', 'sagittal'
        - Axial: 'ax', 'tra', 'axial'
        """
        fn_lower = os.path.splitext(filename)[0].lower()
        tokens = [t for t in re.split(r'[_.\-\s]+', fn_lower) if t]

        if any(t in ["co", "cor", "coronal"] for t in tokens) or "_co_" in fn_lower or "coronal" in fn_lower:
            return "Coronal"
        elif any(t in ["sa", "sag", "sagittal"] for t in tokens) or "_sa_" in fn_lower or "sagittal" in fn_lower:
            return "Sagittal"
        elif any(t in ["ax", "tra", "axial"] for t in tokens) or "_ax_" in fn_lower or "axial" in fn_lower:
            return "Axial"

        # Direct token check fallback
        if "co" in tokens:
            return "Coronal"
        if "sa" in tokens:
            return "Sagittal"
        if "ax" in tokens:
            return "Axial"

        return "Unknown"

    def find_mask_path(self, base_name_no_ext):
        """Locates the segmentation mask matching the scan filename."""
        search_pattern = os.path.join(self.seg_dir, "**", f"{base_name_no_ext}.*")
        matches = [f for f in glob.glob(search_pattern, recursive=True) if f.lower().endswith(self.valid_extensions)]

        if not matches:
            search_pattern_alt = os.path.join(self.seg_dir, "**", f"*{base_name_no_ext}*")
            matches = [f for f in glob.glob(search_pattern_alt, recursive=True) if f.lower().endswith(self.valid_extensions)]

        for match in matches:
            if "mask" in match.lower() or len(matches) == 1 or "mask" in self.seg_dir.lower():
                return match
        return matches[0] if matches else None

    def segment_skull(self, img_gray):
        """Extracts brain/skull outer contour and convex hull."""
        _, thresh = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_cnt)
        return largest_cnt, hull

    def normalize_sagittal_direction(self, img, mask, skull_cnt):
        """
        Unifies Sagittal scans so Anterior (Face) points Left and Posterior (Occiput) points Right.
        Preserves original scale, resolution, and aspect ratio.
        """
        bx, by, bw, bh = cv2.boundingRect(skull_cnt)
        lower_bound_y = by + int(bh * 0.45)

        # Build lower-half cranial/facial mask to detect facial protrusion asymmetry
        mask_lower = np.zeros_like(img)
        cv2.drawContours(mask_lower, [skull_cnt], -1, 255, -1)
        mask_lower[:lower_bound_y, :] = 0

        M_lower = cv2.moments(mask_lower)
        if M_lower["m00"] > 0:
            cx_lower = M_lower["m10"] / M_lower["m00"]
            cx_midpoint = bx + (bw / 2.0)

            # If lower facial mass center is on the right side of skull bbox -> Flip to Face Left
            if cx_lower > cx_midpoint:
                img_flipped = cv2.flip(img, 1)
                mask_flipped = cv2.flip(mask, 1) if mask is not None else None
                return img_flipped, mask_flipped, True

        return img, mask, False

    def render_overlays(self, img_gray, mask_gray):
        """Overlays brain boundary, skull hull, tumor boundary, tumor hull, and centroids."""
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        skull_cnt, skull_hull = self.segment_skull(img_gray)

        b_centroid = None
        bx, by, bw, bh = 0, 0, img_gray.shape[1], img_gray.shape[0]

        if skull_cnt is not None:
            bx, by, bw, bh = cv2.boundingRect(skull_hull)
            M_b = cv2.moments(skull_hull)
            if M_b["m00"] != 0:
                b_centroid = (int(M_b["m10"] / M_b["m00"]), int(M_b["m01"] / M_b["m00"]))
                cv2.drawMarker(img_rgb, b_centroid, (255, 0, 0), cv2.MARKER_CROSS, 16, 2)

            cv2.drawContours(img_rgb, [skull_cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(img_rgb, [skull_hull], -1, (255, 0, 0), 2)

        tumor_stats = {
            "centroid": None,
            "area": 0.0,
            "hull_area": 0.0,
            "dural_dist": 0.0
        }

        if mask_gray is not None:
            _, t_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
            t_cnts, _ = cv2.findContours(t_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if t_cnts:
                overlay = img_rgb.copy()
                cv2.drawContours(overlay, t_cnts, -1, (255, 255, 0), -1)
                cv2.addWeighted(overlay, 0.35, img_rgb, 0.65, 0, img_rgb)
                cv2.drawContours(img_rgb, t_cnts, -1, (255, 255, 0), 2)

                all_t_pts = np.vstack(t_cnts)
                t_hull = cv2.convexHull(all_t_pts)
                cv2.drawContours(img_rgb, [t_hull], -1, (0, 255, 255), 2)

                tumor_stats["area"] = float(sum(cv2.contourArea(c) for c in t_cnts))
                tumor_stats["hull_area"] = float(cv2.contourArea(t_hull))

                M_t = cv2.moments(all_t_pts)
                if M_t["m00"] != 0:
                    t_cx = int(M_t["m10"] / M_t["m00"])
                    t_cy = int(M_t["m01"] / M_t["m00"])
                    tumor_stats["centroid"] = (t_cx, t_cy)
                    cv2.drawMarker(img_rgb, (t_cx, t_cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

                    if skull_cnt is not None:
                        dist = cv2.pointPolygonTest(skull_cnt, (float(t_cx), float(t_cy)), True)
                        tumor_stats["dural_dist"] = abs(dist)

        return img_rgb, skull_cnt, skull_hull, b_centroid, (bx, by, bw, bh), tumor_stats

    def compute_metrics(self, plane, bbox, b_centroid, t_stats, skull_cnt, skull_hull, flipped):
        bx, by, bw, bh = bbox
        metrics = {
            "plane": plane,
            "sagittal_flipped": flipped,
            "laterality_L": None,
            "cranio_caudal_SI": None,
            "antero_posterior_AP": None,
            "dural_dist_px": t_stats["dural_dist"],
            "area_ratio": 0.0,
            "solidity": 0.0
        }

        brain_area = float(cv2.contourArea(skull_hull)) if skull_hull is not None else 1.0
        if brain_area > 0 and t_stats["area"] > 0:
            metrics["area_ratio"] = t_stats["area"] / brain_area
        if t_stats["hull_area"] > 0:
            metrics["solidity"] = t_stats["area"] / t_stats["hull_area"]

        if t_stats["centroid"] is not None and bw > 0 and bh > 0:
            tcx, tcy = t_stats["centroid"]

            # Laterality L in [-1.0 (Left), +1.0 (Right)] for Axial and Coronal
            if b_centroid is not None and plane in ["Axial", "Coronal"]:
                metrics["laterality_L"] = (tcx - b_centroid[0]) / (bw / 2.0)

            # Cranio-Caudal SI in [0.0 (Base/Inferior), 1.0 (Vertex/Superior)] for Coronal and Sagittal
            if plane in ["Coronal", "Sagittal"]:
                metrics["cranio_caudal_SI"] = (by + bh - tcy) / float(bh)

            # Antero-Posterior AP in [0.0 (Anterior/Front), 1.0 (Posterior/Back)]
            if plane == "Sagittal":
                # With normalized direction (Anterior at left side bx)
                metrics["antero_posterior_AP"] = (tcx - bx) / float(bw)
            elif plane == "Axial":
                # In Axial MRI: top is Anterior, bottom is Posterior
                metrics["antero_posterior_AP"] = (tcy - by) / float(bh)

        return metrics

    def update_plot(self):
        img_name = self.image_files[self.current_idx]
        img_path = os.path.join(self.folder_path, img_name)
        plane = self.parse_view_plane(img_name)

        img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            return

        base_name = os.path.splitext(img_name)[0]
        mask_path = self.find_mask_path(base_name)
        mask_raw = None
        if mask_path and os.path.exists(mask_path):
            loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is not None:
                mask_raw = cv2.resize(loaded_mask, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 1. Left Subplot: Raw Original Scan
        orig_rgb, raw_cnt, _, _, _, _ = self.render_overlays(img_raw, mask_raw)

        # 2. Middle Subplot: Direction-Normalized (Native dimensions maintained)
        norm_img = img_raw.copy()
        norm_mask = mask_raw.copy() if mask_raw is not None else None
        flipped = False

        if plane == "Sagittal" and raw_cnt is not None:
            norm_img, norm_mask, flipped = self.normalize_sagittal_direction(norm_img, norm_mask, raw_cnt)

        norm_rgb, norm_cnt, norm_hull, norm_b_centroid, norm_bbox, norm_t_stats = self.render_overlays(norm_img, norm_mask)
        metrics = self.compute_metrics(plane, norm_bbox, norm_b_centroid, norm_t_stats, norm_cnt, norm_hull, flipped)

        # Render Left Axes (Raw)
        self.ax_orig.clear()
        self.ax_orig.imshow(orig_rgb)
        self.ax_orig.set_title(
            f"[Original Acquisition]\n{img_name}\nPlane: {plane} | Dim: {img_raw.shape[1]}x{img_raw.shape[0]}",
            fontsize=9.5, fontweight='bold'
        )
        self.ax_orig.axis('off')

        # Render Center Axes (Normalized)
        self.ax_norm.clear()
        self.ax_norm.imshow(norm_rgb)
        dir_status = "Flipped (Face Normalized to Left)" if flipped else "Original Orientation (Face Left / Co / Ax)"
        self.ax_norm.set_title(
            f"[Direction-Normalized]\n{dir_status}\nNative Scale: {norm_img.shape[1]}x{norm_img.shape[0]}",
            fontsize=9.5, fontweight='bold'
        )
        self.ax_norm.axis('off')

        # Render Right Axes (Info Card)
        self.ax_text.clear()
        self.ax_text.axis('off')

        lat_val = f"{metrics['laterality_L']:+.3f} [-1:L, +1:R]" if metrics['laterality_L'] is not None else "N/A (Sagittal)"
        si_val  = f"{metrics['cranio_caudal_SI']:.3f} [0:Base, 1:Vertex]" if metrics['cranio_caudal_SI'] is not None else "N/A (Axial)"
        ap_val  = f"{metrics['antero_posterior_AP']:.3f} [0:Ant, 1:Post]" if metrics['antero_posterior_AP'] is not None else "N/A (Coronal)"

        panel_text = (
            f"=== SCAN IDENTIFIERS ===\n"
            f"Index:       [{self.current_idx + 1}/{self.total_images}]\n"
            f"Class:       {self.target_class.upper()}\n"
            f"Plane Code:  {plane.upper()} (co / sa / ax)\n"
            f"File:        {img_name}\n\n"
            f"=== DIRECTION STATUS ===\n"
            f"Sagittal Flip: {'YES (Face -> Left)' if flipped else 'NO (Aligned)'}\n"
            f"Native Scale:  {norm_img.shape[1]} x {norm_img.shape[0]} px\n\n"
            f"=== NORMALIZED METRICS ===\n"
            f"Laterality (L):    {lat_val}\n"
            f"Cranio-Caudal (SI):{si_val}\n"
            f"Antero-Post (AP):  {ap_val}\n"
            f"Dist to Skull:     {metrics['dural_dist_px']:.1f} px\n\n"
            f"=== MORPHOMETRICS ===\n"
            f"Tumor/Brain Ratio: {metrics['area_ratio'] * 100:.2f}%\n"
            f"Tumor Solidity:    {metrics['solidity']:.3f}\n\n"
            f"=== COLOR LEGEND ===\n"
            f"• Green:  Skull Outer Boundary\n"
            f"• Red:    Brain Convex Hull & Centroid (+)\n"
            f"• Yellow: Tumor Boundary & Segmentation\n"
            f"• Cyan:   Tumor Convex Hull & Centroid (+)"
        )

        self.ax_text.text(
            0.02, 0.98, panel_text,
            transform=self.ax_text.transAxes,
            fontsize=9.5,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f7f7f7', alpha=0.95, edgecolor='#bbb')
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

    viewer = InteractiveDirectionNormalizedViewer(
        base_dir=BASE_CLASS_DIR,
        seg_dir=SEG_TASK_DIR,
        target_class=TARGET_CLASS
    )