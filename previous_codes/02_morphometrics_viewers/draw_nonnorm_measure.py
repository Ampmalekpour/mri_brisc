# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : draw_nonnorm_measure.py
# Purpose       :
#   Viewer computing orientation-invariant ('alignment invariant') metrics without flipping.
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


class AlignmentInvariantTumorViewer:
    def __init__(self, base_dir, seg_dir, target_class="glioma"):
        self.folder_path = os.path.join(base_dir, target_class)
        self.seg_dir = seg_dir
        self.target_class = target_class

        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Directory not found: {self.folder_path}")

        self.valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(self.valid_exts)]

        if not self.image_files:
            raise ValueError(f"No valid images found in {self.folder_path}")

        self.current_idx = 0
        self.total_images = len(self.image_files)

        # Setup 2-Panel Display: Native Scan with Visual Overlays (Left) | Interpreted Metrics (Right)
        self.fig, (self.ax_img, self.ax_text) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1.6, 1.0]}
        )
        plt.subplots_adjust(bottom=0.12, left=0.04, right=0.96, top=0.92, wspace=0.10)

        self.ax_text.axis('off')

        # Navigation Buttons
        ax_prev = plt.axes([0.25, 0.03, 0.14, 0.05])
        ax_next = plt.axes([0.55, 0.03, 0.14, 0.05])

        self.btn_prev = Button(ax_prev, '◄ Previous')
        self.btn_next = Button(ax_next, 'Next ►')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_plot()
        plt.show()

    def parse_view_plane(self, filename):
        """Identifies plane from filename tokens without case sensitivity."""
        fn_lower = os.path.splitext(filename)[0].lower()
        tokens = [t for t in re.split(r'[_.\-\s]+', fn_lower) if t]

        if any(t in ["co", "cor", "coronal"] for t in tokens) or "_co_" in fn_lower or "coronal" in fn_lower:
            return "Coronal"
        elif any(t in ["sa", "sag", "sagittal"] for t in tokens) or "_sa_" in fn_lower or "sagittal" in fn_lower:
            return "Sagittal"
        elif any(t in ["ax", "tra", "axial"] for t in tokens) or "_ax_" in fn_lower or "axial" in fn_lower:
            return "Axial"

        if "co" in tokens: return "Coronal"
        if "sa" in tokens: return "Sagittal"
        if "ax" in tokens: return "Axial"
        return "Unknown"

    def find_mask_path(self, base_name):
        search_pattern = os.path.join(self.seg_dir, "**", f"{base_name}.*")
        matches = [f for f in glob.glob(search_pattern, recursive=True) if f.lower().endswith(self.valid_exts)]
        if not matches:
            search_pattern_alt = os.path.join(self.seg_dir, "**", f"*{base_name}*")
            matches = [f for f in glob.glob(search_pattern_alt, recursive=True) if f.lower().endswith(self.valid_exts)]
        for match in matches:
            if "mask" in match.lower() or len(matches) == 1 or "mask" in self.seg_dir.lower():
                return match
        return matches[0] if matches else None

    def segment_brain_skull(self, img_gray):
        _, thresh = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, np.zeros_like(img_gray)

        largest_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_cnt)

        brain_mask = np.zeros_like(img_gray)
        cv2.drawContours(brain_mask, [largest_cnt], -1, 255, -1)
        return largest_cnt, hull, brain_mask

    def compute_invariant_metrics(self, plane, img_shape, brain_cnt, brain_hull, brain_mask, tumor_cnts, tumor_mask):
        h, w = img_shape
        metrics = {
            "plane": plane,
            "has_tumor": False,
            "brain_area_px": 0.0,
            "r_eff_px": 1.0,
            "tumor_area_px": 0.0,
            "area_ratio_pct": 0.0,
            "radial_depth": 0.0,
            "radial_interp": "N/A",
            "dural_dist_px": 0.0,
            "dural_dist_norm": 0.0,
            "dural_interp": "N/A",
            "midline_offset": None,
            "midline_interp": "N/A",
            "elevation_si": None,
            "elevation_interp": "N/A",
            "solidity": 0.0,
            "solidity_interp": "N/A",
            "circularity": 0.0,
            "circularity_interp": "N/A",
            "b_cx": None, "b_cy": None,
            "t_cx": None, "t_cy": None
        }

        if brain_cnt is None:
            return metrics

        # Brain Geometry
        brain_area = float(cv2.contourArea(brain_cnt))
        metrics["brain_area_px"] = brain_area
        r_eff = np.sqrt(brain_area / np.pi) if brain_area > 0 else 1.0
        metrics["r_eff_px"] = r_eff

        bx, by, bw, bh = cv2.boundingRect(brain_hull)
        M_b = cv2.moments(brain_hull)
        if M_b["m00"] != 0:
            b_cx = int(M_b["m10"] / M_b["m00"])
            b_cy = int(M_b["m01"] / M_b["m00"])
            metrics["b_cx"], metrics["b_cy"] = b_cx, b_cy
        else:
            b_cx, b_cy = bx + bw // 2, by + bh // 2

        if not tumor_cnts or tumor_mask is None or np.sum(tumor_mask) == 0:
            return metrics

        metrics["has_tumor"] = True
        all_t_pts = np.vstack(tumor_cnts)
        tumor_hull = cv2.convexHull(all_t_pts)

        t_area = float(sum(cv2.contourArea(c) for c in tumor_cnts))
        t_hull_area = float(cv2.contourArea(tumor_hull))
        t_perimeter = float(sum(cv2.arcLength(c, True) for c in tumor_cnts))

        metrics["tumor_area_px"] = t_area
        metrics["area_ratio_pct"] = (t_area / brain_area) * 100.0 if brain_area > 0 else 0.0

        M_t = cv2.moments(all_t_pts)
        if M_t["m00"] != 0:
            t_cx = int(M_t["m10"] / M_t["m00"])
            t_cy = int(M_t["m01"] / M_t["m00"])
            metrics["t_cx"], metrics["t_cy"] = t_cx, t_cy
        else:
            t_cx, t_cy = b_cx, b_cy

        # -------------------------------------------------------------
        # 1. RADIAL DEPTH (Invariant to Flip, Rotate, Scale)
        # -------------------------------------------------------------
        euc_dist = np.sqrt((t_cx - b_cx)**2 + (t_cy - b_cy)**2)
        r_depth = euc_dist / r_eff
        metrics["radial_depth"] = r_depth
        if r_depth < 0.30:
            metrics["radial_interp"] = "Deep Core (Ventricular / Thalamic)"
        elif r_depth <= 0.75:
            metrics["radial_interp"] = "Mid Parenchyma (Intra-axial White Matter)"
        else:
            metrics["radial_interp"] = "Peripheral Rim (Cortical / Dural Border)"

        # -------------------------------------------------------------
        # 2. DURAL / CALVARIAL PROXIMITY (Distance to Skull Wall)
        # -------------------------------------------------------------
        dist_map = cv2.distanceTransform(brain_mask, cv2.DIST_L2, 5)
        dura_dist_px = float(np.min(dist_map[tumor_mask > 0])) if np.any(tumor_mask > 0) else 0.0
        dura_dist_norm = dura_dist_px / r_eff
        metrics["dural_dist_px"] = dura_dist_px
        metrics["dural_dist_norm"] = dura_dist_norm
        if dura_dist_norm < 0.05:
            metrics["dural_interp"] = "Abutting Dura / Bone (Extra-axial Contact)"
        elif dura_dist_norm < 0.18:
            metrics["dural_interp"] = "Subcortical / Near Surface"
        else:
            metrics["dural_interp"] = "Deep Parenchyma (Insulated by Cortex)"

        # -------------------------------------------------------------
        # 3. ABSOLUTE MIDLINE OFFSET (Axial & Coronal - Flip Invariant)
        # -------------------------------------------------------------
        if plane in ["Axial", "Coronal"] and bw > 0:
            abs_mid = abs(t_cx - b_cx) / (bw / 2.0)
            metrics["midline_offset"] = abs_mid
            if abs_mid < 0.15:
                metrics["midline_interp"] = "Strict Midline Center (Sellar / Falx)"
            elif abs_mid < 0.35:
                metrics["midline_interp"] = "Paramedian / Slight Lateralization"
            else:
                metrics["midline_interp"] = "Unilateral Hemispheric (Strong Lateralization)"

        # -------------------------------------------------------------
        # 4. SKULL-BASE ELEVATION (Coronal & Sagittal - Natural Vertical Axis)
        # -------------------------------------------------------------
        if plane in ["Coronal", "Sagittal"] and bh > 0:
            elev_si = (by + bh - t_cy) / float(bh)
            metrics["elevation_si"] = elev_si
            if elev_si < 0.20:
                metrics["elevation_interp"] = "Skull Base / Sellar Floor (Pituitary Territory)"
            elif elev_si <= 0.70:
                metrics["elevation_interp"] = "Mid-Cerebral / Lobar Height"
            else:
                metrics["elevation_interp"] = "High Convexity / Vertex (Parasagittal Rim)"

        # -------------------------------------------------------------
        # 5. MORPHOLOGY: SOLIDITY & CIRCULARITY (Pure Shape Invariant)
        # -------------------------------------------------------------
        solidity = (t_area / t_hull_area) if t_hull_area > 0 else 0.0
        metrics["solidity"] = solidity
        if solidity >= 0.88:
            metrics["solidity_interp"] = "Compact / Globular / Circumscribed"
        elif solidity >= 0.75:
            metrics["solidity_interp"] = "Moderately Lobulated"
        else:
            metrics["solidity_interp"] = "Irregular / Invasive / Finger-like Infiltration"

        circ = (4.0 * np.pi * t_area) / (t_perimeter**2) if t_perimeter > 0 else 0.0
        metrics["circularity"] = circ
        if circ >= 0.70:
            metrics["circularity_interp"] = "Smooth / Round / Well-defined Margin"
        elif circ >= 0.45:
            metrics["circularity_interp"] = "Wavy / Non-spherical Contour"
        else:
            metrics["circularity_interp"] = "Highly Spiculated / Complex Margin"

        return metrics

    def update_plot(self):
        img_name = self.image_files[self.current_idx]
        img_path = os.path.join(self.folder_path, img_name)
        plane = self.parse_view_plane(img_name)

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return

        base_name = os.path.splitext(img_name)[0]
        mask_path = self.find_mask_path(base_name)
        mask_gray = None
        if mask_path and os.path.exists(mask_path):
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is not None:
                mask_gray = cv2.resize(raw_mask, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 1. Segment Skull & Tumor
        brain_cnt, brain_hull, brain_mask = self.segment_brain_skull(img_gray)
        tumor_cnts, t_mask_bin = [], None

        if mask_gray is not None:
            _, t_mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
            tumor_cnts, _ = cv2.findContours(t_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Extract Metrics
        m = self.compute_invariant_metrics(
            plane, img_gray.shape, brain_cnt, brain_hull, brain_mask, tumor_cnts, t_mask_bin
        )

        # 3. Render Visual Overlays on Scan
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        if brain_cnt is not None:
            cv2.drawContours(img_rgb, [brain_cnt], -1, (0, 255, 0), 2)       # Green skull boundary
            cv2.drawContours(img_rgb, [brain_hull], -1, (255, 0, 0), 2)      # Red brain hull
            if m["b_cx"] is not None:
                cv2.drawMarker(img_rgb, (m["b_cx"], m["b_cy"]), (255, 0, 0), cv2.MARKER_CROSS, 16, 2)

        if m["has_tumor"]:
            overlay = img_rgb.copy()
            cv2.drawContours(overlay, tumor_cnts, -1, (255, 255, 0), -1)      # Yellow tumor fill
            cv2.addWeighted(overlay, 0.35, img_rgb, 0.65, 0, img_rgb)
            cv2.drawContours(img_rgb, tumor_cnts, -1, (255, 255, 0), 2)      # Yellow tumor border

            all_t = np.vstack(tumor_cnts)
            t_hull = cv2.convexHull(all_t)
            cv2.drawContours(img_rgb, [t_hull], -1, (0, 255, 255), 2)        # Cyan tumor hull

            if m["t_cx"] is not None:
                cv2.drawMarker(img_rgb, (m["t_cx"], m["t_cy"]), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                if m["b_cx"] is not None:
                    # Draw white vector from Brain Center to Tumor Center (Radial Vector)
                    cv2.line(img_rgb, (m["b_cx"], m["b_cy"]), (m["t_cx"], m["t_cy"]), (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Display Image
        self.ax_img.clear()
        self.ax_img.imshow(img_rgb)
        self.ax_img.set_title(
            f"Image [{self.current_idx + 1}/{self.total_images}]: {img_name}\n"
            f"Acquisition Plane: {plane.upper()} | Size: {img_gray.shape[1]}x{img_gray.shape[0]} px",
            fontsize=10, fontweight='bold'
        )
        self.ax_img.axis('off')

        # 5. Build Formatted Metrics Card with Visual Interpretations
        self.ax_text.clear()
        self.ax_text.axis('off')

        mid_line_str = f"{m['midline_offset']:.3f}" if m['midline_offset'] is not None else "N/A"
        elev_si_str  = f"{m['elevation_si']:.3f}" if m['elevation_si'] is not None else "N/A"

        panel_text = (
            f"=== SCAN INFORMATION ===\n"
            f"File:       {img_name}\n"
            f"Class:      {self.target_class.upper()}\n"
            f"View Plane: {plane.upper()}\n"
            f"Brain Area: {int(m['brain_area_px']):,} px² (Reff = {m['r_eff_px']:.1f} px)\n"
            f"Tumor Area: {int(m['tumor_area_px']):,} px² ({m['area_ratio_pct']:.2f}% burden)\n\n"

            f"=== RADIAL & BOUNDARY METRICS ===\n"
            f"• Radial Depth (δ): {m['radial_depth']:.3f}\n"
            f"  ↳ [{m['radial_interp']}]\n\n"
            f"• Dural Proximity:  {m['dural_dist_norm']:.3f} ({m['dural_dist_px']:.1f} px)\n"
            f"  ↳ [{m['dural_interp']}]\n\n"

            f"=== SYMMETRY & ELEVATION ===\n"
            f"• |Midline Offset|: {mid_line_str}\n"
            f"  ↳ [{m['midline_interp']}]\n\n"
            f"• Base Elevation:   {elev_si_str}\n"
            f"  ↳ [{m['elevation_interp']}]\n\n"

            f"=== MORPHOMETRIC SHAPE ===\n"
            f"• Solidity:         {m['solidity']:.3f}\n"
            f"  ↳ [{m['solidity_interp']}]\n\n"
            f"• Circularity:      {m['circularity']:.3f}\n"
            f"  ↳ [{m['circularity_interp']}]\n\n"

            f"=== VISUAL OVERLAY KEY ===\n"
            f"• Green: Brain Skull Boundary\n"
            f"• Red:   Brain Hull & Centroid (+)\n"
            f"• White: Radial Centroid Offset Vector\n"
            f"• Yellow:Tumor Segmentation Mask\n"
            f"• Cyan:  Tumor Hull & Centroid (+)"
        )

        self.ax_text.text(
            0.02, 0.98, panel_text,
            transform=self.ax_text.transAxes,
            fontsize=9.2,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#f8f9fa', alpha=0.95, edgecolor='#cccccc')
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
    TARGET_CLASS = "meningioma"

    viewer = AlignmentInvariantTumorViewer(
        base_dir=BASE_CLASS_DIR,
        seg_dir=SEG_TASK_DIR,
        target_class=TARGET_CLASS
    )