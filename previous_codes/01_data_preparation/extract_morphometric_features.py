# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : extract_measures_1.py
# Purpose       :
#   Bulk extraction of per-scan brain/tumor geometry (radial depth, dural distance, midline offset,
#   elevation, area ratio, solidity, circularity) from images + segmentation masks.
#   OUTPUT: brisc_morphometric_features.csv (5000 rows, kept next to this file).
#   NOTE: these 'location' features are what the random forest in 05_spatial_prior_analysis uses.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class BulkTumorFeatureExtractor:
    def __init__(self, base_dir, seg_dir, classes=None):
        self.base_dir = base_dir
        self.seg_dir = seg_dir
        self.classes = classes if classes else ["glioma", "meningioma", "pituitary", "no_tumor"]
        self.valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    def parse_view_plane(self, filename):
        fn_lower = os.path.splitext(filename)[0].lower()
        tokens = [t for t in re.split(r'[_.\-\s]+', fn_lower) if t]

        if any(t in ["co", "cor", "coronal"] for t in tokens) or "_co_" in fn_lower or "coronal" in fn_lower:
            return "Coronal"
        elif any(t in ["sa", "sag", "sagittal"] for t in tokens) or "_sa_" in fn_lower or "sagittal" in fn_lower:
            return "Sagittal"
        elif any(t in ["ax", "tra", "axial"] for t in tokens) or "_ax_" in fn_lower or "axial" in fn_lower:
            return "Axial"
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

    def segment_brain(self, img_gray):
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

    def extract_features(self):
        dataset_records = []

        for target_class in self.classes:
            folder_path = os.path.join(self.base_dir, target_class)
            if not os.path.exists(folder_path):
                print(f"Skipping missing directory: {folder_path}")
                continue

            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(self.valid_exts)]

            print(f"Processing class: {target_class.upper()} ({len(image_files)} images)")

            for img_name in tqdm(image_files, desc=target_class, unit="scan"):
                img_path = os.path.join(folder_path, img_name)
                plane = self.parse_view_plane(img_name)

                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    continue

                h, w = img_gray.shape
                base_name = os.path.splitext(img_name)[0]

                # Base Record
                record = {
                    "filename": img_name,
                    "class": target_class,
                    "plane": plane,
                    "img_w": w,
                    "img_h": h,
                }

                # Find & load mask
                mask_path = self.find_mask_path(base_name)
                mask_gray = None
                if mask_path and os.path.exists(mask_path):
                    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if raw_mask is not None:
                        mask_gray = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Segment brain
                brain_cnt, brain_hull, brain_mask = self.segment_brain(img_gray)

                if brain_cnt is None:
                    record["error"] = "Brain segmentation failed"
                    dataset_records.append(record)
                    continue

                # Brain base metrics
                brain_area = float(cv2.contourArea(brain_cnt))
                r_eff = np.sqrt(brain_area / np.pi) if brain_area > 0 else 1.0

                bx, by, bw, bh = cv2.boundingRect(brain_hull)
                M_b = cv2.moments(brain_hull)
                b_cx = int(M_b["m10"] / M_b["m00"]) if M_b["m00"] != 0 else bx + bw // 2
                b_cy = int(M_b["m01"] / M_b["m00"]) if M_b["m00"] != 0 else by + bh // 2

                record.update({
                    "brain_area": brain_area,
                    "r_eff": r_eff,
                    "b_cx": b_cx,
                    "b_cy": b_cy
                })

                # If no tumor (or no mask found)
                if mask_gray is None or np.sum(mask_gray) == 0:
                    record["has_tumor"] = 0
                    dataset_records.append(record)
                    continue

                # Tumor metrics
                record["has_tumor"] = 1
                _, t_mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                tumor_cnts, _ = cv2.findContours(t_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not tumor_cnts:
                    dataset_records.append(record)
                    continue

                all_t_pts = np.vstack(tumor_cnts)
                tumor_hull = cv2.convexHull(all_t_pts)

                t_area = float(sum(cv2.contourArea(c) for c in tumor_cnts))
                t_hull_area = float(cv2.contourArea(tumor_hull))
                t_perimeter = float(sum(cv2.arcLength(c, True) for c in tumor_cnts))

                M_t = cv2.moments(all_t_pts)
                t_cx = int(M_t["m10"] / M_t["m00"]) if M_t["m00"] != 0 else b_cx
                t_cy = int(M_t["m01"] / M_t["m00"]) if M_t["m00"] != 0 else b_cy

                # 1. Radial Depth
                euc_dist = np.sqrt((t_cx - b_cx) ** 2 + (t_cy - b_cy) ** 2)
                record["radial_depth"] = euc_dist / r_eff

                # 2. Dural Proximity
                # Draw a 1-pixel black border to prevent cv2.distanceTransform overflow
                cv2.rectangle(brain_mask, (0, 0), (w - 1, h - 1), 0, 1)

                dist_map = cv2.distanceTransform(brain_mask, cv2.DIST_L2, 5)
                dura_dist_px = float(np.min(dist_map[t_mask_bin > 0]))
                record["dural_dist_norm"] = dura_dist_px / r_eff

                # 3. Midline Offset (Axial/Coronal)
                if plane in ["Axial", "Coronal"] and bw > 0:
                    record["midline_offset"] = abs(t_cx - b_cx) / (bw / 2.0)
                else:
                    record["midline_offset"] = np.nan

                # 4. Elevation SI (Coronal/Sagittal)
                if plane in ["Coronal", "Sagittal"] and bh > 0:
                    record["elevation_si"] = (by + bh - t_cy) / float(bh)
                else:
                    record["elevation_si"] = np.nan

                # 5. Morphology
                record["area_ratio"] = t_area / brain_area if brain_area > 0 else 0.0
                record["solidity"] = t_area / t_hull_area if t_hull_area > 0 else 0.0
                record["circularity"] = (4.0 * np.pi * t_area) / (t_perimeter ** 2) if t_perimeter > 0 else 0.0

                dataset_records.append(record)

        df = pd.DataFrame(dataset_records)
        return df


if __name__ == "__main__":
    # Point these to your setup
    BASE_CLASS_DIR = r"D:\Datasets\Medical\brisc2025\classification_task\train"
    SEG_TASK_DIR = r"D:\Datasets\Medical\brisc2025\segmentation_task\train"
    OUTPUT_CSV = "brisc_morphometric_features.csv"

    extractor = BulkTumorFeatureExtractor(
        base_dir=BASE_CLASS_DIR,
        seg_dir=SEG_TASK_DIR,
        classes=["glioma", "meningioma", "pituitary", "no_tumor"]
    )

    print("Starting feature extraction pipeline...")
    df_features = extractor.extract_features()

    # Save out the raw data
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExtraction complete! Saved {len(df_features)} records to {OUTPUT_CSV}")

    # Quick sanity check printout
    print("\n--- Quick Stats by Class ---")
    tumor_df = df_features[df_features['has_tumor'] == 1]
    summary = tumor_df.groupby('class')[['radial_depth', 'dural_dist_norm', 'solidity']].mean().round(3)
    print(summary)