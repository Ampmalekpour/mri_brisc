# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : mask_tumor_biharmonic_inpainting_foreground.py
# Purpose       :
#   Creates (a) 'train_mask_inpainted': tumor removed and filled by biharmonic inpainting, and
#   (b) 'train_foreground_only': only the tumor is kept. no_tumor images are copied unchanged.
#   CAVEAT: inpainting leaves blur artefacts; the mask may not cover edema / mass effect.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import glob
import cv2
import numpy as np
import shutil


def create_miccai_diagnostic_datasets():
    # --- Directory Configuration ---
    base_class_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    val_dir = os.path.join(base_class_dir, "train")
    seg_test_dir = r"D:\Datasets\Medical\brisc2025\segmentation_task\train"

    # New target directories for MICCAI tests
    output_inpaint_dir = os.path.join(base_class_dir, "train_mask_inpainted")
    output_foreground_dir = os.path.join(base_class_dir, "train_foreground_only")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    print("Starting MICCAI diagnostic dataset generation...")

    classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

    for cls in classes:
        print(f"\nProcessing class: {cls}")
        src_cls_dir = os.path.join(val_dir, cls)

        # Setup subdirectories
        dst_inpaint_cls = os.path.join(output_inpaint_dir, cls)
        dst_fore_cls = os.path.join(output_foreground_dir, cls)
        os.makedirs(dst_inpaint_cls, exist_ok=True)
        os.makedirs(dst_fore_cls, exist_ok=True)

        all_files = os.listdir(src_cls_dir)
        img_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

        for img_name in img_files:
            src_img_path = os.path.join(src_cls_dir, img_name)
            base_name_no_ext = os.path.splitext(img_name)[0]

            out_inpaint_path = os.path.join(dst_inpaint_cls, img_name)
            out_fore_path = os.path.join(dst_fore_cls, img_name)

            # If no_tumor, copy original to inpaint (it's already healthy).
            # For foreground isolation, copy as well, or you can zero it (we'll copy here to maintain dataset size).
            if cls == "no_tumor":
                shutil.copy2(src_img_path, out_inpaint_path)
                shutil.copy2(src_img_path, out_fore_path)
                continue

            # --- Search for the Mask ---
            search_pattern = os.path.join(seg_test_dir, "**", f"{base_name_no_ext}.*")
            mask_matches = [f for f in glob.glob(search_pattern, recursive=True) if
                            not f.lower().endswith(valid_extensions)]

            if not mask_matches:
                search_pattern_alt = os.path.join(seg_test_dir, "**", f"*{base_name_no_ext}*")
                mask_matches = glob.glob(search_pattern_alt, recursive=True)

            mask_path = None
            for match in mask_matches:
                if any(match.lower().endswith(ext) for ext in valid_extensions):
                    if "mask" in match.lower() or len(mask_matches) == 1 or "mask" in seg_test_dir.lower():
                        mask_path = match
                        break
            if not mask_path and mask_matches:
                mask_path = mask_matches[0]

            if not mask_path:
                print(f"  [Warning] No mask found for {img_name}. Copying original image.")
                shutil.copy2(src_img_path, out_inpaint_path)
                shutil.copy2(src_img_path, out_fore_path)
                continue

            # --- Image Processing Core ---
            img = cv2.imread(src_img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # -------------------------------------------------------------
            # TEST 1: Anatomical Inpainting (The "Healthy Brain" Test)
            # -------------------------------------------------------------
            # Dilate the mask to ensure the tumor's enhancing outer rim is also removed
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # Use Telea algorithm for smooth textural inward propagation
            img_inpainted = cv2.inpaint(img, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            cv2.imwrite(out_inpaint_path, img_inpainted)

            # -------------------------------------------------------------
            # TEST 2: Foreground Isolation (The "Pathology Only" Test)
            # -------------------------------------------------------------
            # Keep only the pixels where the binary_mask is 255, zero everything else
            img_foreground = cv2.bitwise_and(img, img, mask=binary_mask)
            cv2.imwrite(out_fore_path, img_foreground)

    print("\nMICCAI Dataset Modification complete!")
    print(f" -> {output_inpaint_dir}")
    print(f" -> {output_foreground_dir}")


if __name__ == '__main__':
    create_miccai_diagnostic_datasets()