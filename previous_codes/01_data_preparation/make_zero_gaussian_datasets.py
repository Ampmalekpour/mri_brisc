# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : mask_tumor_val_dataset_creation.py
# Purpose       :
#   Creates counterfactual copies of a split: tumor pixels set to 0 ('_mask_zero') or replaced by
#   Gaussian noise ('_mask_gaussian'); no_tumor images are copied unchanged.
#   CAVEAT: the black/noisy blob keeps the tumor's silhouette (location, size, shape) visible.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import glob
import cv2
import numpy as np
import shutil


def create_masked_datasets():
    # --- Directory Configuration ---
    base_class_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    val_dir = os.path.join(base_class_dir, "train")
    seg_test_dir = r"D:\Datasets\Medical\brisc2025\segmentation_task\train"

    # New target directories
    output_zero_dir = os.path.join(base_class_dir, "train_mask_zero")
    output_gaussian_dir = os.path.join(base_class_dir, "train_mask_gaussian")

    # Supported extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    print("Starting dataset modification loop...")

    # Iterate through classes: glioma, meningioma, no_tumor, pituitary
    classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

    for cls in classes:
        print(f"\nProcessing class: {cls}")
        src_cls_dir = os.path.join(val_dir, cls)

        # Setup clean subdirectories in targets
        dst_zero_cls = os.path.join(output_zero_dir, cls)
        dst_gauss_cls = os.path.join(output_gaussian_dir, cls)
        os.makedirs(dst_zero_cls, exist_ok=True)
        os.makedirs(dst_gauss_cls, exist_ok=True)

        # List all images in this class folder
        all_files = os.listdir(src_cls_dir)
        img_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

        for img_name in img_files:
            src_img_path = os.path.join(src_cls_dir, img_name)
            base_name_no_ext = os.path.splitext(img_name)[0]

            # Output paths
            out_zero_path = os.path.join(dst_zero_cls, img_name)
            out_gauss_path = os.path.join(dst_gauss_cls, img_name)

            # If no_tumor, skip mask searching entirely and copy the originals
            if cls == "no_tumor":
                shutil.copy2(src_img_path, out_zero_path)
                shutil.copy2(src_img_path, out_gauss_path)
                continue

            # --- Search for the Mask ---
            # Search recursively inside the segmentation task test folder for matching filenames
            search_pattern = os.path.join(seg_test_dir, "**", f"{base_name_no_ext}.*")
            mask_matches = [f for f in glob.glob(search_pattern, recursive=True) if
                            not f.lower().endswith(valid_extensions)]

            # If explicit name matching needs fallback, look broad
            if not mask_matches:
                search_pattern_alt = os.path.join(seg_test_dir, "**", f"*{base_name_no_ext}*")
                mask_matches = glob.glob(search_pattern_alt, recursive=True)

            # Filter matches to make sure we load an actual image/mask file
            mask_path = None
            for match in mask_matches:
                if any(match.lower().endswith(ext) for ext in valid_extensions):
                    # Guard to avoid accidentally picking up the raw image copy if stored together
                    if "mask" in match.lower() or len(mask_matches) == 1 or "mask" in seg_test_dir.lower():
                        mask_path = match
                        break
            if not mask_path and mask_matches:
                mask_path = mask_matches[0]

            # If no mask is found for tumorous classes, log it and keep original as fallback
            if not mask_path:
                print(f"  [Warning] No mask found for {img_name}. Copying original image.")
                shutil.copy2(src_img_path, out_zero_path)
                shutil.copy2(src_img_path, out_gauss_path)
                continue

            # --- Image Processing Core ---
            # Load the original image and the corresponding mask
            img = cv2.imread(src_img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                print(f"  [Error] Failed to read image or mask for {img_name}. Copying original.")
                shutil.copy2(src_img_path, out_zero_path)
                shutil.copy2(src_img_path, out_gauss_path)
                continue

            # Ensure mask matches image dimensions exactly
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create clean binary mask (thresholding potential anti-aliasing artifacts)
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # --- Version 1: Zero / Blackout Masking ---
            img_zero = img.copy()
            img_zero[binary_mask == 255] = 0
            cv2.imwrite(out_zero_path, img_zero)

            # --- Version 2: Gaussian Noise Masking ---
            img_gaussian = img.copy()

            # Generate Gaussian Noise matching the input distribution profile
            mean = 0
            std_dev = 25  # Standard variance for typical MRI pixel disturbances
            noise = np.random.normal(mean, std_dev, img.shape).astype(np.int16)

            # Add noise and clip values to valid 8-bit image array space [0, 255]
            img_noisy_background = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Replace only the tumor coordinates with the noisy variant
            img_gaussian[binary_mask == 255] = img_noisy_background[binary_mask == 255]
            cv2.imwrite(out_gauss_path, img_gaussian)

    print("\nModification complete! Folders generated successfully:")
    print(f" -> {output_zero_dir}")
    print(f" -> {output_gaussian_dir}")


if __name__ == '__main__':
    create_masked_datasets()