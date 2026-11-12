# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : train_random_forest.py
# Purpose       :
#   Random forest on tumor-only scans: location features alone vs location+shape (80/20 stratified).
#   Tests whether location alone separates the tumor classes. No saved output exists for this run.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main():
    csv_path = "brisc_morphometric_features.csv"

    print(f"=== INITIALIZING RANDOM FOREST BASELINE AUDIT ===")
    print(f"Loading data from: {csv_path}\n")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[CRITICAL ERROR] Could not find {csv_path}. Please ensure it is in the same directory.")
        return

    # We only care about diagnostic confusion between the actual tumors.
    # 'no_tumor' is trivially solved by 'has_tumor == 0', so we drop it to test actual pathology separation.
    df_tumors = df[df['has_tumor'] == 1].copy()

    # Define our feature sets
    spatial_features = ['radial_depth', 'dural_dist_norm']
    shape_features = ['solidity', 'circularity', 'area_ratio']
    all_features = spatial_features + shape_features

    # Drop any rows where contour extraction might have failed and left NaNs in these core features
    df_tumors = df_tumors.dropna(subset=all_features)

    print(f"Dataset ready. Training on {len(df_tumors)} lesion scans...\n")

    X = df_tumors[all_features]
    y = df_tumors['class']

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # TEST 1: PURE SPATIAL PRIORS (Location Only)
    # ---------------------------------------------------------
    print("-" * 60)
    print(" MODEL 1: PURE SPATIAL FEATURES ONLY")
    print(f" Features: {spatial_features}")
    print("-" * 60)

    rf_spatial = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf_spatial.fit(X_train[spatial_features], y_train)

    preds_spatial = rf_spatial.predict(X_test[spatial_features])
    acc_spatial = accuracy_score(y_test, preds_spatial)

    print(f"Accuracy (Location Only): {acc_spatial:.4f}\n")
    print(classification_report(y_test, preds_spatial, digits=4))

    # ---------------------------------------------------------
    # TEST 2: ALL PLANE-AGNOSTIC FEATURES (Location + Shape)
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(" MODEL 2: LOCATION + MORPHOLOGY")
    print(f" Features: {all_features}")
    print("-" * 60)

    rf_full = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf_full.fit(X_train, y_train)

    preds_full = rf_full.predict(X_test)
    acc_full = accuracy_score(y_test, preds_full)

    print(f"Accuracy (Location + Shape): {acc_full:.4f}\n")
    print(classification_report(y_test, preds_full, digits=4))

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" RANDOM FOREST FEATURE IMPORTANCE (MODEL 2)")
    print("=" * 60)

    importances = pd.Series(rf_full.feature_importances_, index=all_features).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"{feat:<20} | {imp:.4f}")

    print("\nInterpretation:")
    print("If Radial Depth and Dural Dist dominate the importance metric, it mathematically")
    print("proves that spatial location is the strongest discriminative signal in the dataset,")
    print("perfectly explaining why the Inpainted CNNs scored so highly.")


if __name__ == "__main__":
    main()