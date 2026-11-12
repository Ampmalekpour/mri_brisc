"""
STEP 1 - Location / shape features for every scan (the 'what a non-CNN can see' table).

For every image in index.csv, brain geometry is estimated from the image (threshold + largest contour)
and, for tumor images, tumor geometry from the segmentation mask. All distances are divided by the
effective brain radius r_eff = sqrt(brain_area / pi), so they are comparable across scans.

 LOCATION  radial_depth      |tumor centroid - brain centroid| / r_eff       (0 = centre, ~1 = rim)
           dural_dist_norm   distance of tumor centroid to the brain boundary / r_eff
           min_edge_dist     closest tumor pixel to the brain boundary / r_eff (meningioma: ~0)
           dy_norm, dx_abs   vertical (up positive) and absolute horizontal offset / r_eff
 SHAPE     area_ratio, solidity, circularity, eccentricity
 Output    outputs/features.csv
Run:  python s01_extract_features.py
"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
from common import brain_mask, centroid, read_mask, read_rgb


def tumor_shape(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    area = float(mask.sum())
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    per = cv2.arcLength(cnt, True)
    ecc = np.nan
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        a, b = max(MA, ma), min(MA, ma)
        ecc = float(np.sqrt(1 - (b / a) ** 2)) if a > 0 else np.nan
    return dict(solidity=cv2.contourArea(cnt) / hull_area if hull_area > 0 else np.nan,
                circularity=4 * np.pi * area / per ** 2 if per > 0 else np.nan, eccentricity=ecc)


def main():
    df = pd.read_csv(C.INDEX_CSV)
    out = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="features"):
        img = read_rgb(r["path"])
        br = brain_mask(img)
        area = float(br.sum())
        rec = dict(path=r["path"], brain_area=area)
        if area < 100:
            out.append(rec); continue
        r_eff = np.sqrt(area / np.pi)
        bcx, bcy = centroid(br)
        rec.update(r_eff=r_eff, b_cx=bcx, b_cy=bcy)
        # acquisition / scanner cues (global intensity statistics, noise level) - NOT biology
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        v = g[br]
        rec.update(int_mean=float(v.mean()), int_std=float(v.std()), int_p5=float(np.percentile(v, 5)),
                   int_p50=float(np.percentile(v, 50)), int_p95=float(np.percentile(v, 95)),
                   bg_mean=float(g[~br].mean()) if (~br).any() else np.nan,
                   noise_lap=float(cv2.Laplacian(g, cv2.CV_32F)[br].var()))
        mask = read_mask(r["mask_path"], img.shape) if r["class"] != "no_tumor" else np.zeros(br.shape, bool)
        if mask.any():
            tcx, tcy = centroid(mask)
            # pad with a 1-px zero border: scans cropped tightly to the head have NO background pixels and
            # distanceTransform would return ~1e34 ("infinite") for them
            dist = cv2.distanceTransform(cv2.copyMakeBorder(br.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
                                         cv2.DIST_L2, 5)[1:-1, 1:-1]
            rec.update(t_cx=tcx, t_cy=tcy, tumor_area=float(mask.sum()),
                       radial_depth=np.hypot(tcx - bcx, tcy - bcy) / r_eff,
                       dural_dist_norm=float(dist[int(round(tcy)), int(round(tcx))]) / r_eff,
                       min_edge_dist=float(dist[mask].min()) / r_eff,
                       dy_norm=(bcy - tcy) / r_eff, dx_abs=abs(tcx - bcx) / r_eff,
                       area_ratio=float(mask.sum()) / area, **tumor_shape(mask))
        out.append(rec)
    feats = pd.DataFrame(out)
    feats.to_csv(C.FEATURES_CSV, index=False)
    full = df.merge(feats, on="path")
    print(full[full["class"] != "no_tumor"].groupby("class")[
        ["radial_depth", "dural_dist_norm", "min_edge_dist", "area_ratio", "solidity"]].mean().round(3))
    print(f"saved {C.FEATURES_CSV}")


if __name__ == "__main__":
    main()
