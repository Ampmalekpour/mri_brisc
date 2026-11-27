"""Step 3. Extract F12–F20, from lesion intensities through boundary contrast."""

import appearance
import config as C
from extraction import run_stage


def extract_row(row, rgb, regions):
    return appearance.extract(regions), None


def main():
    bins = (C.TEXTURE_BINS, *C.TEXTURE_SENSITIVITY_BINS)
    if any(not 2 <= value <= 256 for value in bins) or len(set(bins)) != len(bins):
        raise ValueError("Use distinct texture bin counts between 2 and 256.")
    groups = [f"F{i:02}" for i in range(12, 21)] + [
        f"F15_bins{level}" for level in C.TEXTURE_SENSITIVITY_BINS
    ]
    run_stage(
        "Step 3 | Intensities, texture, radial profiles and boundary contrast", groups, extract_row, stage=3
    )


if __name__ == "__main__":
    main()
