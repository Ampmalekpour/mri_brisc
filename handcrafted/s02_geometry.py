"""Step 2. Extract F01–F11, from acquisition controls through tumor geometry."""

import context
import geometry
from extraction import run_stage


def extract_row(row, rgb, regions):
    return context.image_controls(row, rgb, regions) | geometry.extract(regions), None


def main():
    run_stage(
        "Step 2 | Acquisition, plane, anatomy and lesion geometry",
        [f"F{i:02}" for i in range(1, 12)],
        extract_row,
        stage=2,
    )


if __name__ == "__main__":
    main()
