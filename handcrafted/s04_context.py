"""Step 4. Extract F21–F25, from surrounding tissue through visual descriptors."""

import context
import visual
from extraction import run_stage


def extract_row(row, rgb, regions):
    visual_groups, patches = visual.extract(regions)
    return context.extract(regions) | visual_groups, patches


def main():
    run_stage(
        "Step 4 | Surrounding tissue, classical descriptors, whole MRI and distant tissue",
        [f"F{i:02}" for i in range(21, 26)],
        extract_row,
        stage=4,
        save_patches=True,
    )


if __name__ == "__main__":
    main()
