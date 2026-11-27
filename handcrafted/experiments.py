"""Feature composition for the one-group probes and named experiments."""

from dataclasses import dataclass
from itertools import combinations

import config as C


@dataclass(frozen=True)
class Experiment:
    name: str
    groups: tuple[str, ...]
    cohort: str = "tumor3"
    vocabulary: str = ""  # histogram, vlad, or both
    pca: bool = False


IMAGE_ONLY = ("F01", "F02", "F03", "F24")
ALL = tuple(f"F{i:02}" for i in range(1, 26))
SHAPE = ("F07", "F08", "F09", "F10", "F11:centered_")
TEXTURE = ("F15", "F16", "F17")
INTERNAL = ("F18", "F19")
APPEARANCE = ("F12", "F14", *TEXTURE, *INTERNAL, "F23:tumor_")
INTRINSIC = ("F06:intrinsic", *SHAPE, *APPEARANCE)
POSITION = ("F04", "F05", "F11:image_")
WITH_POSITION = ("F04", "F05", "F11", *INTRINSIC)
LOCAL = ("F13", "F20", "F21", "F22", "F23:context_hog")

NAMED = {
    "E00": Experiment(
        "Small geometry and plane probe", ("F04:centroid", "F06:small", "F07:small", "F05:depth", "F03")
    ),
    "E01": Experiment("Plane only", ("F03",)),
    "E02": Experiment("Acquisition and background", ("F01", "F02")),
    "E03": Experiment("Normalized centroid only", ("F04:centroid",)),
    "E04": Experiment("Lesion size", ("F06",)),
    "E04N": Experiment("Normalized lesion size", ("F06:normalized",)),
    "E05": Experiment("Shape and topology", SHAPE),
    "E06": Experiment("Location and size", ("F04", "F05", "F06")),
    "E07": Experiment(
        "Mask-only location, size and shape", ("F04", "F06:intrinsic", "F07", "F08", "F09", "F10", "F11")
    ),
    "E08": Experiment("Raw intensity", ("F12", "F14")),
    "E09": Experiment("Reference-normalized intensity", ("F13",)),
    "E10": Experiment("Texture", TEXTURE),
    "E11": Experiment("Internal lesion architecture", INTERNAL),
    "E12": Experiment("Boundary appearance", ("F20",)),
    "E13": Experiment("Peritumoral rings", ("F21:ring_only",)),
    "E13H": Experiment("Head-restricted peritumoral rings", ("F21:head_ring_only",)),
    "E13U": Experiment("Unrestricted peritumoral rings", ("F21:unrestricted_ring_only",)),
    "E14": Experiment("Tumor appearance in centered views", APPEARANCE, vocabulary="histogram"),
    "E15": Experiment("Intrinsic tumor appearance, size and shape", INTRINSIC, vocabulary="histogram"),
    "E16": Experiment("Intrinsic tumor features and position", WITH_POSITION, vocabulary="histogram"),
    "E17": Experiment("Tumor features and local context", (*WITH_POSITION, *LOCAL), vocabulary="histogram"),
    "E18": Experiment("Distant tissue and background", ("F02", "F25")),
    "E19": Experiment(
        "Medically motivated features", (*WITH_POSITION, *LOCAL, "F25"), vocabulary="histogram"
    ),
    "E20": Experiment("All handcrafted features", ALL, vocabulary="both"),
    "4C01": Experiment("Four-class plane", ("F03",), "image4"),
    "4C02": Experiment("Four-class acquisition and background", ("F01", "F02"), "image4"),
    "4C03": Experiment("Four-class whole MRI", ("F23:full_hog", "F24"), "image4"),
    "4C04": Experiment("Four-class all image-only features", (*IMAGE_ONLY, "F23:full_hog"), "image4"),
    "V01": Experiment("HOG of centered mask", ("F23:mask_hog",)),
    "V02": Experiment("HOG of centered lesion appearance", ("F23:tumor_hog",)),
    "V03": Experiment("HOG of centered local context", ("F23:context_hog",)),
    "V04": Experiment("HOG of the full MRI", ("F23:full_hog",)),
    "V05": Experiment("Lesion visual word histogram", (), vocabulary="histogram"),
    "V06": Experiment("Lesion VLAD", (), vocabulary="vlad"),
    "P01": Experiment("PCA of centered masks", ("F11:centered_mask",), pca=True),
    "P02": Experiment("PCA of centered lesion appearance", ("F23:tumor_mri_16",), pca=True),
}

EXACT_COLUMNS = {
    "F04:centroid": ("cx_norm", "cy_norm"),
    "F06:small": ("area_px",),
    "F07:small": ("circularity", "solidity", "eccentricity"),
    "F05:depth": ("mean_depth",),
    "F06:normalized": (
        "area_image_fraction",
        "area_head_fraction",
        "width_to_head",
        "height_to_head",
        "width_image_fraction",
        "height_image_fraction",
    ),
}

FAMILIES = {
    "location": POSITION,
    "size_shape": ("F06:intrinsic", *SHAPE),
    "intensity": ("F12", "F14"),
    "texture": TEXTURE,
    "architecture": INTERNAL,
    "boundary": ("F20",),
    "anatomical_context": ("F13", "F21", "F22", "F25"),
    "acquisition": ("F01", "F02", "F03"),
}


def all_experiments():
    experiments = {
        group: Experiment(f"{group} independently", (group,), vocabulary="both" if group == "F23" else "")
        for group in ALL
    }
    experiments.update(NAMED)
    if C.RUN_TEXTURE_SENSITIVITY:
        experiments.update(
            {
                f"T{bins}": Experiment(f"Texture with {bins} gray levels", (group,))
                for bins, group in (
                    (C.TEXTURE_BINS, "F15"),
                    *((b, f"F15_bins{b}") for b in C.TEXTURE_SENSITIVITY_BINS),
                )
            }
        )
    return experiments


def subsets():
    families = list(FAMILIES)
    yield "S000", (), Experiment("Intercept-only baseline", ())
    for count in range(1, len(families) + 1):
        for keys in combinations(families, count):
            code = sum(1 << families.index(name) for name in keys)
            specs = tuple(dict.fromkeys(group for name in keys for group in FAMILIES[name]))
            yield f"S{code:03}", keys, Experiment(" + ".join(keys), specs)
