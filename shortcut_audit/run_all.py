"""
Run the full pipeline in order. Edit config.py first.   python run_all.py [--quick]

 s00 index + hygiene -> s01 features -> s02 cue probes -> s03 train (clean baselines x archs x seeds,
 plus inpaint/crop/mask probes and the relocation mitigation for the FIRST arch) -> s04 typicality ->
 s05 interventions -> s06 XAI -> s07 report.

 --quick : first arch only, first seed only (a sanity pass before the multi-hour full run).
Everything is resumable: already-trained runs are skipped.
"""
import argparse
import subprocess
import sys

import config as C

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
py = sys.executable
archs = C.ARCHS[:1] if a.quick else C.ARCHS
seeds = C.SEEDS[:1] if a.quick else C.SEEDS


def run(*args):
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([py, *args], check=True)


run("s00_build_index.py"); run("s01_extract_features.py"); run("s02_cue_probes.py")
for arch in archs:
    for s in seeds:
        run("s03_train.py", "--arch", arch, "--seed", str(s))
for s in seeds:                                          # probes + mitigation on the first architecture
    for v in ("inpaint", "crop", "mask"):
        run("s03_train.py", "--arch", archs[0], "--variant", v, "--seed", str(s))
    run("s03_train.py", "--arch", archs[0], "--p_relocate", "0.5", "--tag", "relocate", "--seed", str(s))
run("s04_location_typicality.py")
run("s05_interventions.py", "--run", *[f"{ar}_clean_s{s}" for ar in archs for s in seeds],
    *[f"{archs[0]}_clean_relocate_s{s}" for s in seeds])
run("s06_xai.py", "--run", *[f"{ar}_clean_s{s}" for ar in archs for s in seeds[:1]])
run("s07_make_report.py")
