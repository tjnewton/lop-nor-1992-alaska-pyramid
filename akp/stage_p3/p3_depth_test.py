"""P3 does depth classification separate the target from natural sources? A control-only test.

On one open control package (Iron Creek, outside Track A), two detector statistics are compared:
- "amplitude": the matched score over the vector bank (one top depth);
- "depth": the depth-classified score over the depth bank (p3_mag_templates --depths: tops 5, 38,
  120 and 300 m). A position scores only where its best template over all depths has the target
  top (p3_mag_detect.TARGET_TOP); elsewhere it scores 0.
For each statistic it records the null peaks (score > 0) with their score quantiles. It also runs
injection trials of T2 at both typical contrasts (p3_mag_trials, induced, base 300 m, top 38.1 m),
N_PER x SEEDS plants each, recording the score quantiles and the share of plants with a non-zero
score: for the depth statistic, the share classified to the target depth. Writes p3_depth_test.csv
(statistic, quantity, dk, value). Run: python -m akp.stage_p3.p3_depth_test
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from akp.paths import data_dir, data_glob
from akp.stage_p3 import p3_mag_detect as det
from akp.stage_p3.p3_lines import read_package

HERE = Path(__file__).parent
BANKS = {"amplitude": "templates_vector.npz", "depth": "templates_vector_depths.npz"}
DKS = (0.0241, 0.0691)  # T2 typical contrasts (p3_g1a_decision.csv)
SEEDS, N_PER = (1, 2), 25


def package():
    return [p for p in data_glob(data_dir("p3/lines/open"), "*.zip") if "iron-creek" in p.name][0]


def run(args):
    stat, seed, dk = args
    det.BANK = BANKS[stat]
    det._bank.cache_clear()
    from akp.stage_p3.p3_mag_trials import trials

    _, df = next(read_package(package()))
    return trials(df, "MAGIGRF", "ALTR", True, 400.0, N_PER, seed, 300.0, 38.1, dk).assign(statistic=stat, dk=dk)


def main() -> None:
    _, df = next(read_package(package()))
    rows = []
    for stat, bank in BANKS.items():
        det.BANK = bank
        det._bank.cache_clear()
        pk = det.detect(df, "MAGIGRF", "ALTR", True)
        pk = pk[pk.score > 0]
        rows.append(dict(statistic=stat, quantity="null_peaks", dk=0.0, value=len(pk)))
        for q in (0.5, 0.9, 0.99):
            rows.append(dict(statistic=stat, quantity=f"null_score_q{q}", dk=0.0, value=round(float(pk.score.quantile(q)), 1)))
    with ProcessPoolExecutor(4) as ex:
        tr = pd.concat(ex.map(run, [(s, seed, dk) for s in BANKS for seed in SEEDS for dk in DKS]))
    for (stat, dk), g in tr.groupby(["statistic", "dk"]):
        rows += [dict(statistic=stat, quantity="plants", dk=dk, value=int(len(g))),
                 dict(statistic=stat, quantity="plant_share_scored", dk=dk, value=round(float((g.score > 0).mean()), 4)),
                 dict(statistic=stat, quantity="plant_score_median", dk=dk, value=round(float(g.score.median()), 1))]
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_depth_test.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
