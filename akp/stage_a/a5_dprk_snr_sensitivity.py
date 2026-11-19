"""A5 clutter against the usable-record threshold: how far the in-box clutter medians move with the screen.

Why: an independent re-derivation could not reproduce the clutter medians of a5_dprk_site_spread.csv from the
manuscript's text. The committed medians keep only "usable" records: covered, without gaps, not clipped, and with a P
peak at least SNR_SEEN times the pre-P level (a5_dprk_network, the screen of A1). The manuscript did not state the rule.
This module shows what the rule does. No waveform is read.

Inputs: a5_dprk_levels.csv (one row per station, event and band: snr, covered, gaps, clipped) and a5_dprk_clutter.csv
(one row per station, event, band and lag: the clutter ratio and A2b's validity flag), both written by a5_dprk_network.
Method: for each threshold in THRESHOLDS (0 = no screen on the P peak), a record is kept if it is covered, has no gap,
is not clipped and its snr is at least the threshold. Over the kept records of the Denali-Nome box, in the lag window
the manuscript uses, the valid clutter ratios give n, the median and the fifth percentile, per band, for all events
pooled ("all") and per event. One further row per band, threshold "none_no_validity", drops the validity rule as well.
Outputs: a5_dprk_snr_sensitivity.csv; a5_dprk_snr_sensitivity_checks.csv (per band, the smallest and largest pooled
median over the thresholds, whether the committed one is reproduced, and the largest simulated share at 1 km or more
over the smallest pooled median of its band, from a8_smoothing.csv and a8_generous_shares.csv).
Self-check: at the committed threshold the pooled and per-event medians equal a5_dprk_site_spread.csv.
Run: PYTHONPATH=. conda run -n swe-akp python -m akp.stage_a.a5_dprk_snr_sensitivity
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_a import a5_dprk_network as net

HERE = Path(__file__).parent
BOX, LAG = "mckinley_nome", "5-15"  # the box and lag window of the manuscript's clutter
COMMITTED = float(net.a5.SNR_SEEN)
THRESHOLDS = (0.0, 3.0, COMMITTED, 10.0)  # P peak over pre-P level; 0 = no screen
FAR_M = 1000.0  # the paper's statement concerns ranges of 1 km or more


def main() -> None:
    lv = pd.read_csv(HERE / "a5_dprk_levels.csv", dtype={"band": str})
    cl = pd.read_csv(HERE / "a5_dprk_clutter.csv", dtype={"band": str, "lag": str})
    cl = cl[(cl.box == BOX) & (cl.lag == LAG)]
    sound = lv[lv.covered.eq(True) & lv.gaps.fillna(1).eq(0) & lv.clipped.eq(False)]
    rows = []
    for thr in THRESHOLDS:
        keep = sound[sound.snr >= thr][["event", "net", "sta", "band"]]
        c = cl.merge(keep, on=["event", "net", "sta", "band"])
        for label, valid_only in ((f"{thr:g}", True),) + ((("none_no_validity", False),) if thr == 0 else ()):
            d = c[c.valid] if valid_only else c
            for (event, band), g in pd.concat([d, d.assign(event="all")]).groupby(["event", "band"]):
                rows.append(dict(threshold=label, event=event, band=band, n=len(g), median=float(g.ratio.median()),
                                 p05=float(np.percentile(g.ratio, 5))))
    out = pd.DataFrame(rows)
    ref = pd.read_csv(HERE / "a5_dprk_site_spread.csv", dtype={"band": str, "lag": str})
    ref = ref[(ref.box == BOX) & (ref.lag == LAG)].set_index(["event", "band"])["median"]
    mine = out[out.threshold == f"{COMMITTED:g}"].set_index(["event", "band"])["median"]
    assert len(mine) == len(ref) and np.allclose(mine.reindex(ref.index), ref, rtol=1e-12), "the committed medians are not reproduced"
    out.round(6).to_csv(HERE / "a5_dprk_snr_sensitivity.csv", index=False)
    pooled = out[(out.event == "all") & (out.threshold != "none_no_validity")]
    checks = [dict(check="committed_threshold", value=COMMITTED), dict(check="committed_medians_reproduced", value=True)]
    for band, g in pooled.groupby("band"):
        checks += [dict(check=f"pooled_median_min_{band}", value=round(float(g["median"].min()), 4)),
                   dict(check=f"pooled_median_max_{band}", value=round(float(g["median"].max()), 4))]
    # The comparison the paper rests on, against the lowest pooled median any threshold gives: the simulated
    # scattered share at 1 km or more, every case old and new, all azimuths, as measured and smoothed the clutter's way.
    low = pooled.groupby("band")["median"].min()
    sh = pd.concat([pd.read_csv(HERE / f, dtype={"band": str})[["case", "range_m", "band", "share", "share_smoothed"]]
                    for f in ("a8_smoothing.csv", "a8_generous_shares.csv")])
    sh = sh[sh.range_m >= FAR_M]
    for col in ("share", "share_smoothed"):
        worst = float((sh[col] / sh.band.map(low)).max())
        checks += [dict(check=f"largest_{col}_over_lowest_median_beyond_1km", value=round(worst, 3)),
                   dict(check=f"{col}_below_lowest_median_beyond_1km", value=bool(worst < 1))]
    checks.append(dict(check="cases_compared", value=" ".join(sorted(sh.case.unique()))))
    pd.DataFrame(checks).to_csv(HERE / "a5_dprk_snr_sensitivity_checks.csv", index=False)
    print(out[out.event == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
