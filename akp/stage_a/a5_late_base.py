"""A5 late windows, base width: what a body with a 460 m base would change in the paper's headline ranges and bounds.

Why: versions of the claim found after the simulations give the sides of the base as "1510 ft" (460 m; Supplement One,
section S1.11). The largest simulated base is 400 m. No case was run at 460 m; this module bounds its effect instead.

Method (fixed before any scaled number was computed; a description, not a new rule):
- the effect of base width as simulated: the ratio of the 400 m high-contrast case (R3) to the 300 m one at the same ground
  (thawed_1200), per band and receiver range (akp/stage_p4/s10_base_ratio.csv). Read as a power of the base, its exponent
  is k = ln(ratio) / ln(400/300), taken as zero where the ratio is below one, so the scaling never lowers a ratio.
- three scalings of every simulated row to a 460 m base from its case's own base b (400 m for R3, R4, R5; 300 m otherwise):
  "simulated" (460/b)**k with k of the row's band and range; "linear" 460/b; "squared" (460/b)**2. The last two are the
  assumed scalings of the inventory of receivers (Supplement Two); they have no support in the simulations.
- with the scaled rows, the paper's own functions of a5_late_windows: the crossing range at the pooled median of the first
  window (way valid), the bound by spreading alone at every station that s11_versions carries outward, each band's own bound
  window by window against the 1992 record, and the least Q (thresholds()).
- self-check: with no scaling, every quantity equals the committed one (a5_late_checks.csv, s11_ranges.csv).

Output: a5_late_base.csv (one row per scaling and quantity). Run: PYTHONPATH=. python akp/stage_a/a5_late_base.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from akp.stage_a import a5_late_windows as aw

HERE = aw.HERE
BASE_STATED_M = 460.0  # "1510 ft", the sides of the base in a version of 2012 (C19); s12_units.csv shows it as 460 m
BASE_OF = {"R3": 400.0, "R4": 400.0, "R5": 400.0}  # simulated base, m; every other case 300 m
DEFAULT_BASE_M = 300.0


def exponents() -> pd.DataFrame:
    b = pd.read_csv(HERE.parent / "stage_p4" / "s10_base_ratio.csv")
    return b.assign(k=np.log(b.ratio.clip(lower=1.0)) / np.log(400.0 / 300.0))[["band", "range_m", "k"]]


def scale(sim: pd.DataFrame, how: str) -> pd.DataFrame:
    base = sim.case.map(BASE_OF).fillna(DEFAULT_BASE_M)
    if how == "none":
        f = pd.Series(1.0, index=sim.index)
    elif how == "linear":
        f = BASE_STATED_M / base
    elif how == "squared":
        f = (BASE_STATED_M / base) ** 2
    else:  # simulated: the exponent of the row's band at the nearest tabled range
        ex = exponents()
        k = [float(ex[ex.band == b].iloc[(ex[ex.band == b].range_m - r).abs().argmin()].k) for b, r in zip(sim.band, sim.range_m)]
        f = (BASE_STATED_M / base) ** np.array(k)
    return sim.assign(ratio=sim.ratio * f, ratio_measured=sim.ratio_measured * f)


def quantities(sim: pd.DataFrame) -> dict:
    lv = pd.read_csv(HERE / "a5_late_levels.csv", dtype={"band": str, "lag": str})
    st, meas = aw.starts(sim), aw.starts(sim.assign(ratio=sim.ratio_measured))
    cross = aw.crossing(sim, lv, None, [])
    cross_m = aw.crossing(sim.assign(ratio=sim.ratio_measured), lv, None, [])
    pairs = aw.bound_pairs()
    bounds = np.array([aw.bound_at(km * 1e3, st)[0] for km in pairs.km])
    bounds_m = np.array([aw.bound_at(km * 1e3, meas)[0] for km in pairs.km])
    bb = pd.read_csv(HERE / "a5_late_by_band.csv", dtype={"band": str, "lag": str})
    far = sim[sim.range_m >= aw.START_M].assign(k=lambda d: d.ratio * np.sqrt(d.range_m)).groupby("band").k.max()
    bb = bb.assign(bound_band=bb.band.map(far) / np.sqrt(bb.km * 1e3))
    thw, _ = aw.thresholds(bb[bb.kind != "open waveform"], float(meas.k.max()))
    pick = lambda c, name: float(c[c.level == name].range_m.max())
    return dict(range_median_m=pick(cross, "pooled median, way valid"), range_median_measured_m=pick(cross_m, "pooled median, way valid"),
                bound_min=float(bounds.min()), bound_max=float(bounds.max()), bound_measured_min=float(bounds_m.min()),
                bound_measured_max=float(bounds_m.max()), q_least_min=float(thw.q_least.min()),
                q_least_faster_than_1_kms=float(thw[thw.speed_below_kms >= 1.0].q_least.min()))


def selfcheck(q: dict) -> None:
    c = pd.read_csv(HERE / "a5_late_checks.csv").set_index("check").value.astype(float)
    cm = pd.read_csv(HERE / "a5_late_as_measured_checks.csv").set_index("check").value.astype(float)
    s11 = pd.read_csv(HERE.parent / "stage_p4" / "s11_ranges.csv").set_index("quantity").range_m
    same = lambda a, b: abs(a - b) <= 1e-6 * max(1.0, abs(b))
    assert same(q["range_median_m"], float(s11["range_median_coda"])), "unscaled crossing range is the committed one"
    assert same(q["bound_min"], c["bound_min"]) and same(q["bound_max"], c["bound_max"]), "unscaled bounds are the committed ones"
    assert same(q["bound_measured_min"], cm["bound_min"]) and same(q["bound_measured_max"], cm["bound_max"]), "as measured too"
    near = lambda a, b: abs(a / b - 1) <= 1e-3  # the 1992 level is read back from a table written to six decimals; Q magnifies it
    assert near(q["q_least_min"], c["threshold_q_least_min"]), "unscaled least Q is the committed one"
    assert near(q["q_least_faster_than_1_kms"], c["threshold_q_least_for_faster_than_1_kms"])


def main() -> None:
    sim = aw.simulated()
    rows = []
    for how in ("none", "simulated", "linear", "squared"):
        q = quantities(scale(sim, how))
        if how == "none":
            selfcheck(q)
        rows += [dict(scaling=how, quantity=k, value=round(v, 6)) for k, v in q.items()]
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "a5_late_base.csv", index=False)
    print(out.pivot(index="quantity", columns="scaling", values="value")[["none", "simulated", "linear", "squared"]].to_string())


if __name__ == "__main__":
    main()
