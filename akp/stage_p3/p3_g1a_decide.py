"""P3 G1a decision per class and sensor, from the forward bounds and cited contrasts.

Rule: a class x sensor combination goes ahead if the M1 bound allows P_d >= G1_FLOOR
for some base, top and line offset, at the class's typical contrast, in the along-line median
clutter (cols_ns_median). This is checked separately for low-flown helicopter surveys (clearance
<= HELI_MAX_M) and for high fixed-wing surveys (FIXED_WING_M).

Magnetic contrasts (susceptibility minus a sedimentary host, Telford et al. 1990 Table 3.1,
doi:10.1017/cbo9781139167932, PDF p. 94, averages):
- T1, felsic stone: granite 2.5e-3 and average acidic igneous 8e-3;
- T2, mafic: average basic igneous 25e-3 and basalt 70e-3;
- host: the average of 48 sedimentary rocks, 0.9e-3 (unconsolidated sands and silts are about
  1e-5 to 1e-3: Hinze et al. 2012, doi:10.1017/cbo9780511843129, PDF p. 276).
Induced magnetization only. Basalt remanence (Q >= 10: Hinze PDF p. 270) can raise or cancel the
anomaly, so G1b's template bank must span the magnetization direction.

EM classes carry their contrasts in p3_em_forward. T4's hosts (100 and 300 ohm m) cover thawed
ground and frozen fine-grained ground, which is also about 100 ohm m or less (doi:10.1029/2011GL050079,
PDF p. 3). Writes p3_g1a_decision.csv (cls, sensor, platform, contrast, pd_best, passes, note).
Run: python -m akp.stage_p3.p3_g1a_decide
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from akp.stage_p3.p3_mag_forward import DK_REF, G1_FLOOR

HERE = Path(__file__).parent
HOST = 0.9e-3
MAG_CLASSES = {"T1": (2.5e-3, 8e-3), "T2": (25e-3, 70e-3)}
HELI_MAX_M, FIXED_WING_M = 60.0, 300.0
NOISE = "cols_ns_median"


def main() -> None:
    mf = pd.read_csv(HERE / "p3_mag_forward.csv").query("noise == @NOISE and spacing_m == 400")
    rows = []
    for cls, (lo, hi) in MAG_CLASSES.items():
        for label, kappa in (("typical_low", lo), ("typical_high", hi)):
            dk = kappa - HOST
            for platform, sel in (("helicopter", mf.clearance_m <= HELI_MAX_M), ("fixed_wing", mf.clearance_m == FIXED_WING_M)):
                g = mf[sel]
                pd_ = pd.Series(norm.cdf(g.snr_ref * dk / DK_REF - g.z), index=g.index)
                best = g.loc[pd_.idxmax()]
                rows.append(dict(cls=cls, sensor="magnetics", platform=platform, contrast=f"{label} dk={dk:.4f} SI",
                                 pd_best=round(float(pd_.max()), 4), passes=bool(pd_.max() >= G1_FLOOR),
                                 note=f"best at base {best.base_m:.0f} m, top {best.top_m:.0f} m, clearance {best.clearance_m:.0f} m, offset {best.offset_m:.0f} m"))
    ef = pd.read_csv(HERE / "p3_em_forward.csv").query("noise == @NOISE")
    for (cls, host), g in ef.groupby(["cls", "rho_host"]):
        pd_ = pd.Series(norm.cdf(g.snr_best - g.z), index=g.index)
        best = g.loc[pd_.idxmax()]
        rows.append(dict(cls=cls, sensor="helicopter_fdem", platform="helicopter",
                         contrast=f"body {best.rho_body:.0f} ohm m in host {host:.0f} ohm m",
                         pd_best=round(float(pd_.max()), 4), passes=bool(pd_.max() >= G1_FLOOR),
                         note=f"best at base {best.base_m:.0f} m, top {best.top_m:.0f} m, clearance {best.clearance_m:.0f} m, offset {best.offset_m:.0f} m, {best.best_freq_hz:.0f} Hz"))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_g1a_decision.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
