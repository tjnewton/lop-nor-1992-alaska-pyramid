"""P3 G1a (EM): offset-averaged detection probability and swath width per class, from p3_em_forward.csv.

Same rule as p3_g1a_mag (single nearest line, d uniform on [0, spacing / 2], SNR interpolated in
log space between offsets and 0 beyond the largest, P_d = Phi(SNR - z)). Only the conservative
statistic is used: the best single frequency, in the along-line noise (cols_ns). Classes carry
their contrasts in their definitions (p3_em_forward), so no contrast scaling applies. A class passes
G1a for a geometry when its offset-averaged P_d >= G1_FLOOR. Writes p3_g1a_em.csv.
Run: python -m akp.stage_p3.p3_g1a_em
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from akp.stage_p3.p3_g1a_mag import N_D, snr_at
from akp.stage_p3.p3_mag_forward import G1_FLOOR, OFFSETS_M

HERE = Path(__file__).parent
SPACING_M = 400.0  # the DGGS helicopter EM standard (P2c FINE_KM)


def main() -> None:
    f = pd.read_csv(HERE / "p3_em_forward.csv")
    f = f[f.noise.str.startswith("cols_ns")]
    keys = ["cls", "rho_body", "rho_host", "base_m", "top_m", "clearance_m", "r_fp_m", "noise"]
    off = np.array(OFFSETS_M)
    d = np.linspace(0.0, SPACING_M / 2, N_D)
    rows = []
    for k, g in f.groupby(keys):
        g = g.sort_values("offset_m")
        assert np.allclose(g.offset_m.to_numpy(), off)
        p = norm.cdf(snr_at(off, g.snr_best.to_numpy(), d) - float(g.z.iloc[0]))
        ok = p >= G1_FLOOR
        rows.append(dict(zip(keys, k), spacing_m=SPACING_M, pd_single=round(float(p.mean()), 4),
                         half_swath_m=float(d[ok].max()) if ok.any() else 0.0, passes=bool(p.mean() >= G1_FLOOR)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_g1a_em.csv", index=False)
    print(out.groupby(["cls", "rho_host", "noise"]).agg(pd_max=("pd_single", "max"), pd_median=("pd_single", "median"),
                                                        n_pass=("passes", "sum"), n=("passes", "size")).to_string())


if __name__ == "__main__":
    main()
