"""P3 G1a (magnetics): detection probability against contrast, averaged over line offset; swath widths.

From p3_mag_forward.csv, the matched-filter SNR at DK_REF scales linearly with the contrast dk.
SNR is interpolated in log space between OFFSETS_M, and set to 0 beyond the largest offset (a
conservative choice). A body lies at offset d from its nearest line and at spacing - d from the
next line, with d uniform on [0, spacing / 2]. Two statistics:
- single: the nearer line alone, P_d = Phi(SNR(d) - z);
- joint: both lines with independent noise, SNR = sqrt(SNR(d)^2 + SNR(spacing - d)^2). This is
  optimistic, because clutter is correlated across lines at long wavelengths.
For each geometry, noise model, spacing and dk in DK_GRID it writes:
- the offset-averaged P_d (single and joint);
- half_swath_m: the largest d with single-line P_d >= G1_FLOOR. G0 counts the strip within this
  distance of a flight line as searched for that contrast.
Writes p3_g1a_mag.csv. The G1a decision per class applies each class's cited
contrast range (p3_PROPERTIES.md) to this table. Run: python -m akp.stage_p3.p3_g1a_mag
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from akp.stage_p3.p3_mag_forward import DK_REF, G1_FLOOR, OFFSETS_M

HERE = Path(__file__).parent
DK_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)  # SI
N_D = 201


def snr_at(off: np.ndarray, snr_ref: np.ndarray, d: np.ndarray) -> np.ndarray:
    v = np.interp(d, off, np.log(np.maximum(snr_ref, 1e-12)))
    return np.where(d <= off.max(), np.exp(v), 0.0)


def main() -> None:
    f = pd.read_csv(HERE / "p3_mag_forward.csv")
    keys = ["base_m", "top_m", "turn_deg", "clearance_m", "noise", "spacing_m"]
    off = np.array(OFFSETS_M)
    rows = []
    for k, g in f.groupby(keys):
        g = g.sort_values("offset_m")
        assert np.allclose(g.offset_m.to_numpy(), off)
        z, sp = float(g.z.iloc[0]), float(g.spacing_m.iloc[0])
        d = np.linspace(0.0, sp / 2, N_D)
        s_near = snr_at(off, g.snr_ref.to_numpy(), d)
        s_far = snr_at(off, g.snr_ref.to_numpy(), sp - d)
        for dk in DK_GRID:
            a = dk / DK_REF
            pd_single = norm.cdf(a * s_near - z)
            pd_joint = norm.cdf(a * np.hypot(s_near, s_far) - z)
            ok = pd_single >= G1_FLOOR
            rows.append(dict(zip(keys, k), dk_si=dk, pd_single=round(float(pd_single.mean()), 4),
                             pd_joint=round(float(pd_joint.mean()), 4),
                             half_swath_m=float(d[ok].max()) if ok.any() else 0.0))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_g1a_mag.csv", index=False)
    s = out[(out.turn_deg == 0) & (out.top_m == out.top_m.max()) & (out.noise == "cols_ns_median") & (out.spacing_m == 400)]
    print(s.pivot_table(index=["base_m", "clearance_m"], columns="dk_si", values="pd_single").round(2).to_string())


if __name__ == "__main__":
    main()
