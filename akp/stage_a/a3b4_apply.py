"""A3b-4 coverage thresholds and the strong-contrast correction.

r*_Rg = (E / C)^2, so an amplitude factor s scales r*_Rg by s^2. This writer reports, per host
velocity, the distance at which box coverage reaches the 1% materiality threshold for each
station set, and the amplitude factor s_crit that would carry the largest r*_Rg there. When
a3b4_strong.csv exists (the FD strong-contrast stage), it also applies C_Rg = FD / Born-Rg in
place of the A3b-1 factor carried by a3b4_cases.csv and recomputes coverage.
Writes a3b4_apply.csv. Run: python -m akp.stage_a.a3b4_apply
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MATERIALITY = 0.01
PAIRS = {"cavity": "near_void", "dense_stiff": "dense_stiff"}


def r_at_fraction(d: np.ndarray, frac: float) -> float:
    """Smallest radius at which the fraction of box points within it of a site reaches frac."""
    return float(np.sort(d)[int(np.ceil(frac * len(d))) - 1])


def r_with_q(r_corr_km: float, freq: float, c_r: float, q: float) -> float:
    """Detection radius (km) with Rg attenuation: sqrt(r*/r) exp(-pi f r / (Q c_R)) = 1."""
    from scipy.optimize import brentq
    r0 = r_corr_km * 1e3
    if not np.isfinite(q):
        return r_corr_km

    def g(r):
        return 0.5 * np.log(r0 / r) - np.pi * freq * r / (q * c_r)

    return float(brentq(g, 1e-6 * r0, r0)) / 1e3


def q_critical(cases: pd.DataFrame, sets: dict) -> list[dict]:
    """Diagnostic (added after the stage's numbers): the Rg Q at which coverage falls to 1%."""
    from akp.stage_a.a3_detectability import VP_VS
    from akp.stage_a.a3b4_rg import rayleigh_speed
    rows = []
    for vp in (1200.0,):
        sub = cases[(cases.host_vp == vp) & np.isfinite(cases.r_corr_km)]
        c_r = rayleigh_speed(vp, vp / VP_VS)
        for name, d in sets.items():
            def cover(q):
                return max(float((d <= r_with_q(r, f, c_r, q)).mean()) for r, f in zip(sub.r_corr_km, sub.freq_hz))
            lo, hi = 1.0, 1e6
            if cover(hi) < MATERIALITY:
                continue
            for _ in range(60):  # bisection in log Q
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if cover(mid) < MATERIALITY else (lo, mid)
            rows.append(dict(check=f"q_crit_{name}_vp_{int(vp)}", value=round(float(hi), 1)))
    return rows


REALISTIC_Q = {"generous_100": 100.0, "sediment_50": 50.0, "sediment_20": 20.0}  # a3_0_PROPERTIES.md


def realistic_q(cases: pd.DataFrame, sets: dict) -> list[dict]:
    """The realistic-Q case (defined in A3b-4; Q values set in A3-0 before this ran)."""
    from akp.stage_a.a3_detectability import VP_VS
    from akp.stage_a.a3b4_rg import rayleigh_speed
    rows = []
    for vp in (1200.0,):
        sub = cases[(cases.host_vp == vp) & np.isfinite(cases.r_corr_km)]
        c_r = rayleigh_speed(vp, vp / VP_VS)
        for tag, q in REALISTIC_Q.items():
            r = [r_with_q(rc, f, c_r, q) for rc, f in zip(sub.r_corr_km, sub.freq_hz)]
            rows.append(dict(check=f"rq_max_r_km_{tag}_vp_{int(vp)}", value=round(max(r), 3)))
            for name, d in sets.items():
                frac = max(float((d <= x).mean()) for x in r)
                rows.append(dict(check=f"rq_box_frac_{name}_{tag}_vp_{int(vp)}", value=round(frac, 4)))
    return rows


def main() -> None:
    cases = pd.read_csv(HERE / "a3b4_cases.csv")
    box = pd.read_csv(HERE / "a1_box_coverage.csv")
    sets = {"recorders": box.km_nearest_recorder.to_numpy(), "operating": box.km_nearest_operating.to_numpy()}
    rows = []
    for name, d in sets.items():
        rows.append(dict(check=f"r_1pct_km_{name}", value=round(r_at_fraction(d, MATERIALITY), 2)))
    for vp in (850.0, 1200.0, 1500.0):
        r_max = float(cases[cases.host_vp == vp].r_star_rg_km.max())
        for name, d in sets.items():
            s = np.sqrt(r_at_fraction(d, MATERIALITY) / r_max)
            rows.append(dict(check=f"s_crit_{name}_vp_{int(vp)}", value=round(float(s), 3)))
    strong = HERE / "a3b4_strong.csv"
    if strong.exists():
        from akp.stage_a.a3b4_rg import qs_factor  # the same factor the sweep applied
        st = pd.read_csv(strong)
        for contrast, label in PAIRS.items():
            qs = qs_factor(contrast)
            for vp in sorted(st.host_vp.unique()):
                c = float(st[(st.label == label) & (st.host_vp == vp)].C_rg.median())
                if not np.isfinite(c):
                    raise ValueError(f"no strong-stage C_Rg for {label} at {vp} m/s")
                rows.append(dict(check=f"C_rg_{label}_vp_{int(vp)}", value=round(c, 3)))
                m = (cases.contrast == contrast) & (cases.host_vp == vp)
                cases.loc[m, "r_corr_km"] = cases.loc[m, "r_star_rg_km"] * (c / qs) ** 2
        for vp in sorted(st.host_vp.unique()):
            sub = cases[cases.host_vp == vp]
            rows.append(dict(check=f"max_r_corr_km_vp_{int(vp)}", value=round(float(sub.r_corr_km.max()), 3)))
            for name, d in sets.items():
                frac = max(float((d <= r).mean()) for r in sub.r_corr_km)
                rows.append(dict(check=f"max_box_frac_corr_{name}_vp_{int(vp)}", value=round(frac, 4)))
        rows += q_critical(cases, sets)
        rows += realistic_q(cases, sets)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "a3b4_apply.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
