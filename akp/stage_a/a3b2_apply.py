"""A3b-2 applied to the A3 r* table, by the fourth-run rule.

A3's cavity and dense rows carry linearized Born times the A3b-1 quasi-static factor, so against
A3's amplitudes the calibrated ratio is C / factor. Per contrast, the bound is the largest C / factor
(P->P) over every frequency and every far-field receiver at angles 75-90 deg (geometry-correct
rule, factor_geo) or 75-95 deg (generous rule, factor); "largest" is taken over receivers too, the
reading most favorable to detection. That scale multiplies every row of the contrast. Reported:
the scales, box coverage for both station sets, the margin (the radius at 1% coverage over the
largest bounded r*; below 1 means coverage exceeds 1%) and the A3c imaging criterion at the largest bounded r*, at 850, 1200 and 1500 m/s; and the
falsifier of the geometry-correct statement (largest C / factor_geo at the governing frequencies).
The near-void (-0.9) stands for the cavity (-1); C is 2-D, applied to 3-D bodies.
[Rewritten 2026-11-10: the earlier version implemented the third-run rule (median C
per frequency over all receivers, no angle split); its output is a3b2_run3_apply.csv.]
Writes a3b2_apply.csv. Run: python -m akp.stage_a.a3b2_apply
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from akp.stage_a.a3b4_apply import MATERIALITY, r_at_fraction
from akp.stage_a.a3c_imaging import imaging_fraction, site_matrices

HERE = Path(__file__).parent
PAIRS = {"cavity": "near_void", "dense_stiff": "dense_stiff"}  # A3 contrast -> A3b-2 case
RULES = {"geo": ((75.0, 90.0), "factor_geo", "r_star_geo_qs_km"), "generous": ((75.0, 95.0), "factor", "r_star_qs_km")}
VPS = (850.0, 1200.0, 1500.0)
GOVERNING = (6.67, 9.41)  # Hz: the governing rows' wavelengths (fourth-run definition)


def main() -> None:
    cal = pd.read_csv(HERE / "a3b2_calibration.csv")
    cal = cal[(cal["mode"] == "PP") & cal.far_field & cal.label.isin(PAIRS.values())]
    fac = pd.read_csv(HERE / "a3b_factors.csv").set_index("contrast")
    cases = pd.read_csv(HERE / "a3b_cases_corrected.csv")
    box = pd.read_csv(HERE / "a1_box_coverage.csv")
    sets = {"recorders": box.km_nearest_recorder.to_numpy(), "operating": box.km_nearest_operating.to_numpy()}
    sites = site_matrices()
    rows = []
    for rule, ((lo, hi), fcol, rcol) in RULES.items():
        for contrast, label in PAIRS.items():
            sub = cal[(cal.label == label) & (cal.theta_deg >= lo) & (cal.theta_deg <= hi)]
            i = sub.C.idxmax()
            scale = float(sub.C.max() / fac.loc[contrast, fcol])
            rows += [dict(check=f"scale_{rule}_{contrast}", value=round(scale, 3)),
                     dict(check=f"scale_{rule}_{contrast}_at_f0_hz", value=float(sub.f0_hz[i])),
                     dict(check=f"scale_{rule}_{contrast}_at_theta_deg", value=float(sub.theta_deg[i]))]
            m = cases.contrast == contrast
            cases.loc[m, f"r_ub_{rule}"] = cases.loc[m, rcol] * scale
        for vp in VPS:
            sub = cases[cases.host_vp == vp]
            r_max = float(sub[f"r_ub_{rule}"].max())
            rows.append(dict(check=f"ub_max_r_km_{rule}_vp_{int(vp)}", value=round(r_max, 3)))
            for sname, d in sets.items():
                rows.append(dict(check=f"ub_box_frac_{rule}_{sname}_vp_{int(vp)}", value=round(float((d <= r_max).mean()), 4)))
                rows.append(dict(check=f"margin_{rule}_{sname}_vp_{int(vp)}",
                                 value=round(float(r_at_fraction(d, MATERIALITY)) / r_max, 3)))
            for sname, (D, A, _) in sites.items():
                frac, valdez = imaging_fraction(D, A, r_max)
                rows.append(dict(check=f"ub_imaging_box_frac_{rule}_{sname}_vp_{int(vp)}", value=round(frac, 4)))
                rows.append(dict(check=f"ub_imaging_valdez_{rule}_{sname}_vp_{int(vp)}", value=valdez))
    # falsifier of the geometry-correct statement: at the governing frequencies, C / factor_geo at any
    # angle in 75-90 deg reaches the critical scale for the operating codes (a3b2_critical.csv)
    crit = pd.read_csv(HERE / "a3b2_critical.csv").set_index("check").value
    s_crit = min(float(crit[f"s_crit_geo_operating_vp_{v}"]) for v in (850, 1200))
    gov = cal[cal.f0_hz.isin(GOVERNING) & (cal.theta_deg >= 75.0) & (cal.theta_deg <= 90.0)]
    for contrast, label in PAIRS.items():
        ratio = float(gov[gov.label == label].C.max() / fac.loc[contrast, "factor_geo"])
        rows += [dict(check=f"falsifier_max_C_over_factor_geo_{contrast}", value=round(ratio, 3)),
                 dict(check=f"falsifier_met_{contrast}", value=int(ratio >= s_crit))]
    rows.append(dict(check="falsifier_s_crit_min_operating", value=round(s_crit, 3)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "a3b2_apply.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
