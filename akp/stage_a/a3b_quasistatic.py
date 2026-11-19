"""A3b-1: exact long-wavelength (quasi-static) correction of A3a's Born amplitudes.

For a sphere with ka << 1 the exact scattering of any contrast is the Born expression with the
elastic perturbations replaced by Eshelby strain-concentrated values (Gubernatis et al. 1977;
Eshelby 1957): dK* = dK / (1 + a dK/K), a = 3K/(3K+4mu); dmu* = dmu / (1 + b dmu/mu),
b = 6(K+2mu)/(5(3K+4mu)); drho unchanged. The exact perturbations (not linearized) of the
inclusion enter. A3a used linearized Born (swe.scattering) with relative perturbation r of vp, vs
and density. Factor = max_theta |pattern_qs| / max_theta |pattern_born_linear| per contrast. Then
rescale A3a's r* (proportional to amplitude) and recompute the covered box fraction.
Writes a3b_factors.csv and a3b_checks.csv. Run: python -m akp.stage_a.a3b_quasistatic
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_a import a3_detectability as a3
from swe.scattering import born_radiation_pattern

HERE = Path(__file__).parent
THETAS = np.linspace(0.0, np.pi, 721)
RAYLEIGH_KA = 0.5


def moduli(vp, vs, rho):
    mu = rho * vs**2
    return rho * vp**2 - 4.0 / 3.0 * mu, mu  # K, mu


def qs_pattern(theta, *, vp, vs, rho, rel):
    """Quasi-static exact P->P pattern for a sphere whose vp, vs, rho are all scaled by (1 + rel)."""
    k0, mu0 = moduli(vp, vs, rho)
    k1, mu1 = moduli(vp * (1 + rel), vs * (1 + rel), rho * (1 + rel))
    dk, dmu, drho = k1 - k0, mu1 - mu0, rho * rel
    alpha = 3 * k0 / (3 * k0 + 4 * mu0)
    beta = 6 * (k0 + 2 * mu0) / (5 * (3 * k0 + 4 * mu0))
    dk_s = dk / (1 + alpha * dk / k0)
    dmu_s = dmu / (1 + beta * dmu / mu0)
    dlam_s = dk_s - 2.0 / 3.0 * dmu_s
    c = np.cos(theta)
    return (drho / rho) * c - (dlam_s + 2.0 * dmu_s * c**2) / (rho * vp**2)


def main() -> None:
    vp, rho = 3000.0, a3.DENSITY
    vs = vp / a3.VP_VS
    rows = []
    geo = (THETAS >= a3.GEO_THETA[0] - 1e-12) & (THETAS <= a3.GEO_THETA[1] + 1e-12)
    for cname, rel in a3.CONTRASTS.items():
        born = np.abs(born_radiation_pattern(THETAS, vp=vp, vs=vs, density=rho, dvp=rel * vp, dvs=rel * vs, drho=rel * rho))
        qs = np.abs(qs_pattern(THETAS, vp=vp, vs=vs, rho=rho, rel=rel))
        rows.append(dict(contrast=cname, rel=rel, born_max=round(float(born.max()), 4), qs_max=round(float(qs.max()), 4),
                         factor=round(float(qs.max() / born.max()), 4),
                         theta_qs_max_deg=round(float(np.degrees(THETAS[np.argmax(qs)])), 1),
                         factor_geo=round(float(qs[geo].max() / born[geo].max()), 4)))
    fac = pd.DataFrame(rows)
    fac.to_csv(HERE / "a3b_factors.csv", index=False)

    cases = pd.read_csv(HERE / "a3_cases.csv")
    box = pd.read_csv(HERE / "a1_box_coverage.csv")
    f = cases.contrast.map(fac.set_index("contrast").factor)
    fg = cases.contrast.map(fac.set_index("contrast").factor_geo)
    cases["r_star_qs_km"] = cases.r_star_km * f
    cases["r_star_geo_qs_km"] = cases.r_star_geo_km * fg
    rep_d, op_d = box.km_nearest_recorder.to_numpy(), box.km_nearest_operating.to_numpy()
    cases["box_frac_qs"] = [float((rep_d <= r).mean()) for r in cases.r_star_qs_km]
    cases["box_frac_qs_operating"] = [float((op_d <= r).mean()) for r in cases.r_star_qs_km]
    cases["box_frac_geo"] = [float((rep_d <= r).mean()) for r in cases.r_star_geo_qs_km]
    cases["box_frac_geo_operating"] = [float((op_d <= r).mean()) for r in cases.r_star_geo_qs_km]
    cases.to_csv(HERE / "a3b_cases_corrected.csv", index=False)
    ok = cases[cases.ka < RAYLEIGH_KA]
    checks = [dict(check=f"factor_{r.contrast}", value=r.factor, detail=f"qs {r.qs_max} / born {r.born_max}") for r in fac.itertuples()]
    checks.append(dict(check="rayleigh_cases", value=len(ok), detail="ka < 0.5: correction exact"))
    checks.append(dict(check="max_box_frac_qs_rayleigh", value=round(float(ok.box_frac_qs.max()), 4) if len(ok) else 0.0, detail=""))
    checks.append(dict(check="max_r_star_qs_km_rayleigh", value=round(float(ok.r_star_qs_km.max()), 3) if len(ok) else 0.0, detail=""))
    for v in a3.HOST_VP:
        sub = cases[cases.host_vp == v]
        checks.append(dict(check=f"max_box_frac_qs_vp_{int(v)}", value=round(float(sub.box_frac_qs.max()), 4), detail="all ka (correction exact only for ka < 0.5)"))
        checks.append(dict(check=f"max_r_star_qs_km_vp_{int(v)}", value=round(float(sub.r_star_qs_km.max()), 3), detail=""))
        checks.append(dict(check=f"max_box_frac_qs_operating_vp_{int(v)}", value=round(float(sub.box_frac_qs_operating.max()), 4), detail="recorders + codes operating at the test"))
        checks.append(dict(check=f"max_r_star_geo_qs_km_vp_{int(v)}", value=round(float(sub.r_star_geo_qs_km.max()), 3), detail="theta 75-90 deg"))
        checks.append(dict(check=f"max_box_frac_geo_vp_{int(v)}", value=round(float(sub.box_frac_geo.max()), 4), detail=""))
        checks.append(dict(check=f"max_box_frac_geo_operating_vp_{int(v)}", value=round(float(sub.box_frac_geo_operating.max()), 4), detail=""))
    with (HERE / "a3b_checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "value", "detail"])
        w.writeheader()
        w.writerows(checks)
    print(fac.to_string(index=False))
    for c in checks:
        print(c)


def _selfcheck() -> None:
    vp, vs, rho = 3000.0, 3000.0 / 1.8, 2000.0
    # weak contrast: quasi-static -> linear Born
    born = born_radiation_pattern(THETAS, vp=vp, vs=vs, density=rho, dvp=0.001 * vp, dvs=0.001 * vs, drho=0.001 * rho)
    qs = qs_pattern(THETAS, vp=vp, vs=vs, rho=rho, rel=0.001)
    assert np.allclose(qs, born, rtol=0.01, atol=1e-6), "weak-contrast limit"
    # void limit: dK* = -K (3K + 4 mu) / (4 mu)
    k0, mu0 = moduli(vp, vs, rho)
    alpha = 3 * k0 / (3 * k0 + 4 * mu0)
    assert np.isclose(-k0 / (1 - alpha), -k0 * (3 * k0 + 4 * mu0) / (4 * mu0))


if __name__ == "__main__":
    _selfcheck()
    main()
