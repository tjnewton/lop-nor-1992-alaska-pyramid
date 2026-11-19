"""A3a: could the 1992 recordings have registered scattering from the claimed body?

Grid (fixed before results): faceted square-based body, base 100-500 m,
height 50-300 m (steelman), cavity (all perturbations -100%) or dense/stiff (+100%) contrast,
frequencies 0.5-4 Hz (A2 bands), host P velocity swept (results are functions of it).

For each case: Born scattered/direct amplitude S(r) = k^2 V F(q) p(theta) / (4 pi r), maximized over
scattering angle (|p F| with the body's voxel form factor), from swe.scattering. Detection margin
M(r) = S(r) / C, with C = the IU.COL 1992 clutter ratio at lag 5-15 s in the band (A2), one recorder
(stacking co-located sensors gains nothing against signal-generated clutter). Two angle rules:
the maximum over all angles (generous), and the maximum over theta 75-90 deg, the direction toward
surface stations kilometers away (geometry-correct). Burial depth does not enter this
full-space far-field model.
Detection radius r*: M(r*) = 1. Coverage: fraction of the mckinley_nome box grid (A1) whose nearest
recorder lies within r*, and whether Valdez (C9) lies within r* of its nearest recorder.
Writes a3_cases.csv (one row per case) and a3_checks.csv. Run: python -m akp.stage_a.a3_detectability
"""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from swe.scattering import born_radiation_pattern, scattering_wavenumber, voxel_form_factor

HERE = Path(__file__).parent
BASES = [100.0, 230.0, 500.0]  # m (steelman bounds and the Giza-scale midpoint)
HEIGHTS = [50.0, 168.0, 300.0]  # m (168 m = the registered 550 ft, C3)
CONTRASTS = {"cavity": -1.0, "dense_stiff": +1.0}  # relative perturbation of vp, vs, density
BANDS = {"0.5-1": 1.0, "1-2": 2.0, "2-4": 4.0}  # band -> upper frequency used (most favorable)
HOST_VP = [300.0, 500.0, 850.0, 1000.0, 1200.0, 1500.0, 2000.0, 3000.0, 4500.0, 6000.0]  # m/s, swept
# (850 and 1200 m/s added 2026-11-09, the A3-0 sensitivity case and floor)
GEO_THETA = (np.radians(75.0), np.radians(90.0))  # toward surface stations km away
VP_VS = 1.8  # host vp/vs (Poisson solid ~1.73; sediments higher)
DENSITY = 2000.0  # kg/m3 host (cancels in the relative pattern)
THETAS = np.linspace(0.0, np.pi, 181)  # 1 deg; includes 75 and 90 (2026-11-10: was 2 deg from 0, so 76)
N_VOX = 40


def square_based_body(base: float, height: float):
    """inside(x, y, z) for a faceted square-based apex body centered so its centroid is near the origin."""
    zc = height / 4.0  # centroid height of a square-based apex body above its base

    def inside(x, y, z):
        zz = z + zc  # base at zz = 0, apex at zz = height
        half = 0.5 * base * (1.0 - zz / height)
        return (zz >= 0) & (zz <= height) & (np.abs(x) <= half) & (np.abs(y) <= half)
    pad = 0.05  # voxel bounds must not touch the body (swe.scattering.voxel_form_factor rejects clipping)
    bounds = [(-(0.5 + pad) * base, (0.5 + pad) * base)] * 2 + [(-zc - pad * height, height - zc + pad * height)]
    return inside, bounds


def max_pattern(freq, vp, base, height, rel, theta_range=(0.0, np.pi)):
    """max over theta in theta_range of |pattern * F(q)| for P->P (incident along +z, scattered in x-z)."""
    vs = vp / VP_VS
    k = 2 * np.pi * freq / vp
    inside, bounds = square_based_body(base, height)
    th = THETAS[(THETAS >= theta_range[0] - 1e-12) & (THETAS <= theta_range[1] + 1e-12)]
    q_vec = np.column_stack([k * np.sin(th), np.zeros_like(th), k * (np.cos(th) - 1.0)])
    f, vol = voxel_form_factor(inside, bounds=bounds, q_vector=q_vec, n=N_VOX)
    pat = born_radiation_pattern(th, vp=vp, vs=vs, density=DENSITY, dvp=rel * vp, dvs=rel * vs, drho=rel * DENSITY)
    mag = np.abs(pat * f)
    i = int(np.argmax(mag))
    assert np.allclose(scattering_wavenumber(th, k_in=k, k_out=k), np.linalg.norm(q_vec, axis=1))
    return float(mag[i]), float(th[i]), vol, k


def main() -> None:
    cl = pd.read_csv(HERE / "a2_clutter.csv")
    c92 = cl[(cl.kind == "test1992") & (cl.estimator == "hilbert") & (cl.lag == "5-15")].set_index("band").ratio
    box = pd.read_csv(HERE / "a1_box_coverage.csv")
    st = pd.read_csv(HERE / "a1_stations.csv")
    valdez_r = float(st[st.recorded_test].km_valdez.min())
    rows = []
    for (base, height, (cname, rel), (band, freq), vp) in product(BASES, HEIGHTS, CONTRASTS.items(), BANDS.items(), HOST_VP):
        pmax, theta, vol, k = max_pattern(freq, vp, base, height, rel)
        pgeo, theta_geo, _, _ = max_pattern(freq, vp, base, height, rel, theta_range=GEO_THETA)
        clutter = float(c92[band])
        # M(r) = (k^2 V pmax / (4 pi r)) / C  ->  r* = k^2 V pmax / (4 pi C)
        r_star_m = k**2 * vol * pmax / (4 * np.pi * clutter)
        r_geo_m = k**2 * vol * pgeo / (4 * np.pi * clutter)
        a_eq = (3 * vol / (4 * np.pi)) ** (1 / 3)
        # Fraunhofer distance 2 D^2 / lambda on the body's largest chord (base diagonal or apex to a base
        # corner); 2026-11-10: was D = 2 a_eq, which understates it
        d_max = max(base * np.sqrt(2.0), np.sqrt(base**2 / 2 + height**2))
        fraunhofer = k * d_max**2 / np.pi
        far = lambda r: bool(k * r > 10.0 and r > fraunhofer)  # noqa: E731
        rows.append(dict(base_m=base, height_m=height, contrast=cname, band=band, freq_hz=freq, host_vp=vp,
                         volume_m3=round(vol), ka=round(k * a_eq, 3), theta_max_deg=round(np.degrees(theta), 1),
                         pattern_x_ff=round(pmax, 4), clutter_c92=round(clutter, 4), r_star_km=round(r_star_m / 1e3, 3),
                         r_star_far_field=far(r_star_m), fraunhofer_km=round(fraunhofer / 1e3, 3),
                         box_frac_within_r_star=round(float((box.km_nearest_recorder <= r_star_m / 1e3).mean()), 4),
                         valdez_within_r_star=bool(valdez_r <= r_star_m / 1e3),
                         pattern_geo=round(pgeo, 4), theta_geo_deg=round(np.degrees(theta_geo), 1),
                         r_star_geo_km=round(r_geo_m / 1e3, 3), r_star_geo_far_field=far(r_geo_m)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "a3_cases.csv", index=False)

    # breakeven host velocity per (body, contrast, band): lowest vp at which the whole box is beyond r*
    worst = df.sort_values("r_star_km", ascending=False).iloc[0]
    checks = [
        dict(check="n_cases", value=len(df), detail=""),
        dict(check="max_r_star_km", value=float(worst.r_star_km),
             detail=f"base {worst.base_m} h {worst.height_m} {worst.contrast} {worst.band} Hz vp {worst.host_vp}"),
        dict(check="max_box_frac_within_r_star", value=float(df.box_frac_within_r_star.max()), detail=""),
        dict(check="cases_with_any_box_coverage", value=int((df.box_frac_within_r_star > 0).sum()), detail=""),
        dict(check="cases_valdez_within_r_star", value=int(df.valdez_within_r_star.sum()), detail=""),
        dict(check="cases_r_star_far_field", value=int(df.r_star_far_field.sum()), detail="cases where r* is in Born's far field"),
        dict(check="cases_r_star_geo_far_field", value=int(df.r_star_geo_far_field.sum()),
             detail="cases where the geometry-correct r* is in Born's far field"),
    ]
    checks.append(dict(check="cases_rayleigh", value=int((df.ka < 0.5).sum()), detail="ka < 0.5"))
    checks.append(dict(check="cases_weak_contrast", value=0, detail="every case is +-100%: Born weak-contrast flag off"))
    for vp in HOST_VP:
        sub = df[df.host_vp == vp]
        checks.append(dict(check=f"max_box_frac_vp_{int(vp)}", value=float(sub.box_frac_within_r_star.max()), detail=""))
        checks.append(dict(check=f"max_r_star_km_vp_{int(vp)}", value=float(sub.r_star_km.max()), detail=""))
        checks.append(dict(check=f"min_ka_at_max_r_star_vp_{int(vp)}", value=float(sub.sort_values("r_star_km").ka.iloc[-1]), detail=""))
    with (HERE / "a3_checks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "value", "detail"])
        w.writeheader()
        w.writerows(checks)
    for c in checks:
        print(c)


if __name__ == "__main__":
    inside, bounds = square_based_body(100.0, 60.0)
    assert inside(np.array(0.0), np.array(0.0), np.array(0.0)) and not inside(np.array(49.0), np.array(49.0), np.array(30.0))
    main()
