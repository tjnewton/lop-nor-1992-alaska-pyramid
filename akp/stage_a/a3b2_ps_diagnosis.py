"""A3b-2 fourth run: why the weak check fails on P->S at 75 deg (diagnosis, no new simulation).

Gate (i-a) compares FD against linearized 2-D Born for a body whose vp, vs and density are all
scaled by 1 + rel (rel = 0.05). The Born P->S bracket, sin(theta) [drho/rho - 2 dmu cos(theta) /
(rho vp vs)], has a null near 72.5 deg for vp/vs = 1.8, so at 75 deg it is the small difference of
two O(1) terms, and any O(rel) correction to either term is amplified.

Test: the exact 2-D series (a3b2_cylinder.py, now with P->S) for a circle of the weak triangle's
area and contrast gives the true non-Born amplitude at the same ka. If the FD C tracks the exact
C, the solver is right and the failure comes from the reference (linear Born) at a finite contrast.
Also reported: the share of the shift that comes from linearizing the parameter map alone (Born
with the exact Lame perturbations), the exact C for a ten-times-weaker contrast, and the exact C of
the odd part, (s(+rel) - s(-rel)) / 2. The odd part removes every even order in rel, so it equals
first-order Born up to O(rel^3). That is the candidate weak check for a fifth run.
Finite distance (added later): the same exact quantities evaluated at each receiver's
distance (H_n(k r), not its asymptote) over the far-field Born the gate uses, plus the near-field
factor of linear Born itself (the exact series at rel = 1e-4, at r).
Late arrival: from the archived weak-check and background traces (needs $AKP_DATA), the
delay and size of the late body-tied P->S arrival relative to the direct P->S, the late arrivals
in the background run at the body center, and the incident amplitude's window dependence.
Fifth-run reference: validation of a3b2_numerical.born_2d_finite (large-r limit; a rasterized
circle against the exact linear series at finite r), the triangle's near-field factor, and the
predictions for the fifth run: the exact circle at -rel (the single-sided -5% check).
Writes a3b2_ps_diagnosis.csv (per angle) and a3b2_ps_diagnosis_checks.csv.
Run: python -m akp.stage_a.a3b2_ps_diagnosis
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import hankel1

from akp.stage_a.a3b2_cylinder import c_exact, coefficients
from akp.stage_a.a3b2_numerical import RHO, VP, VS, WEAK, WEAK_TOL
from swe.scattering import born_radiation_pattern

HERE = Path(__file__).parent
WEAKER = 0.1  # factor on rel for the weaker-contrast prediction


def exact_pert_born_ratio(theta, rel, mode):
    """|Born with exact Lame perturbations| / |linear Born|: the parameter-map linearization alone."""
    # exact dmu / mu = (1 + rel)^3 - 1 = 2 dvs/vs + rel to first order, so dvs/vs = ((1+rel)^3 - 1 - rel) / 2
    d = ((1 + rel) ** 3 - 1 - rel) / 2
    ex = born_radiation_pattern(theta, vp=VP, vs=VS, density=RHO, dvp=d * VP, dvs=d * VS, drho=rel * RHO, mode=mode)
    lin = born_radiation_pattern(theta, vp=VP, vs=VS, density=RHO, dvp=rel * VP, dvs=rel * VS, drho=rel * RHO,
                                 mode=mode)
    return np.abs(ex) / np.abs(lin)


LINEAR = 1e-4  # contrast at which the exact series is linear Born to within rounding


def exact_complex(*, f0, a, theta, rel, mode, r=None):
    """Complex exact sum, as c_exact: far field (Green's factor dropped) or at distance r."""
    A, B, kp, ks = coefficients(freq=f0, a=a, vp=VP, vs=VS, rho=RHO,
                                inclusion=(VP * (1 + rel), VS * (1 + rel), RHO * (1 + rel)))
    n = np.arange(len(A))
    k = kp if mode == "PP" else ks
    radial = (-1j) ** n if r is None else hankel1(n, k * r) * np.exp(-1j * k * r)
    if mode == "PP":
        return complex(np.sum(A * radial * np.cos(n * theta)))
    return complex(ks / kp * np.sum(B * radial * np.sin(n * theta)))


def late_arrivals(f0):
    """Checks on late arrivals in the archived fourth-run weak check (envelope peaks)."""
    from scipy.signal import hilbert

    from akp.stage_a import a3b2_numerical as n
    g = n.setup(f0)
    tx, tz = n.load((f0, n.WEAK, 1))
    bx, bz = n.load((f0, None, 1))
    t = np.arange(len(bz)) * g["dt"]
    sx, sz = tx - bx, tz - bz
    th, _ = n.receiver_dirs()
    out = []
    for ti in (0, list(np.degrees(th).round(1)).index(90.0)):
        for ri, r in enumerate(g["rs"]):
            i = (ri * len(th) + ti) * 4
            curl = (sz[:, i] - sz[:, i + 1] - (sx[:, i + 2] - sx[:, i + 3])) / (2 * g["dx"])
            env = np.abs(hilbert(curl))
            t_ps = g["t_inc"] + r / VS
            direct = env[np.abs(t - t_ps) <= 0.5 / f0].max()
            j = int(np.argmax(env * (t > t_ps + 2.0 / f0)))
            tag = f"PS_{np.degrees(th[ti]):.0f}_r{int(r)}"
            out += [dict(check=f"late_delay_s_{tag}", value=float(t[j] - t_ps)),
                    dict(check=f"late_rel_{tag}", value=float(env[j] / direct))]
    env = np.abs(hilbert(bz[:, -1]))
    pk = env[np.abs(t - g["t_inc"]) <= 0.5 / f0].max()
    late = t > g["t_inc"] + 2.0 / f0
    j = int(np.argmax(env * late))
    out += [dict(check="bg_late_max_rel_body_center", value=float(env[j] / pk)),
            dict(check="bg_late_delay_s_body_center", value=float(t[j] - g["t_inc"])),
            dict(check="bottom_edge_ghost_delay_s", value=float(2 * (g["lz"] - g["zs"]) / VP))]
    a1 = n.windowed_amp(bz[:, -1], f0, g["t_inc"], g["dt"], n.WIN_HALF_PERIODS / f0)
    a4 = n.windowed_amp(bz[:, -1], f0, g["t_inc"], g["dt"], 4 * n.WIN_HALF_PERIODS / f0)
    out.append(dict(check="a_inc_4x_over_1x_window", value=float(a4 / a1)))
    return out


def finite_born_checks(f0):
    """Validation of born_2d_finite and its near-field factor for the weak triangle."""
    from akp.stage_a import a3b2_numerical as n
    g = n.setup(f0)
    _, _, base, height, rel, _ = WEAK
    circ = ("circle", "circle", base, height, rel, f0)
    a = np.sqrt(base * height / 2 / np.pi)
    lin = (VP * (1 + LINEAR), VS * (1 + LINEAR), RHO * (1 + LINEAR))
    out, far, circ_err = [], [], []
    for mode in ("PP", "PS"):
        for th_deg in n.THETAS_DEG:
            th = np.radians(th_deg)
            far.append(abs(n.born_2d_finite(WEAK, g, 1e6, th, mode) / n.born_2d(WEAK, g, 1e6, th, mode)[0] - 1))
            for r in g["rs"]:
                ref = float(c_exact(freq=f0, a=a, theta=th, vp=VP, vs=VS, rho=RHO, rel_born=LINEAR, inclusion=lin,
                                    mode=mode, r=r)[0][0])
                cr = n.born_2d_finite(circ, g, r, th, mode) / n.born_2d(circ, g, r, th, mode)[0]
                circ_err.append(abs(cr / ref - 1))
                if th_deg == 75.0:
                    nf = n.born_2d_finite(WEAK, g, r, th, mode) / n.born_2d(WEAK, g, r, th, mode)[0]
                    out.append(dict(check=f"nf_factor_triangle_{mode}_75_r{int(r)}", value=float(nf)))
    out += [dict(check="finite_born_large_r_max_rel_dev", value=float(max(far))),
            dict(check="finite_born_circle_vs_exact_max_rel_dev", value=float(max(circ_err)))]
    return out


def main() -> None:
    _, _, base, height, rel, f0 = WEAK
    a = np.sqrt(base * height / 2 / np.pi)
    cal = pd.read_csv(HERE / "a3b2_run4_calibration.csv")  # fourth-run weak check
    fd = cal[cal.label == "weak_check"]
    rows = []
    for mode in ("PP", "PS"):
        for th_deg in sorted(fd.theta_deg.unique()):
            th = np.radians(th_deg)
            incl = lambda r: (VP * (1 + r), VS * (1 + r), RHO * (1 + r))  # noqa: E731
            c_ex = float(c_exact(freq=f0, a=a, theta=th, vp=VP, vs=VS, rho=RHO, rel_born=rel, inclusion=incl(rel),
                                 mode=mode)[0][0])
            r2 = rel * WEAKER
            c_ex_weak = float(c_exact(freq=f0, a=a, theta=th, vp=VP, vs=VS, rho=RHO, rel_born=r2,
                                      inclusion=incl(r2), mode=mode)[0][0])
            c_lin, e_lin = c_exact(freq=f0, a=a, theta=th, vp=VP, vs=VS, rho=RHO, rel_born=rel, inclusion=incl(rel),
                                   mode=mode)
            born = float(e_lin[0] / c_lin[0])
            odd = abs(exact_complex(f0=f0, a=a, theta=th, rel=rel, mode=mode)
                      - exact_complex(f0=f0, a=a, theta=th, rel=-rel, mode=mode)) / 2
            sub = fd[(fd["mode"] == mode) & (fd.theta_deg == th_deg)]
            k = 2 * np.pi * f0 / (VP if mode == "PP" else VS)
            for _, s in sub.iterrows():
                r = float(s.r_m)
                kw = dict(freq=f0, a=a, theta=th, vp=VP, vs=VS, rho=RHO, mode=mode, r=r)
                c_r = float(c_exact(rel_born=rel, inclusion=incl(rel), **kw)[0][0])
                nf = float(c_exact(rel_born=LINEAR, inclusion=incl(LINEAR), **kw)[0][0])
                born_r = born * np.sqrt(2 / (np.pi * k * r))
                odd_r = abs(exact_complex(f0=f0, a=a, theta=th, rel=rel, mode=mode, r=r)
                            - exact_complex(f0=f0, a=a, theta=th, rel=-rel, mode=mode, r=r)) / 2
                c_neg = float(c_exact(rel_born=-rel, inclusion=incl(-rel), **kw)[0][0])
                rows.append(dict(mode=mode, theta_deg=th_deg, r_m=r, C_fd=s.C, C_exact_circle=c_ex,
                                 fd_over_exact=s.C / c_ex, C_exact_pert_born=float(exact_pert_born_ratio(th, rel, mode)),
                                 C_exact_circle_weaker=c_ex_weak, C_exact_circle_odd=odd / born,
                                 C_exact_circle_r=c_r, nf_factor_linear_r=nf, fd_over_exact_r=s.C / c_r,
                                 C_exact_circle_odd_r=odd_r / born_r, C_exact_circle_r_neg=c_neg))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "a3b2_ps_diagnosis.csv", index=False, float_format="%.5g")

    checks = []
    for mode in ("PP", "PS"):
        d = df[df["mode"] == mode]
        checks += [
            dict(check=f"max_abs_C_fd_minus_1_{mode}", value=float((d.C_fd - 1).abs().max())),
            dict(check=f"max_abs_C_exact_circle_minus_1_{mode}", value=float((d.C_exact_circle - 1).abs().max())),
            dict(check=f"max_abs_fd_over_exact_minus_1_{mode}", value=float((d.fd_over_exact - 1).abs().max())),
            dict(check=f"max_abs_C_exact_pert_born_minus_1_{mode}", value=float((d.C_exact_pert_born - 1).abs().max())),
            dict(check=f"max_abs_C_exact_circle_weaker_minus_1_{mode}",
                 value=float((d.C_exact_circle_weaker - 1).abs().max())),
            dict(check=f"max_abs_C_exact_circle_odd_minus_1_{mode}",
                 value=float((d.C_exact_circle_odd - 1).abs().max())),
            dict(check=f"max_abs_fd_over_exact_r_minus_1_{mode}", value=float((d.fd_over_exact_r - 1).abs().max())),
            dict(check=f"max_C_exact_circle_odd_r_{mode}", value=float(d.C_exact_circle_odd_r.max())),
        ]
        o = d[d.theta_deg != 75.0]
        checks.append(dict(check=f"max_abs_fd_over_exact_r_minus_1_{mode}_not75",
                           value=float((o.fd_over_exact_r - 1).abs().max())))
    for _, rw in df[(df["mode"] == "PS") & (df.theta_deg == 75.0)].iterrows():
        tag = f"PS_75_r{int(rw.r_m)}"
        checks += [dict(check=f"C_exact_circle_r_{tag}", value=float(rw.C_exact_circle_r)),
                   dict(check=f"nf_factor_linear_{tag}", value=float(rw.nf_factor_linear_r)),
                   dict(check=f"fd_over_exact_r_{tag}", value=float(rw.fd_over_exact_r)),
                   dict(check=f"C_exact_circle_odd_r_{tag}", value=float(rw.C_exact_circle_odd_r)),
                   dict(check=f"C_exact_circle_r_neg_{tag}", value=float(rw.C_exact_circle_r_neg))]
    checks += finite_born_checks(f0)
    checks += late_arrivals(f0)
    checks += [dict(check="weak_rel", value=rel), dict(check="weaker_rel", value=rel * WEAKER),
               dict(check="weak_tol", value=WEAK_TOL), dict(check="circle_radius_m", value=float(a)),
               dict(check="ps_born_null_deg", value=float(np.degrees(np.arccos(VP / VS / 6))))]
    with (HERE / "a3b2_ps_diagnosis_checks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "value"])
        w.writeheader()
        w.writerows({**c, "value": round(c["value"], 4)} for c in checks)
    pd.set_option("display.width", 200)
    print(df.groupby(["mode", "theta_deg"]).agg(C_fd=("C_fd", "max"), C_exact=("C_exact_circle", "first"),
                                                 fd_over_exact=("fd_over_exact", "max"),
                                                 pert=("C_exact_pert_born", "first"),
                                                 weaker=("C_exact_circle_weaker", "first"),
                                                 odd=("C_exact_circle_odd", "first")).to_string())
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == "__main__":
    main()
