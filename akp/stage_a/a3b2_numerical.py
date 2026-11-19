"""A3b-2: numerical calibration of Born outside the Rayleigh regime.

2-D elastic velocity-stress FD (Devito, CPU, C backend; parallel processes). Homogeneous host
(vp 2000 m/s), sponge absorbing layers, upgoing plane P wave from a line of explosive sources.
Scattered field = total - body-free run. P and S are separated at each receiver by div v and
curl v (four receivers one cell around each point): |v_P| = |div v| / k_P, |v_S| = |curl v| / k_S.
Each amplitude is the DFT at f0 of the trace in a Tukey window centered on the predicted arrival;
the incident amplitude is vz of the body-free run at the body center, windowed the same way.
C = numerical / linearized 2-D Born for the body as rasterized on the FD grid.

Fourth run:
- frequencies 0.35-9.41 Hz; receivers at theta 75-95 deg and, where the Fraunhofer distance
  2 D^2 / lambda is large, at 1.2 and 1.5 times it;
- windows widened by twice the strong body's arrival spread; a doubled-window sensitivity;
- margins grown until every receiver's window ends before its first boundary reflection;
- gate (i-a) weak check at every angle; gate (i-b) an FD near-void circle against the exact
  series solution (a3b2_cylinder.py); a doubled-sponge sensitivity; traces archived to
  $AKP_DATA/a3b2/run4.
[Runs 1-2 failed the weak check from measurement bugs; run 3 passed, but its application was later
judged weaker than stated.]
Writes a3b2_calibration.csv, a3b2_summary.csv and a3b2_checks.csv.
Run: python -m akp.stage_a.a3b2_numerical
"""
from __future__ import annotations

import csv
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from akp.stage_a import fd2d  # also sets the Devito environment

HERE = Path(__file__).parent
VP, VP_VS, RHO = 2000.0, 1.8, 2000.0
VS = VP / VP_VS
NBL = 100
CELLS_PER_L, MAX_DX = 200, 500.0 / 60  # grid step = min(L / 200, strong-body base / 60)
STRONG = (500.0, 300.0)  # base, height of the strong-contrast body (sets windows and far field)
FREQS = [0.35, 0.5, 1.0, 2.0, 3.0, 6.67, 9.41]
THETAS_DEG = [75.0, 80.0, 85.0, 90.0, 95.0]
NEAR_L, FF_FACTORS = [1.6, 2.0, 2.4], [1.2, 1.5]
SRC_BELOW_L = 0.75
WIN_HALF_PERIODS, DT_PER_DX = 1.25, 1.1e-4
T0_PERIODS = fd2d.T0_PERIODS
WEAK = ("weak_check", "triangle", 60.0, 40.0, 0.05, 2.0)
WEAK_NEG = ("weak_check_neg", "triangle", 60.0, 40.0, -0.05, 2.0)  # fifth run: odd-part partner
CIRCLE_FREQS = [0.5, 1.0, 2.0, 3.0]
CASES = (  # (label, shape, base, height, rel, f0)
    [WEAK]
    + [("near_void", "triangle", *STRONG, -0.9, f) for f in FREQS]
    + [("dense_stiff", "triangle", *STRONG, 1.0, f) for f in FREQS]
    + [("circle_near_void", "circle", *STRONG, -0.9, f) for f in CIRCLE_FREQS]
)
SPONGE_CASE = ("near_void", "triangle", *STRONG, -0.9, 1.0)
WEAK_TOL, CIRCLE_TOL, NULL_FRAC, MATERIAL_FACTOR = 0.20, 0.10, 0.10, 3.0


# ---------------------------------------------------------------- geometry
def receiver_dirs():
    th = np.radians(THETAS_DEG)
    return th, np.column_stack([np.sin(th), -np.cos(th)])  # incident travels -z; r_hat(theta)


NF_FACTORS = [0.3, 0.6]  # near-field receivers (fractions of the Fraunhofer distance), governing f0 only


def fraunhofer(f0):
    return 2 * STRONG[0] ** 2 / (VP / f0)


def distances(f0):
    lam, rff = VP / f0, fraunhofer(f0)
    if FF_FACTORS[0] * rff > NEAR_L[0] * lam:
        return [k * rff for k in NF_FACTORS + FF_FACTORS]
    return [k * lam for k in NEAR_L]


def is_far(f0, r):
    # [Fixed 2026-11-09 before the final measurement: was min(), which counted the near-field
    # receivers at 6.67 and 9.41 Hz as far field.]
    return r >= max(FF_FACTORS[0] * fraunhofer(f0), NEAR_L[0] * VP / f0) - 1e-6


def half_width(f0, mode, theta, factor=1.0):
    """Window half-width (s): pulse plus twice the strong body's arrival spread."""
    v_out = VP if mode == "PP" else VS
    base, height = STRONG
    spread = 0.5 * (base * abs(np.sin(theta)) / v_out + height * abs(1 / VP - np.cos(theta) / v_out))
    return factor * (WIN_HALF_PERIODS / f0 + 2 * spread)


def setup(f0, nbl_factor=1):
    """Grid, time step and positions; margins grown until all windows end before reflections."""
    lam = VP / f0
    dx = min(lam / CELLS_PER_L, MAX_DX)
    th, rhat = receiver_dirs()
    rs = distances(f0)
    rmax = max(rs)
    win_end_p = max(half_width(f0, "PP", t) for t in th)
    t_lead = T0_PERIODS / f0  # Ricker delay
    m = 1.0 * lam
    while True:
        xb, zb = m, m + rmax * max(0.0, -rhat[:, 1].min())  # body; receivers above reach -0.26 r
        lx = xb + rmax * rhat[:, 0].max() + m
        lz = zb + max(SRC_BELOW_L * lam, rmax * max(0.0, rhat[:, 1].max())) + m
        g = dict(f0=f0, lam=lam, dx=dx, xb=xb, zb=zb, zs=zb + SRC_BELOW_L * lam, lx=lx, lz=lz,
                 rs=rs, nbl=NBL * nbl_factor)
        if all_clean(g, factor=1.0):
            break
        m += 0.25 * lam
    t_inc = t_lead + SRC_BELOW_L * lam / VP
    t_end = t_inc + rmax / VS + 2 * max(half_width(f0, "PS", t) for t in th) + 1.0 / f0
    g.update(dt=DT_PER_DX * dx, nt=int(t_end / (DT_PER_DX * dx)) + 1, t_inc=t_inc, win_end_p=win_end_p)
    return g


def reflection_delay(g, rx, rz):
    """Earliest first-order boundary reflection of a P wave from the body to (rx, rz), minus direct (s)."""
    direct = np.hypot(rx - g["xb"], rz - g["zb"])
    images = [(-rx, rz), (2 * g["lx"] - rx, rz), (rx, -rz), (rx, 2 * g["lz"] - rz)]
    return min(np.hypot(ix - g["xb"], iz - g["zb"]) - direct for ix, iz in images) / VP


def clean(g, delay_p, mode, theta, factor):
    """Window ends before the first reflection of the same wave type (div sees P, curl sees S)."""
    delay = delay_p if mode == "PP" else delay_p * VP / VS
    return delay >= half_width(g["f0"], mode, theta, factor) + 1.0 / g["f0"]


def all_clean(g, factor):
    th, rhat = receiver_dirs()
    for r in g["rs"]:
        for t, (ux, uz) in zip(th, rhat):
            d = reflection_delay(g, g["xb"] + r * ux, g["zb"] + r * uz)
            if not (clean(g, d, "PP", t, factor) and clean(g, d, "PS", t, factor)):
                return False
    return True


def body_mask(xx, zz, shape, base, height, xb, zb):
    if shape == "circle":  # same area as the triangle, centered on the centroid
        a = np.sqrt(base * height / 2 / np.pi)
        return (xx - xb) ** 2 + (zz - zb) ** 2 <= a**2
    return fd2d.triangle_mask(xx, zz, base, height, xb, zb)


def grid_xz(g):
    nb = g["nbl"]
    nx, nz = int(round(g["lx"] / g["dx"])) + 1 + 2 * nb, int(round(g["lz"] / g["dx"])) + 1 + 2 * nb
    x0 = nb * g["dx"]
    return nx, nz, x0, np.arange(nx) * g["dx"] - x0, np.arange(nz) * g["dx"] - x0


def receivers(g):
    """Four points one cell around each (distance, angle) receiver, then the body center."""
    th, rhat = receiver_dirs()
    h = g["dx"]
    pts = []
    for r in g["rs"]:
        for ux, uz in rhat:
            x, z = g["xb"] + r * ux, g["zb"] + r * uz
            pts += [(x + h, z), (x - h, z), (x, z + h), (x, z - h)]
    return np.array(pts + [(g["xb"], g["zb"])])


# ---------------------------------------------------------------- FD
def run(job):
    """One FD run; job = (f0, case or None, nbl_factor). Archives and returns (vx, vz)."""
    f0, case, nbl_factor = job
    if ARCHIVE_RUN == "run4" and archive_path(job).exists():
        raise RuntimeError(f"{archive_path(job)} exists; the run4 archive is read-only (set AKP_A3B2_RUN)")
    g = setup(f0, nbl_factor)
    nx, nz, x0, xs, zs = grid_xz(g)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    vp = np.full((nx, nz), VP)
    vs = vp / VP_VS
    rho = np.full((nx, nz), RHO)
    if case is not None:
        _, shape, base, height, rel, _ = case
        m = body_mask(xx, zz, shape, base, height, g["xb"], g["zb"])
        vp[m] *= 1 + rel
        vs[m] *= 1 + rel
        rho[m] *= 1 + rel
    lam, mu = rho * (vp**2 - 2 * vs**2), rho * vs**2
    del vp, vs, xx, zz
    src_x = np.arange(-x0 + g["dx"], (nx - 1) * g["dx"] - x0 - g["dx"], g["dx"])
    pts = receivers(g) + x0
    vx, vz = fd2d.run(dx=g["dx"], dt=g["dt"], nt=g["nt"], lam=lam, mu=mu, rho=rho,
                      damp=fd2d.sponge(nx, nz, g["nbl"]), src_x=src_x + x0, src_z=g["zs"] + x0, f0=f0, rec=pts)
    np.savez_compressed(archive_path(job), vx=vx, vz=vz, dt=g["dt"], pts=pts - x0)
    return vx, vz


ARCHIVE_RUN = os.environ.get("AKP_A3B2_RUN", "run4")  # run4 is read-only; a new run sets its own folder


def run_fifth(job):
    """run() with the fifth-run archive (a Pool worker re-imports this module, resetting ARCHIVE_RUN)."""
    global ARCHIVE_RUN
    ARCHIVE_RUN = "run5"
    return run(job)


def archive_path(job, run=None):
    from akp.paths import data_dir
    f0, case, nbl_factor = job
    out = data_dir("a3b2") / (run or ARCHIVE_RUN)
    out.mkdir(parents=True, exist_ok=True)
    tag = "bg" if case is None else f"{case[0]}_{case[1]}"
    return out / f"{tag}_f{f0:g}_nbl{nbl_factor}.npz"


def load(job, run=None):
    z = np.load(archive_path(job, run))
    return z["vx"], z["vz"]


def windowed_amp(trace, f0, t_arr, dt, half):
    return abs(fd2d.tukey_dft(trace, f0, t_arr, dt, half))


# ---------------------------------------------------------------- Born and exact
def born_2d(case, g, r, theta, mode):
    from swe.scattering import born_radiation_pattern
    _, shape, base, height, rel, f0 = case
    k = 2 * np.pi * f0 / VP
    pat = born_radiation_pattern(theta, vp=VP, vs=VS, density=RHO, dvp=rel * VP, dvs=rel * VS, drho=rel * RHO,
                                 mode=mode)
    k_out = k if mode == "PP" else 2 * np.pi * f0 / VS
    q = k_out * np.array([np.sin(theta), -np.cos(theta)]) - k * np.array([0.0, -1.0])
    _, _, _, xs, zs = grid_xz(g)
    sx = xs[np.abs(xs - g["xb"]) <= base]
    sz = zs[np.abs(zs - g["zb"]) <= base]
    xx, zz = np.meshgrid(sx, sz, indexing="ij")
    m = body_mask(xx, zz, shape, base, height, g["xb"], g["zb"])
    area = float(m.sum() * g["dx"] ** 2)
    F = np.mean(np.exp(-1j * (q[0] * (xx[m] - g["xb"]) + q[1] * (zz[m] - g["zb"]))))
    amp = k_out**2 * area * abs(pat * F) * 0.25 * np.sqrt(2.0 / (np.pi * k_out * r))
    return float(amp), float(k * np.sqrt(area / np.pi)), float(k_out * r)


def born_2d_finite(case, g, r, theta, mode):
    """Linear Born of the rasterized body at the receiver distance (fifth-run gate i-a reference).

    Sums the exact linear point-scatterer field (a3b2_cylinder.point_coefficients) over the body's
    grid cells, each with its area dx^2 and the incident phase, with H_n(k R) at the true cell-to-
    receiver distance R, so the near field is kept. Same normalization as born_2d, which it
    approaches as r grows.
    """
    from scipy.special import hankel1

    from akp.stage_a.a3b2_cylinder import point_coefficients
    _, shape, base, height, rel, f0 = case
    an, bn, kp, ks = point_coefficients(freq=f0, vp=VP, vs=VS, rho=RHO)
    _, _, _, xs, zs = grid_xz(g)
    sx = xs[np.abs(xs - g["xb"]) <= base]
    sz = zs[np.abs(zs - g["zb"]) <= base]
    xx, zz = np.meshgrid(sx, sz, indexing="ij")
    m = body_mask(xx, zz, shape, base, height, g["xb"], g["zb"])
    cx, cz = xx[m], zz[m]
    rx, rz = g["xb"] + r * np.sin(theta), g["zb"] - r * np.cos(theta)
    dxr, dzr = rx - cx, rz - cz
    R = np.hypot(dxr, dzr)
    th = np.arctan2(dxr, -dzr)  # from the incident direction (-z) toward +x, as receiver_dirs
    inc = np.exp(-1j * kp * (cz - g["zb"]))  # incident phase, traveling -z
    n = np.arange(len(an))[:, None]
    if mode == "PP":
        field = np.sum(inc * np.sum(an[:, None] * hankel1(n, kp * R) * np.cos(n * th), axis=0))
        scale = 1.0
    else:
        field = np.sum(inc * np.sum(bn[:, None] * hankel1(n, ks * R) * np.sin(n * th), axis=0))
        scale = ks / kp
    return float(scale * abs(field) * rel * g["dx"] ** 2)


def measure(case, g, tx, tz, bx, bz, *, label=None, finite=False):
    """Rows per receiver: numerical, Born, C, doubled-window C and the reflection-clean flags.

    finite=True compares with born_2d_finite (the fifth-run gate i-a reference) instead of the
    far-field born_2d; label overrides the case label in the rows.
    """
    f0, dx, dt = g["f0"], g["dx"], g["dt"]
    kp, ks = 2 * np.pi * f0 / VP, 2 * np.pi * f0 / VS
    a_inc = windowed_amp(bz[:, -1], f0, g["t_inc"], dt, WIN_HALF_PERIODS / f0)
    a_inc2 = windowed_amp(bz[:, -1], f0, g["t_inc"], dt, 2 * WIN_HALF_PERIODS / f0)
    sx, sz = tx - bx, tz - bz
    th, rhat = receiver_dirs()
    rows, i = [], 0
    for r in g["rs"]:
        for t, (ux, uz) in zip(th, rhat):
            div = (sx[:, i] - sx[:, i + 1] + sz[:, i + 2] - sz[:, i + 3]) / (2 * dx)
            curl = (sz[:, i] - sz[:, i + 1] - (sx[:, i + 2] - sx[:, i + 3])) / (2 * dx)
            delay = reflection_delay(g, g["xb"] + r * ux, g["zb"] + r * uz)
            for mode, trace, kk, v in (("PP", div, kp, VP), ("PS", curl, ks, VS)):
                t_arr = g["t_inc"] + r / v
                h1, h2 = half_width(f0, mode, t), half_width(f0, mode, t, 2.0)
                num = windowed_amp(trace, f0, t_arr, dt, h1) / kk / a_inc
                num2 = windowed_amp(trace, f0, t_arr, dt, h2) / kk / a_inc2
                born, ka, kr = born_2d(case, g, r, t, mode)
                if finite:
                    born = born_2d_finite(case, g, r, t, mode)
                rows.append(dict(label=label or case[0], shape=case[1], mode=mode, f0_hz=f0, theta_deg=round(np.degrees(t), 1),
                                 r_m=round(r, 1), kr=round(kr, 2), ka=round(ka, 3), dx_m=round(dx, 3),
                                 far_field=bool(is_far(f0, r)), numerical=num, born=born, C=num / born,
                                 C_win2=num2 / born, win2_clean=bool(clean(g, delay, mode, t, 2.0))))
            i += 4
    return rows


def circle_exact(g, r, theta, f0):
    from akp.stage_a.a3b2_cylinder import c_exact
    a = np.sqrt(STRONG[0] * STRONG[1] / 2 / np.pi)
    _, exact = c_exact(freq=f0, a=a, theta=theta, vp=VP, vs=VS, rho=RHO, rel_born=-0.9,
                       inclusion=(0.1 * VP, 0.1 * VS, 0.1 * RHO))
    k = 2 * np.pi * f0 / VP
    return float(exact[0] * np.sqrt(2 / (np.pi * k * r)))


# ---------------------------------------------------------------- main
def main(from_archive: bool = False, fifth: bool = False) -> None:
    """Run every FD job (or, with from_archive, reload the archived traces) and measure.

    fifth=True: the fourth run's traces from the read-only run4 archive, plus the -5% weak
    check (archive run5, computed if absent); gate (i-a) becomes the odd-part check against
    born_2d_finite.
    """
    import pandas as pd
    jobs = [(f, None, 1) for f in FREQS] + [(c[5], c, 1) for c in CASES]
    jobs += [(SPONGE_CASE[5], None, 2), (SPONGE_CASE[5], SPONGE_CASE, 2)]
    neg = (WEAK_NEG[5], WEAK_NEG, 1)
    if fifth:
        global ARCHIVE_RUN
        out = {j: load(j, "run4") for j in jobs}
        ARCHIVE_RUN = "run5"  # run() archives the new jobs here; run4 stays read-only
        bg2 = (WEAK[5], None, 1)
        new = [j for j in (neg, bg2) if not archive_path(j).exists()]
        if new:
            with Pool(len(new)) as pool:
                pool.map(run_fifth, new)
        out[neg] = load(neg)
        # the odd part subtracts a new-build trace from a run4 trace: the new build must reproduce run4
        for a, b in zip(load(bg2), out[bg2]):
            assert np.array_equal(a, b), "this build does not reproduce the run4 background at 2 Hz"
    elif from_archive:
        out = {j: load(j) for j in jobs}
    else:
        cost = {f: (lambda g: g["nt"] * np.prod(grid_xz(g)[:2]))(setup(f)) for f in FREQS}
        order = sorted(range(len(jobs)), key=lambda j: -cost[jobs[j][0]])  # largest first
        with Pool(10) as pool:
            res = pool.map(run, [jobs[j] for j in order], chunksize=1)
        out = {jobs[j]: res[n] for n, j in enumerate(order)}
    rows = []
    for c in CASES:
        g = setup(c[5])
        rows += measure(c, g, *out[(c[5], c, 1)], *out[(c[5], None, 1)])
    if fifth:  # odd part (s+ - s-) / 2 on the vx, vz traces, before div / curl; same windows, a_inc and background
        g = setup(WEAK[5])
        (tpx, tpz), (tnx, tnz), (bx, bz) = out[(WEAK[5], WEAK, 1)], out[neg], out[(WEAK[5], None, 1)]
        rows += measure(WEAK, g, bx + (tpx - tnx) / 2, bz + (tpz - tnz) / 2, bx, bz, label="weak_check_odd",
                        finite=True)
        rows += measure(WEAK_NEG, g, tnx, tnz, bx, bz)  # single-sided -5%, context
    sp = measure(SPONGE_CASE, setup(1.0, 2), *out[(1.0, SPONGE_CASE, 2)], *out[(1.0, None, 2)])
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "a3b2_calibration.csv", index=False)
    near = df[~df.far_field & (df["mode"] == "PP") & df.label.isin(["near_void", "dense_stiff"])]
    df = df[df.far_field]
    summ = (df[df["shape"] == "triangle"].groupby(["label", "mode", "f0_hz", "theta_deg"])
            .agg(C=("C", "median"), C_win2=("C_win2", "median"), ka=("ka", "first"),
                 C_spread=("C", lambda c: float(c.max() / c.min()))).reset_index())
    summ.to_csv(HERE / "a3b2_summary.csv", index=False)
    checks = []
    weak = df[df.label == "weak_check"]
    for mode in ("PP", "PS"):  # in the fifth run the single-sided check is context; gate (i-a) is the odd part
        err = float(np.abs(weak[weak["mode"] == mode].C - 1).max())
        checks.append(dict(check=f"weak_check_max_rel_err_{mode}", value=round(err, 3),
                           passed="" if fifth else bool(err <= WEAK_TOL)))
    if fifth:
        for lab in ("weak_check_odd", "weak_check_neg"):
            sub = df[df.label == lab]
            for mode in ("PP", "PS"):
                err = float(np.abs(sub[sub["mode"] == mode].C - 1).max())
                checks.append(dict(check=f"{lab}_max_rel_err_{mode}", value=round(err, 3),
                                   passed=bool(err <= WEAK_TOL) if lab == "weak_check_odd" else ""))
    circ = df[(df.label == "circle_near_void") & (df["mode"] == "PP")].copy()
    circ["exact"] = [circle_exact(setup(rw.f0_hz), rw.r_m, np.radians(rw.theta_deg), rw.f0_hz) for rw in circ.itertuples()]
    circ["peak"] = circ.groupby(["f0_hz", "r_m"]).exact.transform("max")
    use = circ[circ.exact >= NULL_FRAC * circ.peak]
    err = float(np.abs(use.numerical / use.exact - 1).max())
    checks.append(dict(check="circle_max_rel_err", value=round(err, 3), passed=bool(err <= CIRCLE_TOL)))
    checks.append(dict(check="circle_receivers_used", value=int(len(use)), passed=""))
    circ.to_csv(HERE / "a3b2_circle.csv", index=False)
    strong = df[df.label.isin(["near_void", "dense_stiff"])]
    clean = strong[strong.win2_clean]
    checks.append(dict(check="window_sens_max_rel", value=round(float(np.abs(clean.C_win2 / clean.C - 1).max()), 3),
                       passed=""))
    checks.append(dict(check="window_sens_receivers", value=int(len(clean)), passed=""))
    base_1hz = strong[(strong.label == "near_void") & (strong.f0_hz == 1.0) & (strong["mode"] == "PP")]
    sp = pd.DataFrame(sp)
    sp = sp[(sp["mode"] == "PP") & sp.far_field]
    checks.append(dict(check="sponge_C_ratio_max_rel", value=round(float(np.abs(sp.C.to_numpy() / base_1hz.C.to_numpy() - 1).max()), 3),
                       passed=""))
    for lab in ("near_void", "dense_stiff"):  # inside the Fraunhofer distance: FD / far-field Born
        sub = near[near.label == lab]
        checks.append(dict(check=f"nearfield_max_C_PP_{lab}", value=round(float(sub.C.max()), 3), passed=""))
    pp = summ[summ["mode"] == "PP"]
    for lab in ("near_void", "dense_stiff"):
        for rule, lo, hi in (("geo", 75.0, 90.0), ("all", 75.0, 95.0)):
            sub = pp[(pp.label == lab) & (pp.theta_deg >= lo) & (pp.theta_deg <= hi)]
            checks.append(dict(check=f"max_C_PP_{lab}_{rule}", value=round(float(sub.C.max()), 3), passed=""))
        at90 = pp[(pp.label == lab) & (pp.theta_deg == 90.0)]
        for r in at90.itertuples():
            checks.append(dict(check=f"C_PP_{lab}_f{r.f0_hz:g}_t90", value=round(float(r.C), 3), passed=""))
    with (HERE / "a3b2_checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "value", "passed"])
        w.writeheader()
        w.writerows(checks)
    for ch in checks:
        print(ch)


def _dryrun() -> None:
    """Print each frequency's grid size, steps and receiver distances (no FD)."""
    for f in FREQS:
        g = setup(f)
        nx, nz = grid_xz(g)[:2]
        print(f, round(g["dx"], 2), nx, nz, f"{nx * nz / 1e6:.1f}M", g["nt"], f"{nx * nz * g['nt'] / 1e11:.2f}e11",
              [round(r) for r in g["rs"]], round(g["xb"]), round(g["lx"]), round(g["lz"]))


if __name__ == "__main__":
    import sys
    if "--dryrun" in sys.argv:
        _dryrun()
    else:
        main(from_archive="--from-archive" in sys.argv, fifth="--fifth" in sys.argv)
