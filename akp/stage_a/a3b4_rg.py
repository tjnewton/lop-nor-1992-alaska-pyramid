"""A3b-4: P-to-Rg conversion by a buried body.

Homogeneous elastic half-space (z down, free surface z = 0), vertically incident plane P with its
free-surface reflection: u_z = u0 cos(k_a z), eps_zz = -u0 k_a sin(k_a z) (u0 = direct P at the
surface). Born scattering into the fundamental Rayleigh mode (Rg) by reciprocity: the receiver-side
Green's function is the Rayleigh pole of Lamb's problem (unit surface force at the receiver),
evaluated in closed form by residues. Born integrand per unit volume (linearized perturbations,
as swe.scattering):
    S = w^2 drho u_z g_z - dlam eps_zz (div g) - 2 dmu eps_zz d(g_z)/dz.
2-D (line body): u_sc = sum S dA, with g the 2-D Lamb pole field. 3-D (compact body): each voxel
is a point scatterer, g_3D = g_2D * sqrt(k_R / (2 pi R)) exp(-i pi/4) (stationary phase), so
|u_sc(R)| = E / sqrt(R) in the far field, and the A3a detection rule |u_sc| / u0 = C gives
r*_Rg = (E / C)^2 with C the A2 clutter ratio. No Rg attenuation (the generous bound).

Solver check (gate i): the 2-D model against Devito 2-D elastic FD with a free surface (vacuum
formalism), weak small body at two depths; and the FD Rayleigh velocity against the exact root.
Writes a3b4_checks.csv, a3b4_fd.csv and a3b4_cases.csv. Run: python -m akp.stage_a.a3b4_rg
"""
from __future__ import annotations

import csv
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from akp.stage_a import fd2d
from akp.stage_a.fd2d import T0_PERIODS, triangle_mask, tukey_dft, tukey_window

HERE = Path(__file__).parent
WEAK_TOL, VEL_TOL, MATERIALITY = 0.20, 0.02, 0.01
APEX_DEPTHS = [30.0, 200.0]  # A3 burial range (as A3b-3)
AZIMUTHS = np.radians([0.0, 15.0, 30.0, 45.0])  # square body: 0-45 deg covers all directions


# ---------------------------------------------------------------- half-space Rayleigh wave
def rayleigh_speed(vp: float, vs: float) -> float:
    """Exact Rayleigh speed of a homogeneous half-space (root of the Rayleigh equation)."""
    def f(x):  # x = c / vs
        return (2 - x**2) ** 2 - 4 * np.sqrt(1 - x**2 * vs**2 / vp**2) * np.sqrt(1 - x**2)
    return float(brentq(f, 0.5, 0.999999) * vs)


class Lamb:
    """Rayleigh pole of a unit vertical surface force (+z, into the medium) on a half-space.

    Field for propagation away from the force along +r: g_r(z) e^{i k r}, g_z(z) e^{i k r}, in 2-D
    (line force per unit length). The x < 0 branch of the 2-D field is (-g_r, g_z).
    """

    def __init__(self, *, vp: float, vs: float, rho: float, freq: float):
        self.w = 2 * np.pi * freq
        self.c = rayleigh_speed(vp, vs)
        self.k = self.w / self.c
        ka, kb = self.w / vp, self.w / vs
        self.ka = ka
        k = self.k
        self.na, self.nb = np.sqrt(k**2 - ka**2), np.sqrt(k**2 - kb**2)
        mu = rho * vs**2

        def F(kk):
            return (2 * kk**2 - kb**2) ** 2 - 4 * kk**2 * np.sqrt(kk**2 - ka**2) * np.sqrt(kk**2 - kb**2)

        h = 1e-6 * k
        dF = (F(k + h) - F(k - h)) / (2 * h)
        # residues of the transformed potentials (phi = A e^{-na z}, psi = B e^{-nb z}), times i
        self.A = 1j * (-(2 * k**2 - kb**2) / (mu * dF))
        self.B = 1j * (2j * k * self.na / (mu * dF))

    def g(self, z):
        """(g_r, g_z, d g_z / dz) at depth z (without the e^{i k r} factor)."""
        k, na, nb, A, B = self.k, self.na, self.nb, self.A, self.B
        ea, eb = np.exp(-na * z), np.exp(-nb * z)
        gr = 1j * k * A * ea + nb * B * eb
        gz = -na * A * ea + 1j * k * B * eb
        dgz = na**2 * A * ea - 1j * k * nb * B * eb
        return gr, gz, dgz


def perturbations(vp, vs, rho, rel):
    """Linearized (drho, dlam, dmu) for relative perturbation rel of vp, vs and density."""
    drho, dvp, dvs = rel * rho, rel * vp, rel * vs
    dmu = drho * vs**2 + 2 * rho * vs * dvs
    dlam = drho * (vp**2 - 2 * vs**2) + 2 * rho * (vp * dvp - 2 * vs * dvs)
    return drho, dlam, dmu


def integrand(lamb: Lamb, z, drho, dlam, dmu):
    """Born integrand S(z) per unit volume, for u0 = 1 and the receiver-side field g (no phase)."""
    gr, gz, dgz = lamb.g(z)
    uz, ezz = np.cos(lamb.ka * z), -lamb.ka * np.sin(lamb.ka * z)
    div_g = 1j * lamb.k * gr + dgz  # horizontal divergence of an outgoing field ~ i k g_r (far field)
    return lamb.w**2 * drho * uz * gz - dlam * ezz * div_g - 2 * dmu * ezz * dgz


def rg_2d(lamb: Lamb, xs, zs, dA, x_rec, drho, dlam, dmu):
    """|scattered Rg u_z| / u0 at a surface receiver x_rec for a 2-D body of cells (xs, zs)."""
    s = integrand(lamb, zs, drho, dlam, dmu)
    return float(np.abs(np.sum(s * np.exp(1j * lamb.k * np.abs(x_rec - xs))) * dA))


def rg_3d_E(lamb: Lamb, xyz, dV, drho, dlam, dmu, r_ref=1.0e5):
    """max over azimuth of E = |u_sc(R)| sqrt(R) / u0, from a voxel sum at reference distance r_ref."""
    s = integrand(lamb, xyz[:, 2], drho, dlam, dmu) * dV
    best = 0.0
    for az in AZIMUTHS:
        rx, ry = r_ref * np.cos(az), r_ref * np.sin(az)
        R = np.hypot(rx - xyz[:, 0], ry - xyz[:, 1])
        u = np.sum(s * np.sqrt(lamb.k / (2 * np.pi * R)) * np.exp(1j * (lamb.k * R - np.pi / 4)))
        best = max(best, float(np.abs(u) * np.sqrt(r_ref)))
    return best


# ---------------------------------------------------------------- 2-D FD (free surface)
VP_VS, RHO = 1.8, 2000.0
NBL, NVAC = 100, 8
CELLS_PER_L = 333
Z_SRC_L, WIN_HALF_PERIODS = 0.75, 1.25
DEPTHS_LR = [0.15, 0.4]  # weak-body centroid depth in Rayleigh wavelengths
STRONG = (500.0, 300.0, 30.0)  # base, height, apex depth of the governing row (A3b-4 strong stage)
STRONG_CASES = {"near_void": -0.9, "dense_stiff": 1.0}
STRONG_VPS = [850.0, 1200.0]
STABILITY_TOL = 0.20


def fd_config(*, vp, f0, offsets_lr, x_body_l, lz_l, lx_l=None, margin_l=None, half_extra=0.0, tail=None):
    """Grid, time step and geometry; lengths in host P wavelengths L = vp / f0."""
    lam = vp / f0
    dx = lam / CELLS_PER_L
    c_r = rayleigh_speed(vp, vp / VP_VS)
    lr = c_r / f0
    lx = (lx_l if lx_l is not None else x_body_l + offsets_lr[-1] * lr / lam + margin_l) * lam
    half = WIN_HALF_PERIODS / f0 + half_extra
    t_end = T0_PERIODS / f0 + Z_SRC_L * lam / vp + offsets_lr[-1] * lr / c_r + (tail if tail is not None else half + 1 / f0)
    dt = 0.22 * dx / vp  # 0.8 of the SO-8 staggered CFL limit for a body at twice the host vp
    return dict(vp=vp, f0=f0, dx=dx, dt=dt, nt=int(t_end / dt) + 1, lam=lam, lr=lr, c_r=c_r, half=half,
                offsets_lr=list(offsets_lr), xb=x_body_l * lam, lx=lx, lz=lz_l * lam, zs=Z_SRC_L * lam)


def gate_config():  # the first run's check (unchanged)
    return fd_config(vp=2000.0, f0=2.0, offsets_lr=[6.0, 8.0, 10.0], x_body_l=2.0, lx_l=8.4, lz_l=3.5, tail=3.0 / 2.0)


def strong_config(vp):  # strong stage: far receivers, body-spread windows, 2.5 L margins, 4.5 L depth
    return fd_config(vp=vp, f0=2.0, offsets_lr=[10.0, 14.0, 18.0], x_body_l=2.5, lz_l=4.5, margin_l=2.5,
                     half_extra=2 * STRONG[0] / rayleigh_speed(vp, vp / VP_VS))


def weak_body(cfg):
    """(base, height, rel) of the weak check: 120 x 80 m at L = 1000 m, scaled with L."""
    return 0.12 * cfg["lam"], 0.08 * cfg["lam"], 0.05


def fd_run(job):
    """job = (cfg, body); body None (body-free) or (base, height, rel, zb). Returns surface vz."""
    cfg, body = job
    dx, VP = cfg["dx"], cfg["vp"]
    nx = int(round(cfg["lx"] / dx)) + 1 + 2 * NBL
    nz = int(round(cfg["lz"] / dx)) + 1 + NBL + NVAC
    x0, z0 = NBL * dx, NVAC * dx  # physical origin (surface at z = 0)
    xx, zz = np.meshgrid(np.arange(nx) * dx - x0, np.arange(nz) * dx - z0, indexing="ij")
    vp = np.full((nx, nz), VP)
    vs = vp / VP_VS
    rho = np.full((nx, nz), RHO)
    if body is not None:
        base, height, rel, zb = body
        m = triangle_mask(xx, zz, base, height, cfg["xb"], zb)
        vp[m] *= 1 + rel
        vs[m] *= 1 + rel
        rho[m] *= 1 + rel
    lamf = rho * (vp**2 - 2 * vs**2)
    muf = rho * vs**2
    vac = zz < 0  # vacuum formalism: no stiffness above the surface
    lamf[vac] = 0.0
    muf[vac] = 0.0
    del vp, vs, xx, zz
    src_x = np.arange(-x0 + dx, (nx - 1) * dx - x0 - dx, dx)
    rx = np.array([cfg["xb"]] + [cfg["xb"] + o * cfg["lr"] for o in cfg["offsets_lr"]])
    _, vz = fd2d.run(dx=dx, dt=cfg["dt"], nt=cfg["nt"], lam=lamf, mu=muf, rho=rho,
                     damp=fd2d.sponge(nx, nz, NBL, top=False), src_x=src_x + x0, src_z=cfg["zs"] + z0,
                     f0=cfg["f0"], rec=np.column_stack([rx + x0, np.full(len(rx), z0)]))
    return vz


def compare(cfg, bg, tr, body):
    """Per receiver: FD Rg amplitude, 2-D model and ratio, for body (base, height, rel, zb)."""
    f0, dt, dx, VP = cfg["f0"], cfg["dt"], cfg["dx"], cfg["vp"]
    vs = VP / VP_VS
    t_p = T0_PERIODS / f0 + cfg["zs"] / VP  # direct P at the surface
    u0 = abs(tukey_dft(bg[:, 0], f0, t_p, dt, WIN_HALF_PERIODS / f0))
    lamb = Lamb(vp=VP, vs=vs, rho=RHO, freq=f0)
    base, height, rel, zb = body
    drho, dlam, dmu = perturbations(VP, vs, RHO, rel)
    n = int(np.ceil(base / dx)) + 2
    ax = np.arange(-n, n + 1) * dx
    xx, zz = np.meshgrid(ax + cfg["xb"], ax + zb, indexing="ij")
    m = triangle_mask(xx, zz, base, height, cfg["xb"], zb)
    sc = tr - bg
    t_body = T0_PERIODS / f0 + (cfg["zs"] - zb) / VP
    rows = []
    for j, o in enumerate(cfg["offsets_lr"], start=1):
        R = o * cfg["lr"]
        t_rg = t_body + R / cfg["c_r"]
        num = abs(tukey_dft(sc[:, j], f0, t_rg, dt, cfg["half"])) / u0
        model = rg_2d(lamb, xx[m], zz[m], dx * dx, cfg["xb"] + R, drho, dlam, dmu)
        rows.append(dict(offset_lr=o, kr=round(lamb.k * R, 1), numerical=num, model=model, ratio=num / model))
    return rows, sc


def fd_check(out):
    """First-run gate: compare FD with the 2-D model; returns (rows, checks)."""
    cfg = gate_config()
    bg = out[None]
    base, height, rel = WEAK_GATE
    rows, errs, speeds = [], [], []
    for dlr in DEPTHS_LR:
        zb = dlr * cfg["lr"]
        rr, sc = compare(cfg, bg, out[dlr], (base, height, rel, zb))
        for r in rr:
            rows.append(dict(depth_lr=dlr, **r))
            errs.append(abs(r["ratio"] - 1))
        t_body = T0_PERIODS / cfg["f0"] + (cfg["zs"] - zb) / cfg["vp"]
        ys = []
        for j in (1, len(cfg["offsets_lr"])):
            R = cfg["offsets_lr"][j - 1] * cfg["lr"]
            y, _ = tukey_window(sc[:, j], t_body + R / cfg["c_r"], cfg["dt"], 2.0 / cfg["f0"])
            ys.append((R, y))
        (r1, y1), (r2, y2) = ys
        xc = np.correlate(y2, y1, mode="full")
        i = int(np.argmax(xc))
        frac = 0.5 * (xc[i - 1] - xc[i + 1]) / (xc[i - 1] - 2 * xc[i] + xc[i + 1])  # parabolic peak
        speeds.append((r2 - r1) / ((i - (len(y1) - 1) + frac) * cfg["dt"]))
    vel_err = max(abs(c / cfg["c_r"] - 1) for c in speeds)
    checks = [
        dict(check="weak_rg_max_rel_err", value=round(max(errs), 3), passed=bool(max(errs) <= WEAK_TOL)),
        dict(check="rayleigh_speed_exact_ms", value=round(cfg["c_r"], 2), passed=""),
        dict(check="rayleigh_speed_fd_max_rel_err", value=round(vel_err, 4), passed=bool(vel_err <= VEL_TOL)),
    ]
    return rows, checks


WEAK_GATE = (120.0, 80.0, 0.05)  # the first run's weak body at L = 1000 m


def qs_factor(contrast: str) -> float:
    """A3b-1 factor carried by the Born-level Rg model: the cavity only, as defined."""
    if contrast != "cavity":
        return 1.0
    import pandas as pd
    fac = pd.read_csv(HERE / "a3b_factors.csv").set_index("contrast")
    return max(float(fac.loc["cavity", "factor"]), float(fac.loc["cavity", "factor_geo"]))


# ---------------------------------------------------------------- 3-D sweep on the A3 grid
def square_body_voxels(base, height, apex_depth, step):
    """Voxel centers of an apex-up square-based body (apex at apex_depth), and the voxel volume."""
    n = max(4, int(np.ceil(base / step)))
    h = base / n
    nzv = max(4, int(np.ceil(height / step)))
    hz = height / nzv
    c = (np.arange(n) + 0.5) * h - base / 2
    zc = (np.arange(nzv) + 0.5) * hz  # height above the base
    X, Y, Zu = np.meshgrid(c, c, zc, indexing="ij")
    half = 0.5 * base * (1 - Zu / height)
    m = (np.abs(X) <= half) & (np.abs(Y) <= half)
    depth = apex_depth + height - Zu  # base deepest
    return np.column_stack([X[m], Y[m], depth[m]]), h * h * hz


def sweep():
    import pandas as pd

    from akp.stage_a import a3_detectability as a3
    cases = pd.read_csv(HERE / "a3_cases.csv")
    clutter = cases.groupby("band").clutter_c92.first()
    box = pd.read_csv(HERE / "a1_box_coverage.csv")
    rep_d, op_d = box.km_nearest_recorder.to_numpy(), box.km_nearest_operating.to_numpy()
    rows = []
    for base, height, (cname, rel), (band, freq), vp, apex in product(
            a3.BASES, a3.HEIGHTS, a3.CONTRASTS.items(), a3.BANDS.items(), a3.HOST_VP, APEX_DEPTHS):
        vs = vp / a3.VP_VS
        lamb = Lamb(vp=vp, vs=vs, rho=a3.DENSITY, freq=freq)
        step = min(2 * np.pi / lamb.k / 10, base / 8, height / 8)
        xyz, dV = square_body_voxels(base, height, apex, step)
        drho, dlam, dmu = perturbations(vp, vs, a3.DENSITY, rel)
        E = rg_3d_E(lamb, xyz, dV, drho, dlam, dmu) * qs_factor(cname)
        r_km = (E / clutter[band]) ** 2 / 1e3
        # far-field flag (2026-11-10): k_R r > 10 and r beyond 2 D^2 / lambda_R, D the largest chord
        d_max = max(base * np.sqrt(2.0), np.sqrt(base**2 / 2 + height**2))
        far = bool(lamb.k * r_km * 1e3 > 10.0 and r_km * 1e3 > lamb.k * d_max**2 / np.pi)
        rows.append(dict(base_m=base, height_m=height, contrast=cname, band=band, freq_hz=freq, host_vp=vp,
                         apex_depth_m=apex, E=E, r_star_rg_km=r_km, r_star_rg_far_field=far,
                         box_frac_rg=float((rep_d <= r_km).mean()),
                         box_frac_rg_operating=float((op_d <= r_km).mean())))
    return pd.DataFrame(rows)


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sweep_checks(df):
    checks = []
    for vp in (850.0, 1200.0, 1500.0):
        sub = df[df.host_vp == vp]
        checks.append(dict(check=f"max_r_star_rg_km_vp_{int(vp)}", value=round(float(sub.r_star_rg_km.max()), 3),
                           passed=""))
        checks.append(dict(check=f"far_field_at_max_r_star_rg_vp_{int(vp)}",
                           value=int(sub.loc[sub.r_star_rg_km.idxmax(), "r_star_rg_far_field"]), passed=""))
        for col, tag in (("box_frac_rg", "recorders"), ("box_frac_rg_operating", "operating")):
            checks.append(dict(check=f"max_box_frac_rg_{tag}_vp_{int(vp)}", value=round(float(sub[col].max()), 4),
                               passed=bool(sub[col].max() <= MATERIALITY) if vp == 1200.0 else ""))
    return checks


def main() -> None:
    """First run: the gate FD check, then the 3-D sweep."""
    cfg = gate_config()
    base, height, rel = WEAK_GATE
    jobs = [None] + DEPTHS_LR
    with Pool(len(jobs)) as pool:
        res = pool.map(fd_run, [(cfg, None if d is None else (base, height, rel, d * cfg["lr"])) for d in jobs])
    rows, checks = fd_check(dict(zip(jobs, res)))
    write_csv(HERE / "a3b4_fd.csv", rows)
    df = sweep()
    df.to_csv(HERE / "a3b4_cases.csv", index=False)
    checks += sweep_checks(df)
    write_csv(HERE / "a3b4_checks.csv", checks, ["check", "value", "passed"])
    for ch in checks:
        print(ch)


def main_sweep() -> None:
    """Re-run only the 3-D sweep, keeping the first run's FD gate rows in a3b4_checks.csv."""
    import pandas as pd
    old = pd.read_csv(HERE / "a3b4_checks.csv", keep_default_na=False)
    old["passed"] = old["passed"].replace("nan", "")
    gate = old[~old.check.str.contains("_rg_km_|box_frac_rg|far_field_at_max")].to_dict("records")
    df = sweep()
    df.to_csv(HERE / "a3b4_cases.csv", index=False)
    checks = gate + sweep_checks(df)
    write_csv(HERE / "a3b4_checks.csv", checks, ["check", "value", "passed"])
    for ch in checks:
        print(ch)


def main_strong() -> None:
    """Strong-contrast stage: C_Rg for the governing geometry, with a weak re-run at 1200 m/s."""
    base, height, apex = STRONG
    zb = apex + 2 * height / 3  # centroid of the apex-up triangle
    jobs = []
    for vp in STRONG_VPS:
        cfg = strong_config(vp)
        jobs.append((cfg, None))
        jobs += [(cfg, (base, height, rel, zb)) for rel in STRONG_CASES.values()]
        if vp == 1200.0:
            wb = weak_body(cfg)
            jobs += [(cfg, (*wb, d * cfg["lr"])) for d in DEPTHS_LR]
    with Pool(5) as pool:
        res = pool.map(fd_run, jobs, chunksize=1)
    rows, checks = [], []
    for vp in STRONG_VPS:
        idx = [i for i, (c, _) in enumerate(jobs) if c["vp"] == vp]
        bg = res[idx[0]]
        cfg = jobs[idx[0]][0]
        for i in idx[1:]:
            body = jobs[i][1]
            label = "weak_check" if body[2] == 0.05 else ("near_void" if body[2] < 0 else "dense_stiff")
            rr, _ = compare(cfg, bg, res[i], body)
            for r in rr:
                rows.append(dict(label=label, host_vp=vp, zb_m=round(body[3], 1), C_rg=r["ratio"], **r))
    df = __import__("pandas").DataFrame(rows)
    df.to_csv(HERE / "a3b4_strong.csv", index=False)
    weak = df[df.label == "weak_check"]
    err = float(np.abs(weak.C_rg - 1).max())
    checks.append(dict(check="strong_stage_weak_max_rel_err", value=round(err, 3), passed=bool(err <= WEAK_TOL)))
    for (lab, vp), sub in df[df.label != "weak_check"].groupby(["label", "host_vp"]):
        spread = float(sub.C_rg.max() / sub.C_rg.min() - 1)
        checks.append(dict(check=f"C_rg_{lab}_vp_{int(vp)}", value=round(float(sub.C_rg.median()), 3), passed=""))
        checks.append(dict(check=f"C_rg_spread_{lab}_vp_{int(vp)}", value=round(spread, 3),
                           passed=bool(spread <= STABILITY_TOL)))
    write_csv(HERE / "a3b4_strong_checks.csv", checks, ["check", "value", "passed"])
    for ch in checks:
        print(ch)


def _selfcheck() -> None:
    # Poisson solid: c_R / vs = 0.9194 (textbook value)
    assert abs(rayleigh_speed(np.sqrt(3.0), 1.0) - 0.9194) < 1e-3
    # the Rayleigh eigenfunction is traction-free at the surface
    L = Lamb(vp=2000.0, vs=2000.0 / 1.8, rho=2000.0, freq=2.0)
    mu, lam = 2000.0 * (2000.0 / 1.8) ** 2, 2000.0 * (2000.0**2 - 2 * (2000.0 / 1.8) ** 2)
    gr, gz, dgz = L.g(0.0)
    h = 1e-4
    dgr = (L.g(h)[0] - L.g(-h)[0]) / (2 * h)
    tau_xz = mu * (dgr + 1j * L.k * gz)
    tau_zz = lam * (1j * L.k * gr + dgz) + 2 * mu * dgz
    scale = mu * abs(L.k * gz)
    assert abs(tau_xz) < 1e-6 * scale and abs(tau_zz) < 1e-6 * scale, (tau_xz, tau_zz)
    print("a3b4 selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--sweep" in sys.argv:
        main_sweep()
    elif "--strong" in sys.argv:
        main_strong()
    else:
        main()
