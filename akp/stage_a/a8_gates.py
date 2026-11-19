"""A8 (P4 item S5): the three validation gates for the 3-D elastic solver.

Inputs: a8_elastic3d.py (solver), a8_sphere_series.py (exact sphere), a3b4_rg.py (the half-space
Rayleigh pole, `Lamb`). Host: the A3-0 floor, vp 1200 m/s, vp/vs 1.8, 2000 kg/m3 (as A3a).
Tolerances and pass rules are the fixed ones and are not changed here.

Gates one and two share three runs on one grid (no body; weak sphere; stone sphere). A sphere of
radius 100 m (near the equal-volume radius of the claimed body) at the origin of a full space; an
upgoing plane P wave (Ricker, peak 2.5 Hz) from a source sheet 150 m below the center; receivers
500 m from the center at 0, 30, 90, 150 and 180 degrees from the propagation direction (forward,
side, back). Scattered P is measured at all five, scattered S at 30, 90 and 150 degrees (S vanishes
on the axis by symmetry, so 0 and 180 degrees have no S ratio). Frequencies 1, 2 and 4 Hz (the upper
edges of the stage's three bands), all from the one broadband run.
  Measured, as in the 2-D stage: scattered field = body run - no-body run; P by div v, S by curl v
  (the component normal to the receiver plane); amplitude = windowed DFT at f, divided by k_P or
  k_S and by the incident amplitude (div v / k_P of the no-body run at the body center).
  Reference, like for like: the reference transfer function (complex, on the FFT frequencies) is
  applied to the measured incident waveform, and the resulting trace goes through the same window
  and DFT. So window truncation acts on both sides alike. The steady-state value |H(f)| is kept in
  the column reference_cw for context.
  - Gate one (weak sphere, all properties +0.5%): reference = the 3-D Born integral over the same
    rasterized body (the solver's volume-fraction raster), at the receiver's own distance:
      div u  = 1/(lam+2mu) sum dV e^{i kp n.x'} g_p [w^2 drho (i kp - 1/R) c
               + i kp (-dlam kp^2 + 2 dmu (c^2 (-kp^2 - 3 i kp/R + 3/R^2) + i kp/R - 1/R^2))]
      curl u = 1/mu sum dV e^{i kp n.x'} g_s (R^ x n) [w^2 drho (i ks - 1/R)
               + 2 i kp dmu c (-ks^2 - 3 i ks/R + 3/R^2)]
    with g = e^{i k R} / (4 pi R), c = n.R^, n the incident direction (derived from the full-space
    Green's tensor; its far field is the swe.scattering pattern). Pass: every ratio within 20% of 1.
  - Gate two (stone sphere: vp, vs, density all doubled, A3's dense, stiff body): reference = the
    exact series at the receiver's distance. Pass: within 20%. The series' own checks come first
    (a8_sphere_series.checks); if they failed, gate two would fall back to the fixed refinement test.
  Conditioning (not a rule; fixed before any run from the references alone): each row carries
  ref_over_max, its reference over the maximum of that mode's reference over all angles at the same
  frequency, and near_null = ref_over_max < 0.10 (the 2-D stage's near-null fraction). The verdict uses every row; the verdict without near-null rows is logged beside it as context.

Gate three (free surface): an explosive point source 50 m deep in a homogeneous half-space;
vertical velocity at ranges 1800-3600 m (first vz level, h/2 below the surface).
  - Rayleigh speed: delay of the windowed vz between the nearest and farthest receivers (cross-
    correlation, parabolic peak), against the root of the Rayleigh equation. Pass: within 2%.
  - Vertical Rayleigh amplitude at 2 and 4 Hz at each range, against the far-field excitation
      |v_z| = |W(f)| h^3 |i k g_r(z_s) + dg_z/dz(z_s)| sqrt(k / (2 pi r)) |g_z(z_r) / g_z(0)|,
    W the DFT of the injected wavelet (moment rate w h^3), g the Rayleigh-pole field of a unit
    vertical surface force (residue of Lamb's problem, a3b4_rg.Lamb), used by reciprocity, with
    the stationary-phase factor from 2-D to 3-D. Source: Lamb (1904, Phil. Trans. A 203, 1); it
    equals the modal form of Aki & Richards (2002, ch. 7): r2(z_r) [k r1(z_s) + r2'(z_s)] M0 /
    (8 c U I1) sqrt(2 / (pi k r)) (leads; not opened; `lamb_vs_modal` checks the two agree).
    Pass: every ratio within 20% of 1.

Sizing (rules in a8_elastic3d): h = 10 m (16.7 points per S wavelength and 15.4 per Rayleigh
wavelength at 4 Hz); windows hold the whole pulse (Ricker half-length 1 / f0, the body's arrival
spread, T_RING of ringing); the sheet's flat half-width is the smallest that keeps its edge wave
out of every window.

Attempts: `--attempt N` (default: the latest in a8_gates.csv, else 1) tags the rows; rows of other
attempts are kept, so the CSVs log every attempt. Traces are archived in $AKP_DATA/a8.
Outputs: a8_gates.csv (one row per gate, quantity, frequency, angle or range) and a8_checks.csv.
Run: OMP_NUM_THREADS=6 PYTHONPATH=. python scripts/mem_guard.py --gb 8 --every 0.5 -- \
       python -m akp.stage_a.a8_gates [--gate surface|sphere|all] [--attempt N] [--from-archive]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from akp.stage_a import a8_elastic3d as fd  # first: sets the Devito environment
from akp.stage_a import a8_sphere_series as series

HERE = Path(__file__).parent
HOST, STONE = series.HOST, series.STONE
WEAK_REL = 0.005  # weak contrast: Born on the raster is within 7% of the exact sphere at every gate row (a8_checks)
WEAK = tuple((1 + WEAK_REL) * v for v in HOST)
TOL, SPEED_TOL, NULL_FRAC = 0.20, 0.02, 0.10  # The 2-D stage's near-null fraction (context only)
H, F0, TAPER = 10.0, 2.5, 0.1
A, R, D_SHEET, T_RING = 100.0, 500.0, 150.0, 0.3
FREQS = (1.0, 2.0, 4.0)
THETAS = (0.0, 30.0, 90.0, 150.0, 180.0)
S_THETAS = (30.0, 90.0, 150.0)
N_INC = np.array([0.0, 0.0, -1.0])  # upgoing (z is down)
SRC_DEPTH, RANGES, RG_FREQS = 50.0, (1800.0, 2400.0, 3000.0, 3600.0), (2.0, 4.0)
SURFACE_DEPTH, SPHERE_MARGIN = 700.0, 200.0  # m: half-space depth above the bottom sponge; sphere receivers to the sponge
SYM_Y = False  # the sphere gate on the half box y >= 0 with a mirror plane (a8_generous, when the full box does not fit memory)
DT_VP = None  # gate three's time step follows this P speed when set (a8_generous, the step of a run that holds a faster body)
# The log of failed attempts (the rows of each attempt stay in the CSVs; the diagnosis is recorded here).
ATTEMPT_NOTES = {
    (3, 1): "FAIL (speed passed; 6 of 8 amplitude rows failed, ratios 0.72-0.92). Two bugs, both in the measurement, "
            "found from the archived traces and one small diagnostic run (vz against the eigenfunction by depth: within 4% "
            "at every level but the first). (1) Receivers were read through Devito's interpolation of a staggered field, "
            "which averages neighbors (1/4, 1/2, 1/4): at the first vz level it mixed in the zero above the surface "
            "(factor 0.75). Fixed in a8_elastic3d.simulate: samples are read by index. (2) The side margins were sized to "
            "the wall, but a grazing Rayleigh wave is reflected from the sponge's inner edge (9-45% of the direct wave in "
            "the archived traces, inside the windows at 3000 and 3600 m). Fixed in surface_setup: margins to the inner edge. "
            "Grid of attempt 1: 461x221x102. The gate, its ranges, frequencies, windows and tolerances are unchanged.",
}
GATE_COLS = ["gate", "attempt", "case", "quantity", "freq_hz", "angle_deg", "range_m", "numerical", "reference",
             "ratio", "tolerance", "passed", "reference_cw", "ratio_cw", "ref_over_max", "near_null"]
CHECK_COLS = ["gate", "attempt", "check", "value", "passed", "detail"]


# ---------------------------------------------------------------- references
def born_div_curl(points, cells, vols, host, inclusion, freq, n=N_INC):
    """Born (div u_sc, curl u_sc) at points (m, 3), per unit incident displacement along n (module docstring)."""
    vp, vs, rho = host
    drho, dlam, dmu = series.perturbations(host, inclusion)
    w = 2 * np.pi * freq
    kp, ks = w / vp, w / vs
    ph = vols * np.exp(1j * kp * (cells @ n))
    D, C = [], []
    for p in np.atleast_2d(points):
        d = p - cells
        Rr = np.linalg.norm(d, axis=1)
        rh = d / Rr[:, None]
        c = rh @ n
        gp, gs = np.exp(1j * kp * Rr) / (4 * np.pi * Rr), np.exp(1j * ks * Rr) / (4 * np.pi * Rr)
        D.append(np.sum(ph * gp * (w**2 * drho * (1j * kp - 1 / Rr) * c + 1j * kp * (
            -dlam * kp**2 + 2 * dmu * (c**2 * (-kp**2 - 3j * kp / Rr + 3 / Rr**2) + 1j * kp / Rr - 1 / Rr**2))))
            / (rho * vp**2))
        amp = ph * gs * (w**2 * drho * (1j * ks - 1 / Rr) + 2j * kp * dmu * c * (-ks**2 - 3j * ks / Rr + 3 / Rr**2))
        C.append((amp[:, None] * np.cross(rh, n)).sum(0) / (rho * vs**2))
    return np.array(D), np.array(C)


def polar(p):
    """(r, theta from the propagation direction) of points (m, 3) about the origin."""
    r = np.linalg.norm(p, axis=1)
    return r, np.arccos(np.clip(p @ N_INC / r, -1, 1))


def series_div_curl(points_p, points_s, inclusion, freq):
    """Exact (div u_sc at points_p, curl_y u_sc at points_s), receivers in the x-z plane (sign of curl dropped)."""
    kp, ks = 2 * np.pi * freq / HOST[0], 2 * np.pi * freq / HOST[1]
    out = []
    for pts, col, k in ((points_p, 0, kp), (points_s, 1, ks)):
        vals = [series.amplitudes(freq=freq, a=A, host=HOST, inclusion=inclusion, theta=th, r=r)[col][0]
                for r, th in zip(*polar(pts))]
        out.append(-k * np.array(vals))
    return out


def lamb_vs_modal(freq=2.0, z_s=SRC_DEPTH):
    """Relative difference between the Lamb-residue excitation and the Aki & Richards modal form."""
    from scipy.integrate import quad

    from akp.stage_a.a3b4_rg import Lamb
    lamb = Lamb(vp=HOST[0], vs=HOST[1], rho=HOST[2], freq=freq)
    k, c = lamb.k, lamb.c
    r1, r2 = (lambda z: lamb.g(z)[0].real), (lambda z: lamb.g(z)[1].imag)  # u_x = r1, u_z = i r2
    i1 = 0.5 * HOST[2] * quad(lambda z: r1(z) ** 2 + r2(z) ** 2, 0, 40 / k, limit=200)[0]
    gr, _, dgz = lamb.g(z_s)
    eps = 1e-3
    modal = abs(r2(0.0) * (k * r1(z_s) + (r2(z_s + eps) - r2(z_s - eps)) / (2 * eps))) / (8 * c * c * i1)  # U = c
    return float(abs(1j * k * gr + dgz) * k / 2 / modal - 1)


def edge_wave_2d():
    """The accuracy rule applied to the 2-D weak check at 2 Hz (a3b2_numerical): delays (s) of the source line's edge
    waves, from the left and right sponge edges, after the plane wave at the body. An earlier run recorded a second
    incident wave there 1.923 s after the first, its source not identified."""
    from akp.stage_a import a3b2_numerical as n2
    g = n2.setup(2.0)
    d = g["zs"] - g["zb"]
    return tuple(float((np.hypot(w, d) - d) / n2.VP) for w in (g["xb"], g["lx"] - g["xb"]))


def rayleigh_reference(freq, r, z_r, z_s=SRC_DEPTH):
    """|v_z| / |W(f)| of the far-field Rayleigh wave (module docstring), and |g_r(0) / g_z(z_r)| (H/V)."""
    from akp.stage_a.a3b4_rg import Lamb
    lamb = Lamb(vp=HOST[0], vs=HOST[1], rho=HOST[2], freq=freq)
    gr, _, dgz = lamb.g(z_s)
    gz_r, gz_0, gr_0 = lamb.g(z_r)[1], lamb.g(0.0)[1], lamb.g(0.0)[0]
    amp = H**3 * abs(1j * lamb.k * gr + dgz) * np.sqrt(lamb.k / (2 * np.pi * r)) * abs(gz_r / gz_0)
    return float(amp), float(abs(gr_0 / gz_r)), lamb


# ---------------------------------------------------------------- sphere runs (gates one and two)
def sphere_setup():
    vp, vs, _ = HOST
    tp = 1 / F0
    pre = {m: tp + A / vp + A / v for m, v in (("P", vp), ("S", vs))}
    post = {m: pre[m] + T_RING for m in pre}
    edge = tp + post["S"] + TAPER  # the edge wave must reach the body this long after the plane wave
    w_flat = float(np.ceil(np.sqrt((vp * edge + D_SHEET) ** 2 - D_SHEET**2) / H) * H)
    zext = R + SPHERE_MARGIN
    g = fd.make_grid(lo=(-w_flat, 0.0 if SYM_Y else -w_flat, -zext), hi=(w_flat, w_flat, zext), h=H, nbl=fd.NBL, sym_y=SYM_Y)
    t0 = tp + 0.1
    tb = t0 + D_SHEET / vp
    dt = fd.time_step(g, STONE[0])
    nt = int((tb + R / vs + post["S"] + TAPER) / dt) + 2
    th = np.radians(THETAS)
    nodes = fd.snap(g, np.vstack([R * np.column_stack([np.sin(th), 0 * th, -np.cos(th)]), np.zeros((1, 3))]))
    return dict(g=g, dt=dt, nt=nt, t0=t0, tb=tb, tp=tp, pre=pre, post=post, w_flat=w_flat, nodes=nodes,
                edge_delay=(np.hypot(w_flat, D_SHEET) - D_SHEET) / vp, edge_needed=edge)


def archive(group, attempt, name):
    from akp.paths import data_dir
    d = data_dir("a8") / f"{group}_att{attempt}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.npz"


def run_or_load(group, attempt, name, runner, args, from_archive):
    """Load the archived traces, or run `runner(*args)` in a fresh process (Devito keeps a finished
    run's arrays alive into the next one, which would double the memory peak) and archive them."""
    path = archive(group, attempt, name)
    if from_archive:
        z = np.load(path)
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1) as pool:
        out = pool.apply(runner, args)
    np.savez_compressed(path, **out)
    print(f"{group}/{name}: {out['npts'] / 1e6:.1f}M points, {out['nt']} steps, {out['seconds']:.0f} s, "
          f"{out['mpts_per_s']:.0f} Mpts/s, peak rss {out['rss_gb']:.2f} GB", flush=True)
    return out


def run_sphere(case):
    s = sphere_setup()
    body, props = (None, None) if case == "bg" else (fd.sphere((0, 0, 0), A), WEAK if case == "weak" else STONE)
    par = fd.effective_params(s["g"], lambda z: HOST, body, props)
    return fd.simulate(s["g"], par, dt=s["dt"], nt=s["nt"], v_ref=HOST[0], rec=fd.ps_points(s["nodes"], H),
                       plane=dict(z=D_SHEET, angle_deg=0.0, vp=HOST[0], f0=F0, t0=s["t0"],
                                  flat=(s["w_flat"], s["w_flat"])))


def measure_sphere(bg, tr, case, s):
    """Gate rows for one sphere case ('weak' -> gate one, 'stone' -> gate two), and check rows."""
    vp, vs, _ = HOST
    dt, nt, tb, tp = s["dt"], s["nt"], s["tb"], s["tp"]
    nodes = s["nodes"][:-1]
    n_nodes = len(s["nodes"])
    div, curl = fd.div_curl(fd.scattered(tr, bg), n_nodes, H)
    inc = fd.div_curl(bg, n_nodes, H)[0][:, -1]  # div v of the incident wave at the body center
    t = np.arange(nt) * dt
    nfft = 2 * nt
    fr = np.fft.rfftfreq(nfft, dt)
    use = (fr > 0) & (fr <= 4 * F0)
    kpf = 2 * np.pi * fr[use] / vp
    # numpy's FFT is the conjugate of the exp(-i w t) convention: Y = conj(H) X, and s = div / (i k_p)
    S = np.fft.rfft(inc * fd.window(t, tb - tp, tb + tp, TAPER), nfft)[use] / (-1j * kpf)
    pts_s = nodes + np.array([0.5, 0.0, 0.5]) * H  # where curl_y is formed
    if case == "weak":
        cells, vols = fd.body_cells(s["g"], fd.sphere((0, 0, 0), A))
        HP = np.array([born_div_curl(nodes, cells, vols, HOST, WEAK, f)[0] for f in fr[use]])
        HS = np.array([born_div_curl(pts_s, cells, vols, HOST, WEAK, f)[1][:, 1] for f in fr[use]])
        inclusion = WEAK
    else:
        both = [series_div_curl(nodes, pts_s, STONE, f) for f in fr[use]]
        HP, HS = np.array([b[0] for b in both]), np.array([b[1] for b in both])
        inclusion = STONE
    fine = np.radians(np.arange(0.0, 180.5, 1.0))
    rows, in_window = [], []
    for mode, traces, Hm, v, pts, thetas in (("P", div, HP, vp, nodes, THETAS), ("S", curl[:, :, 1], HS, vs, pts_s, S_THETAS)):
        r_act, th_act = polar(pts)
        for i, theta in enumerate(THETAS):
            if theta not in thetas:
                continue
            Y = np.zeros(len(fr), complex)
            Y[use] = np.conj(Hm[:, i]) * S
            y = np.fft.irfft(Y, nfft)[:nt]
            lo, hi = tb + r_act[i] / v - s["pre"][mode], tb + r_act[i] / v + s["post"][mode]
            in_window.append(float(np.sum((y * fd.window(t, lo, hi, TAPER)) ** 2) / np.sum(np.fft.irfft(Y, nfft) ** 2)))
            for f in FREQS:
                k = 2 * np.pi * f / v
                a_inc = abs(fd.win_dft(inc, f, tb - tp, tb + tp, dt, TAPER)) / (2 * np.pi * f / vp)
                num = abs(fd.win_dft(traces[:, i], f, lo, hi, dt, TAPER)) / k / a_inc
                ref = abs(fd.win_dft(y, f, lo, hi, dt, TAPER)) / k / a_inc
                j = int(np.argmin(np.abs(fr[use] - f)))
                assert abs(fr[use][j] - f) < 0.5 / (nfft * dt) + 1e-9
                col = 0 if mode == "P" else 1
                cw = abs(series_div_curl(pts[i:i + 1], pts[i:i + 1], inclusion, f)[col][0]) / k if case == "stone" \
                    else abs((born_div_curl(pts[i:i + 1], cells, vols, HOST, WEAK, f)[col]).ravel()[0 if col == 0 else 1]) / k
                pattern = np.abs(series.amplitudes(freq=f, a=A, host=HOST, inclusion=inclusion, theta=fine, r=R)[col])
                rom = float(abs(series.amplitudes(freq=f, a=A, host=HOST, inclusion=inclusion, theta=th_act[i],
                                                  r=r_act[i])[col][0]) / pattern.max())
                rows.append(dict(gate=1 if case == "weak" else 2, case=case, quantity=f"scattered_{mode}", freq_hz=f,
                                 angle_deg=theta, range_m=round(float(r_act[i]), 1), numerical=num, reference=ref,
                                 ratio=num / ref, tolerance=TOL, passed=bool(abs(num / ref - 1) <= TOL),
                                 reference_cw=cw, ratio_cw=num / cw, ref_over_max=rom, near_null=bool(rom < NULL_FRAC)))
    gate = rows[0]["gate"]
    dev = np.array([abs(r["ratio"] - 1) for r in rows])
    good = np.array([not r["near_null"] for r in rows])
    after = t > tb + tp + TAPER
    late = np.abs(inc[after]).max() / np.abs(inc).max()
    onset = float(t[after][np.argmax(np.abs(inc[after]) > 0.01 * np.abs(inc).max())] - tb) if late > 0.01 else float("nan")
    a_div = abs(fd.win_dft(inc, F0, tb - tp, tb + tp, dt, TAPER)) / (2 * np.pi * F0 / vp)
    checks = [
        dict(check=f"gate{gate}_rows", value=len(rows), passed=""),
        dict(check=f"gate{gate}_rows_failed", value=int((dev > TOL).sum()), passed=""),
        dict(check=f"gate{gate}_max_abs_ratio_minus_1", value=float(dev.max()), passed=bool(dev.max() <= TOL),
             detail="Verdict: every row"),
        dict(check=f"gate{gate}_max_abs_ratio_minus_1_without_near_nulls", value=float(dev[good].max()),
             passed=bool(dev[good].max() <= TOL), detail=f"context, not part of the verdict: {int((~good).sum())} near-null rows left out"),
        dict(check=f"gate{gate}_reference_energy_in_window_min", value=min(in_window), passed=bool(min(in_window) > 0.9),
             detail="share of each synthesized reference trace inside its window (convention check)"),
        dict(check=f"gate{gate}_scattered_max_over_incident", value=float(max(r["numerical"] for r in rows)), passed=""),
    ]
    if case == "weak":
        checks += [dict(check="sphere_incident_late_over_peak", value=float(late), passed="",
                        detail="no-body run at the body center, after the incident window: the sheet's edge wave"),
                   dict(check="sphere_incident_late_onset_s", value=onset, passed=bool(onset >= s["edge_delay"] - 2 * tp),
                        detail=f"first time after the pulse that |div v| passes 1% of its peak, from the plane wave's arrival; "
                               f"rule 4a predicts the edge wave centered at {s['edge_delay']:.2f} s, starting about {s['edge_delay'] - tp:.2f} s"),
                   dict(check="sphere_incident_amp_div_over_kp", value=float(a_div), passed="",
                        detail="incident velocity amplitude at f0 from div v / k_P")]
    return rows, checks


def sphere_reference_checks(s):
    """Reference-only checks (no FD): the series' own, and Born on the raster against the weak series."""
    rows = [dict(check=r["check"], value=r["value"], passed=r["passed"], detail=f"tol {r['tol']}") for r in series.checks(A)]
    cells, vols = fd.body_cells(s["g"], fd.sphere((0, 0, 0), A))
    rows.append(dict(check="raster_sphere_volume_rel_err", value=float(vols.sum() / (4 * np.pi * A**3 / 3) - 1), passed=""))
    nodes = s["nodes"][:-1]
    pts_s = nodes + np.array([0.5, 0.0, 0.5]) * H
    isS = [THETAS.index(a) for a in S_THETAS]
    for label, inc in (("1e-4", tuple((1 + 1e-4) * v for v in HOST)), ("gate", WEAK)):
        errs = []
        for f in FREQS:
            kp, ks = 2 * np.pi * f / HOST[0], 2 * np.pi * f / HOST[1]
            bp = np.abs(born_div_curl(nodes, cells, vols, HOST, inc, f)[0]) / kp
            bs = np.abs(born_div_curl(pts_s, cells, vols, HOST, inc, f)[1][:, 1]) / ks
            sp, ss = series_div_curl(nodes, pts_s, inc, f)
            errs += list(bp / (np.abs(sp) / kp) - 1) + list((bs / (np.abs(ss) / ks) - 1)[isS])
        worst = float(np.abs(errs).max())
        rows.append(dict(check=f"born_raster_vs_series_max_rel_diff_contrast_{label}", value=worst,
                         passed=bool(worst < 0.02) if label == "1e-4" else "",
                         detail="finite-distance Born integral on the raster against the exact series at the gate rows; "
                                + ("validates the integral" if label == "1e-4" else "the nonlinearity the weak gate carries")))
    return rows


# ---------------------------------------------------------------- half-space run (gate three)
def surface_setup():
    """Half-space grid. Margins by rule 4(b) of a8_elastic3d, measured to the sponge's inner edge:
    a Rayleigh wave reflected there must start after the last window has closed, so its extra path
    is at least c_R (2 / f0 + 2 TAPER) (the window's half-length plus the pulse's)."""
    vp = HOST[0]
    from akp.stage_a.a3b4_rg import rayleigh_speed
    c_r = rayleigh_speed(vp, HOST[1])
    extra = c_r * (2 / F0 + 2 * TAPER)
    wy = float(np.ceil(np.sqrt((RANGES[-1] + extra) ** 2 - RANGES[-1] ** 2) / 2 / H) * H)  # image across a side edge
    xm = float(np.ceil(extra / 2 / H) * H)  # behind the source and beyond the last receiver
    g = fd.make_grid(lo=(-xm, -wy, 0.0), hi=(RANGES[-1] + xm, wy, SURFACE_DEPTH), h=H, free_surface=True, nbl=fd.NBL)
    dt = fd.time_step(g, DT_VP or vp)
    t0 = 1 / F0 + 0.1
    nt = int((t0 + RANGES[-1] / c_r + 1 / F0 + 3 * TAPER) / dt) + 2
    nodes = fd.snap(g, [(r, 0.0, 0.0) for r in RANGES])
    return dict(g=g, dt=dt, nt=nt, t0=t0, c_r=c_r, nodes=nodes, wy=wy, xm=xm)


def run_surface():
    s = surface_setup()
    par = fd.effective_params(s["g"], lambda z: HOST)
    nat = fd.native(s["nodes"], H)
    return fd.simulate(s["g"], par, dt=s["dt"], nt=s["nt"], v_ref=HOST[0], rec=dict(vx=nat["vx"], vz=nat["vz"]),
                       point=dict(xyz=(0.0, 0.0, SRC_DEPTH), f0=F0, t0=s["t0"]))


def measure_surface(tr, s):
    dt, nt, t0, c_r = s["dt"], s["nt"], s["t0"], s["c_r"]
    t = np.arange(nt) * dt
    half = 1 / F0 + TAPER

    def lohi(r):
        return t0 + r / c_r - half, t0 + r / c_r + half
    y1, y2 = (tr["vz"][:, i] * fd.window(t, *lohi(RANGES[i]), TAPER) for i in (0, -1))
    xc = np.correlate(y2, y1, mode="full")
    i = int(np.argmax(xc))
    frac = 0.5 * (xc[i - 1] - xc[i + 1]) / (xc[i - 1] - 2 * xc[i] + xc[i + 1])  # parabolic peak, as a3b4_rg
    c_fd = (RANGES[-1] - RANGES[0]) / ((i - (nt - 1) + frac) * dt)
    rows = [dict(gate=3, case="half_space", quantity="rayleigh_speed_ms", freq_hz="", angle_deg="",
                 range_m=f"{RANGES[0]:g}-{RANGES[-1]:g}", numerical=c_fd, reference=c_r, ratio=c_fd / c_r,
                 tolerance=SPEED_TOL, passed=bool(abs(c_fd / c_r - 1) <= SPEED_TOL))]
    checks = []
    wav = fd.ricker(t, F0, t0)
    for f in RG_FREQS:
        W = abs(np.sum(wav * np.exp(2j * np.pi * f * t)) * dt)
        for j, r in enumerate(RANGES):
            ref, hv_ref, lamb = rayleigh_reference(f, r, H / 2, SRC_DEPTH)
            az, ax, lag = fd.rayleigh_measure(tr["vx"][:, j], tr["vz"][:, j], f, *lohi(r), dt, TAPER)
            num = az / W
            rows.append(dict(gate=3, case="half_space", quantity="rayleigh_vz_over_source", freq_hz=f, angle_deg="",
                             range_m=r, numerical=num, reference=ref, ratio=num / ref, tolerance=TOL,
                             passed=bool(abs(num / ref - 1) <= TOL)))
            checks.append(dict(check=f"gate3_h_over_v_fd_over_exact_f{f:g}_r{r:g}", value=float(ax / az / hv_ref), passed="",
                               detail=f"particle motion: radial/vertical phase {lag:.1f} deg (a quarter cycle, "
                                      f"less k h / 2 = {np.degrees(lamb.k * H / 2):.1f} deg for the half-cell offset of vx)"))
    amp = [r for r in rows if r["quantity"] == "rayleigh_vz_over_source"]
    dev = max(abs(r["ratio"] - 1) for r in amp)
    checks = [
        dict(check="gate3_rayleigh_speed_rel_err", value=float(c_fd / c_r - 1), passed=rows[0]["passed"],
             detail=f"exact {c_r:.2f} m/s; tolerance {SPEED_TOL}"),
        dict(check="gate3_rows", value=len(rows), passed=""),
        dict(check="gate3_rows_failed", value=int(sum(not r["passed"] for r in rows)), passed=""),
        dict(check="gate3_amp_max_abs_ratio_minus_1", value=float(dev), passed=bool(dev <= TOL), detail="Verdict: every range and frequency"),
    ] + checks
    return rows, checks


# ---------------------------------------------------------------- bookkeeping
def run_checks(name, out, g):
    return [dict(check=f"{name}_grid_shape", value="x".join(str(n) for n in g["shape"]), passed="", detail=f"h {g['h']} m, sponge {g['nbl']} cells"),
            dict(check=f"{name}_points_millions", value=round(out["npts"] / 1e6, 2), passed=""),
            dict(check=f"{name}_time_step_s", value=float(out["dt"]), passed="", detail=f"{int(out['nt'])} steps"),
            dict(check=f"{name}_run_seconds", value=round(float(out["seconds"]), 1), passed="", detail=f"{float(out['mpts_per_s']):.0f} Mpts/s on OMP_NUM_THREADS threads"),
            dict(check=f"{name}_peak_rss_gb", value=round(float(out["rss_gb"]), 2), passed=bool(out["rss_gb"] < 8.0),
                 detail=f"peak of the run's process; {out['rss_gb'] * 1e9 / out['npts']:.0f} bytes per point")]


def merge_csv(path, rows, cols, gates, attempt):
    """Replace the rows of (gates, attempt) in the CSV; rows of other gates and attempts are kept."""
    import pandas as pd
    new = pd.DataFrame(rows).reindex(columns=cols)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        old = old[~(old.gate.astype(str).isin([str(x) for x in gates]) & (old.attempt.astype(int) == attempt))]
        new = pd.concat([old.reindex(columns=cols), new], ignore_index=True)
    new.to_csv(path, index=False)


def latest_attempt(gates):
    import pandas as pd
    path = HERE / "a8_gates.csv"
    if not path.exists():
        return 1
    df = pd.read_csv(path)
    sub = df[df.gate.astype(str).isin([str(x) for x in gates])]
    return int(sub.attempt.max()) if len(sub) else 1


def main(which="all", attempt=None, from_archive=False) -> None:
    do_surface, do_sphere = which in ("surface", "all"), which in ("sphere", "all")
    common = [dict(check="points_per_s_wavelength_at_4hz", value=round(HOST[1] / 4.0 / H, 1), passed=bool(HOST[1] / 4.0 / H >= fd.PPW)),
              dict(check="lamb_residue_vs_modal_rel_diff", value=(lv := lamb_vs_modal()), passed=bool(abs(lv) < 1e-4),
                   detail="gate three's reference, two derivations"),
              dict(check="a3b2_weak_check_edge_wave_delay_right_s", value=round(edge_wave_2d()[1], 3), passed="",
                   detail="2-D stage, weak check at 2 Hz: predicted delay of the source line's edge wave from the right sponge edge; "
                          "an earlier run recorded an unidentified second incident wave at 1.923 s (a lead, not a result)"),
              dict(check="a3b2_weak_check_edge_wave_delay_left_s", value=round(edge_wave_2d()[0], 3), passed="",
                   detail="the same from the left sponge edge (inside the 2 Hz pulse's tail, so not separable there)")]
    if not from_archive:  # absolute source scaling and the P/S stencils, in a full space (not a gate quantity)
        ratio, leak, _ = fd.explosion_check()
        common += [dict(check="explosion_full_space_div_over_analytic_max_dev", value=float(np.abs(ratio - 1).max()),
                        passed=bool(np.abs(ratio - 1).max() < 0.03)),
                   dict(check="explosion_full_space_curl_leak_max", value=float(leak.max()), passed=bool(leak.max() < 0.03))]
        merge_csv(HERE / "a8_checks.csv", [dict(gate="solver", attempt=0, **c) for c in common[4:]], CHECK_COLS, ["solver"], 0)
    merge_csv(HERE / "a8_checks.csv", [dict(gate="reference", attempt=0, **c) for c in common[:4]], CHECK_COLS, ["reference"], 0)
    for (gate, att), note in ATTEMPT_NOTES.items():
        merge_csv(HERE / "a8_checks.csv", [dict(gate=f"log{gate}", attempt=att, check=f"gate{gate}_attempt{att}_diagnosis",
                                                value="", passed=False, detail=note)], CHECK_COLS, [f"log{gate}"], att)
    if do_surface:  # the small case first: its memory per point is checked before the larger grid runs
        att = attempt or latest_attempt([3])
        s = surface_setup()
        tr = run_or_load("surface", att, "half_space", run_surface, (), from_archive)
        rows, checks = measure_surface(tr, s)
        checks = run_checks("surface", tr, s["g"]) + checks + [
            dict(check="surface_points_per_rayleigh_wavelength_at_4hz", value=round(s["c_r"] / 4.0 / H, 1), passed=""),
            dict(check="surface_source_depth_m", value=SRC_DEPTH, passed="", detail="scheme: stress imaging, z order 2-4-6-8 over the top four cells"),
            dict(check="surface_side_half_width_m", value=s["wy"], passed="", detail="receiver line to the side sponges' inner edges (rule 4b)"),
            dict(check="surface_end_margin_m", value=s["xm"], passed="", detail="source and last receiver to the end sponges' inner edges"),
            dict(check="surface_depth_m", value=SURFACE_DEPTH, passed="", detail="to the bottom sponge's inner edge")]
        merge_csv(HERE / "a8_gates.csv", [dict(attempt=att, **r) for r in rows], GATE_COLS, [3], att)
        merge_csv(HERE / "a8_checks.csv", [dict(gate=3, attempt=att, **c) for c in checks], CHECK_COLS, [3], att)
        for r in rows:
            print(r["quantity"], r["freq_hz"], r["range_m"], f"{r['ratio']:.4f}", r["passed"])
    if do_sphere:
        att = attempt or latest_attempt([1, 2])
        s = sphere_setup()
        ref = sphere_reference_checks(s)
        merge_csv(HERE / "a8_checks.csv", [dict(gate="reference", attempt=0, **c) for c in common[:4] + ref], CHECK_COLS, ["reference"], 0)
        series_ok = all(c["passed"] for c in ref if c["check"].startswith("series_"))
        if not series_ok:
            raise SystemExit("the series failed its own checks: gate two must use the fixed grid-refinement test (not implemented: not needed)")
        import pandas as pd  # memory: scale the small case (the half-space run) before the larger grid runs
        small = pd.read_csv(HERE / "a8_checks.csv") if (HERE / "a8_checks.csv").exists() else pd.DataFrame(columns=CHECK_COLS)
        small = small[small.check.isin(["surface_peak_rss_gb", "surface_points_millions"])].sort_values("attempt").groupby("check").value.last()
        per_point = float(small["surface_peak_rss_gb"]) * 1e3 / float(small["surface_points_millions"]) if len(small) == 2 else 26 * 4 * 1.3
        predicted = per_point * np.prod(s["g"]["shape"]) / 1e9
        print(f"sphere grid {s['g']['shape']}, {s['nt']} steps, predicted peak {predicted:.1f} GB ({per_point:.0f} bytes per point)", flush=True)
        if predicted > 8.0 and not from_archive:
            raise SystemExit("predicted memory over the 8 GB limit: shrink the grid")
        bg = run_or_load("sphere", att, "bg", run_sphere, ("bg",), from_archive)
        checks = run_checks("sphere", bg, s["g"]) + [
            dict(check="sphere_predicted_peak_gb_from_small_case", value=round(float(predicted), 2), passed=bool(predicted <= 8.0),
                 detail=f"{per_point:.0f} bytes per point measured on the half-space run"),
            dict(check="sphere_flat_half_width_m", value=s["w_flat"], passed=bool(s["edge_delay"] >= s["edge_needed"]),
                 detail=f"edge wave {s['edge_delay']:.2f} s after the plane wave; windows need {s['edge_needed']:.2f} s"),
            dict(check="sphere_receiver_margin_m", value=SPHERE_MARGIN, passed="",
                 detail="axial receivers to the top and bottom sponge (reached near normal incidence, where it is matched)")]
        rows = []
        for case in ("weak", "stone"):
            tr = run_or_load("sphere", att, case, run_sphere, (case,), from_archive)
            r, c = measure_sphere(bg, tr, case, s)
            rows += r
            checks += c + [dict(check=f"sphere_{case}_run_seconds", value=round(float(tr["seconds"]), 1), passed=""),
                           dict(check=f"sphere_{case}_peak_rss_gb", value=round(float(tr["rss_gb"]), 2), passed=bool(tr["rss_gb"] < 8.0))]
            del tr
        merge_csv(HERE / "a8_gates.csv", [dict(attempt=att, **r) for r in rows], GATE_COLS, [1, 2], att)
        merge_csv(HERE / "a8_checks.csv", [dict(gate="1-2", attempt=att, **c) for c in checks], CHECK_COLS, ["1-2"], att)
        for r in rows:
            print(r["gate"], r["quantity"], r["freq_hz"], r["angle_deg"], f"{r['ratio']:.4f}", f"cw {r['ratio_cw']:.4f}",
                  f"ref/max {r['ref_over_max']:.2f}", r["passed"])


def _selfcheck() -> None:
    """Reference-side asserts (no gate FD): Born's far field, the Lamb / modal identity, the row counts."""
    cells, vols = fd.body_cells(fd.make_grid(lo=(-150,) * 3, hi=(150,) * 3, h=H, nbl=2), fd.sphere((0, 0, 0), A))
    weak = tuple((1 + 1e-4) * v for v in HOST)
    th = np.radians([30.0, 90.0, 150.0])
    far = 1e5 * np.column_stack([np.sin(th), 0 * th, -np.cos(th)])
    for f in (1.0, 4.0):  # the finite-distance integral tends to the far-field Born of a sphere
        kp, ks = 2 * np.pi * f / HOST[0], 2 * np.pi * f / HOST[1]
        D, C = born_div_curl(far, cells, vols, HOST, weak, f)
        bp, bs = series.born_sphere(freq=f, a=A, host=HOST, inclusion=weak, theta=th)
        assert np.allclose(np.abs(D) / kp * 1e5, np.abs(bp), rtol=0.02), (f, np.abs(D) / kp * 1e5 / np.abs(bp))
        assert np.allclose(np.abs(C[:, 1]) / ks * 1e5, np.abs(bs), rtol=0.02), (f, np.abs(C[:, 1]) / ks * 1e5 / np.abs(bs))
    assert abs(lamb_vs_modal()) < 1e-4
    s = sphere_setup()
    assert s["edge_delay"] >= s["edge_needed"] and np.allclose(polar(s["nodes"][:-1])[0], R, rtol=0.01)
    # the measurement chain on synthetic traces (no FD): the exact series field at the receiver stencil
    # points, driven by a Ricker, must come back through measure_sphere with every ratio near one
    nt, dt = s["nt"], s["dt"]
    nfft = 2 * nt
    fr = np.fft.rfftfreq(nfft, dt)
    use = np.flatnonzero((fr > 0) & (fr <= 4 * F0))
    X = np.fft.rfft(fd.ricker(np.arange(nt) * dt, F0, s["tb"]), nfft)
    pts = fd.ps_points(s["nodes"], H)
    for case, inc, tol in (("stone", STONE, 0.02), ("weak", WEAK, 0.10)):  # weak: series against Born, nonlinearity included
        bg, tr = {}, {}
        for d, c in enumerate(("vx", "vy", "vz")):
            Y = N_INC[d] * X[:, None] * np.exp(-2j * np.pi * fr[:, None] * (pts[c] @ N_INC / HOST[0])[None, :])
            bg[c] = np.fft.irfft(Y, nfft, axis=0)[:nt]
            Ysc = np.zeros_like(Y)
            for j in use:
                Ysc[j] = np.conj(series.displacement(freq=fr[j], a=A, host=HOST, inclusion=inc, points=pts[c], axis=N_INC)[:, d]) * X[j]
            tr[c] = bg[c] + np.fft.irfft(Ysc, nfft, axis=0)[:nt]
        tr.update(dt=dt, nt=nt)
        bg.update(dt=dt, nt=nt)
        rows, checks = measure_sphere(bg, tr, case, s)
        dev = max(abs(r["ratio"] - 1) for r in rows)
        print(f"synthetic {case}: {len(rows)} rows, max |ratio - 1| = {dev:.4f}; reference energy in window >= "
              f"{[c['value'] for c in checks if 'energy_in_window' in c['check']][0]:.3f}")
        assert len(rows) == 24 and dev < tol, [(r["quantity"], r["freq_hz"], r["angle_deg"], round(r["ratio"], 3)) for r in rows]
        assert all(c["passed"] for c in checks if "energy_in_window" in c["check"])
    print("a8_gates selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        arg = dict(zip(sys.argv[1::1], sys.argv[2::1]))
        main(which=arg.get("--gate", "all"), attempt=int(arg["--attempt"]) if "--attempt" in arg else None,
             from_archive="--from-archive" in sys.argv)
