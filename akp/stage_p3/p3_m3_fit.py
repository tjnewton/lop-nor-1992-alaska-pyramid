"""P4 M3 stage two: multi-line, multi-family least-squares fit at a detector cluster.

Inputs: the samples of every traverse line within REACH_M of a cluster position (collect: per-line
frames with X, Y, the IGRF-removed field and the altitude channel), and the survey's line spacing.
The data are the raw IGRF-removed values, block-averaged to STEP_M along each line; the background is
this module's own per-line constant and slope, solved with the body (not the detector's regional).
Samples are weighted by a Gaussian of their distance from the cluster (TAPER_M), so that the fit is
of the cluster's own anomaly; every residual sum, R2 and energy below is weighted in that way.

Method. Four families are fitted by least squares with those weights and no noise model:
- pyramid: base PYR_BASES, turn PYR_TURNS, top depth TOPS_M, height p3_mag_forward.HEIGHT_M (fixed);
- plug: a vertical-sided square prism, width and depth extent PLUGS, top depth TOPS_M;
- dike: a DIKE_LENGTH_M long vertical sheet, width and depth extent DIKES, strike DIKE_STRIKES_DEG
  (refined to half that step), top depth TOPS_M, position searched across strike only;
- dipole: a point dipole, depth DIPOLE_DEPTHS_M.
For every geometry the field is linear in a free magnetization vector (three columns), so those and
the per-line background are solved linearly. Geometry and position are searched coarse to fine:
a coarse grid of positions (along the line and across it, out to half the spacing) and of top depths
with every shape; then, twice, position at a finer step for the best shape followed by every shape
and every top depth at that position; then, for the N_POLISH best shapes, a pattern search
(POLISH_ITERS steps) on position and top depth as continuous values. Speed: for each prism shape the tensor U_ij of p3_mag_fast.prism_tensor is
tabulated once in the body frame on (x, y, d), d the height of the sensor above the body's top, on a
grid that is asinh-spaced in x and y and log-spaced in d (one quadrant; the others by symmetry), and
interpolated trilinearly. A turn is a rotation into the body frame, as in p3_mag_fast.response.
The self-check measures the table against p3_mag_fast.response (TABLE_TOL of the peak).

fit() returns one dict of descriptors: residual variance and R2 per family; margin = ln(RSS of the
best other family / RSS of the pyramid family) and the same against each family; the fitted pyramid's
base, top, turn and position offset; its contrast |m| (SI, comparable to p3_g1b_run.MAG_CLASSES), its
component along the inducing field and the angle between m and that field; r2_pyr_ind and k_ind, the
best pyramid with magnetization along the field only (the built case is induced only);
n_lines, n_lines_reach (lines with a sample within base/2 + top + clearance of the fitted position),
share_top_line (share of the fitted anomaly's energy on its strongest line) and the same for the
data; the fitted peak; the best plug, dike and dipole parameters; the fit time.

Outputs: $AKP_DATA/p3/m3/tables_<key>.npy (the tensor tables, built on first use).
Self-check (run: python -m akp.stage_p3.p3_m3_fit): the table matches the analytic model; a synthetic
pyramid on synthetic lines 200 m apart in colored noise prefers the pyramid family and recovers its
base, top and contrast; a synthetic dike prefers the dike family. The same pyramid on lines 400 m
apart is fitted too and its family count returned without an assertion: there one line carries
nearly all of the anomaly, and a plug or a dipole with a free magnetization vector fits it as well.
"""
from __future__ import annotations

import hashlib
import os
import time
from functools import lru_cache

import numpy as np

from akp.paths import data_dir
from akp.stage_p3 import p3_mag_fast as fast
from akp.stage_p3.p3_mag_forward import FIELD, HEIGHT_M
from akp.stage_p3.p3_mag_inject import FT_M, clean_heights

REACH_M = 600.0
TAPER_M = 300.0  # samples are weighted by exp(-r^2 / (2 TAPER_M^2)), r from the cluster: the fit is of the cluster's anomaly, not of its neighbors
STEP_M = 20.0
GAP_M = 60.0  # as p3_mag_detect.GAP_M: a longer gap starts a new run with its own background
MIN_RUN = 8  # binned samples; shorter runs are dropped
MIN_N = 24
N_MAX = 500  # above this the along-line step is widened (closely spaced surveys)
ALONG_STEP_M = 100.0  # coarse position step along the line; across it the step is spacing / 3
A_M, DU, NU = 60.0, 0.06, 66  # horizontal axis: x = A_M sinh(DU i), out to about 1480 m
D0_M, DD, ND = 20.0, 0.05, 72  # vertical axis: d = D0_M exp(DD k), 20 to about 700 m
PYR_BASES = (100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 900.0)
PYR_TURNS = (0.0, 45.0)  # M2: the plan shape is not recoverable, so two turns suffice
PLUGS = tuple((w, e) for w in (100.0, 160.0, 230.0, 320.0, 450.0) for e in (60.0, 110.0, HEIGHT_M, 500.0))
DIKES = ((20.0, HEIGHT_M), (20.0, 1000.0), (100.0, 400.0))
DIKE_LENGTH_M = 3000.0  # as p3_identifiability
DIKE_STRIKES_DEG = tuple(np.arange(0.0, 180.0, 15.0))
TOPS_M = (8.0, 14.0, 20.0, 26.0, 31.0, 38.0, 46.0, 56.0, 70.0, 90.0, 120.0, 160.0, 220.0, 300.0)
DIPOLE_DEPTHS_M = (40.0, 60.0, 85.0, 115.0, 150.0, 200.0, 270.0, 360.0, 500.0)
TOP_RANGE_M = (5.0, 600.0)  # bounds of the continuous depth polish
POLISH_ITERS = 8
N_POLISH = 3  # shapes of a prism family whose position and depth are polished
TABLE_TOL = 0.01  # share of the peak (the M1 tolerance)
SHAPES = [("pyramid", b, HEIGHT_M) for b in PYR_BASES] + [("plug", w, e) for w, e in PLUGS] + [("dike", w, e) for w, e in DIKES]
TID = {fam: [i for i, s in enumerate(SHAPES) if s[0] == fam] for fam in ("pyramid", "plug", "dike")}
B0 = fast.unit(FIELD["inclination"], FIELD["declination"])
SCALE = FIELD["amplitude"] / (4 * np.pi)


def shape_prisms(kind: str, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Prism corners in the body frame, the body's top at z = 0."""
    if kind == "pyramid":
        return fast.layers(a, 0.0)
    half_x = DIKE_LENGTH_M / 2 if kind == "dike" else a / 2
    return np.array([[-half_x, -a / 2, -b]]), np.array([[half_x, a / 2, 0.0]])


def _table_path():
    key = hashlib.sha256(repr((A_M, DU, NU, D0_M, DD, ND, SHAPES, DIKE_LENGTH_M, fast.N_LAYERS)).encode()).hexdigest()[:10]
    return data_dir("p3/m3") / f"tables_{key}.npy"


def build_tables() -> None:
    """U_xx, U_yy, U_xy, U_xz, U_yz of every shape on the quadrant grid: (shape, NU * NU * ND, 5) float32."""
    path = _table_path()
    if path.exists():
        return
    g = A_M * np.sinh(DU * np.arange(NU))
    d = D0_M * np.exp(DD * np.arange(ND))
    obs = np.stack(np.meshgrid(g, g, d, indexing="ij"), -1).reshape(-1, 3)
    out = np.zeros((len(SHAPES), len(obs), 5), np.float32)
    for i, (kind, a, b) in enumerate(SHAPES):
        lo, hi = shape_prisms(kind, a, b)
        for j in range(0, len(obs), 20000):
            u = fast.prism_tensor(obs[j:j + 20000], lo, hi)
            out[i, j:j + 20000] = np.stack([u[:, 0, 0], u[:, 1, 1], u[:, 0, 1], u[:, 0, 2], u[:, 1, 2]], -1)
        print(f"table {i + 1}/{len(SHAPES)} {kind} {a:.0f} {b:.0f}", flush=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, out)
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def tables() -> np.ndarray:
    build_tables()
    return np.asarray(np.load(_table_path(), mmap_mode="r"))  # memory-mapped (workers share the pages), as a plain array view


def columns(tid: int, ang, px, py, tops, x, y, h) -> np.ndarray:
    """TMI per unit magnetization (east, north, up; SI) of shape tid turned by ang (degrees) at
    (px, py), for each top depth: shape (K, T, N, 3), K = len(ang) = len(px) = len(py)."""
    tab = tables()[tid]
    c, s = np.cos(np.radians(ang))[:, None], np.sin(np.radians(ang))[:, None]
    dx, dy = x[None, :] - np.asarray(px)[:, None], y[None, :] - np.asarray(py)[:, None]
    xb, yb = c * dx + s * dy, -s * dx + c * dy
    u = np.clip(np.arcsinh(np.abs(xb) / A_M) / DU, 0.0, NU - 1.001)
    v = np.clip(np.arcsinh(np.abs(yb) / A_M) / DU, 0.0, NU - 1.001)
    w = np.clip(np.log((h[None, :] + np.asarray(tops, float)[:, None]) / D0_M) / DD, 0.0, ND - 1.001)
    iu, iv, iw = u.astype(np.intp), v.astype(np.intp), w.astype(np.intp)
    fu, fv, fw = (u - iu)[:, None, :], (v - iv)[:, None, :], (w - iw)[None, :, :]
    flat = ((iu * NU + iv) * ND)[:, None, :] + iw[None, :, :]
    t = np.zeros(flat.shape + (5,))
    for a, wa in ((0, 1 - fu), (1, fu)):
        for b, wb in ((0, 1 - fv), (1, fv)):
            for k, wk in ((0, 1 - fw), (1, fw)):
                t += (wa * wb * wk)[..., None] * tab.take(flat + (a * NU + b) * ND + k, axis=0)
    sx, sy = np.sign(xb)[:, None, :], np.sign(yb)[:, None, :]
    uxx, uyy, uxy, uxz, uyz = t[..., 0], t[..., 1], t[..., 2] * sx * sy, t[..., 3] * sx, t[..., 4] * sy
    c, s = c[:, :, None], s[:, :, None]
    bx, by, bz = c * B0[0] + s * B0[1], -s * B0[0] + c * B0[1], B0[2]  # the field direction in the body frame
    gx = uxx * bx + uxy * by + uxz * bz
    gy = uxy * bx + uyy * by + uyz * bz
    gz = uxz * bx + uyz * by - (uxx + uyy) * bz
    return SCALE * np.stack([gx * c - gy * s, gx * s + gy * c, gz], -1)


def dipole_columns(px, py, depths, x, y, h) -> np.ndarray:
    """TMI per unit moment vector of a point dipole at (px, py) and each depth: (K, T, N, 3), scaled to nT-like size."""
    rx = np.broadcast_to((x[None, :] - np.asarray(px)[:, None])[:, None, :], (len(px), len(depths), len(x)))
    ry = np.broadcast_to((y[None, :] - np.asarray(py)[:, None])[:, None, :], rx.shape)
    rz = np.broadcast_to((h[None, :] + np.asarray(depths, float)[:, None])[None, :, :], rx.shape)
    r2 = rx * rx + ry * ry + rz * rz
    rb = rx * B0[0] + ry * B0[1] + rz * B0[2]
    r = np.stack([rx, ry, rz], -1)
    return 1e6 * SCALE * (3 * rb[..., None] * r / r2[..., None] - B0) / (r2 ** 1.5)[..., None]


def collect(frames, mag: str, alt: str, alt_in_feet: bool, x0: float, y0: float) -> dict | None:
    """Samples within REACH_M of (x0, y0) from per-line frames (each one line, in fiducial order):
    local x, y (m from the cluster), sensor height h (m), value v, and a run index per sample."""
    step = STEP_M
    while True:
        xs, ys, hs, vs, ids = [], [], [], [], []
        for f in frames:
            X, Y, V, H = (f[c].to_numpy(float) for c in ("X", "Y", mag, alt))
            ok = (np.hypot(X - x0, Y - y0) <= REACH_M) & np.isfinite(V) & np.isfinite(H)
            if ok.sum() < MIN_RUN:
                continue
            X, Y, V = X[ok] - x0, Y[ok] - y0, V[ok]
            H = clean_heights(H[ok] * (FT_M if alt_in_feet else 1.0))
            ds = np.hypot(np.diff(X), np.diff(Y))
            s = np.r_[0.0, np.cumsum(ds)]
            run = np.r_[0, np.cumsum(ds > GAP_M)]
            b = np.floor(s / step).astype(np.int64) + run * (int(s[-1] / step) + 2)
            _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
            mean = lambda a: np.bincount(inv, a) / cnt
            r = np.round(mean(run)).astype(int)
            for k in np.unique(r):
                m = r == k
                if m.sum() >= MIN_RUN:
                    xs.append(mean(X)[m]); ys.append(mean(Y)[m]); hs.append(mean(H)[m]); vs.append(mean(V)[m])
                    ids.append(np.full(m.sum(), len(ids)))
        if not xs or sum(map(len, xs)) < MIN_N:
            return None
        if sum(map(len, xs)) <= N_MAX:
            return dict(x=np.concatenate(xs), y=np.concatenate(ys), h=np.concatenate(hs), v=np.concatenate(vs), run=np.concatenate(ids))
        step *= 1.5


class _Problem:
    def __init__(self, w: dict):
        self.x, self.y, self.h, self.run = w["x"], w["y"], w["h"], w["run"]
        n, nr = len(self.x), int(self.run.max()) + 1
        bg = np.zeros((n, 2 * nr))
        for k in range(nr):
            m = self.run == k
            t = np.hypot(self.x[m] - self.x[m][0], self.y[m] - self.y[m][0])
            bg[m, 2 * k], bg[m, 2 * k + 1] = 1.0, t / max(t.max(), 1.0) - 0.5
        self.sw = np.exp(-(self.x**2 + self.y**2) / (4 * TAPER_M**2))  # square root of the weight
        self.B = np.linalg.qr(bg * self.sw[:, None])[0]  # orthonormal weighted per-run constant and slope
        v = w["v"] * self.sw
        self.yp = v - self.B @ (self.B.T @ v)
        self.tss = float(self.yp @ self.yp)
        near = self.run[np.argmin(np.hypot(self.x, self.y))]
        m = self.run == near
        e = np.array([self.x[m][-1] - self.x[m][0], self.y[m][-1] - self.y[m][0]])
        self.along = e / np.linalg.norm(e)
        self.across = np.array([-self.along[1], self.along[0]])
        self.best: dict[str, tuple] = {}

    def solve(self, fam: str, A: np.ndarray, meta: list) -> None:
        """Least squares for columns A (K, T, N, 3); meta lists the K * T candidates. Keeps the family's best (rss, m, meta)."""
        A = A * self.sw[:, None]
        BtA = self.B.T @ A
        At = np.swapaxes(A, -1, -2)
        G = At @ A - np.swapaxes(BtA, -1, -2) @ BtA
        G = G + (1e-9 * np.trace(G, axis1=-2, axis2=-1) / 3)[..., None, None] * np.eye(3)
        c = At @ self.yp
        m = np.linalg.solve(G, c[..., None])[..., 0]
        rss = np.maximum(self.tss - (c * m).sum(-1), 1e-12 * self.tss)
        self._keep(fam, rss, m, meta)
        if fam == "pyramid":  # magnetization along the inducing field only (the built case)
            a = A @ B0
            g1 = (a * a).sum(-1) - ((self.B.T @ a[..., None])[..., 0] ** 2).sum(-1)
            c1 = a @ self.yp
            k = c1 / np.maximum(g1, 1e-30)
            self._keep("pyramid_ind", np.maximum(self.tss - c1 * k, 1e-12 * self.tss), k[..., None] * B0, meta)

    def _keep(self, fam, rss, m, meta) -> None:
        i, j = np.unravel_index(np.argmin(rss), rss.shape)
        got = (float(rss[i, j]), m[i, j], meta[i * rss.shape[1] + j])
        if fam != "pyramid_ind":  # the search follows the free-vector fit
            self.last, self.last_kt = got, (int(i), int(j))
        if fam not in self.best or got[0] < self.best[fam][0]:
            self.best[fam] = got

    def grid(self, center, step_along: float, step_across: float, n: int) -> np.ndarray:
        k = np.arange(-n, n + 1)
        return np.array([center + i * step_along * self.along + j * step_across * self.across for i in k for j in k])


def _polish(p: _Problem, fam: str, cols, start, steps, meta) -> None:
    """Pattern search on two continuous parameters and a depth: the 9 x 3 neighbors at the current
    steps (the depth step is a ratio); the steps halve when the center stays best. cols(a, b, depths)
    gives the columns for the 9 (a, b) pairs and the 3 depths; meta(a, b, depth) labels a candidate."""
    ab = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
    center, depth, steps, dstep = np.asarray(start[:2], float), float(start[2]), np.asarray(steps[:2], float), float(steps[2])
    for _ in range(POLISH_ITERS):
        q = center + ab * steps
        dd = np.clip(depth * np.exp(dstep * np.arange(-1, 2)), *TOP_RANGE_M)
        p.solve(fam, cols(q[:, 0], q[:, 1], dd), [meta(r[0], r[1], float(d)) for r in q for d in dd])
        if p.last_kt == (4, 1):
            steps, dstep = steps / 2, dstep / 2
        center, depth = q[p.last_kt[0]], float(dd[p.last_kt[1]])


def _prism_family(p: _Problem, fam: str, turns, spacing: float) -> None:
    tops = np.array(TOPS_M)
    sa, sc = ALONG_STEP_M, spacing / 3

    def scan(tids, turns, pos, top_idx) -> list:
        """The best candidate (rss, meta) of each shape, best first."""
        ang, pp = np.repeat(turns, len(pos)), np.tile(pos, (len(turns), 1))
        got = []
        for tid in tids:
            p.solve(fam, columns(tid, ang, pp[:, 0], pp[:, 1], tops[top_idx], p.x, p.y, p.h),
                    [(tid, a, q[0], q[1], float(t)) for a, q in zip(ang, pp) for t in tops[top_idx]])
            got.append((p.last[0], p.last[2]))
        return sorted(got, key=lambda g: g[0])

    tid, turn, px, py, top = scan(TID[fam], turns[:1], p.grid(np.zeros(2), sa, sc, 1), np.arange(1, len(tops), 2))[0][1]
    for div in (2, 4):  # position at a finer step for the current shape, then every shape and top there
        near = np.unique(np.clip(np.searchsorted(tops, top) + np.arange(-1, 2), 0, len(tops) - 1))
        tid, turn, px, py, top = scan([tid], [turn], p.grid(np.array([px, py]), sa / div, sc / div, 2), near)[0][1]
        ranked = scan(TID[fam], turns, np.array([[px, py]]), np.arange(len(tops)))
        tid, turn, px, py, top = ranked[0][1]
    for _, (tid, turn, px, py, top) in ranked[:N_POLISH]:
        _polish(p, fam, lambda a, b, dd: columns(tid, np.full(len(a), turn), a, b, dd, p.x, p.y, p.h),
                (px, py, top), (sa / 8, sc / 8, 0.12), lambda a, b, d: (tid, turn, a, b, d))


def _dike_family(p: _Problem) -> None:
    tops = np.array(TOPS_M)

    def cols(tid, ang, off, tt):
        return columns(tid, ang, -np.sin(np.radians(ang)) * off, np.cos(np.radians(ang)) * off, tt, p.x, p.y, p.h)  # across strike

    def scan(tids, strikes, offs, top_idx) -> list:
        ang, off = np.repeat(strikes, len(offs)), np.tile(offs, len(strikes))
        got = []
        for tid in tids:
            p.solve("dike", cols(tid, ang, off, tops[top_idx]), [(tid, a, o, float(t)) for a, o in zip(ang, off) for t in tops[top_idx]])
            got.append((p.last[0], p.last[2]))
        return sorted(got, key=lambda g: g[0])

    tid, ang, off, top = scan(TID["dike"], np.array(DIKE_STRIKES_DEG), 100.0 * np.arange(-2, 3), np.arange(1, len(tops), 2))[0][1]
    near = np.unique(np.clip(np.searchsorted(tops, top) + np.arange(-1, 2), 0, len(tops) - 1))
    tid, ang, off, top = scan([tid], ang + np.array([-7.5, 0.0, 7.5]), off + 50.0 * np.arange(-2, 3), near)[0][1]
    for _, (tid, ang, off, top) in scan(TID["dike"], np.array([ang]), np.array([off]), np.arange(len(tops))):
        _polish(p, "dike", lambda a, b, dd: cols(tid, a, b, dd), (ang, off, top), (4.0, 25.0, 0.12), lambda a, b, d: (tid, a, b, d))


def _dipole_family(p: _Problem, spacing: float) -> None:
    sa, sc = ALONG_STEP_M, spacing / 3
    depths = np.array(DIPOLE_DEPTHS_M)
    center = np.zeros(2)
    for step, n in ((1.0, 1), (0.5, 2)):
        pos = p.grid(center, sa * step, sc * step, n)
        p.solve("dipole", dipole_columns(pos[:, 0], pos[:, 1], depths, p.x, p.y, p.h), [(q[0], q[1], float(d)) for q in pos for d in depths])
        center = np.array(p.best["dipole"][2][:2])
    _polish(p, "dipole", lambda a, b, dd: dipole_columns(a, b, dd, p.x, p.y, p.h),
            (center[0], center[1], p.best["dipole"][2][2]), (sa / 4, sc / 4, 0.12), lambda a, b, d: (a, b, d))


def fit(w: dict | None, spacing: float) -> dict:
    """Descriptors of the four family fits for a window from collect(). An unusable window gives n = 0."""
    t0 = time.time()
    if w is None:
        return dict(n=0)
    p = _Problem(w)
    n = len(p.x)
    spacing = float(np.clip(spacing, 60.0, 1200.0)) if np.isfinite(spacing) else 400.0
    _prism_family(p, "pyramid", list(PYR_TURNS), spacing)
    _prism_family(p, "plug", [0.0], spacing)
    _dike_family(p)
    _dipole_family(p, spacing)
    rss = {f: p.best[f][0] for f in ("pyramid", "plug", "dike", "dipole", "pyramid_ind")}
    wsum = float((p.sw**2).sum())
    out = dict(n=n, n_lines=int(p.run.max()) + 1, clr_med=float(np.median(p.h)), tss_var=p.tss / wsum)
    for f, key in (("pyramid", "pyr"), ("plug", "plug"), ("dike", "dike"), ("dipole", "dip"), ("pyramid_ind", "pyr_ind")):
        out[f"r2_{key}"], out[f"rv_{key}"] = 1 - rss[f] / p.tss, rss[f] / wsum
    for f, key in (("plug", "plug"), ("dike", "dike"), ("dipole", "dip")):
        out[f"margin_{key}"] = float(np.log(rss[f] / rss["pyramid"]))
    other = min(("plug", "dike", "dipole"), key=rss.get)
    out["margin"] = float(np.log(rss[other] / rss["pyramid"]))
    out["best_family"] = min(("pyramid", "plug", "dike", "dipole"), key=rss.get)
    _, m, (tid, turn, px, py, top) = p.best["pyramid"]
    base = SHAPES[tid][1]
    mag = float(np.linalg.norm(m))
    out.update(pyr_base=base, pyr_top=top, pyr_turn=float(turn), pyr_dx=float(px), pyr_dy=float(py), pyr_off=float(np.hypot(px, py)),
               pyr_k=mag, pyr_kind=float(m @ B0), pyr_angle=float(np.degrees(np.arccos(np.clip(m @ B0 / max(mag, 1e-30), -1, 1)))))
    _, mi, ind = p.best["pyramid_ind"]
    out.update(k_ind=float(mi @ B0), ind_base=SHAPES[ind[0]][1], ind_top=ind[4])
    model = columns(tid, np.array([turn]), [px], [py], [top], p.x, p.y, p.h)[0, 0] @ m
    mp = model * p.sw - p.B @ (p.B.T @ (model * p.sw))
    e_model = np.bincount(p.run, mp * mp)
    e_data = np.bincount(p.run, p.yp * p.yp)
    reach = base / 2 + top + out["clr_med"]
    out.update(n_lines_reach=int(len(np.unique(p.run[np.hypot(p.x - px, p.y - py) <= reach]))),
               share_top_line=float(e_model.max() / max(e_model.sum(), 1e-30)),
               share_top_line_data=float(e_data.max() / max(e_data.sum(), 1e-30)),
               peak_nT=float(np.abs(model).max()))
    plug, dike, dip = p.best["plug"][2], p.best["dike"][2], p.best["dipole"][2]
    out.update(plug_w=SHAPES[plug[0]][1], plug_ext=SHAPES[plug[0]][2], plug_top=plug[4], dike_strike=float((90.0 - dike[1]) % 180),
               dike_w=SHAPES[dike[0]][1], dike_top=dike[3], dip_depth=dip[2], fit_s=time.time() - t0)
    return out


def synthetic_lines(rng, spacing: float = 400.0, clearance: float = 35.0, azimuth: float = 70.0) -> dict:
    """Straight parallel lines (5 m samples) through a disc of REACH_M, at a random cross-line phase,
    with a drape scatter on the clearance: a window as collect() makes it, without values."""
    import pandas as pd

    ux, uy = np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))
    s = np.arange(-1500.0, 1500.0, 5.0)
    frames = []
    for off in rng.uniform(-spacing / 2, spacing / 2) + spacing * np.arange(-3, 4):
        frames.append(pd.DataFrame(dict(X=s * ux - off * uy, Y=s * uy + off * ux, MAG=0.0,
                                        ALT=clearance + 4.0 * np.sin(s / 300.0 + rng.uniform(0, 6)) + rng.normal(0, 1.0, s.size))))
    return collect(frames, "MAG", "ALT", False, 0.0, 0.0)


def colored_noise(rng, w: dict, sigma: float, corr_m: float = 150.0) -> np.ndarray:
    """AR(1) noise along each run (correlation length corr_m) plus a per-run offset and slope."""
    out = np.zeros(len(w["x"]))
    rho = np.exp(-STEP_M / corr_m)
    for k in np.unique(w["run"]):
        m = np.flatnonzero(w["run"] == k)
        e = rng.normal(0, sigma * np.sqrt(1 - rho * rho), m.size)
        for i in range(1, m.size):
            e[i] += rho * e[i - 1]
        out[m] = e + rng.normal(0, 30.0) + rng.normal(0, 20.0) * np.linspace(-1, 1, m.size)
    return out


def self_check() -> dict:
    rng = np.random.default_rng(20261113)
    # 1. the table against the analytic model, off the grid, at sensor heights and tops a fit meets
    worst = 0.0
    for base in PYR_BASES:
        for _ in range(3):
            x, y = rng.uniform(-900, 900, 1500), rng.uniform(-900, 900, 1500)
            x[:300], y[:300] = rng.uniform(-150, 150, 300), rng.uniform(-150, 150, 300)
            h, top, turn = rng.uniform(22.0, 160.0, 1500), float(rng.uniform(8.0, 120.0)), float(rng.uniform(0, 90))
            m = rng.normal(size=3)
            want = fast.prism_tensor(np.c_[x, y, h + top] @ _rot(turn).T, *fast.layers(base, 0.0))
            want = SCALE * np.einsum("nij,j->ni", want, _rot(turn) @ m) @ (_rot(turn) @ B0)
            got = columns(TID["pyramid"][PYR_BASES.index(base)], np.array([turn]), [0.0], [0.0], [top], x, y, h)[0, 0] @ m
            worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
    assert worst <= TABLE_TOL, f"table error {worst:.4f} of the peak exceeds {TABLE_TOL}"
    # and once against p3_mag_fast.response itself (induced, a pyramid at depth)
    x, y, h = rng.uniform(-600, 600, 800), rng.uniform(-600, 600, 800), rng.uniform(25.0, 60.0, 800)
    want = fast.response(x, y, h, 300.0, 38.1, 30.0, 0.05)
    got = columns(TID["pyramid"][PYR_BASES.index(300.0)], np.array([30.0]), [0.0], [0.0], [38.1], x, y, h)[0, 0] @ (0.05 * B0)
    resp_err = float(np.abs(got - want).max() / np.abs(want).max())
    assert resp_err <= TABLE_TOL, f"table against response(): {resp_err:.4f}"
    # 2. a pyramid in colored noise. At 200 m spacing the pyramid family wins and the geometry comes back.
    #    At 400 m one line carries nearly all of the anomaly and the families tie: reported, not asserted.
    n_try, got = 12, {}
    for spacing in (200.0, 400.0):
        wins, base_ok, top_ok, k_ok, secs = 0, 0, 0, 0, []
        for _ in range(n_try):
            w = synthetic_lines(rng, spacing=spacing, azimuth=float(rng.uniform(0, 180)))
            px, py = rng.uniform(-80, 80, 2)
            w["v"] = fast.response(w["x"] - px, w["y"] - py, w["h"], 300.0, 38.1, float(rng.uniform(0, 90)), 0.0691) + colored_noise(rng, w, 2.0)
            d = fit(w, spacing)
            secs.append(d["fit_s"])
            wins += d["best_family"] == "pyramid"
            base_ok += 200.0 <= d["pyr_base"] <= 400.0
            top_ok += 20.0 <= d["pyr_top"] <= 60.0
            k_ok += 0.4 * 0.0691 <= d["pyr_k"] <= 2.5 * 0.0691 and d["pyr_angle"] < 30.0
        got[spacing] = (wins, base_ok, top_ok, k_ok, float(np.median(secs)))
    wins, base_ok, top_ok, k_ok, _ = got[200.0]
    assert wins >= n_try - 2 and base_ok >= n_try - 2 and top_ok >= n_try - 2 and k_ok >= n_try - 3, got
    # 3. a dike in colored noise at 400 m spacing: the dike family wins
    dwins = 0
    for _ in range(n_try):
        w = synthetic_lines(rng, azimuth=float(rng.uniform(0, 180)))
        strike_turn, off = float(rng.uniform(0, 180)), float(rng.uniform(-80, 80))
        lo, hi = np.array([[-DIKE_LENGTH_M / 2, -20.0, -(38.1 + HEIGHT_M)]]), np.array([[DIKE_LENGTH_M / 2, 20.0, -38.1]])
        r = _rot(strike_turn)
        obs = np.c_[w["x"] + np.sin(np.radians(strike_turn)) * off, w["y"] - np.cos(np.radians(strike_turn)) * off, w["h"]]
        f = SCALE * np.einsum("nij,j->ni", fast.prism_tensor(obs @ r.T, lo, hi), r @ B0) @ (r @ B0)
        w["v"] = f * 60.0 / np.abs(f).max() + colored_noise(rng, w, 2.0)
        dwins += fit(w, 400.0)["best_family"] == "dike"
    assert dwins >= n_try - 2, dwins
    res = dict(table_err=round(worst, 5), response_err=round(resp_err, 5), n_try=n_try, pyramid_wins_200m=wins, base_in_range_200m=base_ok,
               top_near_200m=top_ok, contrast_ok_200m=k_ok, pyramid_wins_400m=got[400.0][0], dike_wins_400m=dwins,
               fit_s_median_200m=round(got[200.0][4], 3), fit_s_median_400m=round(got[400.0][4], 3))
    print("p3_m3_fit self-check OK", res)
    return res


def _rot(turn: float) -> np.ndarray:
    c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


if __name__ == "__main__":
    self_check()
