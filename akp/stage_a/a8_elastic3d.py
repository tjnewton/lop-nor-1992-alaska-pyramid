"""A8 (P4 item S5): three-dimensional elastic velocity-stress FD wrapper.

Inputs: a box (m; x, y horizontal, z down), a grid step, a depth profile of (vp, vs, rho), an
optional body (sphere or square apex-up body) with its own (vp, vs, rho), a source (a plane P wave
or a buried explosive point source) and receiver points.

Method (built on the stage's two-dimensional code, fd2d.py / a3b2_numerical.py):
- Solver: Devito, velocity-stress on the standard staggered grid, space order 8, leapfrog in time,
  CPU. OpenMP through the environment's libomp (Apple clang has no -fopenmp of its own).
- Material: effective parameters at each staggered position from sub-cell sampling (NS^3 samples per
  cell): arithmetic mean of density at the velocity points, harmonic mean of the bulk and shear
  moduli at the stress points (Moczo et al. 2002, BSSA 92, 3042; a lead, not opened). The
  "rasterized body" is this volume-fraction raster; `body_cells` returns it for the Born integral.
- Sponge: every field is multiplied each step by exp(-sigma dt), sigma = SIGMA_MAX xi^2 over NBL
  cells (xi from 0 at the inner edge to 1 at the wall), on all faces, or on the sides and bottom
  when there is a free surface. Damping velocity and stress alike keeps the impedance real, so a
  normally incident wave is not reflected by the damping gradient whatever its wavelength.
  SIGMA_MAX = 3 A v_ref / (2 L) gives a round-trip amplitude factor exp(-A), A = SPONGE_A, for a
  wave of speed v_ref at normal incidence.
- Free surface: stress imaging, not a vacuum layer. The surface is the plane of the normal-stress
  nodes (z = 0). There tau_zz = 0; tau_xx and tau_yy use d_z v_z = -lam / (lam + 2 mu) (d_x v_x +
  d_y v_y); tau_xz and tau_yz are imaged as odd functions (one ghost plane), which is all the
  second-order stencil at the surface needs. The z-derivative order grows 2, 4, 6, 8 over the top
  four cells so no stencil reaches above the surface; horizontal derivatives stay order 8.
  Why not the vacuum layer of the 2-D stage: with Devito's arithmetic averaging the shear modulus
  half a cell above the surface is not zero, and the 2-D stage needed 333 cells per P wavelength
  for a 0.7% Rayleigh speed; three dimensions cannot afford that. Gate three measures what this
  scheme delivers at the production points per wavelength.
- Plane P wave: a horizontal sheet of explosive sources (equal stress-rate injection into the
  three normal stresses, a gradient field, so it radiates P only), fired with the delay
  x sin(angle) / vp for an upgoing wave at `angle` from the vertical, with a Ricker wavelet
  (band-limited). The sheet is flat inside `flat` and tapers (cos^2) to zero at the walls. Inside
  the causal cone of the flat part the wave is exactly plane; the taper's edge wave reaches a point
  at horizontal distance W from the taper and height d above the sheet (sqrt(W^2 + d^2) - d) / vp
  after the plane wave. Callers size W from that (rule 4).
- Point source: an explosive point source on a node (stress-rate injection w(t) into the normal
  stresses of one node = moment rate w h^3).
- Receivers: each velocity component is read as the grid sample at its own staggered point, by
  index (Devito's interpolation of a staggered field averages neighboring samples; see `simulate`).
  P and S are separated as in the 2-D code, by div v and curl v, here with the fourth-order
  staggered stencil on recorded points (`ps_points`, `div_curl`): |v_P| = |div v| / k_P,
  |v_S| = |curl v| / k_S. Amplitudes are DFTs in a window (`win_dft`: flat between t_lo and t_hi,
  cosine tapers outside). Rayleigh waves at the surface are measured by their window (r / c_R) and
  particle motion (`rayleigh_measure`: vertical and radial amplitudes and their phase lag).
- Scattered field: `scattered(with_body, without_body)`, two runs on the identical grid and step.
- Attenuation (optional, `q=dict(q=Q, f0=f)`; without it the elastic equations are built exactly as
  before): one standard linear solid with memory variables (Robertsson, Blanch & Symes 1994,
  Geophysics 59, 1444; Blanch, Robertsson & Symes 1995, Geophysics 60, 176; leads, not opened; the
  form is Devito's viscoelastic example reduced to equal Q for P and S). With equal Q every modulus
  carries the same relaxation function, so each stress component has one memory variable r driven
  by its own elastic stress rate E:  d sigma/dt = kappa (tau_e / tau_s) E + r,
  dr/dt = -(r + kappa (tau_e / tau_s - 1) E) / tau_s, r stepped by the trapezoid rule and stored
  once (no time buffer). tau_s = 1 / (a w0), tau_e = a / w0, a = sqrt(1 + 1/Q^2) + 1/Q, w0 = 2 pi f0:
  Q(f) has its minimum, Q, at f0 and rises away from it as Q (f/f0 + f0/f) / 2 (`sls_q`). kappa
  scales the relaxed moduli so that the phase speeds at f0 are the model's stated speeds. The
  free surface is unchanged (tau_zz = 0, and the plane-stress ratio is real when Q is equal).
- Mirror plane (optional, `sym_y`): when the model, the body and the source are symmetric about
  y = 0, only y >= 0 is computed. Four ghost planes below y = 0 are refilled after each velocity
  and each stress update: even images for v_x, v_z, the normal stresses and tau_xz; odd images for
  v_y, tau_xy and tau_yz. The stencils are symmetric, so this is the full problem, not an
  approximation; it removes one sponge and the sheet edge nearest the body.

Sizing rules (callers apply them; a8_gates.py records the values):
  1. grid step h <= lambda_min / PPW, lambda_min = lowest shear (or Rayleigh) speed / highest
     frequency measured, PPW = 12 at least;
  2. time step dt = DT_SAFETY h / (sqrt(3) * 1.2863 * vp_max) (1.2863 = sum of the order-8
     staggered weights; DT_SAFETY = 0.8);
  3. duration = wavelet delay + travel time to the last window + window length;
  4. margins from travel times: every measurement window must close before (a) the sheet's edge
     wave, scattered by the body, arrives; (b) a reflection of the measured wave from the inner edge
     of a sponge arrives (extra path / speed). The sponge is matched only near normal incidence: a
     wave that grazes it is reflected from its inner edge (gate three, first attempt), so the
     reflecting plane in rule (b) is the sponge's inner edge, not the wall.

Outputs: functions only (`simulate` returns traces and the run's size, time and memory).
Run: python -m akp.stage_a.a8_elastic3d  (self-check: rasterizer, stencils, three small FD runs)
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _devito_env() -> None:
    """OpenMP for Devito's JIT with Apple clang and the environment's libomp; plain C elsewhere."""
    prefix = Path(sys.prefix)
    if sys.platform == "darwin" and (prefix / "lib" / "libomp.dylib").exists():
        omp = f"-L{prefix}/lib -lomp"
        os.environ.update(DEVITO_LANGUAGE="openmp", DEVITO_ARCH="custom", CC="clang",
                          CFLAGS=f"-Xclang -fopenmp -I{prefix}/include",
                          LDFLAGS=f"{omp} -Wl,-rpath,{prefix}/lib", OMP_LDFLAGS=omp)
    os.environ.setdefault("DEVITO_LOGGING", "ERROR")
    os.environ.setdefault("OMP_NUM_THREADS", "6")


_devito_env()

SO = 8
W8 = (1.19628906, -0.0797526042, 0.0095703125, -0.000697544643)  # order-8 staggered weights
DT_SAFETY, PPW = 0.8, 12
NBL, SPONGE_A = 30, 7.0
NS = 8  # sub-cell samples per axis for the effective parameters
KF = 1  # z index of the free surface (one ghost plane above it)
JS = SO // 2  # y index of the mirror plane y = 0 (SO / 2 ghost planes below it)


# ---------------------------------------------------------------- grid, bodies, parameters
def make_grid(*, lo, hi, h, free_surface=False, sym_y=False, nbl=NBL):
    """Physical box lo..hi (m) plus nbl sponge cells on each absorbing face.

    z = 0 is the free surface (free_surface); y = 0 is a mirror plane (sym_y: the box starts at y = 0).
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if (free_surface and lo[2] != 0.0) or (sym_y and lo[1] != 0.0):
        raise ValueError("a free surface needs the box to start at z = 0, a mirror plane at y = 0")
    n = np.round((hi - lo) / h).astype(int) + 1 + 2 * nbl
    origin = lo - nbl * h
    if free_surface:
        n[2] += KF - nbl
        origin[2] = -KF * h
    if sym_y:
        n[1] += JS - nbl
        origin[1] = -JS * h
    return dict(shape=tuple(int(v) for v in n), origin=origin, h=float(h), nbl=nbl, fs=bool(free_surface),
                sym=bool(sym_y))


def snap(g, pts):
    """Nearest grid nodes to pts (n, 3)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    return g["origin"] + np.round((pts - g["origin"]) / g["h"]) * g["h"]


def sphere(center, a):
    c = np.asarray(center, float)
    return dict(inside=lambda x, y, z: (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= a * a,
                lo=c - a, hi=c + a)


def square_body(*, x, y, apex_depth, base, height):
    """Apex-up square-based body: apex at depth apex_depth, base (side `base`) at apex_depth + height."""
    def inside(xx, yy, zz):
        up = zz - apex_depth
        half = 0.5 * base * up / height
        return (up >= 0) & (up <= height) & (np.abs(xx - x) <= half) & (np.abs(yy - y) <= half)
    return dict(inside=inside, lo=np.array([x - base / 2, y - base / 2, apex_depth]),
                hi=np.array([x + base / 2, y + base / 2, apex_depth + height]))


def _moduli(vp, vs, rho):
    mu = rho * vs**2
    return rho * vp**2 - 4 * mu / 3, mu  # bulk, shear


def _cell_means(g, idx, stag, profile, body=None, props=None):
    """Sub-cell means over cells idx = ((i0, i1), (j0, j1), (k0, k1)) centered at node + stag * h.

    Returns (mean rho, mean 1/K, mean 1/mu, body volume fraction). Above a free surface the profile
    is read at z = 0, so the surface plane carries the solid's own values. A flat face of the body
    is located to h / NS (it falls between sub-samples).
    """
    (i0, i1), jk = idx[0], idx[1:]
    if i1 - i0 > 8:  # slabs along x bound the memory of the sub-cell arrays
        parts = [_cell_means(g, ((a, min(a + 8, i1)), *jk), stag, profile, body, props) for a in range(i0, i1, 8)]
        return tuple(np.concatenate(q, axis=0) for q in zip(*parts))
    h, off = g["h"], (np.arange(NS) + 0.5) / NS - 0.5
    ax = [g["origin"][d] + ((np.arange(a, b)[:, None] + stag[d] + off[None, :]) * h).ravel()
          for d, (a, b) in enumerate(idx)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij", sparse=True)
    shape = tuple(len(a) for a in ax)
    vp, vs, rho = (np.asarray(q, float) for q in profile(np.maximum(Z, 0.0) if g["fs"] else Z))
    K, mu = _moduli(vp, vs, rho)
    rho, Ki, mi = (np.broadcast_to(q, shape) for q in (rho, 1 / K, 1 / mu))
    frac = np.zeros(shape)
    if body is not None:
        m = np.broadcast_to(body["inside"](X, Y, Z), shape)
        Kb, mub = _moduli(*props)
        rho, Ki, mi = np.where(m, props[2], rho), np.where(m, 1 / Kb, Ki), np.where(m, 1 / mub, mi)
        frac = m.astype(float)
    n = [b - a for a, b in idx]
    return tuple(q.reshape(n[0], NS, n[1], NS, n[2], NS).mean(axis=(1, 3, 5)) for q in (rho, Ki, mi, frac))


_STAG = dict(bx=(0.5, 0, 0), by=(0, 0.5, 0), bz=(0, 0, 0.5), node=(0, 0, 0),
             mxy=(0.5, 0.5, 0), mxz=(0.5, 0, 0.5), myz=(0, 0.5, 0.5))


def _body_box(g, body):
    lo = np.floor((body["lo"] - g["origin"]) / g["h"]).astype(int) - 1
    hi = np.ceil((body["hi"] - g["origin"]) / g["h"]).astype(int) + 2
    return tuple((max(int(a), 0), min(int(b), n)) for a, b, n in zip(lo, hi, g["shape"]))  # clipped to the grid


def effective_params(g, profile, body=None, props=None):
    """Effective parameter arrays at the staggered positions; profile(z) -> (vp, vs, rho) (vectorized).

    Returns dict(bx, by, bz, lam, mu, mxy, mxz, myz) of float32 arrays of the grid shape.
    """
    nz = g["shape"][2]
    out = {}
    box = _body_box(g, body) if body is not None else None
    for name, stag in _STAG.items():
        rho, Ki, mi, _ = _cell_means(g, ((0, 1), (0, 1), (0, nz)), stag, profile)
        vals = [np.broadcast_to(q.astype(np.float32), g["shape"]).copy() for q in (rho, Ki, mi)]
        if body is not None:
            sl = tuple(slice(a, b) for a, b in box)
            for arr, q in zip(vals, _cell_means(g, box, stag, profile, body, props)[:3]):
                arr[sl] = q
        rho, Ki, mi = vals
        if name[0] == "b":
            out[name] = 1 / rho
        elif name == "node":
            out["mu"] = 1 / mi
            out["lam"] = 1 / Ki - 2 / (3 * mi)
        else:
            out[name] = 1 / mi
    return out


def body_cells(g, body):
    """The rasterized body: node positions (n, 3) and volumes (volume fraction * h^3) of its cells."""
    box = _body_box(g, body)
    frac = _cell_means(g, box, (0, 0, 0), lambda z: (1.0, 0.5, 1.0), body, (1.0, 0.5, 1.0))[3]
    ii = np.argwhere(frac > 0)
    xyz = g["origin"] + (ii + np.array([b[0] for b in box])) * g["h"]
    return xyz, frac[frac > 0] * g["h"] ** 3


def time_step(g, vp_max):
    return DT_SAFETY * g["h"] / (np.sqrt(3.0) * sum(abs(w) for w in W8) * vp_max)


def sponge_profiles(g, dt, v_ref):
    """Per-step damping factors along x, y, z (the product is applied to every field)."""
    nb, out = g["nbl"], []
    sigma_max = 3 * SPONGE_A * v_ref / (2 * nb * g["h"])
    xi = (nb - np.arange(nb)) / nb
    for d, n in enumerate(g["shape"]):
        f = np.ones(n)
        f[-nb:] = np.exp(-sigma_max * xi[::-1] ** 2 * dt)
        if not ((d == 2 and g["fs"]) or (d == 1 and g.get("sym"))):
            f[:nb] = np.exp(-sigma_max * xi**2 * dt)
        out.append(f.astype(np.float32))
    return out


def sls_constants(q, f0, dt=None):
    """One standard linear solid with minimum Q = q at f0: relaxation times, modulus scale, step weights."""
    w0 = 2 * np.pi * f0
    a = np.sqrt(1 + 1 / q**2) + 1 / q
    tau_s, tau_e = 1 / (a * w0), a / w0
    kappa = float(np.real(1 / np.sqrt((1 + 1j * a) / (1 + 1j / a))) ** 2)  # phase speed at f0 = the stated speed
    out = dict(tau_s=tau_s, tau_e=tau_e, kappa=kappa)
    if dt is not None:  # r_new = c1 r + c2 E; sigma += dt (g_e E + g_r r), the trapezoid mean of old and new r
        th = dt / (2 * tau_s)
        c1, c2 = (1 - th) / (1 + th), -2 * th * kappa * (tau_e / tau_s - 1) / (1 + th)
        out.update(c1=c1, c2=c2, g_e=kappa * tau_e / tau_s + 0.5 * c2, g_r=0.5 * (1 + c1))
    return out


def sls_q(freq, q, f0):
    """(Q, phase speed over the stated speed) of that solid at frequency freq."""
    c = sls_constants(q, f0)
    w = 2 * np.pi * np.asarray(freq, float)
    m = c["kappa"] * (1 + 1j * w * c["tau_e"]) / (1 + 1j * w * c["tau_s"])
    return np.real(m) / np.imag(m), 1 / np.real(1 / np.sqrt(m))


def ricker(t, f0, t0):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


# ---------------------------------------------------------------- the FD run
def simulate(g, par, **kw):
    """`_simulate`, then Devito's symbol cache is cleared so the field arrays are freed before the next run."""
    from devito import clear_cache
    out = _simulate(g, par, **kw)
    clear_cache()
    return out


def _simulate(g, par, *, dt, nt, v_ref, plane=None, point=None, rec=None, q=None):
    """One run (it empties `par`). Returns dict(vx, vy, vz (nt, n) traces, seconds, mpts_per_s, rss_gb, npts).

    plane = dict(z, angle_deg, vp, f0, t0, flat=(wx, wy), center=(x, y)): sheet at depth z (snapped
      to a node plane); the wavelet center passes the sheet point under `center` at t0;
    point = dict(xyz, f0, t0): explosive point source (snapped to a node);
    rec = dict(vx=pts, vy=pts, vz=pts): sample points (n, 3) of each component (use `ps_points` or
      `native`); v_ref: the speed the sponge is tuned to (host vp).
    """
    from devito import (Eq, Function, Grid, Operator, SparseTimeFunction, SubDomain, TensorTimeFunction,
                        VectorTimeFunction, exp)
    nx, ny, nz = g["shape"]
    h, fs = g["h"], g["fs"]

    class Layer(SubDomain):  # z-planes k0..k1 (inclusive), all x and y
        def __init__(self, name, k0, k1):
            self.name, self._thick = name, (k0, nz - 1 - k1)
            super().__init__()

        def define(self, dimensions):
            xd, yd, zd = dimensions
            return {xd: xd, yd: yd, zd: ("middle", *self._thick)}

    class YPlane(SubDomain):  # one y-plane, all x and z
        def __init__(self, name, j):
            self.name, self._thick = name, (j, ny - 1 - j)
            super().__init__()

        def define(self, dimensions):
            xd, yd, zd = dimensions
            return {xd: xd, yd: ("middle", *self._thick), zd: zd}

    sym = g.get("sym", False)
    yplanes = [YPlane(f"ym{m}", JS - m) for m in range(1, JS + 1)] if sym else []
    layers = {}
    if fs:  # (name, first plane, last plane, z order of the node-level updates, of the half-level updates)
        spec = [("surf", KF, KF, 2, 2), ("l1", KF + 1, KF + 1, 2, 4), ("l2", KF + 2, KF + 2, 4, 6),
                ("l3", KF + 3, KF + 3, 6, 8), ("deep", KF + 4, nz - 1, 8, 8)]
        layers = {name: (Layer(name, k0, k1), zn, zh) for name, k0, k1, zn, zh in spec}
        layers["ghost"] = (Layer("ghost", KF - 1, KF - 1), 0, 0)
    if plane is not None:
        ks = int(round((plane["z"] - g["origin"][2]) / h))
        layers["sheet"] = (Layer("sheet", ks, ks), 0, 0)
    grid = Grid(shape=g["shape"], extent=tuple((n - 1) * h for n in g["shape"]), origin=tuple(g["origin"]),
                subdomains=tuple(v[0] for v in layers.values()) + tuple(yplanes), dtype=np.float32)
    x, y, z = grid.dimensions
    v = VectorTimeFunction(name="v", grid=grid, space_order=SO, time_order=1)
    tau = TensorTimeFunction(name="tau", grid=grid, space_order=SO, time_order=1)
    stag = dict(bx=x, by=y, bz=z, lam=None, mu=None, mxy=(x, y), mxz=(x, z), myz=(y, z))
    P = {}
    for name, st in stag.items():
        P[name] = Function(name=name, grid=grid, space_order=0, **({} if st is None else dict(staggered=st)))
        P[name].data[:] = par.pop(name)  # par is consumed: no second copy of the model in memory
    damp = 1
    for dim, prof in zip((x, y, z), sponge_profiles(g, dt, v_ref)):  # 1-D factors: no 3-D damping array
        f = Function(name=f"dm{dim.name}", dimensions=(dim,), shape=(len(prof),), space_order=2)
        f.data_with_halo[:] = np.pad(prof, f.halo[0], mode="edge")  # Devito averages it to the staggered points
        damp = damp * f
    s = grid.stepping_dim.spacing
    vx, vy, vz = v
    txx, tyy, tzz, txy, txz, tyz = tau[0, 0], tau[1, 1], tau[2, 2], tau[0, 1], tau[0, 2], tau[1, 2]
    fx, fy, fz = vx.forward, vy.forward, vz.forward
    lam, mu = P["lam"], P["mu"]
    l2m = lam + 2 * mu

    def upd(f, rhs, sub):
        return Eq(f.forward, damp * (f + s * rhs), subdomain=sub)

    mem = {}
    if q is not None:  # one memory variable per stress component, at that component's own points
        sls = sls_constants(q["q"], q["f0"], dt)
        for f, st in ((txx, None), (tyy, None), (tzz, None), (txy, (x, y)), (txz, (x, z)), (tyz, (y, z))):
            mem[f] = Function(name=f"mem{len(mem)}", grid=grid, space_order=0, **({} if st is None else dict(staggered=st)))

    def supd(f, rhs, sub):  # stress update: elastic, or with the standard linear solid's memory variable
        if q is None:
            return [upd(f, rhs, sub)]
        r = mem[f]
        return [Eq(f.forward, damp * (f + s * (sls["g_e"] * rhs + sls["g_r"] * r)), subdomain=sub),
                Eq(r, sls["c1"] * r + sls["c2"] * rhs, subdomain=sub)]

    def vel(sub, zn, zh):
        return [upd(vx, P["bx"] * (txx.dx + txy.dy + txz.dz(fd_order=zn)), sub),
                upd(vy, P["by"] * (txy.dx + tyy.dy + tyz.dz(fd_order=zn)), sub),
                upd(vz, P["bz"] * (txz.dx + tyz.dy + tzz.dz(fd_order=zh)), sub)]

    def stress(sub, zn, zh, surface=False):
        if surface:  # tau_zz = 0 on the surface plane
            c = lam * lam / l2m
            normal = (supd(txx, (l2m - c) * fx.dx + (lam - c) * fy.dy, sub)
                      + supd(tyy, (lam - c) * fx.dx + (l2m - c) * fy.dy, sub) + [Eq(tzz.forward, 0.0, subdomain=sub)])
        else:
            ez = fz.dz(fd_order=zn)
            normal = (supd(txx, l2m * fx.dx + lam * (fy.dy + ez), sub) + supd(tyy, l2m * fy.dy + lam * (fx.dx + ez), sub)
                      + supd(tzz, l2m * ez + lam * (fx.dx + fy.dy), sub))
        return (normal + supd(txy, P["mxy"] * (fx.dy + fy.dx), sub)
                + supd(txz, P["mxz"] * (fx.dz(fd_order=zh) + fz.dx), sub)
                + supd(tyz, P["myz"] * (fy.dz(fd_order=zh) + fz.dy), sub))

    def mirror(fields):  # ghost plane JS - m takes the image of the plane (or half-plane) m above y = 0
        out = []
        for m, sub in enumerate(yplanes, start=1):
            for f, sign in fields:  # the odd fields are the ones staggered in y: their image is 2m - 1 samples up
                shift = 2 * m - 1 if sign < 0 else 2 * m
                out.append(Eq(f.forward, sign * f.forward.subs(y, y + shift * y.spacing), subdomain=sub))
        return out

    if fs:
        names = ("surf", "l1", "l2", "l3", "deep")
        eqs = [e for n in names for e in vel(*layers[n])] + mirror([(vx, 1), (vy, -1), (vz, 1)])
        eqs += [e for n in names for e in stress(*layers[n], surface=(n == "surf"))]
        gh = layers["ghost"][0]  # odd images of the shear tractions about the surface
        eqs += [Eq(t.forward, -t.forward.subs(z, z + z.spacing), subdomain=gh) for t in (txz, tyz)]
    else:
        eqs = vel(None, SO, SO) + mirror([(vx, 1), (vy, -1), (vz, 1)]) + stress(None, SO, SO)
    eqs += mirror([(txx, 1), (tyy, 1), (tzz, 1), (txz, 1), (txy, -1), (tyz, -1)])

    if plane is not None:
        ang = np.radians(plane["angle_deg"])
        cx, cy = plane.get("center", (0.0, 0.0))
        X = g["origin"][0] + np.arange(nx) * h - cx
        Y = g["origin"][1] + np.arange(ny) * h - cy

        def taper(c, flat, n):  # 1 inside |c| <= flat, cos^2 to zero at the wall
            edge = np.maximum(np.abs(c[0]), np.abs(c[-1]))
            return np.cos(0.5 * np.pi * np.clip((np.abs(c) - flat) / max(edge - flat, h), 0, 1)) ** 2
        amp = Function(name="amp", dimensions=(x, y), shape=(nx, ny), space_order=0)
        amp.data[:] = np.outer(taper(X, plane["flat"][0], nx), taper(Y, plane["flat"][1], ny))
        tdel = Function(name="tdel", dimensions=(x, y), shape=(nx, ny), space_order=0)
        tdel.data[:] = (plane["t0"] + X * np.sin(ang) / plane["vp"])[:, None]
        if tdel.data.min() < 1.0 / plane["f0"]:
            raise ValueError("plane-wave t0 too early: the wavelet would start before t = 0 at the sheet's edge")
        arg = (np.pi * plane["f0"] * ((grid.time_dim + 1) * s - tdel)) ** 2
        w = (1 - 2 * arg) * exp(-arg)
        eqs += [Eq(t.forward, t.forward + s * amp * w, subdomain=layers["sheet"][0]) for t in (txx, tyy, tzz)]
    t_ax = np.arange(nt) * dt
    if point is not None:
        src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=nt, coordinates=snap(g, point["xyz"]))
        src.data[:, 0] = ricker(t_ax, point["f0"], point["t0"])
        for t in (txx, tyy, tzz):
            eqs += src.inject(field=t.forward, expr=s * src)
    recs = {}
    for d, (comp, f) in enumerate((("vx", vx), ("vy", vy), ("vz", vz))):
        if rec and rec.get(comp) is not None and len(rec[comp]):
            # Read the grid sample itself. Devito's interpolation of a staggered field averages the
            # neighbors along the staggered axis (weights 1/4, 1/2, 1/4 at a sample's own point), which
            # low-passes the trace and, at the first v_z level under a free surface, mixes in the zero
            # above it. The indexified field at the node with the same array index is the exact sample.
            idx = (np.asarray(rec[comp], float) - g["origin"]) / h - 0.5 * np.eye(3)[d]
            if np.abs(idx - np.round(idx)).max() > 1e-3:
                raise ValueError(f"{comp} receivers must sit on that component's own staggered points")
            r = SparseTimeFunction(name=f"rcv{comp}", grid=grid, npoint=len(idx), nt=nt,
                                   coordinates=g["origin"] + np.round(idx) * h)
            recs[comp] = r
            eqs += r.interpolate(expr=f.indexify())
    op = Operator(eqs)
    t0 = time.time()
    op(dt=dt, time_M=nt - 2)
    sec = time.time() - t0
    out = {c: np.array(r.data) for c, r in recs.items()}
    for f in list(v) + [txx, tyy, tzz, txy, txz, tyz]:
        if not np.isfinite(f.data[0, ::7, ::7, ::7]).all():
            raise FloatingPointError("the run diverged")
    out["vmax_end"] = float(max(np.abs(f.data[(nt - 1) % 2, ::3, ::3, ::3]).max() for f in v))  # what is left in the box
    npts = nx * ny * nz
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    out.update(seconds=sec, npts=npts, mpts_per_s=npts * (nt - 1) / sec / 1e6, rss_gb=rss, dt=dt, nt=nt)
    return out


def scattered(with_body, without_body):
    """Scattered traces: the run with the body less the same run without it (identical grid and step)."""
    assert with_body["dt"] == without_body["dt"] and with_body["nt"] == without_body["nt"]
    return {c: with_body[c] - without_body[c] for c in ("vx", "vy", "vz") if c in with_body}


# ---------------------------------------------------------------- receivers: P / S separation
# (output, component, direction, center offset in cells, sign): div at the node; each curl component
# at its own staggered point, so every sample is a native point of its component.
_DERIVS = [("div", 0, 0, (0, 0, 0), 1), ("div", 1, 1, (0, 0, 0), 1), ("div", 2, 2, (0, 0, 0), 1),
           ("cx", 2, 1, (0, .5, .5), 1), ("cx", 1, 2, (0, .5, .5), -1),
           ("cy", 0, 2, (.5, 0, .5), 1), ("cy", 2, 0, (.5, 0, .5), -1),
           ("cz", 1, 0, (.5, .5, 0), 1), ("cz", 0, 1, (.5, .5, 0), -1)]
_S4 = ((-1.5, 1 / 24), (-0.5, -9 / 8), (0.5, 9 / 8), (1.5, -1 / 24))  # fourth-order staggered derivative


def ps_points(nodes, h):
    """Sample points dict(vx, vy, vz) for div and curl at each node (n, 3), in `div_curl`'s order."""
    pts = {0: [], 1: [], 2: []}
    for p in np.atleast_2d(nodes):
        for _, comp, d, off, _ in _DERIVS:
            for sh, _ in _S4:
                q = p + np.array(off) * h
                q[d] += sh * h
                pts[comp].append(q)
    return dict(vx=np.array(pts[0]), vy=np.array(pts[1]), vz=np.array(pts[2]))


def div_curl(tr, n_nodes, h):
    """(div (nt, n), curl (nt, n, 3)) from traces sampled at `ps_points`."""
    comp = [tr["vx"], tr["vy"], tr["vz"]]
    nt = comp[0].shape[0]
    out = dict(div=np.zeros((nt, n_nodes)), cx=np.zeros((nt, n_nodes)), cy=np.zeros((nt, n_nodes)),
               cz=np.zeros((nt, n_nodes)))
    pos = [0, 0, 0]
    for i in range(n_nodes):
        for name, c, _, _, sign in _DERIVS:
            for _, w in _S4:
                out[name][:, i] += sign * w / h * comp[c][:, pos[c]]
                pos[c] += 1
    return out["div"], np.stack([out["cx"], out["cy"], out["cz"]], axis=-1)


def native(nodes, h):
    """Each component's own staggered point next to each node: vx at +h/2 in x, and so on."""
    nodes = np.atleast_2d(np.asarray(nodes, float))
    return {c: nodes + np.eye(3)[d] * h / 2 for d, c in enumerate(("vx", "vy", "vz"))}


def window(t, t_lo, t_hi, taper):
    """1 between t_lo and t_hi, cosine tapers of length `taper` outside."""
    u = np.clip((t - (t_lo - taper)) / taper, 0, 1) * np.clip(((t_hi + taper) - t) / taper, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def win_dft(trace, f, t_lo, t_hi, dt, taper):
    """Complex DFT at f (exp(+i w t) kernel, so a phase follows the exp(-i w t) convention)."""
    t = np.arange(trace.shape[0]) * dt
    w = window(t, t_lo, t_hi, taper) * np.exp(2j * np.pi * f * t) * dt
    return np.tensordot(w, trace, axes=(0, 0))


def rayleigh_measure(v_rad, v_z, f, t_lo, t_hi, dt, taper):
    """Rayleigh wave by its window and particle motion: (|v_z|, |v_rad|, phase of v_rad / v_z in degrees).

    For a retrograde Rayleigh wave at the surface (z down) the radial motion leads or lags the
    vertical by a quarter cycle: the phase is +90 or -90 degrees, not 0 or 180 as for a body wave.
    """
    az, ar = win_dft(v_z, f, t_lo, t_hi, dt, taper), win_dft(v_rad, f, t_lo, t_hi, dt, taper)
    return abs(az), abs(ar), float(np.degrees(np.angle(ar / az)))


# ---------------------------------------------------------------- self-check
HOST = (1200.0, 1200.0 / 1.8, 2000.0)


def explosion_check(f0=3.0, h=10.0, sym_y=False):
    """Full-space explosion: |div v| against k_p^2 |W| h^3 / (4 pi rho vp^2 r), and the curl leak.

    sym_y: the same problem on the half box y >= 0, the source on the mirror plane."""
    vp, vs, rho = HOST
    g = make_grid(lo=(-300, 0 if sym_y else -300, -300), hi=(300, 300, 300), h=h, nbl=20, sym_y=sym_y)
    par = effective_params(g, lambda z: HOST)
    dt = time_step(g, vp)
    nodes = snap(g, [(150.0, 0.0, 0.0), (120.0, 120.0, -120.0)])
    nt = int((1.5 / f0 + 210.0 / vp + 1.5 / f0) / dt) + 1
    out = simulate(g, par, dt=dt, nt=nt, v_ref=vp, point=dict(xyz=(0, 0, 0), f0=f0, t0=1.5 / f0),
                   rec=ps_points(nodes, h))
    div, curl = div_curl(out, len(nodes), h)
    t = np.arange(nt) * dt
    W = abs(np.sum(ricker(t, f0, 1.5 / f0) * np.exp(2j * np.pi * f0 * t)) * dt)
    kp, ks = 2 * np.pi * f0 / vp, 2 * np.pi * f0 / vs
    r = np.linalg.norm(nodes, axis=1)
    num = np.abs(win_dft(div, f0, 0.0, nt * dt, dt, 1e-9))
    ref = kp**2 * W * h**3 / (4 * np.pi * rho * vp**2 * r)
    leak = np.linalg.norm(np.abs(win_dft(curl, f0, 0.0, nt * dt, dt, 1e-9)), axis=-1) / ks / (num / kp)
    return num / ref, leak, out


def plane_check(angle_deg=20.0, f0=10.0, h=10.0):
    """Tilted plane P wave: apparent horizontal speed, polarization angle and the S leak (curl)."""
    vp, vs, _ = HOST
    g = make_grid(lo=(-600, -600, -150), hi=(600, 600, 250), h=h, nbl=20)
    par = effective_params(g, lambda z: HOST)
    dt = time_step(g, vp)
    nodes = snap(g, [(-50.0, 0.0, 0.0), (50.0, 0.0, 0.0)])
    t0 = 1.5 / f0 + 800.0 * np.sin(np.radians(angle_deg)) / vp
    nt = int((t0 + 200.0 / vp + 1.2 / f0) / dt) + 1
    rec = ps_points(nodes, h)
    nat = native(nodes, h)
    rec = {c: np.vstack([rec[c], nat[c]]) for c in rec}
    out = simulate(g, par, dt=dt, nt=nt, v_ref=vp, rec=rec,
                   plane=dict(z=150.0, angle_deg=angle_deg, vp=vp, f0=f0, t0=t0, flat=(500.0, 500.0)))
    div, curl = div_curl({c: out[c][:, :-2] for c in ("vx", "vy", "vz")}, 2, h)
    a = win_dft(div, f0, 0.0, nt * dt, dt, 1e-9)
    slow = np.angle(a[1] / a[0]) / (2 * np.pi * f0) / 100.0  # s/m along x
    ax, az = (abs(win_dft(out[c][:, -2], f0, 0.0, nt * dt, dt, 1e-9)) for c in ("vx", "vz"))
    kp, ks = 2 * np.pi * f0 / vp, 2 * np.pi * f0 / vs
    leak = np.linalg.norm(np.abs(win_dft(curl, f0, 0.0, nt * dt, dt, 1e-9)), axis=-1) / ks / (np.abs(a) / kp)
    return slow * vp / np.sin(np.radians(angle_deg)), np.degrees(np.arctan2(ax, az)), leak


def surface_check(f0=3.0, h=10.0):
    """Free surface, explosion at 50 m: the run stays bounded after the waves have left the box."""
    vp = HOST[0]
    g = make_grid(lo=(-300, -300, 0), hi=(300, 300, 300), h=h, nbl=20, free_surface=True)
    par = effective_params(g, lambda z: HOST)
    dt = time_step(g, vp)
    nt = int(4.0 / dt)
    pts = native(snap(g, [(200.0, 0.0, 0.0)]), h)
    out = simulate(g, par, dt=dt, nt=nt, v_ref=vp, point=dict(xyz=(0, 0, 50.0), f0=f0, t0=1.5 / f0), rec=pts)
    vz = np.abs(out["vz"][:, 0])
    return float(vz[-nt // 4:].max() / vz.max())


def _selfcheck() -> None:
    g = make_grid(lo=(-200, -200, -200), hi=(200, 200, 200), h=10.0, nbl=5)
    xyz, vol = body_cells(g, sphere((0, 0, 0), 100.0))
    assert abs(vol.sum() / (4 * np.pi * 1e6 / 3) - 1) < 2e-3, vol.sum()
    assert np.abs((xyz * vol[:, None]).sum(0) / vol.sum()).max() < 0.1
    gf = make_grid(lo=(-200, -200, 0), hi=(200, 200, 300), h=10.0, nbl=5, free_surface=True)
    xyz, vol = body_cells(gf, square_body(x=0, y=0, apex_depth=42.0, base=300.0, height=156.0))
    assert abs(vol.sum() / (300.0**2 * 156.0 / 3) - 1) < 1e-2, vol.sum()  # the flat base falls between sub-samples
    assert abs((xyz[:, 2] * vol).sum() / vol.sum() - (42.0 + 0.75 * 156.0)) < 0.5  # centroid 3/4 down from the apex
    stone = tuple(2 * q for q in HOST)
    par = effective_params(g, lambda z: HOST, sphere((0, 0, 0), 100.0), stone)
    c = tuple(n // 2 for n in g["shape"])
    assert np.isclose(par["mu"][c], stone[2] * stone[1] ** 2, rtol=1e-5) and np.isclose(par["bx"][0, 0, 0], 1 / HOST[2])
    assert par["mxz"].max() <= stone[2] * stone[1] ** 2 * 1.0001 and par["mxz"].min() >= HOST[2] * HOST[1] ** 2 * 0.9999
    # stencils: a plane P wave has no curl, a plane S wave no divergence
    h, k = 10.0, 2 * np.pi / 150.0
    n = np.array([0.36, 0.48, 0.8])
    pol_s = np.cross(n, [0, 0, 1.0]) / np.linalg.norm(np.cross(n, [0, 0, 1.0]))
    node = np.array([[30.0, -20.0, 10.0]])
    pts = ps_points(node, h)
    for pol, want_div, want_curl in ((n, k, 0.0), (pol_s, 0.0, k)):
        tr = {c: (pol[d] * np.exp(1j * k * pts[c] @ n))[None, :] for d, c in enumerate(("vx", "vy", "vz"))}
        dr, cr = div_curl({c: q.real for c, q in tr.items()}, 1, h)
        di, ci = div_curl({c: q.imag for c, q in tr.items()}, 1, h)
        assert abs(abs(dr[0, 0] + 1j * di[0, 0]) - want_div) < 2e-3 * k
        assert abs(np.linalg.norm(cr[0, 0] + 1j * ci[0, 0]) - want_curl) < 2e-3 * k
    t = np.arange(4000) * 1e-3
    assert abs(abs(win_dft(np.exp(-2j * np.pi * 2.0 * t), 2.0, 1.0, 3.0, 1e-3, 0.1)) - 2.1) < 1e-6  # flat 2 s + two half tapers
    ratio, leak, out = explosion_check()
    print("explosion |div v| / analytic", ratio, "curl leak", leak, f"{out['mpts_per_s']:.0f} Mpts/s")
    assert np.abs(ratio - 1).max() < 0.03 and leak.max() < 0.03
    speed, pol, leak = plane_check()
    print("plane wave: apparent speed / (vp / sin)", speed, "polarization deg", pol, "curl leak", leak)
    assert abs(speed - 1) < 0.02 and abs(pol - 20.0) < 1.0
    assert leak.max() < 2e-3  # 1e-2 when the receivers were read through Devito's staggered interpolation
    qf, cf = sls_q([1.5, 3.0, 6.0], 50.0, 3.0)  # the solid's Q: 50 at f0, 62.5 an octave either side; speed unchanged at f0
    assert np.allclose(qf, [62.5, 50.0, 62.5], rtol=1e-3) and abs(cf[1] - 1) < 1e-12 and cf[0] < 1 < cf[2]
    tail = surface_check()
    print("free surface: late / peak |vz|", tail)
    assert tail < 0.05
    ratio_s, leak_s, _ = explosion_check(sym_y=True)
    print("mirror plane: explosion on the plane, |div v| / analytic", ratio_s, "curl leak", leak_s)
    assert np.abs(ratio_s - ratio).max() < 2e-3 and leak_s.max() < 0.03  # the half box reproduces the full one
    print("a8_elastic3d selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
