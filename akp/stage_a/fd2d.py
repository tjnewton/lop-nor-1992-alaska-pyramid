"""Shared 2-D elastic velocity-stress FD for A3b-2 and A3b-4.

Devito, CPU C backend, space order 8, first order in time. A line of explosive sources (equal
injection into tau_xx and tau_zz) fires a Ricker wavelet; vx and vz are recorded at given points.
The model comes in as Lame and density arrays, so the free surface (vacuum formalism, lam = mu = 0)
and the body are the caller's business. The sponge is the profile both stages used:
d = 1 - 0.0035 ((nb - i) / nb)^2 * 100 / 10 on each damped edge.
Extracted unchanged from a3b2_numerical.run and a3b4_rg.fd_run (2026-11-10); the fifth A3b-2 run
checks that the extracted code reproduces the fourth run's archived traces.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("DEVITO_LANGUAGE", "C")
os.environ.setdefault("DEVITO_LOGGING", "ERROR")

SO = 8
SPONGE_SCALE = 100  # the sponge strength reference (both stages' NBL)
T0_PERIODS = 1.5  # Ricker delay, in periods


def triangle_mask(xx, zz, base, height, xb, zb):
    """Apex-up triangle (apex toward -z) with its centroid at (xb, zb)."""
    zz_up = zb + height / 3.0 - zz
    half = 0.5 * base * (1.0 - zz_up / height)
    return (zz_up >= 0) & (zz_up <= height) & (np.abs(xx - xb) <= half)


def ricker(t, f0):
    a = (np.pi * f0 * (t - T0_PERIODS / f0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def sponge(nx, nz, nb, *, top=True):
    """Damping array on the sides and bottom (index -1 in z), plus the top unless top=False."""
    d = np.ones((nx, nz))
    for i in range(nb):
        f = 1.0 - 0.0035 * ((nb - i) / nb) ** 2 * SPONGE_SCALE / 10
        d[i, :] *= f
        d[-1 - i, :] *= f
        if top:
            d[:, i] *= f
        d[:, -1 - i] *= f
    return d


def tukey_window(trace, t_arr, dt, half):
    """(windowed trace, times): a Tukey (alpha 0.5) window of half-width `half` centered at t_arr."""
    from scipy.signal.windows import tukey
    t = np.arange(len(trace)) * dt
    sel = (t >= t_arr - half) & (t <= t_arr + half)
    w = np.zeros_like(trace)
    w[sel] = tukey(int(sel.sum()), alpha=0.5)
    return w * trace, t


def tukey_dft(trace, f0, t_arr, dt, half):
    """Complex DFT at f0 of the Tukey-windowed trace."""
    y, t = tukey_window(trace, t_arr, dt, half)
    return complex(np.sum(y * np.exp(-2j * np.pi * f0 * t)) * dt)


def run(*, dx, dt, nt, lam, mu, rho, damp, src_x, src_z, f0, rec):
    """One run on a grid of lam.shape with spacing dx; coordinates are grid coordinates (from 0).

    src_x: source x positions on the line z = src_z; rec: (n, 2) receiver (x, z). Returns (vx, vz),
    each (nt, n).
    """
    from devito import (Eq, Function, Grid, Operator, SparseTimeFunction, TensorTimeFunction,
                        VectorTimeFunction, diag, div, grad)
    nx, nz = lam.shape
    grid = Grid(shape=(nx, nz), extent=((nx - 1) * dx, (nz - 1) * dx))
    lam_ = Function(name="lam", grid=grid, space_order=SO)
    mu_ = Function(name="mu", grid=grid, space_order=SO)
    b = Function(name="b", grid=grid, space_order=SO)
    lam_.data[:], mu_.data[:], b.data[:] = lam, mu, 1.0 / rho
    dmp = Function(name="damp", grid=grid)
    dmp.data[:] = damp
    v = VectorTimeFunction(name="v", grid=grid, space_order=SO, time_order=1)
    tau = TensorTimeFunction(name="tau", grid=grid, space_order=SO, time_order=1)
    s = grid.stepping_dim.spacing
    u_v = Eq(v.forward, dmp * (v + s * b * div(tau)))
    u_t = Eq(tau.forward, dmp * (tau + s * (lam_ * diag(div(v.forward))
                                            + mu_ * (grad(v.forward) + grad(v.forward).transpose(inner=False)))))
    t = np.arange(nt) * dt
    src = SparseTimeFunction(name="src", grid=grid, npoint=len(src_x), nt=nt,
                             coordinates=np.column_stack([src_x, np.full(len(src_x), src_z)]))
    src.data[:] = ricker(t, f0)[:, None]
    recz = SparseTimeFunction(name="recz", grid=grid, npoint=len(rec), nt=nt, coordinates=rec)
    recx = SparseTimeFunction(name="recx", grid=grid, npoint=len(rec), nt=nt, coordinates=rec)
    terms = src.inject(field=tau.forward[0, 0], expr=s * src) + src.inject(field=tau.forward[1, 1], expr=s * src)
    terms += recz.interpolate(expr=v[1]) + recx.interpolate(expr=v[0])
    Operator([u_v, u_t] + terms)(dt=dt, time_M=nt - 2)
    return np.array(recx.data), np.array(recz.data)


def _selfcheck() -> None:
    xx, zz = np.meshgrid(np.linspace(-50, 50, 201), np.linspace(-50, 50, 201), indexing="ij")
    m = triangle_mask(xx, zz, 60.0, 40.0, 0.0, 0.0)
    assert abs(m.mean() * 100.0**2 - 1200.0) / 1200.0 < 0.02  # area base * height / 2
    assert abs(zz[m].mean()) < 0.5  # centroid at zb
    d = sponge(300, 300, 100)
    assert d[150, 150] == 1.0 and d[0, 150] < d[50, 150] < 1.0 and np.allclose(d, d.T)
    assert sponge(300, 300, 100, top=False)[150, 0] == 1.0
    t = np.arange(4000) * 1e-3
    tr = np.cos(2 * np.pi * 2.0 * t)
    assert abs(abs(tukey_dft(tr, 2.0, 2.0, 1e-3, 1.0)) - 0.5 * 2.0 * 0.75) < 0.02  # 0.5 * width * Tukey mean
    print("fd2d selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
