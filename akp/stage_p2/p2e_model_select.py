"""P2e / B-gate route 2: does a faceted body explain a flag better than smooth landforms?

At a flag, a window of the DTM (half-width WINDOW_BASES bases, at the flag's working
resolution) is fitted, each model on top of a free plane, by grid search over the nonlinear
parameters and least squares for the linear ones (plane and amplitude):
- facet: square faceted landform (swe.topo.faceted_template): base, rotation, center;
- cone: circular cone (a pingo-like mound): radius, center;
- dome: circular Gaussian: width, center;
- ridge: elongated Gaussian: two widths, orientation, center.
Each fit's BIC is n ln(RSS / n) + k ln n. The criterion is
delta_bic = min(BIC of the smooth models) - BIC(facet); positive favors the facet. Residuals
of real terrain are correlated, so delta_bic is not a calibrated probability: its cut is set
on the control flags, never read as significance.
Run the self-check: python -m akp.stage_p2.p2e_model_select
"""
from __future__ import annotations

import numpy as np
from swe.topo import faceted_template

WINDOW_BASES = 1.0  # window half-width, in bases
SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)  # size grid, in bases
ROTS = tuple(range(0, 90, 10))  # deg: a square repeats every 90
RIDGE_ROTS = tuple(range(0, 180, 10))  # deg: an elongated dome repeats every 180
SHIFTS = (-0.2, 0.0, 0.2)  # center grid, in bases
K = {"facet": 8, "cone": 7, "dome": 7, "ridge": 9}  # plane 3 + amplitude 1 + nonlinear


def _fit(z: np.ndarray, shapes: list[np.ndarray]) -> float:
    """Smallest RSS of z ~ plane + a * shape over the candidate shapes (z, shapes flattened).

    One projection for all shapes: with the plane removed from z and from each shape (an
    orthonormal plane basis), RSS = |z'|^2 - (s'.z')^2 / |s'|^2, the exact least-squares fit.
    """
    n = z.size
    m = int(round(np.sqrt(n)))
    yy, xx = np.mgrid[0:m, 0:m]
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), xx.ravel(), yy.ravel()]))
    zr = z - q @ (q.T @ z)
    s = np.column_stack(shapes)
    sr = s - q @ (q.T @ s)
    ss = np.einsum("ij,ij->j", sr, sr)
    proj = np.where(ss > 1e-12 * n, (sr.T @ zr) ** 2 / np.where(ss > 0, ss, 1.0), 0.0)
    return float(zr @ zr - proj.max())


def _place(shape: np.ndarray, m: int, dr: int, dc: int) -> np.ndarray:
    """`shape` centered at (m//2 + dr, m//2 + dc) in an m x m zero window."""
    out = np.zeros((m, m))
    k0, k1 = shape.shape
    r0, c0 = m // 2 + dr - k0 // 2, m // 2 + dc - k1 // 2
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + k0, m), min(c0 + k1, m)
    if rs < re and cs < ce:
        out[rs:re, cs:ce] = shape[rs - r0:re - r0, cs - c0:ce - c0]
    return out


def model_bics(window: np.ndarray, base: float, res: float) -> dict[str, float]:
    m = window.shape[0]
    assert window.shape == (m, m) and m % 2 == 1 and np.isfinite(window).all()
    z = window.ravel() - np.mean(window)
    n = z.size
    shifts = [int(round(s * base / res)) for s in SHIFTS]
    cents = [(dr, dc) for dr in shifts for dc in shifts]
    yy, xx = (np.mgrid[0:m, 0:m] - m // 2) * res
    shapes: dict[str, list[np.ndarray]] = {"facet": [], "cone": [], "dome": [], "ridge": []}
    for sc in SCALES:
        b = sc * base
        for rot in ROTS:
            t = faceted_template(n_sides=4, base=b, height=1.0, dx=res, rotation=rot)
            shapes["facet"] += [_place(t, m, dr, dc).ravel() for dr, dc in cents]
        for dr, dc in cents:
            r = np.hypot(xx - dc * res, yy - dr * res)
            shapes["cone"].append(np.clip(1 - r / (b / 2), 0, None).ravel())
            shapes["dome"].append(np.exp(-(r ** 2) / (2 * (b / 4) ** 2)).ravel())
            for ratio in (0.33, 0.5):
                for rot in RIDGE_ROTS:
                    th = np.radians(rot)
                    u = (xx - dc * res) * np.cos(th) + (yy - dr * res) * np.sin(th)
                    v = -(xx - dc * res) * np.sin(th) + (yy - dr * res) * np.cos(th)
                    shapes["ridge"].append(np.exp(-(u ** 2) / (2 * (b / 4) ** 2) - v ** 2 / (2 * (ratio * b / 4) ** 2)).ravel())
    return {k: n * np.log(max(_fit(z, v), 1e-12) / n) + K[k] * np.log(n) for k, v in shapes.items()}


def delta_bic(window: np.ndarray, base: float, res: float) -> float:
    b = model_bics(window, base, res)
    return min(b["cone"], b["dome"], b["ridge"]) - b["facet"]


def _selfcheck() -> None:
    rng = np.random.default_rng(1)
    res, base = 5.0, 100.0
    m = int(2 * WINDOW_BASES * base / res) | 1
    yy, xx = (np.mgrid[0:m, 0:m] - m // 2) * res
    noise = lambda: 0.02 * xx + rng.normal(0, 0.3, (m, m))  # noqa: E731
    facet = noise() + _place(faceted_template(n_sides=4, base=base, height=6.0, dx=res, rotation=25.0), m, 0, 0)
    cone = noise() + 6.0 * np.clip(1 - np.hypot(xx, yy) / (base / 2), 0, None)
    dome = noise() + 6.0 * np.exp(-(xx ** 2 + yy ** 2) / (2 * (base / 4) ** 2))
    th = np.radians(135.0)  # a ridge in the second half-circle of orientations
    u, v = xx * np.cos(th) + yy * np.sin(th), -xx * np.sin(th) + yy * np.cos(th)
    ridge = noise() + 6.0 * np.exp(-(u ** 2) / (2 * (base / 4) ** 2) - v ** 2 / (2 * (0.33 * base / 4) ** 2))
    cases = [("facet", facet), ("cone", cone), ("dome", dome), ("ridge", ridge)]
    d = {k: delta_bic(w, base, res) for k, w in cases}
    assert d["facet"] > 10 and d["cone"] < 0 and d["dome"] < 0 and d["ridge"] < 0, d
    print("p2e_model_select self-check OK:", {k: round(v, 1) for k, v in d.items()})


if __name__ == "__main__":
    _selfcheck()
