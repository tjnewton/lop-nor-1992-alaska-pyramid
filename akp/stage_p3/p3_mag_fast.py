"""P4 M1: a fast analytic forward model of the pyramid's magnetic anomaly.

The pyramid is a stack of N_LAYERS uniformly magnetized rectangular prisms, each the mean square
section of its layer. The field of one prism is closed-form: B_i = (B0 / 4 pi) k sum_j U_ij m_j, with
U_ij the second derivatives of the prism's 1/r volume integral (the gravity-gradient kernels of
Nagy et al. 2000, summed over the eight corners), k the effective susceptibility and m the unit
magnetization direction. A turn of the body is a rotation of the receivers and of the vectors into
the body frame. Conventions match p3_mag_inject.response: receivers relative to the plant, heights
above flat ground, x east, y north, z up, magnetization dk along the inducing field plus q * dk
along a remanent direction, TMI as the projection on the inducing field.

response() is a drop-in for p3_mag_inject.response and needs no mesh. Validation has two parts,
because the SimPEG model is itself a staircase of 10 m cells:
- the kernel: the prism formula on SimPEG's own active cells must reproduce SimPEG to TOLERANCE;
- the geometry: the layered pyramid against SimPEG at the working cell and at half of it, to show
  that the remaining difference shrinks as SimPEG's staircase is refined.
Self-check (run: python -m akp.stage_p3.p3_mag_fast): the tensor is trace-free; a small prism far
away matches a point dipole.
Validation (run: python -m akp.stage_p3.p3_mag_fast validate): writes p3_mag_fast_check.csv, one row
per case, over receivers on a line pattern, as shares of the SimPEG peak: same_cells_diff_share
(the kernel check, at the working cell) and max_diff_share at the working cell and at half of it,
with the run times. kernel_ok is same_cells_diff_share <= TOLERANCE.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from akp.stage_p3.p3_mag_forward import FIELD, HEIGHT_M

HERE = Path(__file__).parent
N_LAYERS = 56  # 3 m layers: the staircase error is far below the validation tolerance
TOLERANCE = 0.01  # Kill above this share of the peak
CHUNK = 500  # prisms per kernel call in the same-cells check (memory)


def unit(inc: float, dec: float) -> np.ndarray:
    i, d = np.radians(inc), np.radians(dec)
    return np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), -np.sin(i)])  # x east, y north, z up


def _log(a, b2):
    """ln(a + r) for r = sqrt(a^2 + b2), without cancellation when a < 0."""
    r = np.sqrt(a * a + b2)
    return np.where(a >= 0, np.log(a + r, where=(a + r) > 0, out=np.zeros_like(r)), np.log(b2 / (r - a), where=b2 > 0, out=np.zeros_like(r)))


def _atan(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, np.arctan(num / den), np.sign(num) * np.pi / 2)


def prism_tensor(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """U_ij at obs (n, 3), summed over prisms with corners lo and hi (each (p, 3)): shape (n, 3, 3)."""
    out = np.zeros((len(obs), 3, 3))
    for i, xs in enumerate((lo[:, 0], hi[:, 0])):
        for j, ys in enumerate((lo[:, 1], hi[:, 1])):
            for k, zs in enumerate((lo[:, 2], hi[:, 2])):
                sign = (-1.0) ** (i + j + k + 1)  # + at the (hi, hi, hi) corner
                x, y, z = xs[None, :] - obs[:, 0:1], ys[None, :] - obs[:, 1:2], zs[None, :] - obs[:, 2:3]
                r = np.sqrt(x * x + y * y + z * z)
                out[:, 0, 0] += sign * (-_atan(y * z, x * r)).sum(1)
                out[:, 1, 1] += sign * (-_atan(x * z, y * r)).sum(1)
                out[:, 2, 2] += sign * (-_atan(x * y, z * r)).sum(1)
                out[:, 0, 1] += sign * _log(z, x * x + y * y).sum(1)
                out[:, 0, 2] += sign * _log(y, x * x + z * z).sum(1)
                out[:, 1, 2] += sign * _log(x, y * y + z * z).sum(1)
    out[:, 1, 0], out[:, 2, 0], out[:, 2, 1] = out[:, 0, 1], out[:, 0, 2], out[:, 1, 2]
    return out


def layers(base: float, top: float, n: int = N_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """Prism corners (lo, hi) of the layered pyramid in its own frame: apex at depth top, base at top + HEIGHT_M."""
    d = np.linspace(0.0, HEIGHT_M, n + 1)
    # each layer takes the square whose area is the layer's mean section: half-width from the mean of half^2
    h0, h1 = 0.5 * base * d[:-1] / HEIGHT_M, 0.5 * base * d[1:] / HEIGHT_M
    half = np.sqrt((h0 * h0 + h0 * h1 + h1 * h1) / 3.0)
    lo = np.c_[-half, -half, -(top + d[1:])]
    hi = np.c_[half, half, -(top + d[:-1])]
    return lo, hi


def response(xr, yr, hr, base, top, turn, dk, q=0.0, rem=(0.0, 0.0), field: dict | None = None) -> np.ndarray:
    """TMI (nT) at receivers (relative coordinates, heights above ground) for one pyramid, in field (default FIELD)."""
    field = FIELD if field is None else field
    b0 = unit(field["inclination"], field["declination"])
    m = dk * b0 + q * dk * unit(*rem)
    c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # to the body frame, as p3_mag_forward.pyramid_cells
    obs = np.c_[np.asarray(xr, float), np.asarray(yr, float), np.asarray(hr, float)] @ rot.T
    u = prism_tensor(obs, *layers(base, top))
    b = field["amplitude"] / (4 * np.pi) * np.einsum("nij,j->ni", u, rot @ m)
    return b @ (rot @ b0)


def simpeg_cells_response(xr, yr, hr, base, top, turn, dk, q, rem, cell: float) -> np.ndarray:
    """TMI from the prism formula on the very cells p3_mag_inject.response activates (the kernel check)."""
    import discretize

    from akp.stage_p3.p3_mag_forward import pyramid_cells

    half_w = 0.5 * base * np.sqrt(2) + 2 * cell
    n_xy, dz = int(np.ceil(2 * half_w / cell)), 0.6 * cell
    n_z = int(np.ceil((top + HEIGHT_M + dz) / dz))
    mesh = discretize.TensorMesh([[(cell, n_xy)], [(cell, n_xy)], [(dz, n_z)]], origin=[-n_xy * cell / 2, -n_xy * cell / 2, -n_z * dz])
    cc = mesh.cell_centers[pyramid_cells(mesh, base, top, turn)]
    d = np.array([cell, cell, dz]) / 2
    b0 = unit(FIELD["inclination"], FIELD["declination"])
    m = dk * b0 + q * dk * unit(*rem)
    obs = np.c_[xr, yr, hr]
    u = sum(prism_tensor(obs, cc[i:i + CHUNK] - d, cc[i:i + CHUNK] + d) for i in range(0, len(cc), CHUNK))
    return FIELD["amplitude"] / (4 * np.pi) * np.einsum("nij,j->ni", u, m) @ b0


def self_check() -> None:
    rng = np.random.default_rng(0)
    obs = np.c_[rng.uniform(-800, 800, 200), rng.uniform(-800, 800, 200), rng.uniform(20, 300, 200)]
    u = prism_tensor(obs, *layers(300.0, 40.0))
    assert np.abs(np.trace(u, axis1=1, axis2=2)).max() < 1e-9 * np.abs(u).max(), "the tensor must be trace-free outside the body"
    # a 2 m cube 500 m and more away against a point dipole of the same moment
    lo, hi = np.array([[-1.0, -1.0, -101.0]]), np.array([[1.0, 1.0, -99.0]])
    far = np.c_[rng.uniform(-500, 500, 50), rng.uniform(-500, 500, 50), rng.uniform(400, 600, 50)]
    mvec = np.array([0.3, -0.5, 0.8])
    got = np.einsum("nij,j->ni", prism_tensor(far, lo, hi), mvec)
    r = far - np.array([0.0, 0.0, -100.0])
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    want = 8.0 * (3 * (r @ mvec)[:, None] * r / rn**2 - mvec) / rn**3
    assert np.abs(got - want).max() < 1e-3 * np.abs(want).max(), "far field must be the dipole's"
    print("p3_mag_fast self-check OK")


def validate() -> None:
    import pandas as pd

    from akp.stage_p3 import p3_mag_inject as inj

    rng = np.random.default_rng(20261113)
    rows = []
    cases = [(300.0, 38.1, 0.0, 0.0), (200.0, 30.48, 30.0, 0.0), (400.0, 45.72, 45.0, 2.0), (300.0, 45.72, 17.0, 8.0)]
    for base, top, turn, q in cases:
        # receivers as a survey sees the body: lines 400 m apart, samples every 10 m, clearance near 40 m with scatter
        ys = np.arange(-1200.0, 1201.0, 400.0) + rng.uniform(-200, 200)
        x = np.tile(np.arange(-1500.0, 1501.0, 10.0), len(ys))
        y = np.repeat(ys, len(x) // len(ys))
        h = 40.0 + rng.normal(0, 5, x.size)
        rem = (float(np.degrees(np.arcsin(rng.uniform(-1, 1)))), float(rng.uniform(0, 360)))
        t0 = time.time()
        fast = response(x, y, h, base, top, turn, 0.03, q, rem)
        t_fast = time.time() - t0
        row = dict(base=base, top=top, turn=turn, q=q, receivers=x.size, fast_s=round(t_fast, 3))
        same = simpeg_cells_response(x, y, h, base, top, turn, 0.03, q, rem, inj.CELL_M)
        for cell in (inj.CELL_M, inj.CELL_M / 2):
            keep, inj.CELL_M = inj.CELL_M, cell
            try:
                t0 = time.time()
                ref = inj.response(x, y, h, base, top, turn, 0.03, q, rem)
                row[f"simpeg_cell{cell:g}_s"] = round(time.time() - t0, 2)
            finally:
                inj.CELL_M = keep
            row[f"max_diff_share_cell{cell:g}"] = round(float(np.abs(fast - ref).max() / np.abs(ref).max()), 5)
            row[f"peak_nT_cell{cell:g}"] = round(float(np.abs(ref).max()), 3)
            if cell == inj.CELL_M:
                row["same_cells_diff_share"] = round(float(np.abs(same - ref).max() / np.abs(ref).max()), 6)
        row["peak_nT_fast"] = round(float(np.abs(fast).max()), 3)
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out["kernel_ok"] = out.same_cells_diff_share <= TOLERANCE
    assert out.kernel_ok.all(), "the prism kernel does not reproduce SimPEG on its own cells"
    out.to_csv(HERE / "p3_mag_fast_check.csv", index=False)


if __name__ == "__main__":
    self_check()
    if sys.argv[1:] == ["validate"]:
        validate()
