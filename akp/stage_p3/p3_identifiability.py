"""P4 M2: how much shape information magnetic line data carry about the target.

Question: with flight lines SPACINGS_M apart and a sensor CLEARANCES_M above the ground, can a
survey tell the pyramid from another body that fits the same anomaly? The answer is a discrimination
index d' per alternative: the norm of what is left of the pyramid's anomaly after the best-fitting
alternative is taken out, over the noise. d' below 1 means no discrimination; above D_CLEAR it is
clear. The alternative's free parameters are profiled out (a grid search, with the amplitude by
least squares), so d' is the distance to the nearest member of each family, not to one fixed shape.

Target: the headline body of the trials' prior center (base BASE_M, top TOP_M, height
p3_mag_forward.HEIGHT_M), induced magnetization at the T2 high contrast, at a random turn and a
random cross-line offset (N_DRAWS draws per cell, seeded). Lines run east-west through
+-HALF_SPAN_M; each line is sampled as p3_mag_forward does (DX, NPTS).

Alternatives:
- twin: the pyramid averaged over all turns, an axisymmetric body with the same volume at every
  depth. What separates it from the pyramid is the four-fold part of the plan shape alone;
- sphere: a point dipole below the apex, depth profiled;
- plug: a vertical-sided square prism with the same top, width and depth extent profiled;
- dike: a long thin vertical sheet through the apex, strike and width profiled.

Noise cases:
- instrument: white, p3_mag_forward.WHITE_NT per sample. No survey does better than this;
- clutter_median and clutter_p90: the along-line clutter spectrum of the control survey
  (p3_mag_forward._psds, cols_ns), through p3_mag_forward.snr. Lines are combined in quadrature,
  which treats the clutter on neighboring lines as independent and so overstates d'.
The amplitude fit is unweighted least squares in every case.

Writes p3_identifiability.csv: one row per spacing, clearance, alternative and noise case, with the
median and the 10th and 90th percentiles of d' over the draws, the share of draws above D_CLEAR, and
the median anomaly peak. Self-check: the pyramid against itself gives zero; at close spacing the
dike is farther from the pyramid than the twin is. Run: python -m akp.stage_p3.p3_identifiability
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_p3 import p3_mag_fast as fast
from akp.stage_p3.p3_mag_forward import DX, FIELD, HEIGHT_M, NPTS, WHITE_NT, _psds, snr

HERE = Path(__file__).parent
SPACINGS_M = (100.0, 200.0, 400.0, 800.0)
CLEARANCES_M = (30.0, 60.0, 100.0, 190.0)
BASE_M, TOP_M, DK = 300.0, 38.1, 0.0691  # the prior center and the T2 high contrast (p3_g1b_run)
HALF_SPAN_M = 1200.0
N_DRAWS = 40
N_TWIN_TURNS = 16
D_CLEAR = 3.0
SEED = 20261113
SPHERE_DEPTHS_M = tuple(TOP_M + HEIGHT_M * f for f in (0.3, 0.45, 0.6, 0.75, 0.9))
PLUG_WIDTHS_M = (100.0, 150.0, 200.0, 250.0, 300.0)
PLUG_EXTENTS_M = (60.0, 120.0, HEIGHT_M)
DIKE_STRIKES_DEG = tuple(np.arange(0.0, 180.0, 15.0))
DIKE_WIDTHS_M = (20.0, 60.0)
DIKE_LENGTH_M = 3000.0


def _field(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray, turn: float = 0.0) -> np.ndarray:
    """Induced TMI at unit contrast for prisms lo..hi given in a frame turned by turn degrees."""
    b0 = fast.unit(FIELD["inclination"], FIELD["declination"])
    c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    u = fast.prism_tensor(obs @ rot.T, lo, hi)
    return FIELD["amplitude"] / (4 * np.pi) * np.einsum("nij,j->ni", u, rot @ b0) @ (rot @ b0)


def sphere(obs: np.ndarray, depth: float) -> np.ndarray:
    """Induced TMI of a unit-moment point dipole at depth below the origin."""
    b0 = fast.unit(FIELD["inclination"], FIELD["declination"])
    r = obs - np.array([0.0, 0.0, -depth])
    rn = np.linalg.norm(r, axis=1)
    return (3 * (r @ b0) ** 2 / rn**2 - 1.0) / rn**3


def residual(target: np.ndarray, family: list[np.ndarray]) -> np.ndarray:
    """The target less its best least-squares multiple of the best member of the family."""
    best = None
    for a in family:
        r = target - a * (target @ a) / (a @ a)
        if best is None or r @ r < best @ best:
            best = r
    return best


def d_prime(res: np.ndarray, n_lines: int, psds: dict) -> dict:
    lines = res.reshape(n_lines, NPTS)
    out = {"instrument": float(np.sqrt((res**2).sum()) / WHITE_NT)}
    for name, key in (("clutter_median", "cols_ns_median"), ("clutter_p90", "cols_ns_p90")):
        out[name] = float(np.sqrt(sum(snr(ln, psds[key]) ** 2 for ln in lines)))
    return out


def one_draw(rng, spacing: float, clearance: float, psds: dict) -> list[dict]:
    u, turn = rng.uniform(-spacing / 2, spacing / 2), rng.uniform(0.0, 90.0)
    ys = u + spacing * np.arange(np.ceil((-HALF_SPAN_M - u) / spacing), np.floor((HALF_SPAN_M - u) / spacing) + 1)
    x = (np.arange(NPTS) - NPTS / 2) * DX
    obs = np.c_[np.tile(x, len(ys)), np.repeat(ys, NPTS), np.full(NPTS * len(ys), clearance)]
    lo, hi = fast.layers(BASE_M, TOP_M)
    target = DK * _field(obs, lo, hi, turn)
    families = {
        "twin": [np.mean([_field(obs, lo, hi, t) for t in np.linspace(0.0, 90.0, N_TWIN_TURNS, endpoint=False)], axis=0)],
        "sphere": [sphere(obs, d) for d in SPHERE_DEPTHS_M],
        "plug": [_field(obs, np.array([[-w / 2, -w / 2, -(TOP_M + e)]]), np.array([[w / 2, w / 2, -TOP_M]]), turn)
                 for w in PLUG_WIDTHS_M for e in PLUG_EXTENTS_M],
        "dike": [_field(obs, np.array([[-DIKE_LENGTH_M / 2, -w / 2, -(TOP_M + HEIGHT_M)]]), np.array([[DIKE_LENGTH_M / 2, w / 2, -TOP_M]]), s)
                 for s in DIKE_STRIKES_DEG for w in DIKE_WIDTHS_M],
        "self": [target.copy()],
    }
    rows = []
    for alt, fam in families.items():
        for noise, d in d_prime(residual(target, fam), len(ys), psds).items():
            rows.append(dict(alternative=alt, noise=noise, d=d, peak_nT=float(np.abs(target).max()), n_lines=len(ys)))
    return rows


def main() -> None:
    psds = _psds()
    rng = np.random.default_rng(SEED)
    out = []
    for spacing in SPACINGS_M:
        for clearance in CLEARANCES_M:
            d = pd.DataFrame([r for _ in range(N_DRAWS) for r in one_draw(rng, spacing, clearance, psds)])
            assert d[d.alternative == "self"].d.max() < 1e-6, "the pyramid must be at zero distance from itself"
            for (alt, noise), g in d[d.alternative != "self"].groupby(["alternative", "noise"]):
                out.append(dict(spacing_m=spacing, clearance_m=clearance, alternative=alt, noise=noise, n_draws=len(g),
                                d_median=round(float(g.d.median()), 3), d_q10=round(float(g.d.quantile(0.1)), 3),
                                d_q90=round(float(g.d.quantile(0.9)), 3), share_clear=round(float((g.d >= D_CLEAR).mean()), 3),
                                peak_nT_median=round(float(g.peak_nT.median()), 2)))
            print(f"spacing {spacing:.0f} clearance {clearance:.0f} done", flush=True)
    res = pd.DataFrame(out)
    near = res[(res.spacing_m == SPACINGS_M[0]) & (res.clearance_m == CLEARANCES_M[0]) & (res.noise == "instrument")].set_index("alternative").d_median
    assert near["dike"] > near["twin"], "at close spacing a dike must be farther from the pyramid than its axisymmetric twin"
    res.to_csv(HERE / "p3_identifiability.csv", index=False)
    print(res[res.noise != "clutter_p90"].pivot_table(index=["noise", "alternative", "clearance_m"], columns="spacing_m", values="d_median").round(2).to_string())


if __name__ == "__main__":
    main()
