"""P4 C2: what the public gravity stations could say about the two classes no other sensor reaches.

Dense stone and a hollow body have no magnetic or EM signature (p3_g1a_decision.csv); gravity senses
both. This closes the gravity column of the ledger with a number, generously.

Inputs:
- stations: USGS Data Series 264, AlaskaGravityData/AkGrav2006.txt (fixed columns; latitude and
  longitude in degrees and minutes, NAD27), extracted to $AKP_DATA/p4/gravity/ds264/extracted;
- the Track A prior and the cover-weighted prior (p3_prior.npz, p3_cover.npz).

Method:
- the body is the center of the trials' prior (BASE_M, TOP_M, p3_mag_forward.HEIGHT_M), as point
  masses on a CELL_M grid. Its vertical attraction at the ground is computed against horizontal
  distance from the axis, for each density contrast in CONTRASTS (TARGETS gives the reasoning);
- r_detect is the largest distance at which the anomaly still reaches G_MIN_MGAL. A station inside
  r_detect of the axis is counted as having "sensed" the body. This is generous twice over: one
  station cannot tell a compact body from regional structure, and G_MIN_MGAL ignores terrain and
  elevation errors;
- searched mass = sum over stations in Track A of the prior mass of the station's 1 km cell times
  the share of the cell within r_detect of the station (capped at the whole cell).
Writes p3_gravity_closure.csv: one row per target and threshold, with the peak anomaly, r_detect,
the station counts and the searched mass under both priors.
Self-check: far from the body the point-mass sum equals a point mass of the same total.
Run: python -m akp.stage_p3.p3_gravity_closure
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer

from akp.paths import data_dir
from akp.stage_p3.p3_mag_forward import HEIGHT_M
from akp.stage_p3.p3_prior import out_dir as prior_dir

HERE = Path(__file__).parent
BIG_G = 6.674e-11
BASE_M, TOP_M, CELL_M = 300.0, 38.1, 5.0  # the trials' prior center (p3_identifiability)
# Density contrasts in kg/m3 against wet unconsolidated fill at 2000 (p3_PROPERTIES, density rows: Telford Table 2.2).
TARGETS = {"T1_felsic_stone": 640.0,  # granite average 2640
           "T2_mafic_stone": 990.0,  # basalt average 2990
           "T5_hollow": -2000.0}  # an empty body
G_MIN_MGAL = (0.2, 0.5, 1.0)  # station anomaly thresholds: optimistic, plausible, typical reconnaissance accuracy (inputs)


def body() -> tuple[np.ndarray, float]:
    """Cell centers (n, 3; z up, ground at 0) of the pyramid and the cell volume."""
    n_xy = int(np.ceil(BASE_M / CELL_M))
    c = (np.arange(n_xy) + 0.5) * CELL_M - n_xy * CELL_M / 2
    z = -(TOP_M + (np.arange(int(np.ceil(HEIGHT_M / CELL_M))) + 0.5) * CELL_M)
    xx, yy, zz = np.meshgrid(c, c, z, indexing="ij")
    half = 0.5 * BASE_M * (-zz - TOP_M) / HEIGHT_M
    keep = (np.abs(xx) <= half) & (np.abs(yy) <= half) & (-zz - TOP_M <= HEIGHT_M)
    return np.c_[xx[keep], yy[keep], zz[keep]], CELL_M**3


def gz_profile(r: np.ndarray, cells: np.ndarray, vol: float) -> np.ndarray:
    """Vertical attraction (mGal per kg/m3 of contrast) at the ground, at distances r along x from the axis."""
    d = np.stack([r[:, None] - cells[None, :, 0], -cells[None, :, 1].repeat(len(r), 0), -cells[None, :, 2].repeat(len(r), 0)], axis=-1)
    return BIG_G * vol * (d[..., 2] / np.linalg.norm(d, axis=-1) ** 3).sum(1) * 1e5


def stations() -> pd.DataFrame:
    f = data_dir("p4/gravity/ds264") / "extracted/AlaskaGravityData/AkGrav2006.txt"
    lat, lon = [], []
    with open(f, encoding="latin-1") as fh:
        for line in fh:
            try:
                lat.append(float(line[9:11]) + float(line[12:17]) / 60)
                lon.append(-(float(line[18:21]) + float(line[22:27]) / 60))
            except ValueError:
                continue
    x, y = Transformer.from_crs("EPSG:4267", "EPSG:3338", always_xy=True).transform(np.array(lon), np.array(lat))
    return pd.DataFrame(dict(x=x, y=y))


def main() -> None:
    cells, vol = body()
    far = gz_profile(np.array([5000.0]), cells, vol)[0]
    zc = -cells[:, 2].mean()
    point = BIG_G * vol * len(cells) * zc / (5000.0**2 + zc**2) ** 1.5 * 1e5
    assert abs(far - point) < 0.01 * abs(point), "far from the body the sum must equal a point mass"
    r = np.arange(0.0, 2001.0, 5.0)
    unit = gz_profile(r, cells, vol)
    pr, cv = np.load(prior_dir() / "p3_prior.npz"), np.load(prior_dir() / "p3_cover.npz")
    land, ta = pr["land"], pr["track_a"]
    idx = np.full(land.shape, -1)
    idx[land] = np.arange(land.sum())
    st = stations()
    col, row = ~Affine(*pr["transform"]) * (st.x.to_numpy(), st.y.to_numpy())
    row, col = np.floor(row).astype(int), np.floor(col).astype(int)
    ok = (row >= 0) & (row < land.shape[0]) & (col >= 0) & (col < land.shape[1])
    cell = np.full(len(st), -1)
    cell[ok] = idx[row[ok], col[ok]]
    cell = cell[cell >= 0]
    cell = cell[ta[cell]]
    per_cell = np.bincount(cell, minlength=land.sum())
    rows = []
    for name, drho in TARGETS.items():
        g = unit * drho
        for g_min in G_MIN_MGAL:
            reach = r[np.abs(g) >= g_min]
            r_det = float(reach.max()) if reach.size else 0.0
            share = np.minimum(1.0, per_cell * np.pi * r_det**2 / 1e6)  # share of each 1 km cell within r_det of a station
            rows.append(dict(target=name, contrast_kg_m3=drho, g_min_mgal=g_min, peak_mgal=round(float(g[0]), 3),
                             r_detect_m=r_det, stations_statewide=len(st), stations_track_a=int(per_cell.sum()),
                             track_a_cells=int(ta.sum()), cells_with_station=int((per_cell > 0).sum()),
                             searched_mass_W_cover=round(float((cv["mass_cover"] * share).sum()), 5),
                             searched_mass_W_none=round(float((pr["mass"] * share).sum()), 5)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_gravity_closure.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
