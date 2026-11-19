"""P2b DEM screening detector: faceted-landform template correlation on IfSAR 5 m DTM tiles.

For each base B in BASES, the DTM is block-averaged to the working resolution RES[B], its
Laplacian curvature (Gaussian-smoothed, sigma = SIGMA_FRAC * B) is correlated with the
curvature of a square faceted template of base B (swe.topo, the method of SWE PR #5) at each
rotation in ROTATIONS, and the score map is the maximum over rotations. Flags are the local
maxima of a score map (neighborhood about one base) at or above S_MIN, each with its base,
rotation, score and map position (EPSG:3338).

Calibration reads control tiles only. `open_tile` refuses any path outside
$AKP_DATA/p2b/ifsar_controls unless `claim_zone=True`, which only the frozen claim-zone run
(after the D-entry freeze) may pass. The self-check runs on synthetic surfaces.
Run the self-check: python -m akp.stage_p2.p2b_detector
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import maximum_filter
from swe.topo import curvature, faceted_template, template_correlation

from akp.paths import data_dir

BASES = (50.0, 100.0, 200.0, 400.0)  # m (input: steelman base range)
RES = {50.0: 5.0, 100.0: 5.0, 200.0: 10.0, 400.0: 20.0}  # m: at least 10 cells per base
ROTATIONS = tuple(float(r) for r in range(0, 90, 15))  # deg; a square repeats every 90
SIGMA_FRAC = 0.1  # curvature smoothing, in bases
S_MIN = 0.5  # lowest score recorded as a flag
# Tiles are processed in BLOCK x BLOCK native-cell blocks read with a HALO, keeping the flags
# whose position falls in the block's core: a whole mosaic tile (up to ~24k x 14k cells) peaks
# at tens of GB in one process. HALO exceeds the largest influence radius (smoothing 4 sigma +
# template half-extent + peak neighborhood, about 820 m at the 400 m base), and both are
# multiples of every coarsening factor, so blocked flags equal whole-tile flags.
BLOCK = 2048
HALO = 200
GDAL_CACHE_MB = 256  # GDAL's default block cache is 5% of RAM per process


def influence_cells() -> int:
    """Native cells over which a flag depends on the DTM, at the largest base: smoothing
    (4 sigma) + template half-extent (circumradius + pad) + peak neighborhood; rounded up to a
    multiple of 4 (every coarsening factor divides the halo, so blocks align)."""
    b = max(BASES)
    sigma = SIGMA_FRAC * b
    meters = 4 * sigma + (b / np.sqrt(2) + 4 * sigma + RES[b]) + b / 2
    cells = int(np.ceil(meters / 5.0))
    return -(-cells // 4) * 4


def configure(*, rotation_step: float = 15.0, sigma_frac: float = 0.1) -> None:
    """Set the design parameters (for exploration on control data; defaults)."""
    global ROTATIONS, SIGMA_FRAC, HALO
    rotations = tuple(float(r) for r in np.arange(0.0, 90.0, rotation_step))
    if (rotations, sigma_frac) == (ROTATIONS, SIGMA_FRAC):
        return  # unchanged: keep the template cache
    ROTATIONS = rotations
    SIGMA_FRAC = sigma_frac
    HALO = max(200, influence_cells())
    template_curvature.cache_clear()


@dataclass(frozen=True)
class Grid:
    z: np.ndarray  # elevation, NaN off data
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    res: float


def open_tile(path: Path, *, claim_zone: bool = False):
    path = Path(path).resolve()
    controls = data_dir("p2b/ifsar_controls").resolve()
    if not claim_zone and controls not in path.parents:
        raise PermissionError(f"Calibration may read control tiles only, not {path}")
    return rasterio.open(path)


def read_native(ds, window=None) -> Grid:
    """The tile (or a window of it) at its native resolution; nodata -> NaN."""
    win = window or rasterio.windows.Window(0, 0, ds.width, ds.height)
    a = ds.read(1, window=win, masked=True)
    return Grid(np.ma.filled(a.astype(np.float64), np.nan), ds.window_transform(win), ds.crs, abs(ds.res[0]))


def coarsen(g: Grid, res: float) -> Grid:
    """Block mean to `res` (an integer multiple of the grid's); a block with any NaN is NaN."""
    f = max(1, int(round(res / g.res)))
    if f == 1:
        return g
    h, w = (g.z.shape[0] // f) * f, (g.z.shape[1] // f) * f
    z = g.z[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    return Grid(z, g.transform * rasterio.Affine.scale(f, f), g.crs, g.res * f)


def blocks(ds, block: int = BLOCK, halo: int | None = None):
    """(window read with halo, core window) pairs covering the tile."""
    halo = HALO if halo is None else halo
    for r0 in range(0, ds.height, block):
        for c0 in range(0, ds.width, block):
            rr0, cc0 = max(0, r0 - halo), max(0, c0 - halo)
            rr1, cc1 = min(ds.height, r0 + block + halo), min(ds.width, c0 + block + halo)
            core = rasterio.windows.Window(c0, r0, min(block, ds.width - c0), min(block, ds.height - r0))
            yield rasterio.windows.Window(cc0, rr0, cc1 - cc0, rr1 - rr0), core


def in_core(f: dict, ds, core) -> bool:
    col, row = ~ds.transform * (f["tx"], f["ty"])
    return (core.row_off <= row < core.row_off + core.height
            and core.col_off <= col < core.col_off + core.width)


def tile_flags(g: Grid, s_min: float = S_MIN) -> list[dict]:
    """Flags at every base, from a native-resolution grid."""
    return [f for base in BASES for f in flags(coarsen(g, RES[base]), base, s_min)]


@lru_cache(maxsize=256)
def template_curvature(base: float, res: float, rotation: float) -> np.ndarray:
    sigma = SIGMA_FRAC * base
    pad = int(4 * sigma / res) + 1
    t = faceted_template(n_sides=4, base=base, height=1.0, dx=res, rotation=rotation, pad=pad)
    return curvature(t, dx=res, dy=res, sigma=sigma, mask_edges=False)


def score_map(g: Grid, base: float) -> tuple[np.ndarray, np.ndarray]:
    """Max over rotations of the curvature-template correlation, and the arg-max rotation."""
    field = curvature(g.z, dx=g.res, dy=g.res, sigma=SIGMA_FRAC * base)
    best = np.full(g.z.shape, np.nan)
    rot = np.full(g.z.shape, np.nan)
    for r in ROTATIONS:
        c = template_correlation(field, template_curvature(base, g.res, r))
        better = np.isfinite(c) & ~(c <= best)  # NaN best is beaten by any finite c
        best[better], rot[better] = c[better], r
    return best, rot


@lru_cache(maxsize=16)
def _to3338(crs_wkt: str) -> Transformer:
    return Transformer.from_crs(crs_wkt, "EPSG:3338", always_xy=True)


def flags(g: Grid, base: float, s_min: float = S_MIN) -> list[dict]:
    """Local maxima of the score map at or above s_min, positioned in EPSG:3338."""
    s, rot = score_map(g, base)
    k = int(base / g.res) | 1
    filled = np.where(np.isfinite(s), s, -np.inf)
    peak = (filled == maximum_filter(filled, size=k, mode="constant", cval=-np.inf)) & (filled >= s_min)
    rows, cols = np.nonzero(peak)
    if rows.size == 0:
        return []
    xs, ys = rasterio.transform.xy(g.transform, rows, cols)
    x, y = _to3338(g.crs.to_wkt()).transform(np.asarray(xs), np.asarray(ys))
    return [dict(base=base, x=float(xi), y=float(yi), score=float(s[r, c]), rotation=float(rot[r, c]),
                 tx=float(txi), ty=float(tyi))
            for xi, yi, txi, tyi, r, c in zip(x, y, xs, ys, rows, cols)]


def _selfcheck() -> None:
    """A planted faceted landform on rough synthetic terrain is the top flag at its base."""
    rng = np.random.default_rng(0)
    n, res, base = 400, 5.0, 100.0
    yy, xx = np.mgrid[0:n, 0:n] * res
    terrain = 0.02 * xx + 15.0 * np.sin(xx / 700.0) * np.cos(yy / 900.0)  # slope and broad hills
    terrain += rng.normal(0, 0.3, (n, n))
    t = faceted_template(n_sides=4, base=base, height=8.0, dx=res, rotation=20.0)
    r0 = c0 = 200 - t.shape[0] // 2
    terrain[r0:r0 + t.shape[0], c0:c0 + t.shape[1]] += t
    g = Grid(terrain, rasterio.transform.from_origin(0, n * res, res, res), rasterio.crs.CRS.from_epsg(3338), res)
    f = sorted(flags(g, base, s_min=0.0), key=lambda d: -d["score"])
    c = coarsen(g, 10.0)
    assert c.z.shape == (n // 2, n // 2) and c.res == 10.0 and abs(c.z[0, 0] - g.z[:2, :2].mean()) < 1e-12
    top = f[0]
    assert abs(top["x"] - 200 * res) <= base / 4 and abs(top["y"] - (n - 200) * res) <= base / 4, top
    assert top["score"] > 0.6, top
    assert HALO >= influence_cells(), (HALO, influence_cells())
    print(f"p2b_detector self-check OK: top flag at the plant, score {top['score']:.2f}, rotation {top['rotation']:.0f}")


if __name__ == "__main__":
    _selfcheck()
