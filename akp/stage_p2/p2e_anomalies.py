"""P2e: P2b flags to anomalies, each with its model-selection delta_bic.

Flags are merged greedily by score: a flag starts an anomaly unless its center lies within half
the larger base of an anomaly already kept, so each anomaly keeps its
highest-scoring flag. The anomaly's window (half-width WINDOW_BASES bases, at the flag base's
working resolution, on the detector's coarse grid) is read from its tile and scored by
p2e_model_select.delta_bic. A window that leaves the tile or holds nodata gets NaN.
Run the self-check: python -m akp.stage_p2.p2e_anomalies
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import Transformer
from rasterio.windows import Window

from akp.stage_p2 import p2b_detector as det
from akp.stage_p2 import p2e_model_select as ms


def merge(flags: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for f in sorted(flags, key=lambda f: -f["score"]):
        if all(np.hypot(f["x"] - k["x"], f["y"] - k["y"]) > max(f["base"], k["base"]) / 2 for k in kept):
            kept.append(f)
    return kept


@lru_cache(maxsize=8)
def _from3338(crs_wkt: str) -> Transformer:
    return Transformer.from_crs("EPSG:3338", crs_wkt, always_xy=True)


def window(ds, f: dict) -> np.ndarray | None:
    """The m x m model-selection window centered on the flag's coarse cell, or None."""
    res = det.RES[f["base"]]
    k = max(1, int(round(res / abs(ds.res[0]))))
    m = 2 * int(round(ms.WINDOW_BASES * f["base"] / res)) + 1
    tx, ty = _from3338(ds.crs.to_wkt()).transform(f["x"], f["y"])
    col, row = ~ds.transform * (tx, ty)  # the flag sits at a coarse cell center
    c0 = int(round(col - k / 2)) - (m // 2) * k
    r0 = int(round(row - k / 2)) - (m // 2) * k
    if c0 < 0 or r0 < 0 or c0 + m * k > ds.width or r0 + m * k > ds.height:
        return None
    z = det.coarsen(det.read_native(ds, Window(c0, r0, m * k, m * k)), res).z
    return z if z.shape == (m, m) and np.isfinite(z).all() else None


def anomalies(ds, flags: list[dict]) -> list[dict]:
    out = []
    for f in merge(flags):
        w = window(ds, f)
        d = ms.delta_bic(w, f["base"], det.RES[f["base"]]) if w is not None else float("nan")
        out.append({**f, "delta_bic": d})
    return out


def _selfcheck() -> None:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from swe.topo import faceted_template

    fl = [dict(base=100.0, x=0.0, y=0.0, score=0.7), dict(base=50.0, x=20.0, y=0.0, score=0.9),
          dict(base=50.0, x=60.0, y=0.0, score=0.6)]
    assert [f["score"] for f in merge(fl)] == [0.9, 0.6], merge(fl)  # 0.7 lies within 25 m of 0.9

    rng = np.random.default_rng(2)
    n, dx = 400, 2.5  # native grid finer than the 5 m working resolution, so coarsening is exercised
    z = rng.normal(0, 0.2, (n, n))
    t = faceted_template(n_sides=4, base=100.0, height=6.0, dx=dx, rotation=20.0)
    r = c = 200 - t.shape[0] // 2
    z[r:r + t.shape[0], c:c + t.shape[1]] += t
    prof = dict(driver="GTiff", width=n, height=n, count=1, dtype="float64", crs="EPSG:3338",
                transform=from_origin(1000.0, 2000.0, dx, dx))
    with MemoryFile() as mem, mem.open(**prof) as ds:
        ds.write(z, 1)
        x, y = rasterio.transform.xy(ds.transform, 200, 200, offset="ul")  # a 5 m cell center: 2x2 native
        a = anomalies(ds, [dict(base=100.0, x=x, y=y, score=0.9)])
        edge = window(ds, dict(base=100.0, x=1000.0 + 5.0, y=2000.0 - 5.0, score=0.9))
    assert a[0]["delta_bic"] > 10 and edge is None, (a, edge)
    print("p2e_anomalies self-check OK: delta_bic", round(a[0]["delta_bic"], 1))


if __name__ == "__main__":
    _selfcheck()
