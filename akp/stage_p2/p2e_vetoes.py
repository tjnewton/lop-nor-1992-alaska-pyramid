"""P2e / B-gate vetoes: mapped geology and cataloged landforms that could produce a flag.

Read statewide, in EPSG:3338, so control tiles and the searched area are vetoed alike:
- igneous: SIM 3340 polygons with a major lithology (nsalith RANK "Major" or "Indeterminate,
  major") that is igneous (volcanic, plutonic or hypabyssal); radius HALF_BASE + GEO_U;
- arcs: SIM 3340 caldera or crater rims and moraine lines (ARC_CODES); radius HALF_BASE + GEO_U;
- vents: every DGGS MP 153 Quaternary vent; radius HALF_BASE + VENT_U[Loc_conf].
Radii are fixed from the sources before any P2e control run, never tuned on results.
`veto_shapes()` gives the buffered geometries; `fraction()` the vetoed share of each 1 km
strata-grid cell, rasterized at FINE m. main() writes p2e_vetoes.csv: unvetoed land per stratum
(control tiles from the P2b control-run cells, searched area from the strata grid), plus a
"_total" row whose blind fraction is the vetoed share of searched land.
Run: python -m akp.stage_p2.p2e_vetoes
"""
from __future__ import annotations

import json
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import mapping, shape

from akp.paths import data_dir
from akp.stage_p2.p2b_injection import TARGET_BASES

HERE = Path(__file__).parent
HALF_BASE = max(TARGET_BASES) / 2  # m
GEO_U = 0.5e-3 * 250_000  # m: half a millimeter at SIM 3340's stated nominal scale (metadata)
VENT_U = {"1": 500.0, "2": 2000.0, "3": 10_000.0}  # m: upper end of each Loc_conf range (MP 153 report, p. 7)
ARC_CODES = (13, 14, 19, 28, 48)  # moraine or till margin, caldera or crater rim (+ inferred), moraine limit, crest
FINE = 100.0  # m
GDB = "p2a/maps/sim3340_gdb/sim3340_gdb/AKgeol_web_gdb/geologic_data/AKStategeol.gdb"


def igneous_classes(gdb) -> set[int]:
    lith = pyogrio.read_dataframe(gdb, layer="nsalith", read_geometry=False)
    major = lith.RANK.isin(["Major", "Indeterminate, major"]) & (lith.LITH1 == "Igneous")
    return set(lith.NSACLASS[major].astype(int))


def veto_shapes() -> dict[str, np.ndarray]:
    gdb = data_dir(GDB)
    ign_classes = ",".join(str(c) for c in sorted(igneous_classes(gdb)))
    poly = pyogrio.read_dataframe(gdb, layer="AKStategeol_poly", columns=["NSACLASS"],
                                  where=f"NSACLASS IN ({ign_classes})")  # read only what is kept
    arc = pyogrio.read_dataframe(gdb, layer="AKStategeol_arc", columns=["ARC_CODE"],
                                 where=f"ARC_CODE IN ({','.join(map(str, ARC_CODES))})")
    assert poly.crs.to_epsg() == 3338 and arc.crs.to_epsg() == 3338
    ign, arcs = poly.geometry.values, arc.geometry.values
    from akp.stage_p2.p2a_strata import TO3338
    from akp.stage_p2.p2e_vents import vents

    v = pd.DataFrame(vents())
    x, y = TO3338.transform(v.LongDegree.astype(float).to_numpy(), v.LatDegree.astype(float).to_numpy())
    r = HALF_BASE + v.Loc_conf.astype(str).map(VENT_U).to_numpy(dtype=float)
    return {"igneous": shapely.buffer(np.asarray(ign), HALF_BASE + GEO_U),
            "arcs": shapely.buffer(np.asarray(arcs), HALF_BASE + GEO_U),
            "vents": shapely.buffer(shapely.points(x, y), r)}


def fraction(shapes, transform, shape_hw, strip: int = 100) -> np.ndarray:
    """Vetoed share of each cell of a grid (cell size a multiple of FINE), strip by strip."""
    k = int(round(transform.a / FINE))
    out = np.zeros(shape_hw, dtype=np.float32)
    fine = transform * transform.scale(1 / k, 1 / k)
    tree = shapely.STRtree(shapes)
    for r0 in range(0, shape_hw[0], strip):
        h = min(strip, shape_hw[0] - r0)
        t = fine * fine.translation(0, r0 * k)
        x0, y1 = t * (0, 0)
        x1, y0 = t * (shape_hw[1] * k, h * k)
        hit = tree.query(shapely.box(x0, y0, x1, y1))
        if hit.size == 0:
            continue
        m = rasterize(((g, 1) for g in shapes[hit]), out_shape=(h * k, shape_hw[1] * k), transform=t,
                      fill=0, dtype="uint8")
        out[r0:r0 + h] = m.reshape(h, k, shape_hw[1], k).mean(axis=(1, 3))
    return out


def main() -> None:
    from akp.stage_p2.p2a_strata import strata_grid, to_3338
    from akp.stage_p2.p2b_controls_run import stratum_namer

    transform, s = strata_grid()
    searched_cells, rare, name = stratum_namer(s)
    sh = veto_shapes()
    print({k: len(v) for k, v in sh.items()}, flush=True)
    f = fraction(np.concatenate(list(sh.values())), transform, s.shape)
    np.save(data_dir("p2e") / "veto_fraction.npy", f)

    zones = [shape(z["geometry"]) for z in json.loads((HERE / "p2a_searched_area.geojson").read_text())["features"]]
    insearch = ~geometry_mask([mapping(to_3338(z)) for z in zones], s.shape, transform) & (s >= 0)
    rows, cols = np.nonzero(insearch)
    srch: dict[str, list[float]] = {}
    for r, c in zip(rows, cols):
        e = srch.setdefault(name(r, c), [0.0, 0.0])
        e[0] += 1.0
        e[1] += 1.0 - float(f[r, c])
    assert abs(sum(e[0] for e in srch.values()) - searched_cells.sum()) < 0.5, "searched land differs from p2a_strata"

    from akp.stage_p2.p2b_controls_run import compact_cells

    ctrl: dict[str, list[float]] = {}
    tiles = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str})
    for t in sorted(tiles.tile_id, key=int):
        res = json.loads((data_dir("p2b/control_run/tiles") / f"{t}.json").read_text())
        for r, c, a in compact_cells([res["cells"]]):
            r, c = int(r), int(c)
            k = name(r, c)
            fr = float(f[r, c]) if 0 <= r < s.shape[0] and 0 <= c < s.shape[1] else 0.0
            e = ctrl.setdefault(k, [0.0, 0.0])
            e[0] += a
            e[1] += a * (1.0 - fr)
        del res
    keys = sorted((set(srch) | set(ctrl)) - {"offland"})
    out = pd.DataFrame([dict(stratum=k, control_km2=ctrl.get(k, [0, 0])[0], control_unvetoed_km2=ctrl.get(k, [0, 0])[1],
                             searched_km2=srch.get(k, [0, 0])[0], searched_unvetoed_km2=srch.get(k, [0, 0])[1])
                        for k in keys])
    tot = out.drop(columns="stratum").sum()
    out = pd.concat([out, pd.DataFrame([dict(stratum="_total", **tot)])], ignore_index=True)
    out.round(1).to_csv(HERE / "p2e_vetoes.csv", index=False)
    blind = 1 - tot.searched_unvetoed_km2 / tot.searched_km2
    print(f"blind fraction {blind:.3f}; control vetoed {1 - tot.control_unvetoed_km2 / tot.control_km2:.3f}; "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


def _selfcheck() -> None:
    from rasterio.transform import from_origin

    disk = shapely.buffer(shapely.points([5_000.0], [5_000.0]), 1_500.0)
    f = fraction(disk, from_origin(0.0, 10_000.0, 1_000.0, 1_000.0), (10, 10), strip=3)
    assert abs(f.sum() - np.pi * 1.5 ** 2) < 0.05 * np.pi * 1.5 ** 2, f.sum()
    print("p2e_vetoes self-check OK: disk km2", round(float(f.sum()), 3))


if __name__ == "__main__":
    import sys

    _selfcheck() if "--selfcheck" in sys.argv else main()
