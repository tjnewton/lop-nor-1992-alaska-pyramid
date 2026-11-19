"""P2a: draw the control tiles.

Pool: p2a_tile_pool.csv tiles outside the exclusion (in_probe_box == 0). With SAME_PROJECTS, only
tiles of the IfSAR projects that also cover the searched area; otherwise every IfSAR project
(all are the same 5 m DTM product; each tile's project and date stay in the output).
Target: the searched area's land, split by stratum (p2a_strata.csv, unit "searched"). Strata
under MIN_SHARE of the searched land are pooled into one "rare" stratum, on both sides.
Draw: the pool in a random order fixed by SEED; a tile is accepted when (a) its bounding box
overlaps no accepted tile by more than MAX_OVERLAP of its own area (vintages of the same
ground count once) and (b) at least ACCEPT of its land falls in strata whose target is not yet
met. The draw stops when every reachable stratum's target is met and the control land is at
least the searched land, or the pool is exhausted.
Writes p2a_controls.csv (the drawn tiles, in draw order) and p2a_controls_summary.csv (check,
value: totals, bytes, per-stratum target and achieved land, L1 share distance).
Thematic maps and TNM metadata only; no raster tile is opened.
Run: python -m akp.stage_p2.p2a_controls
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box, shape

HERE = Path(__file__).parent
SEED = 20261110
MAX_OVERLAP = 0.05
ACCEPT = 0.2
MIN_SHARE = 0.005
SAME_PROJECTS = False


def main() -> None:
    tiles = pd.read_csv(HERE / "p2a_tile_pool.csv")
    tiles["key"] = tiles.tile_id.astype(str)
    strata = pd.read_csv(HERE / "p2a_strata.csv")
    wide = strata.pivot_table(index="unit", columns="stratum", values="cells", fill_value=0)
    land_cols = [c for c in wide.columns if c != "_total"]
    share = wide.loc["searched", land_cols] / wide.loc["searched", land_cols].sum()
    rare = [c for c in land_cols if share[c] < MIN_SHARE]
    wide["rare"] = wide[rare].sum(axis=1)
    wide = wide.drop(columns=rare)
    names = [c for c in wide.columns if c != "_total"]
    target_cells = wide.loc["searched", names].astype(float)
    target_land = float(target_cells.sum())

    zones = [shape(f["geometry"]) for f in json.loads((HERE / "p2a_searched_area.geojson").read_text())["features"]]
    boxes = {t.key: box(t.min_lon, t.min_lat, t.max_lon, t.max_lat) for t in tiles.itertuples()}
    searched_projects = {t.project for t in tiles.itertuples() if any(z.intersects(boxes[t.key]) for z in zones)}
    pool = tiles[(tiles.in_probe_box == 0) & (tiles.project.isin(searched_projects) | (not SAME_PROJECTS))]
    order = np.random.default_rng(SEED).permutation(len(pool))

    deficit = target_cells.copy()
    accepted, got = [], pd.Series(0.0, index=names)
    for i in order:
        t = pool.iloc[i]
        c = wide.loc[t.key, names].astype(float) if t.key in wide.index else pd.Series(0.0, index=names)
        land = float(c.sum())
        if land == 0:
            continue
        b = boxes[t.key]
        if any(b.intersection(boxes[a]).area > MAX_OVERLAP * b.area for a in accepted):
            continue
        if float(np.minimum(c, deficit.clip(lower=0)).sum()) < ACCEPT * land:
            continue
        accepted.append(t.key)
        got += c
        deficit -= c
        if (deficit <= 0).all() and got.sum() >= target_land:
            break

    drawn = pool.set_index("key").loc[accepted].reset_index(drop=True)
    drawn.insert(0, "draw", range(1, len(drawn) + 1))
    drawn.to_csv(HERE / "p2a_controls.csv", index=False)
    share_t, share_g = target_cells / target_land, got / max(got.sum(), 1.0)
    rows = [("seed", SEED), ("accept", ACCEPT), ("min_share", MIN_SHARE),
            ("same_projects", int(SAME_PROJECTS)), ("rare_strata_pooled", len(rare)),
            ("pool_tiles", len(pool)), ("pool_projects", int(pool.project.nunique())),
            ("drawn_tiles", len(drawn)), ("drawn_bytes_gb", round(drawn.bytes.sum() / 1e9, 2)),
            ("target_land_km2", round(target_land)), ("control_land_km2", round(float(got.sum()))),
            ("control_over_target_land", round(float(got.sum()) / target_land, 3)),
            ("strata_target", int((target_cells > 0).sum())),
            ("strata_short", int((deficit > 0).sum())),
            ("l1_share_distance", round(float((share_t - share_g).abs().sum()), 3))]
    rows += [(f"target_km2|{s}", int(target_cells[s])) for s in names if target_cells[s] > 0]
    rows += [(f"control_km2|{s}", int(got[s])) for s in names if target_cells[s] > 0 or got[s] > 0]
    pd.DataFrame(rows, columns=["check", "value"]).to_csv(HERE / "p2a_controls_summary.csv", index=False)
    for r in rows[:15]:
        print(*r)


if __name__ == "__main__":
    main()
