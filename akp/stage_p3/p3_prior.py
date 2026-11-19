"""P3 Track A prior: the claim-weighted mixture.

Fixed before any P3 survey data is read. Built on the P2a 1 km grid (EPSG:3338, equal area, so a
cell is 1 km2), land cells only (p2a_strata.strata_grid >= 0). Each component is normalized to
unit mass over land; the mixture is the W-weighted average. Components:
- band: C1 and C2 ("between Mt McKinley and Nome"); C3 quotes it too, and C7 retells C1-C4.
  Uniform over the Denali-Nome band (p2a_searched_area.geojson).
- c3_belt: the C3 guest's "within 50 miles of Denali", a 45-55 mile belt, "a little bit
  northwest". Uniform in area over the annulus, times a von Mises weight in azimuth centered on
  north-west (C3_AZ, concentration KAPPA: a mild lean, as "a little bit" reads).
- c4_point: C4's "about 60 miles west of Mount McKinley ... on a straight-line bearing to Nome"
  and its map's "60 to 70 miles". An isotropic Gaussian at C4_MILES along the Denali-Nome
  geodesic, standard deviation C4_SD_MILES.
- valdez: C9, a lead from a recollection of TV coverage. Uniform over the Valdez box, at
  half weight (a lead, not a located source).
C10 (Carey Lake) is an unopened lead with an unverified location, so it has weight 0 until its
source is read. Track A is the union of the component supports, with c4_point cut at SUPPORT_SD.
W_EQUAL (every located statement at weight 1) is kept beside W as a robustness check for G0.
Writes $AKP_DATA/p3/prior/p3_prior.npz (per-cell mass for W and W_EQUAL, each component's own unit
mass as comp_<name>, the Track A mask, and the grid transform) and p3_prior.csv (check, component, value). The cover factor is p3_cover's.
Run: python -m akp.stage_p3.p3_prior
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape

from akp.paths import data_dir
from akp.stage_p2.p2a_strata import strata_grid, to_3338
from akp.zones import ANCHORS

HERE = Path(__file__).parent
MILE_KM = 1.609344
C3_MILES = (45.0, 55.0)  # C3: the 45-55 mile belt
C3_AZ = 315.0  # north-west, degrees from north
KAPPA = 1.0  # von Mises concentration: the north-west cell weighs e^2 (about 7.4) times the south-east one
C4_MILES = 65.0  # middle of C4's 60 and 60-70 miles
C4_SD_MILES = 5.0  # covers 55-75 miles at two standard deviations
SUPPORT_SD = 3.0
W = {"band": 1.0, "c3_belt": 1.0, "c4_point": 1.0, "valdez": 0.5}
W_EQUAL = {"band": 1.0, "c3_belt": 1.0, "c4_point": 1.0, "valdez": 1.0}
G = Geod(ellps="WGS84")
TO4326 = Transformer.from_crs("EPSG:3338", "EPSG:4326", always_xy=True)


def out_dir() -> Path:
    return data_dir("p3/prior")


def components():
    """(transform, land mask, {name: unnormalized weight per land cell}, {name: support per land cell})."""
    transform, s = strata_grid()
    land = s >= 0
    rows, cols = np.nonzero(land)
    x, y = transform * (cols + 0.5, rows + 0.5)
    lon, lat = TO4326.transform(np.asarray(x), np.asarray(y))
    n = lon.size
    dla, dlo = ANCHORS["denali"]
    nla, nlo = ANCHORS["nome"]
    az, _, d = G.inv(np.full(n, dlo), np.full(n, dla), lon, lat)
    d_mi = d / 1e3 / MILE_KM
    zones = {f["properties"]["zone"]: to_3338(shape(f["geometry"]))
             for f in json.loads((HERE.parent / "stage_p2/p2a_searched_area.geojson").read_text())["features"]}
    inside = {k: geometry_mask([mapping(g)], land.shape, transform, invert=True)[land] for k, g in zones.items()}
    belt = (d_mi >= C3_MILES[0]) & (d_mi <= C3_MILES[1])
    az_nome, _, _ = G.inv(dlo, dla, nlo, nla)
    c4lo, c4la, _ = G.fwd(dlo, dla, az_nome, C4_MILES * MILE_KM * 1e3)
    _, _, d4 = G.inv(np.full(n, c4lo), np.full(n, c4la), lon, lat)
    d4_mi = d4 / 1e3 / MILE_KM
    raw = {"band": inside["denali_nome_band"].astype(float),
           "c3_belt": belt * np.exp(KAPPA * np.cos(np.radians(az - C3_AZ))),
           "c4_point": np.exp(-0.5 * (d4_mi / C4_SD_MILES) ** 2),
           "valdez": inside["valdez_box"].astype(float)}
    support = {"band": inside["denali_nome_band"], "c3_belt": belt,
               "c4_point": d4_mi <= SUPPORT_SD * C4_SD_MILES, "valdez": inside["valdez_box"]}
    return transform, land, raw, support


def mixture(raw: dict, w: dict) -> np.ndarray:
    tot = sum(w.values())
    return sum(w[k] * raw[k] / raw[k].sum() for k in w) / tot


def main() -> None:
    transform, land, raw, support = components()
    track_a = np.logical_or.reduce(list(support.values()))
    m, m_eq = mixture(raw, W), mixture(raw, W_EQUAL)
    rows = []
    add = lambda check, comp, value: rows.append(dict(check=check, component=comp, value=value))
    for k in W:
        f = raw[k] / raw[k].sum()
        add("weight", k, W[k])
        add("weight_equal", k, W_EQUAL[k])
        add("support_land_km2", k, int(support[k].sum()))
        add("mass_in_own_support", k, round(float(f[support[k]].sum()), 4))
        add("mass_in_band", k, round(float(f[support["band"]].sum()), 4))
    add("track_a_land_km2", "_all", int(track_a.sum()))
    add("mixture_mass_in_track_a", "_all", round(float(m[track_a].sum()), 4))
    add("mixture_mass_in_band", "_all", round(float(m[support["band"]].sum()), 4))
    add("mixture_equal_mass_in_band", "_all", round(float(m_eq[support["band"]].sum()), 4))
    add("land_cells", "_all", int(land.sum()))
    out = pd.DataFrame(rows)
    assert abs(m.sum() - 1) < 1e-9 and abs(m_eq.sum() - 1) < 1e-9
    assert out.query("check == 'mass_in_band' and component == 'band'").value.iloc[0] == 1.0
    comps = {f"comp_{k}": (raw[k] / raw[k].sum()).astype(np.float32) for k in W}
    np.savez_compressed(out_dir() / "p3_prior.npz", mass=m.astype(np.float32), mass_equal=m_eq.astype(np.float32),
                        track_a=track_a, land=land, transform=np.array(transform)[:6], **comps)
    out.to_csv(HERE / "p3_prior.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
