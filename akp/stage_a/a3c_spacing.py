"""A3c: station spacing (not distance to the box) and distances from claim points.

The imaging argument needs the spacing between stations, not the distance from a box point to its
nearest station. For the sites that recorded the test, and for those plus the sites operating at
it (a1_stations.csv, one position per site_id), it computes each site's geodesic distance to its
nearest other site: minimum and median. It compares them with the largest steelman base
(MAX_BASE_KM, an image must resolve the body, so the wavelength and the sampling must be at most
about its size) and with the shortest half-wavelength A3c used (a3c_checks.csv).
Claim points (akp.zones.ANCHORS): the C4 point, C4_MILES from Denali along the geodesic toward
Nome, and Denali itself (the C3 belt center). For each, the nearest recording and operating site.
C3 area (C3: "within 50 miles of Denali", a 45-55 mile belt, "a little bit northwest"):
the operating sites inside the C3_DISC_MILES disc; and over the belt (points every BELT_STEP_KM in
distance and 1 degree in azimuth), the share within the Rg detection radii (a3b4_apply.csv: cited
attenuation, Q = 100; and none) of a recording site and of any operating site, and the nearest operating site to the belt
and to its north-west quadrant (azimuth 270-360).
Writes a3c_spacing.csv (check, value, note). Run: python -m akp.stage_a.a3c_spacing
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from akp.zones import ANCHORS

HERE = Path(__file__).parent
MAX_BASE_KM = 0.5  # largest A3 steelman base (A3 Method)
C4_MILES = 60.0  # C4: "about 60 miles west of Mount McKinley... on a straight-line bearing to Nome"
MILE_KM = 1.609344
C3_DISC_MILES = 50.0
C3_BELT_MILES = (45.0, 55.0)
BELT_STEP_KM = 1.0
G = Geod(ellps="WGS84")


def sites(evidence: tuple[str, ...]) -> pd.DataFrame:
    st = pd.read_csv(HERE / "a1_stations.csv")
    st = st[st.evidence.isin(evidence)].dropna(subset=["lat", "lon"])
    return st.groupby("site_id", as_index=False).agg(lat=("lat", "first"), lon=("lon", "first"),
                                                     codes=("code", lambda c: "/".join(sorted(map(str, c)))))


def nearest_neighbor_km(s: pd.DataFrame) -> np.ndarray:
    lat, lon = s.lat.to_numpy(), s.lon.to_numpy()
    out = np.empty(len(s))
    for i in range(len(s)):
        _, _, d = G.inv(np.full(len(s), lon[i]), np.full(len(s), lat[i]), lon, lat)
        d[i] = np.inf
        out[i] = d.min() / 1e3
    return out


def nearest_site(lat: float, lon: float, s: pd.DataFrame) -> tuple[float, str]:
    _, _, d = G.inv(np.full(len(s), lon), np.full(len(s), lat), s.lon.to_numpy(), s.lat.to_numpy())
    i = int(np.argmin(d))
    return d[i] / 1e3, s.codes.iloc[i]


def main() -> None:
    half_s = float(pd.read_csv(HERE / "a3c_checks.csv").set_index("check").value["half_s_wavelength_km_4hz_vp_1200"])
    rows = []
    sets = {"recorders": ("recorded_test",), "recorders_and_operating": ("recorded_test", "operating_at_test")}
    for name, ev in sets.items():
        s = sites(ev)
        nn = nearest_neighbor_km(s)
        rows += [dict(check=f"n_sites_{name}", value=len(s), note=""),
                 dict(check=f"nn_spacing_min_km_{name}", value=round(float(nn.min()), 2), note=""),
                 dict(check=f"nn_spacing_median_km_{name}", value=round(float(np.median(nn)), 2), note=""),
                 dict(check=f"nn_min_over_max_base_{name}", value=round(float(nn.min()) / MAX_BASE_KM, 1), note=""),
                 dict(check=f"nn_median_over_max_base_{name}", value=round(float(np.median(nn)) / MAX_BASE_KM, 1), note=""),
                 dict(check=f"nn_min_over_half_s_{name}", value=round(float(nn.min()) / half_s, 1), note="")]
    dlat, dlon = ANCHORS["denali"]
    nlat, nlon = ANCHORS["nome"]
    az, _, _ = G.inv(dlon, dlat, nlon, nlat)
    c4lon, c4lat, _ = G.fwd(dlon, dlat, az, C4_MILES * MILE_KM * 1e3)
    for point, (la, lo) in (("c4_point", (c4lat, c4lon)), ("denali", (dlat, dlon))):
        for name, ev in sets.items():
            km, codes = nearest_site(la, lo, sites(ev))
            rows.append(dict(check=f"km_{point}_to_nearest_{name}", value=round(km, 1), note=codes))
    op = sites(sets["recorders_and_operating"])
    _, _, dd = G.inv(np.full(len(op), dlon), np.full(len(op), dlat), op.lon.to_numpy(), op.lat.to_numpy())
    inside = op[dd / 1e3 <= C3_DISC_MILES * MILE_KM]
    rows.append(dict(check="c3_disc_operating_sites", value=len(inside), note="/".join(inside.codes)))
    apply = pd.read_csv(HERE / "a3b4_apply.csv").set_index("check").value
    radii = {"rg_cited_q100": float(apply["rq_max_r_km_generous_100_vp_1200"]), "rg_unattenuated": float(apply["max_r_corr_km_vp_1200"])}
    r_km = np.arange(C3_BELT_MILES[0] * MILE_KM, C3_BELT_MILES[1] * MILE_KM + 1e-9, BELT_STEP_KM)
    azs = np.arange(0.0, 360.0, 1.0)
    rr, aa = np.meshgrid(r_km, azs)
    blon, blat, _ = G.fwd(np.full(rr.size, dlon), np.full(rr.size, dlat), aa.ravel(), rr.ravel() * 1e3)
    w = rr.ravel()  # area weight of a polar sample
    for name, ev in sets.items():
        ss = sites(ev)
        near = np.full(rr.size, np.inf)
        for la, lo in zip(ss.lat, ss.lon):
            _, _, d = G.inv(blon, blat, np.full(rr.size, lo), np.full(rr.size, la))
            near = np.minimum(near, d / 1e3)
        for k, r in radii.items():
            rows.append(dict(check=f"c3_belt_frac_within_{k}_{name}", value=round(float(w[near <= r].sum() / w.sum()), 4),
                             note=f"radius {r} km"))
    nw = (aa.ravel() >= 270.0)
    rows += [dict(check="c3_belt_min_km_to_operating", value=round(float(near.min()), 1), note=""),
             dict(check="c3_belt_nw_min_km_to_operating", value=round(float(near[nw].min()), 1), note="")]
    out = pd.DataFrame(rows)
    assert (out.value >= 0).all()
    out.to_csv(HERE / "a3c_spacing.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
