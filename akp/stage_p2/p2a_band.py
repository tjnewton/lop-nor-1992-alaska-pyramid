"""P2a: the searched area -- the Denali-Nome band and the Valdez box -- as geometry and area.

The band is the set of points whose geodesic distances to the Denali and Nome anchors
(akp.zones.ANCHORS) add up to at most FACTOR times the Denali-Nome distance: an ellipse on the
WGS84 ellipsoid with the anchors as foci. Its boundary is found by root-finding the focal sum
along N_RAYS geodesic rays from the midpoint (the sum grows monotonically outward from an
interior point of a convex region), then measured with pyproj's geodesic polygon area. Areas
include water (Norton Sound near Nome); the land part comes with the thematic maps.
Writes p2a_band.csv (check, value) and p2a_searched_area.geojson (both zones, for later
intersections). Run: python -m akp.stage_p2.p2a_band
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Geod
from scipy.optimize import brentq

from akp.zones import ANCHORS, BOXES

HERE = Path(__file__).parent
FACTOR = 1.05  # Focal-sum factor (author choice)
N_RAYS = 1440
GEOD = Geod(ellps="WGS84")


def focal_sum(lon, lat) -> float:
    (la1, lo1), (la2, lo2) = ANCHORS["denali"], ANCHORS["nome"]
    return GEOD.inv(lon, lat, lo1, la1)[2] + GEOD.inv(lon, lat, lo2, la2)[2]


def band_boundary(factor: float = FACTOR, n: int = N_RAYS):
    (la1, lo1), (la2, lo2) = ANCHORS["denali"], ANCHORS["nome"]
    d = GEOD.inv(lo1, la1, lo2, la2)[2]
    (lon0, lat0), = GEOD.npts(lo1, la1, lo2, la2, 1)  # geodesic midpoint
    target = factor * d
    pts = []
    for az in np.linspace(0.0, 360.0, n, endpoint=False):
        def g(s):
            lon, lat, _ = GEOD.fwd(lon0, lat0, az, s)
            return focal_sum(lon, lat) - target
        s = brentq(g, 0.0, target, xtol=1.0)  # 1 m
        lon, lat, _ = GEOD.fwd(lon0, lat0, az, s)
        pts.append((lon, lat))
    return d, pts


def area_km2(pts) -> float:
    lons, lats = zip(*pts)
    a, _ = GEOD.polygon_area_perimeter(lons, lats)
    return abs(a) / 1e6


def box_ring(b: dict):
    """Lat/lon box as a ring: the parallels are densified (not geodesics); meridians are."""
    w, e, s, n = b["minlongitude"], b["maxlongitude"], b["minlatitude"], b["maxlatitude"]
    xs = np.linspace(w, e, 200)
    return [*((x, n) for x in xs), *((x, s) for x in xs[::-1])]


def main() -> None:
    d, band = band_boundary()
    a_band = area_km2(band)
    a_coarse = area_km2(band_boundary(n=N_RAYS // 4)[1])
    valdez = box_ring(BOXES["valdez"])
    a_valdez = area_km2(valdez)
    lats = [p[1] for p in band]
    lons = [p[0] for p in band]
    # checks: boundary on the focal-sum level set; anchors inside; near the planar ellipse pi*a*b
    assert max(abs(focal_sum(*p) / (FACTOR * d) - 1) for p in band) < 1e-6
    assert all(focal_sum(lo, la) < FACTOR * d for la, lo in ANCHORS.values() if (la, lo) != ANCHORS["valdez"])
    semi_a, c = FACTOR * d / 2, d / 2
    planar = np.pi * semi_a * np.sqrt(semi_a**2 - c**2) / 1e6
    rows = [
        ("denali_nome_km", round(d / 1e3, 3)),
        ("focal_sum_factor", FACTOR),
        ("band_semi_major_km", round(FACTOR * d / 2e3, 3)),
        ("band_area_km2", round(a_band)),
        ("band_area_rel_change_quarter_rays", f"{abs(a_coarse / a_band - 1):.1e}"),
        ("band_semi_minor_planar_km", round(np.sqrt(semi_a**2 - c**2) / 1e3, 3)),
        ("band_area_over_planar_ellipse", round(a_band / planar, 4)),
        ("band_min_lat", round(min(lats), 3)), ("band_max_lat", round(max(lats), 3)),
        ("band_min_lon", round(min(lons), 3)), ("band_max_lon", round(max(lons), 3)),
        ("band_inside_probe_box", int(all(
            BOXES["mckinley_nome"]["minlatitude"] <= la <= BOXES["mckinley_nome"]["maxlatitude"]
            and BOXES["mckinley_nome"]["minlongitude"] <= lo <= BOXES["mckinley_nome"]["maxlongitude"]
            for lo, la in band))),
        ("valdez_box_area_km2", round(a_valdez)),
        ("searched_area_km2", round(a_band + a_valdez)),
    ]
    with (HERE / "p2a_band.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("check", "value"))
        w.writerows(rows)
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"zone": "denali_nome_band", "focal_sum_factor": FACTOR},
         "geometry": {"type": "Polygon", "coordinates": [[*map(list, band), list(band[0])]]}},
        {"type": "Feature", "properties": {"zone": "valdez_box"},
         "geometry": {"type": "Polygon", "coordinates": [[*map(list, valdez), list(valdez[0])]]}},
    ]}
    (HERE / "p2a_searched_area.geojson").write_text(json.dumps(gj, separators=(",", ":")))
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()
