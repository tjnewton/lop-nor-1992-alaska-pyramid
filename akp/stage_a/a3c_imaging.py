"""A3c: where could the 1992 recordings have *imaged* the body?

Imaging (minimal, generous): >= MIN_RECORDERS recorder sites within the corrected detection radius r*
(A3b-1) of the body, with an azimuthal span >= MIN_SPAN_DEG about the body. Evaluated on the
mckinley_nome box grid (A1) and at Valdez, for each host velocity's largest r*, for two station sets
(reporters; reporters + codes operating at the test) and two angle rules (generous all-angle r*;
geometry-correct theta 75-90 deg r*). Also reports the spatial sampling of the recorder sites
against lambda/2. Writes a3c_checks.csv.
Run: python -m akp.stage_a.a3c_imaging
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from akp.stage_a import a3_detectability as a3
from akp.zones import ANCHORS

HERE = Path(__file__).parent
GEOD = Geod(ellps="WGS84")
MIN_RECORDERS = 3
MIN_SPAN_DEG = 90.0


def azimuth_span(az: np.ndarray) -> float:
    """Smallest arc (degrees) containing all azimuths."""
    if len(az) < 2:
        return 0.0
    a = np.sort(np.mod(az, 360.0))
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(360.0 - gaps.max())


def imaging_fraction(D, A, r):
    """Fraction of points (all but the last, Valdez) and Valdez flag meeting the A3c criterion at radius r."""
    ok = []
    for i in range(D.shape[0]):
        near = D[i] <= r
        ok.append(near.sum() >= MIN_RECORDERS and azimuth_span(A[i][near]) >= MIN_SPAN_DEG)
    ok = np.array(ok)
    return float(ok[:-1].mean()), int(ok[-1])


def site_matrices() -> dict:
    """Per station set: (D km, A azimuth deg, n sites) from each box point (and Valdez, last) to each site."""
    st = pd.read_csv(HERE / "a1_stations.csv")
    box = pd.read_csv(HERE / "a1_box_coverage.csv")
    pts = list(zip(box.lat, box.lon)) + [ANCHORS["valdez"]]
    sets = {"reporters": st[st.recorded_test], "operating": st[st.recorded_test | st.operating_at_test]}
    out = {}
    for sname, sub in sets.items():
        rec = sub.drop_duplicates("site_id")  # one per site
        D = np.empty((len(pts), len(rec)))
        A = np.empty_like(D)
        for i, (la, lo) in enumerate(pts):
            az, _, d = GEOD.inv(np.full(len(rec), lo), np.full(len(rec), la), rec.lon.to_numpy(), rec.lat.to_numpy())
            D[i], A[i] = d / 1000.0, az
        out[sname] = (D, A, len(rec))
    return out


def main() -> None:
    cases = pd.read_csv(HERE / "a3b_cases_corrected.csv")
    rows = []
    for sname, (D, A, n_sites) in site_matrices().items():
        # spatial sampling: the median distance from a box point to its nearest recorder site, against half
        # the shortest wavelength (most favorable band) -- coherent imaging needs sampling below lambda/2
        rows.append(dict(check=f"median_nearest_site_km_{sname}", value=round(float(np.median(D[:-1].min(axis=1))), 2), detail=""))
        for vp in sorted(cases.host_vp.unique()):
            c = cases[cases.host_vp == vp]
            for rname, col in (("generous", "r_star_qs_km"), ("geo", "r_star_geo_qs_km")):
                frac, valdez = imaging_fraction(D, A, float(c[col].max()))
                tag = f"{sname}_{rname}_vp_{int(vp)}"
                rows.append(dict(check=f"imaging_box_frac_{tag}", value=round(frac, 4), detail=f"r* {c[col].max():.3f} km"))
                rows.append(dict(check=f"imaging_valdez_{tag}", value=valdez, detail=""))
        rows.append(dict(check=f"recorder_sites_{sname}", value=n_sites, detail=""))
    rows.append(dict(check="half_wavelength_km_4hz_vp_1200", value=round(1200.0 / 4.0 / 2.0 / 1000.0, 3),
                     detail="lambda/2 at the most favorable band edge and the A3-0 floor"))
    # [Added 2026-11-09: the shortest wavelength is S, not P, so the binding requirement
    # is lambda_S / 2; the P row above is kept for continuity.]
    rows.append(dict(check="half_s_wavelength_km_4hz_vp_1200",
                     value=round(1200.0 / a3.VP_VS / 4.0 / 2.0 / 1000.0, 4),
                     detail="lambda_S/2 at the most favorable band edge and the A3-0 floor"))
    # kept for continuity with the first A3c run (reporters, generous r*)
    for vp in sorted(cases.host_vp.unique()):
        rows.append(dict(check=f"imaging_box_frac_vp_{int(vp)}",
                         value=next(r["value"] for r in rows if r["check"] == f"imaging_box_frac_reporters_generous_vp_{int(vp)}"),
                         detail="= reporters, generous r*"))
        rows.append(dict(check=f"imaging_valdez_vp_{int(vp)}",
                         value=next(r["value"] for r in rows if r["check"] == f"imaging_valdez_reporters_generous_vp_{int(vp)}"),
                         detail=""))
    rows.append(dict(check="recorder_sites", value=next(r["value"] for r in rows if r["check"] == "recorder_sites_reporters"), detail=""))
    with (HERE / "a3c_checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "value", "detail"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if any(k in r["check"] for k in ("_vp_850", "_vp_1200", "_vp_1500", "median", "recorder_sites_", "half")):
            print(r)


if __name__ == "__main__":
    assert azimuth_span(np.array([350.0, 10.0])) == 20.0 and azimuth_span(np.array([0.0, 120.0, 240.0])) == 240.0
    main()
