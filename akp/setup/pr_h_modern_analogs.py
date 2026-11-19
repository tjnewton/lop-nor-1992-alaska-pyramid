"""S5-h: modern analogs for Track A (no data request needed).

Writes four CSVs:
- pr_h_events.csv     analog sources (event parameters fetched from the USGS catalog, located by the
                      date/position anchors below) and M>=6.5 teleseisms (30-95 deg) during the ZE 2019
                      node deployment (the A3 dense-array upper-bound test bed).
- pr_h_coverage.csv   real stations operating in each probe box at each analog origin; network "*"
                      = unique codes; sites_1992 = which in-box 1992 ISC-reporting codes were operating.
- pr_h_same_site.csv  for each 1992 ISC-reporting code (pr_b_isc_arrivals.csv): every real EarthScope
                      epoch with that code, and its distance from the ISC-derived 1992 position.
- pr_h_temp_networks.csv  every network ever operating inside each probe box: start year, size, extent.
Run: python -m akp.setup.pr_h_modern_analogs
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime as T
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth, locations2degrees

from akp.zones import BOXES

HERE = Path(__file__).parent
# Anchors are search inputs (date + approximate epicenter); all parameters come from the catalog.
ANCHORS = {
    "dprk_2017_explosion": ("2017-09-03", 41.3, 129.1, 5.0),
    "xinjiang_2016_eq": ("2016-12-08", 43.8, 86.4, 5.5),
}
ZE_SPAN = (T("2019-02-11"), T("2019-03-27"))
ZE_CENTER = (63.5, -149.5)  # approximate centroid of the in-box ZE 2019 nodes (pr_c_ze2019.csv extent)
AK_QUERY = dict(minlatitude=51, maxlatitude=72, minlongitude=-180, maxlongitude=-120)


def write(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    usgs = Client("USGS", timeout=300)
    es = Client(base_url="https://service.earthscope.org", timeout=600)
    events = []
    for name, (day, lat, lon, mmin) in ANCHORS.items():
        cat = usgs.get_events(starttime=T(day), endtime=T(day) + 86400, latitude=lat, longitude=lon,
                              maxradius=1.0, minmagnitude=mmin)
        assert len(cat) == 1, f"{name}: {len(cat)} events"
        e = cat[0]; o = e.preferred_origin() or e.origins[0]; m = e.preferred_magnitude() or e.magnitudes[0]
        events.append(dict(kind="analog", name=name, time=str(o.time), lat=o.latitude, lon=o.longitude,
                           mag_type=m.magnitude_type, mag=m.mag,
                           dist_to_box_deg=round(locations2degrees(o.latitude, o.longitude, *ZE_CENTER), 1)))
    cat = usgs.get_events(starttime=ZE_SPAN[0], endtime=ZE_SPAN[1], minmagnitude=6.5)
    for e in sorted(cat, key=lambda e: e.origins[0].time):
        o = e.preferred_origin() or e.origins[0]; m = e.preferred_magnitude() or e.magnitudes[0]
        d = locations2degrees(o.latitude, o.longitude, *ZE_CENTER)
        if 30.0 <= d <= 95.0:
            events.append(dict(kind="ze2019_teleseism", name=str(o.time)[:16], time=str(o.time), lat=o.latitude,
                               lon=o.longitude, mag_type=m.magnitude_type, mag=m.mag, dist_to_box_deg=round(d, 1)))
    write("pr_h_events.csv", events)

    isc = pd.read_csv(HERE / "pr_b_isc_arrivals.csv")
    sites_1992 = set(isc.loc[isc["in_mckinley_nome_box"], "station"])  # 1992 ISC-reporting codes in the box
    cov = []
    for ev in (e for e in events if e["kind"] == "analog"):
        t = T(ev["time"])
        for zone, box in BOXES.items():
            nets = {}
            try:
                for n in es.get_stations(starttime=t, endtime=t + 3600, level="station", **box):
                    if n.code != "SY":
                        nets.setdefault(n.code, set()).update(s.code for s in n)
            except FDSNNoDataException:
                pass
            for code in sorted(nets):
                cov.append(dict(event=ev["name"], zone=zone, network=code, n_stations=len(nets[code])))
            allc = set().union(*nets.values()) if nets else set()
            cov.append(dict(event=ev["name"], zone=zone, network="*", n_stations=len(allc),
                            sites_1992=";".join(sorted(allc & sites_1992))))
    write("pr_h_coverage.csv", cov)

    arr = pd.read_csv(HERE / "pr_b_isc_arrivals.csv").drop_duplicates("station")
    same = []
    for _, a in arr.sort_values("station").iterrows():
        try:
            inv = es.get_stations(station=a["station"], level="station", **AK_QUERY)
            hits = [(n.code, s) for n in inv for s in n if n.code != "SY"]
        except FDSNNoDataException:
            hits = []
        if not hits:
            same.append(dict(station=a["station"], network="", start="", end="", sep_km="", in_1992_box=a["in_mckinley_nome_box"]))
        for net, s in hits:
            sep = gps2dist_azimuth(a["est_lat"], a["est_lon"], s.latitude, s.longitude)[0] / 1000.0
            same.append(dict(station=a["station"], network=net, start=str(s.start_date)[:10],
                             end=str(s.end_date)[:10] if s.end_date else "open", sep_km=round(sep, 1),
                             in_1992_box=a["in_mckinley_nome_box"]))
    write("pr_h_same_site.csv", same)

    temp = []
    for zone, box in BOXES.items():
        for n in es.get_stations(level="station", **box):
            if n.code == "SY" or not len(n):
                continue
            lats = [s.latitude for s in n]; lons = [s.longitude for s in n]
            temp.append(dict(zone=zone, network=n.code, start_year=n.start_date.year if n.start_date else "",
                             n_stations=len(n), min_lon=round(min(lons), 2), max_lon=round(max(lons), 2),
                             min_lat=round(min(lats), 2), max_lat=round(max(lats), 2)))
    write("pr_h_temp_networks.csv", sorted(temp, key=lambda r: (r["zone"], -r["n_stations"])))
    for ev in events:
        print(ev)
    print([r for r in cov if r["network"] == "*"])


if __name__ == "__main__":
    main()
