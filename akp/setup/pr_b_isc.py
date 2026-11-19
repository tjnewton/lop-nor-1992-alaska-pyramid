"""S5-b: ISC bulletin for the 1992-05-21 Lop Nor test, and which stations in Alaska / NW Canada reported it.

Station positions are first estimated from each arrival's geocentric (distance, azimuth) (good to
tens of km: enough to classify region, not to prove co-location), then replaced by EarthScope station-service
coordinates at the 1992 epoch where the code resolves to a real (non-SY) station there;
column earthscope_1992 records 'SY only' / 'none' otherwise.
Writes pr_b_isc_event.csv (origin + magnitudes) and pr_b_isc_arrivals.csv (in-region arrivals, exact
duplicates dropped; `onset` False marks amplitude readings such as AMS/pmax).
Run: python -m akp.setup.pr_b_isc
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from obspy import UTCDateTime as T
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from akp.zones import BOXES, in_box, in_recording_region

WINDOW = (T("1992-05-21T04:55:00"), T("1992-05-21T05:05:00"))
HERE = Path(__file__).parent
AMPLITUDE_PHASES = {"AMS", "pmax", "AML", "IAmb", "IAMs_20", "IVmB_BB", "IVMs_BB"}  # amplitude readings, not onsets


F_WGS84 = 1 / 298.257223563


def forward(lat, lon, dist_deg, az_deg):
    """Destination point (geographic degrees) for an ISC geocentric distance and azimuth.

    Bulletin distances are geocentric, so the great circle is walked on geocentric latitudes
    and the result converted back; skipping the conversion biases Alaska positions ~0.2 deg.
    """
    k = (1 - F_WGS84) ** 2
    p1 = math.atan(k * math.tan(math.radians(lat)))
    l1, d, a = map(math.radians, (lon, dist_deg, az_deg))
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(a))
    l2 = l1 + math.atan2(math.sin(a) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(math.atan(math.tan(p2) / k)), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def main() -> None:
    cat = Client("ISC", timeout=300).get_events(starttime=WINDOW[0], endtime=WINDOW[1],
                                                minmagnitude=5, includearrivals=True)
    assert len(cat) == 1, f"expected one event, got {len(cat)}"
    ev = cat[0]
    o = ev.preferred_origin() or ev.origins[0]
    with (HERE / "pr_b_isc_event.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["origin_time", "latitude", "longitude", "depth_m", "mag_type", "mag", "n_arrivals_total"])
        for m in ev.magnitudes:
            w.writerow([o.time, o.latitude, o.longitude, o.depth, m.magnitude_type, m.mag, len(o.arrivals)])
    picks = {p.resource_id: p for p in ev.picks}
    es = Client(base_url="https://service.earthscope.org", timeout=300)
    rows = []
    for a in o.arrivals:
        if a.distance is None or a.azimuth is None:
            continue
        lat, lon = forward(o.latitude, o.longitude, a.distance, a.azimuth)
        if not in_recording_region(lat, lon):
            continue
        p = picks[a.pick_id]
        rows.append(dict(station=p.waveform_id.station_code, isc_network=p.waveform_id.network_code,
                         phase=a.phase, pick_time=str(p.time), distance_deg=a.distance, azimuth_deg=a.azimuth,
                         time_residual_s=a.time_residual, est_lat=round(lat, 3), est_lon=round(lon, 3)))
    coords = {}
    for sta in sorted({r["station"] for r in rows}):
        try:
            inv = es.get_stations(station=sta, starttime=WINDOW[0], endtime=WINDOW[1], level="station")
            hits = [(n.code, s.latitude, s.longitude) for n in inv for s in n]
        except FDSNNoDataException:
            hits = []
        coords[sta] = hits
    for r in rows:
        hits = coords[r["station"]]
        # keep only resolutions consistent with the arrival geometry (same code can exist elsewhere)
        hits = [h for h in hits if abs(h[1] - r["est_lat"]) < 2 and abs(((h[2] - r["est_lon"]) + 180) % 360 - 180) < 4]
        real = [h for h in hits if h[0] != "SY"]  # SY = synthetic network, not a real station epoch
        r["earthscope_1992"] = ";".join(sorted({h[0] for h in real})) or ("SY only" if hits else "none")
        r["fdsn_lat"] = real[0][1] if real else ""
        r["fdsn_lon"] = real[0][2] if real else ""
        r["in_mckinley_nome_box"] = in_box(r["est_lat"], r["est_lon"], BOXES["mckinley_nome"])
    for r in rows:
        r["onset"] = r["phase"] not in AMPLITUDE_PHASES
    rows = [dict(t) for t in dict.fromkeys(tuple(r.items()) for r in rows)]  # ISC repeats some picks verbatim
    rows.sort(key=lambda r: (r["distance_deg"], r["station"], r["phase"], r["pick_time"]))
    with (HERE / "pr_b_isc_arrivals.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"origin {o.time} {o.latitude} {o.longitude}; {len(o.arrivals)} arrivals total, "
          f"{len(rows)} in region ({sum(r['onset'] for r in rows)} onsets) from {len(coords)} station codes")


if __name__ == "__main__":
    assert abs(forward(0, 0, 90, 90)[1] - 90) < 1e-9 and abs(forward(0, 0, 0, 0)[0]) < 1e-9
    assert abs(forward(45.0, 0, 0, 0)[0] - 45.0) < 1e-9  # zero distance returns the start point
    main()
