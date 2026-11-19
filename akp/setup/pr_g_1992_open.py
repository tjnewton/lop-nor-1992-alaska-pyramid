"""S5-g: open recordings of the 1992-05-21 test (no data request needed).

For every real (non-SY) station operating at the test hour in Alaska / NW Canada, plus the GSN
networks IU and II worldwide, ask EarthScope for vertical waveforms around the predicted P
(iasp91). Origin comes from the committed ISC probe CSV. Raw miniSEED for stations that have
data is saved to $AKP_DATA/probes/test1992 (kilobytes). Writes pr_g_1992_open.csv: one row per
vertical trace (has_data True) or per station without data. `snr_p` is a peak-amplitude ratio in
0.8-3 Hz, P window [-2, 20] s over noise window [-280, -20] s: a screening number, not a measurement
for the paper. Run: python -m akp.setup.pr_g_1992_open
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy import read
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

from akp.paths import data_dir
from akp.zones import in_recording_region

HERE = Path(__file__).parent
OUT = Path(__file__).with_suffix(".csv")
REGION_QUERIES = [dict(minlatitude=51, maxlatitude=72, minlongitude=-180, maxlongitude=-120),
                  dict(minlatitude=51, maxlatitude=56, minlongitude=170, maxlongitude=180)]
GSN = ("IU", "II")


def main() -> None:
    ev = pd.read_csv(HERE / "pr_b_isc_event.csv").iloc[0]
    o, elat, elon = T(ev["origin_time"]), float(ev["latitude"]), float(ev["longitude"])
    c = Client(base_url="https://service.earthscope.org", timeout=600)
    stations = {}
    for q in REGION_QUERIES:
        for n in c.get_stations(starttime=o, endtime=o + 3600, level="station", **q):
            for s in n:
                if n.code != "SY" and in_recording_region(s.latitude, s.longitude):
                    stations[(n.code, s.code)] = (s.latitude, s.longitude, True)
    for n in c.get_stations(network=",".join(GSN), starttime=o, endtime=o + 3600, level="station"):
        for s in n:
            stations.setdefault((n.code, s.code), (s.latitude, s.longitude, False))
    taup = TauPyModel("iasp91")
    outdir = data_dir("probes/test1992")
    rows = []
    for net in sorted({k[0] for k in stations}):
        codes = sorted(k[1] for k in stations if k[0] == net)
        path = outdir / f"{net}.mseed"
        try:
            # one window that covers P at every distance (0-180 deg) is too long; use per-network
            # window spanning that network's P times
            tps = []
            for sta in codes:
                lat, lon, _ = stations[(net, sta)]
                arr = taup.get_travel_times(0.0, locations2degrees(elat, elon, lat, lon), ["P", "Pdiff", "PKP"])
                tps.append(arr[0].time if arr else np.nan)
            t0, t1 = o + np.nanmin(tps) - 300, o + np.nanmax(tps) + 600
            if not path.exists():
                c.get_waveforms(net, ",".join(codes), "*", "?HZ", t0, t1, filename=str(path))
            st = read(str(path))
        except FDSNNoDataException:
            st = []
        have = {tr.stats.station for tr in st}
        for sta in codes:
            lat, lon, regional = stations[(net, sta)]
            dist = locations2degrees(elat, elon, lat, lon)
            arr = taup.get_travel_times(0.0, dist, ["P", "Pdiff", "PKP"])
            base = dict(network=net, station=sta, lat=lat, lon=lon, dist_deg=round(dist, 2), in_region=regional,
                        phase=arr[0].name if arr else "")
            if sta not in have:
                rows.append(dict(base, channel="", fs="", has_data=False, snr_p=""))
                continue
            for tr in (t for t in st if t.stats.station == sta):
                snr = ""
                if arr and tr.stats.sampling_rate >= 10:
                    f = tr.copy().detrend("demean").filter("bandpass", freqmin=0.8, freqmax=3.0)
                    t = f.times(reftime=o + arr[0].time)
                    nz, sg = f.data[(t > -280) & (t < -20)], f.data[(t > -2) & (t < 20)]
                    if nz.size and sg.size:
                        snr = round(float(np.abs(sg).max() / np.abs(nz).max()), 1)
                rows.append(dict(base, channel=tr.stats.channel, fs=tr.stats.sampling_rate, has_data=True, snr_p=snr))
    rows.sort(key=lambda r: (not r["in_region"], r["dist_deg"], r["station"], r["channel"]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    reg = [r for r in rows if r["in_region"]]
    print(f"{len(stations)} stations; in region with data: "
          f"{sorted({(r['network'], r['station']) for r in reg if r['has_data']})}; "
          f"global with data: {len({(r['network'], r['station']) for r in rows if r['has_data'] and not r['in_region']})}")


if __name__ == "__main__":
    main()
