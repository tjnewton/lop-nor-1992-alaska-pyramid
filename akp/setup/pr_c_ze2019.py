"""S5-c: ZE 2019 nodal deployment -- metadata, open access, and measured size of sample node-days.

Downloads three node-days (first / middle / last station by code, one day) as the raw server
miniSEED to $AKP_DATA/probes/ze2019, reads HEADERS ONLY (no sample values are looked at), measures bytes, and projects the full-deployment size from
station-days in the channel epochs. Not a bulk download.
Writes pr_c_ze2019.csv (one row per sampled node-day + a projection row).
Run: python -m akp.setup.pr_c_ze2019
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from obspy import UTCDateTime as T
from obspy import read
from obspy.clients.fdsn import Client

from akp.paths import data_dir
from akp.zones import BOXES, in_box

NET = "ZE"
SPAN = (T("2019-02-11"), T("2019-03-27"))  # DOI 10.7914/SN/ZE_2019 deployment window (lead; checked below)
DAY = T("2019-03-01")
OUT = Path(__file__).with_suffix(".csv")


def main() -> None:
    c = Client(base_url="https://service.earthscope.org", timeout=600)
    inv = c.get_stations(network=NET, starttime=SPAN[0], endtime=SPAN[1], level="channel")
    stas = sorted({(s.code, s.latitude, s.longitude) for n in inv for s in n})
    chans = [(s.code, ch) for n in inv for s in n for ch in s]
    station_days = 0.0
    per_sta = {}
    for code, ch in chans:
        t0 = max(ch.start_date, SPAN[0])
        t1 = min(ch.end_date or SPAN[1], SPAN[1])
        per_sta.setdefault(code, []).append(max(0.0, (t1 - t0) / 86400.0))
    sdays = {k: max(v) for k, v in per_sta.items()}  # channels of one station share an epoch
    station_days = sum(sdays.values())
    n_in_box = {z: sum(in_box(lat, lon, b) for _, lat, lon in stas) for z, b in BOXES.items()}
    rates = sorted({ch.sample_rate for _, ch in chans})
    codes = sorted({ch.code for _, ch in chans})
    first = min(ch.start_date for _, ch in chans)
    last = max(ch.end_date for _, ch in chans if ch.end_date)

    outdir = data_dir("probes/ze2019")
    pick = [stas[0], stas[len(stas) // 2], stas[-1]]
    rows = []
    for sta, slat, slon in pick:
        path = outdir / f"{NET}.{sta}.{DAY.strftime('%Y%j')}.mseed"
        if not path.exists():
            c.get_waveforms(NET, sta, "*", "*", DAY, DAY + 86400, filename=str(path))
        st = read(str(path), headonly=True)
        rows.append(dict(kind="node_day", station=sta, day=DAY.date, n_stations=1, station_days=1.0,
                         lat=slat, lon=slon,
                         bytes=path.stat().st_size,
                         n_traces=len(st), channels=";".join(sorted({tr.stats.channel for tr in st})),
                         n_samples=sum(tr.stats.npts for tr in st),
                         encoding=";".join(sorted({str(tr.stats.mseed.encoding) for tr in st}))))
    mean_b = statistics.mean(r["bytes"] for r in rows)
    rows.append(dict(kind="projection", station="", day="", n_stations=len(stas), station_days=round(station_days, 1),
                     bytes=round(mean_b * station_days), n_traces="", channels=";".join(codes),
                     n_samples="", encoding=(f"rates={rates}; epochs {first.date}..{last.date}; "
                                             f"in_box={n_in_box}; min/max node-day bytes="
                                             f"{min(r['bytes'] for r in rows)}/{max(r['bytes'] for r in rows)}")))
    for z, b in BOXES.items():
        zd = sum(sdays[code] for code, lat, lon in stas if in_box(lat, lon, b))
        ll = [(lat, lon) for _, lat, lon in stas if in_box(lat, lon, b)]
        ext = (f"node extent lat {min(a for a, _ in ll):.2f}..{max(a for a, _ in ll):.2f}, "
               f"lon {min(o for _, o in ll):.2f}..{max(o for _, o in ll):.2f}") if ll else ""
        rows.append(dict(kind=f"projection_{z}", station="", day="", n_stations=n_in_box[z], station_days=round(zd, 1),
                         bytes=round(mean_b * zd), n_traces="", channels="", n_samples="", encoding=ext,
                         min_lon=min((o for _, o in ll), default=""), max_lon=max((o for _, o in ll), default="")))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
