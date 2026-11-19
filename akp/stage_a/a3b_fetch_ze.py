"""A3b data: one teleseism window of the ZE 2019 nodes inside the probe box.

Event: the M>=6.5, 30-95 deg teleseism during the deployment (pr_h_events.csv, kind=ze2019_teleseism).
Window: predicted P (iasp91) -120 s to +300 s, vertical channel DPZ only, every ZE 2019 node
inside the mckinley_nome box. Raw miniSEED per node under $AKP_DATA/a3b/ze2019_event (cached);
StationXML for all nodes in the same folder. Writes a3b_fetch_ze.csv (node, lat, lon, bytes,
status). Run: python -m akp.stage_a.a3b_fetch_ze
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime as T
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

from akp.paths import data_dir
from akp.zones import BOXES, in_box

HERE = Path(__file__).parent
PRE, POST = 120.0, 300.0


def main() -> None:
    ev = pd.read_csv(HERE.parent / "setup" / "pr_h_events.csv")
    e = ev[ev.kind == "ze2019_teleseism"].iloc[0]
    o = T(e.time)
    c = Client(base_url="https://service.earthscope.org", timeout=600)
    out = data_dir("a3b/ze2019_event")
    inv_path = out / "ZE2019_box_nodes.xml"
    if inv_path.exists():
        from obspy import read_inventory
        inv = read_inventory(str(inv_path))
    else:
        inv = c.get_stations(network="ZE", starttime=o, endtime=o + 3600, channel="DPZ", level="response",
                             **BOXES["mckinley_nome"])
        inv.write(str(inv_path), format="STATIONXML")
    taup = TauPyModel("iasp91")
    rows = []
    for net in inv:
        for sta in net:
            if not in_box(sta.latitude, sta.longitude, BOXES["mckinley_nome"]):
                continue
            d = locations2degrees(e.lat, e.lon, sta.latitude, sta.longitude)
            tp = o + taup.get_travel_times(0.0, d, ["P"])[0].time
            path = out / f"ZE.{sta.code}.DPZ.mseed"
            status = "cached"
            if not path.exists():
                try:
                    c.get_waveforms("ZE", sta.code, "*", "DPZ", tp - PRE, tp + POST, filename=str(path))
                    status = "ok"
                except FDSNNoDataException:
                    status = "no data"
            rows.append(dict(station=sta.code, lat=sta.latitude, lon=sta.longitude, dist_deg=round(d, 3),
                             p_time=str(tp), bytes=path.stat().st_size if path.exists() else 0, status=status))
    with (HERE / "a3b_fetch_ze.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["bytes"] > 0 for r in rows)
    print(f"{ok}/{len(rows)} nodes with data, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
