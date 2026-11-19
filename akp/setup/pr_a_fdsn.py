"""S5-a: EarthScope FDSN access + station counts in each probe box, 1992 test epoch vs recent.

Re-derives the brainstorm's station-count leads from committed code.
Writes akp/setup/pr_a_fdsn.csv: one row per (zone, epoch, network), plus a network="*" row with the
count of UNIQUE real station codes (a code can sit in several networks). SY (synthetic) is excluded
from "*" and its count recorded on its own row. Run: python -m akp.setup.pr_a_fdsn
"""
from __future__ import annotations

import csv
from pathlib import Path

import obspy
from obspy import UTCDateTime as T
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from akp.zones import BOXES

BASE_URL = "https://service.earthscope.org"
EXCLUDE = {"SY"}  # synthetic-seismogram network, not real stations
# Fixed epochs (not wall clock) so re-runs query the same windows.
EPOCHS = {"test_1992": T("1992-05-21T05:00:00"), "recent_2026": T("2026-11-01T00:00:00")}
OUT = Path(__file__).with_suffix(".csv")


def main() -> None:
    client = Client(base_url=BASE_URL, timeout=300)
    rows = []
    for zone, box in BOXES.items():
        for ename, t in EPOCHS.items():
            try:
                inv = client.get_stations(starttime=t, endtime=t + 3600, level="station", **box)
            except FDSNNoDataException:
                inv = None
            nets = {}
            for net in inv or []:
                nets.setdefault(net.code, set()).update(s.code for s in net)
            for code in sorted(nets):
                rows.append(dict(zone=zone, epoch=ename, epoch_utc=str(t), network=code,
                                 excluded=code in EXCLUDE, n_stations=len(nets[code]),
                                 stations=";".join(sorted(nets[code]))))
            real = set().union(*[v for k, v in nets.items() if k not in EXCLUDE]) if nets else set()
            rows.append(dict(zone=zone, epoch=ename, epoch_utc=str(t), network="*", excluded=False,
                             n_stations=len(real), stations=";".join(sorted(real))))  # unique codes, SY excluded
            if not nets:
                rows.append(dict(zone=zone, epoch=ename, epoch_utc=str(t), network="",
                                 excluded=False, n_stations=0, stations=""))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"obspy {obspy.__version__} base_url={BASE_URL} -> {OUT.name} ({len(rows)} rows)")
    for r in rows:
        if r["network"] == "*":
            print(r["zone"], r["epoch"], r["network"], r["n_stations"])


if __name__ == "__main__":
    main()
