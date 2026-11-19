"""A2b: explosion-only clutter reference at IU.COLA (defined before any numbers).

Every DPRK explosion with mb >= 4.5 recorded at IU.COLA, measured exactly as in A2
(a2_wavefield.envelopes and the same bands, lag windows and P-peak rule). Valid cells need the
reference clutter envelope above SNR_FACTOR x its own pre-event noise envelope. F1b compares the
1992 COL ratio with [min / BRACKET, max * BRACKET] of the valid references. Also reports, per cell,
COLA earthquake median / COL 1992 (from a2_clutter.csv). Writes a2b_clutter.csv, a2b_checks.csv.
Run: python -m akp.stage_a.a2b_explosions
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from akp.paths import data_dir
from akp.stage_a import a2_wavefield as a2
from swe.io import catalog_to_dataframe, get_events

HERE = Path(__file__).parent
SITE = (41.3, 129.1)  # approximate test-site center: search anchor only; parameters come from the catalog
MB_MIN = 4.5
SNR_FACTOR = 2.0
BRACKET = 2.0


def main() -> None:
    c = Client(base_url=a2.ES, timeout=600)
    cat = catalog_to_dataframe(get_events(starttime=T("2006-01-01"), endtime=T("2018-01-01"), latitude=SITE[0],
                                          longitude=SITE[1], maxradius=0.5, minmagnitude=4.0))
    cat = cat[cat.magnitude >= MB_MIN].sort_values("time")
    wf_dir, resp_dir = data_dir("a2/wf"), data_dir("a2/resp")
    inv = c.get_stations(network="IU", station="COLA", level="station")
    sta_xy = {("IU", "COLA"): (inv[0][0].latitude, inv[0][0].longitude)}
    rows = []
    for _, e in cat.iterrows():
        r = dict(kind="explosion", name=f"dprk_{e.time[:10]}", time=T(e.time), lat=e.latitude, lon=e.longitude,
                 depth_km=0.0, net="IU", sta="COLA")
        try:
            st, resp, tp, dist = a2.fetch(c, r, wf_dir, resp_dir, sta_xy)
        except (FDSNNoDataException, FileNotFoundError, IndexError) as exc:
            rows.append(dict(event=r["name"], mag=e.magnitude, status=f"no data: {type(exc).__name__}"))
            continue
        tr = st.copy().merge(method=0)[0]
        if np.ma.isMaskedArray(tr.data):
            tr.data = tr.data.filled(0)
        nyq = tr.stats.sampling_rate / 2.0
        tr.detrend("demean").taper(0.02)
        tr.remove_response(inventory=resp, output="VEL", pre_filt=(0.2, 0.3, 0.9 * nyq, 0.98 * nyq), water_level=None)
        for fmin, fmax in a2.BANDS:
            if fmax >= 0.95 * nyq:
                continue
            t, hil, _ = a2.envelopes(tr, fmin, fmax)
            rel = t - (tp - tr.stats.starttime)
            win = (rel >= a2.PEAK[0]) & (rel <= a2.PEAK[1])
            ipk = np.flatnonzero(win)[np.argmax(hil[win])]
            tpk, peak = rel[ipk], hil[ipk]
            noise = float(np.median(hil[(rel >= a2.NOISE[0]) & (rel <= a2.NOISE[1])]))
            for a, b in a2.LAGS:
                w = hil[(rel >= tpk + a) & (rel <= tpk + b)]
                med = float(np.median(w))
                rows.append(dict(event=r["name"], mag=e.magnitude, status="ok", band=f"{fmin:g}-{fmax:g}",
                                 lag=f"{a:g}-{b:g}", ratio=med / peak, noise_ratio=noise / peak,
                                 valid=bool(med > SNR_FACTOR * noise)))
    ref = pd.DataFrame(rows)
    ref.to_csv(HERE / "a2b_clutter.csv", index=False)

    cl = pd.read_csv(HERE / "a2_clutter.csv")
    h = cl[cl.estimator == "hilbert"]
    t92 = h[h.kind == "test1992"].set_index(["band", "lag"]).ratio
    eq = h[(h.kind == "teleseism") & (h.sta == "COLA")].groupby(["band", "lag"]).ratio.median()
    ok = ref[(ref.status == "ok")]
    outside, cells, bias = [], 0, []
    for (band, lag), v in t92.items():
        vals = ok[(ok.band == band) & (ok.lag == lag) & ok.valid].ratio
        bias.append(eq.get((band, lag), np.nan) / v)
        if vals.empty:
            continue
        cells += 1
        lo, hi = vals.min() / BRACKET, vals.max() * BRACKET
        if not lo <= v <= hi:
            outside.append(f"{band}Hz/{lag}s:{v:.3f} vs [{lo:.3f},{hi:.3f}] (n={len(vals)})")
    checks = [
        dict(check="n_reference_explosions", value=int(ok.event.nunique()), of=len(cat), triggered="",
             detail=";".join(sorted(ok.event.unique()))),
        dict(check="valid_cells", value=cells, of=len(t92), triggered="", detail=""),
        dict(check="F1b_outside_cells", value=len(outside), of=cells,
             triggered=bool(cells and len(outside) >= cells / 2), detail="; ".join(outside)),
        dict(check="eq_over_1992_min", value=round(float(np.nanmin(bias)), 2), of="", triggered="", detail="COLA earthquake median / COL 1992, over cells"),
        dict(check="eq_over_1992_median", value=round(float(np.nanmedian(bias)), 2), of="", triggered="", detail=""),
        dict(check="eq_over_1992_max", value=round(float(np.nanmax(bias)), 2), of="", triggered="", detail=""),
    ]
    checks.append(dict(check="gate_pass", value=int(bool(cells) and not checks[2]["triggered"]), of="", triggered="", detail=""))
    with (HERE / "a2b_checks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    for ch in checks:
        print({k: str(v)[:300] for k, v in ch.items()})


if __name__ == "__main__":
    main()
