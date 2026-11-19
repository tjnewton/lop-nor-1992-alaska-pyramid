"""A1: the 1992 stations.

Which stations in Alaska / NW Canada recorded the 1992-05-21 test, or were operating in May
1992 and could have; where they were; what they recorded with; how far from each claim anchor.

Sources (all open, no request):
- ISC arrivals for the test (akp/setup/pr_b_isc_arrivals.csv): who reported it.
- ISC bulletin census: station codes with arrivals for any event located in the region during
  May 1992 (operating that month), via swe.io.get_events.
- International Registry of Seismograph Stations (ISC / WDC Denver): coordinates by code
  (region search; HTML cached under $AKP_DATA/a1).
- EarthScope station metadata at the test hour via swe.io.get_stations_in_region (SY excluded):
  network and channels.
- akp/setup/pr_g_1992_open.csv (open waveforms) and pr_h_same_site.csv (modern same-site).

Writes a1_stations.csv (one row per code with any 1992 evidence; site_id groups aliases),
a1_checks.csv (gate checks and gap accounting), a1_line_coverage.csv (points along the Denali-Nome
line) and a1_box_coverage.csv (a grid over the mckinley_nome box): distance to the nearest station
that recorded the test and to the nearest station operating at the test.
Run: python -m akp.stage_a.a1_stations
"""
from __future__ import annotations

import csv
import html
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy import read_events
from pyproj import Geod

from akp.paths import data_dir
from akp.zones import ANCHORS, BOXES, in_box, in_recording_region
from swe.io import get_events, get_stations_in_region, inventory_to_dataframe

HERE = Path(__file__).parent
SETUP = HERE.parent / "setup"
GEOD = Geod(ellps="WGS84")
# Region as (minlat, maxlat, minlon, maxlon) boxes; the antimeridian needs two.
REGION = [(51.0, 72.0, -180.0, -120.0), (51.0, 56.0, 170.0, 180.0)]
CENSUS = (T("1992-05-01"), T("1992-06-01"))
SEARCH_KM = 300.0  # Falsifier radius (input)
TOL_KM = 2.0  # Position tolerance (input)
LINE_STEP_KM = 10.0  # sampling of the Denali-Nome line for coverage
SNR_SEEN = 5.0
AROUND_TEST_DAYS = 3  # operating_at_test: picks within this many days on both sides of the test (input)
ALIAS_KM = 0.5  # codes this close (or linked by the registry prime code) are one site (input)
BOX_STEP = (0.1, 0.2)  # (lat, lon) degrees: grid for the mckinley_nome box coverage  # screening P SNR above which an open waveform counts as recording the test (input)
REGISTRY_URL = "http://www.isc.ac.uk/cgi-bin/stations"
ROW_RE = re.compile(
    r"^ (?P<code>\S+)\s+(?P<name>.+?)\s+(?P<lat>-?\d+\.\d{3,})\s+(?P<lon>-?\d+\.\d{3,})"
    r"(?:\s+(?P<elev>-?\d+\.\d+))?(?P<rest>.*)$"
)


def km(lat1, lon1, lat2, lon2):
    return GEOD.inv(lon1, lat1, lon2, lat2)[2] / 1000.0


def registry(cache: Path) -> pd.DataFrame:
    """Registry rows for the region (code, name, lat, lon, elev, prime, status, region)."""
    rows = []
    for i, (la0, la1, lo0, lo1) in enumerate(REGION):
        page = cache / f"registry_{i}.html"
        if not page.exists():
            q = urllib.parse.urlencode(dict(stnsearch="RECT", stn_bot_lat=la0, stn_top_lat=la1,
                                            stn_left_lon=lo0, stn_right_lon=lo1))
            with urllib.request.urlopen(f"{REGISTRY_URL}?{q}", timeout=300) as r:
                page.write_bytes(r.read())
        text = html.unescape(re.sub(r"<[^>]*>", " ", page.read_text("latin-1")))
        lines = text.splitlines()
        for j, ln in enumerate(lines):
            m = ROW_RE.match(ln)
            if not m:
                continue
            rest = m["rest"].split()
            status = rest[-1] if rest and rest[-1] in ("Open", "Closed") else ""
            prime = next((t for t in rest if t not in ("Open", "Closed") and not re.fullmatch(r"-?\d+\.?\d*", t)), "")
            region = lines[j + 1].strip() if j + 1 < len(lines) else ""
            rows.append(dict(code=m["code"], reg_name=m["name"].strip(), reg_lat=float(m["lat"]),
                             reg_lon=float(m["lon"]), reg_elev=float(m["elev"]) if m["elev"] else np.nan,
                             reg_prime=prime, reg_status=status, reg_region=region))
    df = pd.DataFrame(rows)
    return df[[in_recording_region(a, o) for a, o in zip(df.reg_lat, df.reg_lon)]].drop_duplicates()


def census(cache: Path) -> tuple[dict[str, list[str]], int]:
    """(station code -> sorted pick dates, events scanned) for regional events in May 1992.

    The ISC QuakeML is cached under $AKP_DATA/a1, so a later bulletin revision cannot silently
    change the counts.
    """
    days: dict[str, set[str]] = {}
    n_events = 0
    for i, (la0, la1, lo0, lo1) in enumerate(REGION):
        path = cache / f"census_{i}.xml"
        if path.exists():
            cat = read_events(str(path))
        else:
            cat = get_events(starttime=CENSUS[0], endtime=CENSUS[1], client="ISC", includearrivals=True,
                             minlatitude=la0, maxlatitude=la1, minlongitude=lo0, maxlongitude=lo1)
            cat.write(str(path), format="QUAKEML")
        n_events += len(cat)
        for ev in cat:
            for p in ev.picks:
                days.setdefault(p.waveform_id.station_code, set()).add(p.time.date.isoformat())
    return {k: sorted(v) for k, v in days.items()}, n_events


def earthscope_1992(t0: T) -> pd.DataFrame:
    frames = []
    for la0, la1, lo0, lo1 in REGION:
        inv = get_stations_in_region(starttime=t0, endtime=t0 + 3600, minlatitude=la0, maxlatitude=la1,
                                     minlongitude=lo0, maxlongitude=lo1, level="channel",
                                     client="https://service.earthscope.org")
        frames.append(inventory_to_dataframe(inv, level="channel"))
    ch = pd.concat(frames, ignore_index=True)
    ch = ch[[in_recording_region(a, o) for a, o in zip(ch.latitude, ch.longitude)]]
    return (ch.assign(chan=lambda d: d.channel + "@" + d.sample_rate.map("{:g}".format))
              .groupby("station")
              .agg(es_network=("network", lambda s: ";".join(sorted(set(s)))),
                   es_lat=("latitude", "first"), es_lon=("longitude", "first"),
                   es_channels=("chan", lambda s: ";".join(sorted(set(s)))))
              .reset_index().rename(columns={"station": "code"}))


def earthscope_epochs() -> dict[str, tuple[T, T | None]]:
    """Code -> (earliest start, latest end or None) over all real EarthScope epochs in the region."""
    out: dict[str, tuple[T, T | None]] = {}
    for la0, la1, lo0, lo1 in REGION:
        inv = get_stations_in_region(starttime=T("1900-01-01"), endtime=T("2100-01-01"), minlatitude=la0,
                                     maxlatitude=la1, minlongitude=lo0, maxlongitude=lo1, level="station",
                                     client="https://service.earthscope.org")
        for net in inv:
            for sta in net:
                s0, e0 = out.get(sta.code, (sta.start_date, sta.end_date))
                end = None if (e0 is None or sta.end_date is None) else max(e0, sta.end_date)
                out[sta.code] = (min(s0, sta.start_date), end)
    return out


def dist_to_line(lat, lon, a, b, step_km=1.0):
    """Minimum distance (km) from a point to the geodesic segment a-b (sampled every step_km)."""
    n = int(km(*a, *b) / step_km)
    pts = [a] + [(p[1], p[0]) for p in GEOD.npts(a[1], a[0], b[1], b[0], n)] + [b]
    la = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts])
    return float(np.min(GEOD.inv(np.full_like(lo, lon), np.full_like(la, lat), lo, la)[2]) / 1000.0)


def main() -> None:
    cache = data_dir("a1")
    ev = pd.read_csv(SETUP / "pr_b_isc_event.csv").iloc[0]
    t0 = T(ev["origin_time"])
    arr = pd.read_csv(SETUP / "pr_b_isc_arrivals.csv")
    rep = (arr.groupby("station")
              .agg(isc_phases=("phase", lambda s: ";".join(sorted(set(s)))),
                   isc_est_lat=("est_lat", "first"), isc_est_lon=("est_lon", "first"),
                   isc_dist_deg=("distance_deg", "min"))
              .reset_index().rename(columns={"station": "code"}))
    reg = registry(cache)
    cen, n_census = census(cache)
    es = earthscope_1992(t0)
    g = pd.read_csv(SETUP / "pr_g_1992_open.csv")
    open_wf = set(g.loc[g.in_region & g.has_data, "station"])
    # P visible in the open waveform (screening SNR from pr_g) also counts as "recorded the test"
    seen_wf = set(g.loc[g.in_region & g.has_data & (pd.to_numeric(g.snr_p, errors="coerce") >= SNR_SEEN), "station"])
    same = pd.read_csv(SETUP / "pr_h_same_site.csv")
    same = (same[same.sep_km <= 0.5].groupby("station")
                .apply(lambda d: ";".join(f"{n}({s[:4]}-{e[:4] if e != 'open' else 'open'})"
                                          for n, s, e in zip(d.network, d.start, d.end)), include_groups=False)
                .rename("modern_same_site").reset_index().rename(columns={"station": "code"}))

    reg_codes = set(reg.code)
    codes = set(rep.code) | (set(cen) & reg_codes) | set(es.code)
    rows = pd.DataFrame({"code": sorted(codes)})
    rows = rows.merge(rep, on="code", how="left").merge(es, on="code", how="left").merge(same, on="code", how="left")
    # registry: if a code has several rows in the region, keep the one nearest any other position we have
    regs = []
    for code in rows.code:
        r = reg[reg.code == code]
        if len(r) > 1:
            ref = rows.loc[rows.code == code, ["isc_est_lat", "isc_est_lon", "es_lat", "es_lon"]].iloc[0]
            la, lo = (ref.isc_est_lat, ref.isc_est_lon) if pd.notna(ref.isc_est_lat) else (ref.es_lat, ref.es_lon)
            if pd.notna(la):
                r = r.iloc[[int(np.argmin([km(la, lo, a, o) for a, o in zip(r.reg_lat, r.reg_lon)]))]]
        regs.append(r.iloc[0].to_dict() | {"reg_rows": len(reg[reg.code == code])} if len(r) else {"code": code, "reg_rows": 0})
    rows = rows.merge(pd.DataFrame(regs), on="code", how="left")
    test_day = t0.date
    rows["census_first"] = rows.code.map(lambda c: cen[c][0] if c in cen else None)
    rows["census_last"] = rows.code.map(lambda c: cen[c][-1] if c in cen else None)
    rows["census_days"] = rows.code.map(lambda c: len(cen.get(c, []))).astype(int)

    def around(c):
        ds = [T(d).date for d in cen.get(c, [])]
        before = any(0 <= (test_day - d).days <= AROUND_TEST_DAYS for d in ds)
        after = any(0 <= (d - test_day).days <= AROUND_TEST_DAYS for d in ds)
        return before and after
    rows["operating_at_test"] = rows.code.map(around)
    rows["epoch_spans_test"] = rows.es_network.notna()
    rows["recorded_by"] = [";".join(k for k, ok in (("isc", isc), ("waveform", c in seen_wf)) if ok)
                           for c, isc in zip(rows.code, rows.isc_phases.notna())]
    rows["recorded_test"] = rows.recorded_by != ""
    rows["evidence"] = np.select(
        [rows.recorded_test, rows.operating_at_test, rows.census_days > 0],
        ["recorded_test", "operating_at_test", "operating_may1992_not_at_test"], "metadata_1992_only")
    rows["waveforms_open"] = rows.code.isin(open_wf)
    for src in ("reg", "es", "isc_est"):
        if f"{src}_lat" not in rows:
            rows[f"{src}_lat"] = np.nan
    rows["lat"] = rows.reg_lat.fillna(rows.es_lat).fillna(rows.isc_est_lat)
    rows["lon"] = rows.reg_lon.fillna(rows.es_lon).fillna(rows.isc_est_lon)
    rows["pos_source"] = np.where(rows.reg_lat.notna(), "registry", np.where(rows.es_lat.notna(), "earthscope", "isc_estimate"))

    def sep(a, b):
        return [km(*x) if not np.isnan(np.sum(x)) else np.nan
                for x in zip(rows[f"{a}_lat"], rows[f"{a}_lon"], rows[f"{b}_lat"], rows[f"{b}_lon"])]
    rows["sep_reg_isc_km"] = np.round(sep("reg", "isc_est"), 2)
    rows["sep_reg_es_km"] = np.round(sep("reg", "es"), 2)
    rows["km_denali_nome_line"] = [round(dist_to_line(a, o, ANCHORS["denali"], ANCHORS["nome"]), 1) for a, o in zip(rows.lat, rows.lon)]
    rows["km_valdez"] = [round(km(a, o, *ANCHORS["valdez"]), 1) for a, o in zip(rows.lat, rows.lon)]
    for z, b in BOXES.items():
        rows[f"in_{z}"] = [in_box(a, o, b) for a, o in zip(rows.lat, rows.lon)]
    # sites: codes linked by the registry prime code or within ALIAS_KM are one site
    parent = {c: c for c in rows.code}
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for c, prime in zip(rows.code, rows.reg_prime):
        if isinstance(prime, str) and prime in parent:
            union(c, prime)
    xy = list(zip(rows.code, rows.lat, rows.lon))
    for i, (c1, a1, o1) in enumerate(xy):
        for c2, a2, o2 in xy[i + 1:]:
            if abs(a1 - a2) < 0.02 and km(a1, o1, a2, o2) < ALIAS_KM:
                union(c1, c2)
    rows["site_id"] = rows.code.map(find)
    rows = rows.sort_values(["evidence", "km_denali_nome_line"]).reset_index(drop=True)
    cols = ["code", "site_id", "evidence", "recorded_test", "recorded_by", "operating_at_test", "epoch_spans_test",
            "census_days", "census_first", "census_last", "waveforms_open", "lat", "lon", "pos_source",
            "reg_name", "reg_region", "reg_status", "reg_prime", "reg_rows", "es_network", "es_channels",
            "isc_phases", "isc_dist_deg", "sep_reg_isc_km", "sep_reg_es_km", "km_denali_nome_line",
            "km_valdez", "in_mckinley_nome", "in_valdez", "modern_same_site"]
    rows[cols].to_csv(HERE / "a1_stations.csv", index=False)

    # gate checks
    r = rows[rows.recorded_test]
    near_reg = reg[[min(dist_to_line(a, o, ANCHORS["denali"], ANCHORS["nome"]), km(a, o, *ANCHORS["valdez"])) <= SEARCH_KM
                    for a, o in zip(reg.reg_lat, reg.reg_lon)]]
    unlisted_all = sorted(set(near_reg.code) - set(rows.code))
    listed_xy = list(zip(rows.lat, rows.lon))
    nr = near_reg.drop_duplicates("code").set_index("code")
    aliases = [c for c in unlisted_all
               if nr.loc[c, "reg_prime"] in set(rows.code)
               or any(abs(nr.loc[c, "reg_lat"] - a) < 0.02 and km(nr.loc[c, "reg_lat"], nr.loc[c, "reg_lon"], a, o) < ALIAS_KM
                      for a, o in listed_xy)]
    unlisted = [c for c in unlisted_all if c not in aliases]
    epochs = earthscope_epochs()
    post = [c for c in unlisted if c in epochs and epochs[c][0] > t0]
    excl = [c for c in unlisted if c in epochs and epochs[c][0] <= t0]  # EarthScope epochs exist, none at the test hour
    unknown = [c for c in unlisted if c not in epochs]
    g2_bad = r[r.sep_reg_isc_km > TOL_KM]
    both = rows[rows.sep_reg_es_km.notna()]
    indep = both[both.sep_reg_es_km > 0.005]  # 0.00 km pairs share one metadata source (not independent)
    checks = [
        dict(check="G1_reporters_in_registry", value=int(r.reg_lat.notna().sum()), of=len(r),
             passed=bool(r.reg_lat.notna().all()), detail=";".join(r.loc[r.reg_lat.isna(), "code"])),
        dict(check="G2_inversion_selfcheck_registry_vs_isc_estimate", value=int((r.sep_reg_isc_km <= TOL_KM).sum()), of=int(r.sep_reg_isc_km.notna().sum()),
             passed=g2_bad.empty, detail=";".join(f"{c}:{s}km" for c, s in zip(g2_bad.code, g2_bad.sep_reg_isc_km))),
        dict(check="G3_registry_vs_earthscope_within_tol", value=int((both.sep_reg_es_km <= TOL_KM).sum()), of=len(both),
             passed=bool((both.sep_reg_es_km <= TOL_KM).all()),
             detail=";".join(f"{c}:{s}km" for c, s in zip(both.code, both.sep_reg_es_km) if s > TOL_KM)),
        dict(check="G3_independent_pairs", value=len(indep), of=len(both), passed="",
             detail=f"max sep {indep.sep_reg_es_km.max():.2f} km" if len(indep) else ""),
        dict(check="G3_independent_max_sep_km", value=round(float(indep.sep_reg_es_km.max()), 2) if len(indep) else "",
             of="", passed="", detail=""),
        dict(check="positions_untested_registry_only", value=int(((rows.pos_source == "registry") & rows.es_network.isna()
                                                                  & rows.isc_phases.isna()).sum()), of=len(rows), passed="",
             detail="registry position with no second source"),
        dict(check="sites_recorded_test", value=int(rows.loc[rows.recorded_test, "site_id"].nunique()),
             of=int(rows.recorded_test.sum()), passed="", detail="distinct sites / codes"),
        dict(check="codes_not_operating_at_test", value=int((rows.census_days > 0).sum() - (rows.operating_at_test).sum()),
             of=int((rows.census_days > 0).sum()), passed="", detail="census codes without picks around the test"),
        dict(check="G4_reporters_in_census", value=int((r.census_days > 0).sum()), of=len(r),
             passed=bool((r.census_days > 0).all()), detail=";".join(r.loc[r.census_days == 0, "code"])),
        dict(check="census_regional_events_may1992", value=n_census, of="", passed="", detail="ISC bulletin events in the region, 1992-05-01..06-01"),
        dict(check="gap_registry_unlisted_within_radius", value=len(unlisted_all), of=len(near_reg), passed="",
             detail="registry codes within the search radius with no 1992 evidence"),
        dict(check="gap_aliases_of_listed_sites", value=len(aliases), of=len(unlisted_all), passed="",
             detail=";".join(aliases)),
        dict(check="gap_post1992_by_earthscope_start", value=len(post), of=len(unlisted), passed="", detail=";".join(post)),
        dict(check="gap_earthscope_epochs_exclude_test", value=len(excl), of=len(unlisted), passed="", detail=";".join(excl)),
        dict(check="gap_no_epoch_information", value=len(unknown), of=len(unlisted), passed="", detail=";".join(unknown)),
    ]
    # coverage along the Denali-Nome line: nearest recorder / nearest May-1992 operating station
    a, b = ANCHORS["denali"], ANCHORS["nome"]
    n = int(km(*a, *b) / LINE_STEP_KM)
    pts = [a] + [(q[1], q[0]) for q in GEOD.npts(a[1], a[0], b[1], b[0], n)] + [b]
    rec = rows[rows.recorded_test]
    op = rows[rows.recorded_test | rows.operating_at_test]
    cov = []
    for i, (la, lo) in enumerate(pts):
        dr = [km(la, lo, x, y) for x, y in zip(rec.lat, rec.lon)]
        do = [km(la, lo, x, y) for x, y in zip(op.lat, op.lon)]
        cov.append(dict(point=i, lat=round(la, 4), lon=round(lo, 4), km_from_denali=round(km(*a, la, lo), 1),
                        nearest_recorder=rec.code.iloc[int(np.argmin(dr))], km_nearest_recorder=round(min(dr), 1),
                        nearest_operating=op.code.iloc[int(np.argmin(do))], km_nearest_operating=round(min(do), 1)))
    pd.DataFrame(cov).to_csv(HERE / "a1_line_coverage.csv", index=False)
    b = BOXES["mckinley_nome"]
    grid = []
    for la in np.arange(b["minlatitude"], b["maxlatitude"] + 1e-9, BOX_STEP[0]):
        for lo in np.arange(b["minlongitude"], b["maxlongitude"] + 1e-9, BOX_STEP[1]):
            dr = GEOD.inv(np.full(len(rec), lo), np.full(len(rec), la), rec.lon.to_numpy(), rec.lat.to_numpy())[2] / 1000.0
            do = GEOD.inv(np.full(len(op), lo), np.full(len(op), la), op.lon.to_numpy(), op.lat.to_numpy())[2] / 1000.0
            grid.append(dict(lat=round(la, 3), lon=round(lo, 3), nearest_recorder=rec.code.iloc[int(np.argmin(dr))],
                             km_nearest_recorder=round(float(dr.min()), 1), km_nearest_operating=round(float(do.min()), 1)))
    pd.DataFrame(grid).to_csv(HERE / "a1_box_coverage.csv", index=False)

    with (HERE / "a1_checks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    print(rows.evidence.value_counts().to_dict())
    for c in checks:
        print({k: (str(v)[:160]) for k, v in c.items()})


if __name__ == "__main__":
    assert abs(dist_to_line(63.0, -150.0, (63.0, -151.0), (63.0, -149.0))) < 0.5  # point on the segment
    assert abs(km(0, 0, 0, 1) - 111.32) < 0.05  # 1 degree of longitude at the equator
    main()
