"""P4 S9: which seismic receivers have ever stood in Track A, and how much prior lies near them.

Inputs (station lists fetched 2026-11-14 into $AKP_DATA/p4/seismic_census; area 60.2-65.8 N, 167.2-144.2 W):
- earthscope_station.txt: fdsnws station, level station, every epoch, restricted included;
- fedcatalog_station.txt: the EarthScope federated catalog (other data centers);
- raspberryshake_station.txt: the Raspberry Shake station service, network AM;
- the ISC station registry cached by stage A (a1_stations.registry; positions, no epochs);
- s9_active_receivers.csv: receivers of the free active-source surveys (s9_active_receivers.py);
- the Track A prior and the cover-weighted prior (p3_prior.npz, p3_cover.npz);
- a8_production.csv, a8_attenuation.csv (simulated share of the incident P against range) and
  a5_dprk_site_spread.csv (clutter measured at in-box sites).
Rules fixed in advance. Nothing here reads a waveform.

Method:
- a receiver is one position (rounded to 1e-3 degree); the synthetic network SY is left out. A
  position read from a scanned report that lies more than ISOLATED_KM from every other position of
  its report is dropped and counted. Each source has a position uncertainty (UNCERTAINTY_KM);
- every Track A cell is split into SUB x SUB points that share its prior mass, so that the radii
  below the cell size are not set by the grid. Exposure(r) is the mass of the points within r of
  any receiver, for all receivers and for passive stations, under four priors (claim-weighted,
  cover-weighted, and the two with equal claim weights), and again with r enlarged by each
  receiver's position uncertainty;
- the high-prior half is the smallest set of Track A cells holding half of the cover-weighted prior;
- the detection range of a ground case and band is the largest tabulated range at which the
  simulated share (forward, slow window, vertical) is at least the clutter of that band in the
  first window after P. Two clutter levels (median site, cleanest twentieth) and two share tables
  (elastic; thawed ground with attenuation at Q_UPPER). This is the rule, kept as a table read;
- the sensitivity table takes the most exposed case (CASE, BAND), fits share = A r^b between
  FIT_KM, and solves A k r^b exp(-pi f r / (Q U)) = clutter for the range, for every combination of
  clutter level (cleanest twentieth of records; cleanest record; each times the square root of the
  repeatable share of the coda at College), body factor k (as simulated; the largest base of the
  trials' prior, by the ratio of bases and by its square) and attenuation (none, Q_GENEROUS, Q_UPPER);
- the reopening test: the exposure of all receivers, with position uncertainty, under each
  prior, at the largest range among the rows with Q_GENEROUS, against REOPEN.
Not simulated, so not in the table: the slower thawed host in the highest band, any higher band,
any body other than the center of the trials' prior.
Writes s9_receivers.csv (receivers within NEAR_KM of Track A), s9_exposure.csv (the radii),
s9_sensitivity.csv, s9_components.csv, s9_high_prior.csv, s9_detection_range.csv and s9_checks.csv.
Self-check: one receiver at a cell center reaches every point of the cell at three quarters of
the cell size and not all of them at a quarter; the query area holds Track A and MARGIN_KM.
Run: python -m akp.stage_p4.s9_census
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from akp.paths import data_dir
from akp.stage_a.a1_stations import registry
from akp.stage_p3.p3_prior import out_dir as prior_dir

HERE = Path(__file__).parent
STAGE_A = HERE.parent / "stage_a"
RADII_KM = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0)
ACTIVE_REACH_KM = 0.2  # Half of the largest base in the trials' prior
SUB = 10  # points per cell side
NEAR_KM = 50.0  # receivers listed in s9_receivers.csv
Q_UPPER = 50.0  # the upper sediment value
BANDS = ("0.5-1", "1-2", "2-4")
CASE, BAND, F_HZ = "thawed_1200", "2-4", 3.0  # the most exposed simulated case and its band center (a8_attenuation.CENTRE_HZ)
FIT_KM = (1.0, 2.6)  # Inside the model, clear of its absorbing edge
Q_GENEROUS = 100.0  # the stage's generous value (a8_attenuation.Q_VALUES)
SIM_BASE_M, MAX_BASE_M = 300.0, 400.0  # the simulated body (center of the trials' prior) and the prior's largest base
REOPEN = 0.05  # Kept
ISOLATED_KM = 5.0
MARGIN_KM = 50.0  # The query area must hold Track A and this margin
QUERY = (60.2, 65.8, -167.2, -144.2)  # lat south, lat north, lon west, lon east of the station queries
UNCERTAINTY_KM = {"earthscope": 0.1, "fedcatalog": 0.1, "raspberryshake": 1.0, "isc_registry": 0.5, "ofr": 0.5, "galena_2000": 0.05, "tact_1985": 0.1}  # inputs:
# archive metadata; public Raspberry Shake positions are offset on purpose; registry positions are rounded and partly pre-GPS; scanned tables;
# the Galena report gives its survey as no better than 15 ft under tree cover
PRIORS = ("prior", "cover", "equal", "equal_cover")
TO_AK = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
TO_LL = Transformer.from_crs("EPSG:3338", "EPSG:4326", always_xy=True)


def fdsn_text(path: Path, source: str) -> pd.DataFrame:
    rows = []
    for ln in path.read_text(errors="replace").splitlines():
        f = [v.strip() for v in ln.split("|")]
        if ln.startswith("#") or len(f) < 8:
            continue
        rows.append(dict(source=source, network=f[0], station=f[1], lat=float(f[2]), lon=float(f[3]), first=f[6][:4], last=f[7][:4]))
    return pd.DataFrame(rows)


def receivers() -> tuple[pd.DataFrame, int]:
    """One row per receiver position, and the number of isolated scanned positions dropped."""
    d = data_dir("p4/seismic_census")
    parts = [fdsn_text(d / "earthscope_station.txt", "earthscope"), fdsn_text(d / "fedcatalog_station.txt", "fedcatalog"),
             fdsn_text(d / "raspberryshake_station.txt", "raspberryshake")]
    reg = registry(data_dir("a1"))
    parts.append(pd.DataFrame(dict(source="isc_registry", network="", station=reg.code, lat=reg.reg_lat, lon=reg.reg_lon, first="", last="")))
    act = pd.read_csv(HERE / "s9_active_receivers.csv")
    ax, ay = TO_AK.transform(act.lon.to_numpy(), act.lat.to_numpy())
    keep = np.zeros(len(act), bool)
    for rep in act.report.unique():
        k = np.flatnonzero(act.report == rep)
        if not rep.startswith("ofr_"):  # surveyed positions, not scans
            keep[k] = True
            continue
        dist = cKDTree(np.c_[ax[k], ay[k]]).query(np.c_[ax[k], ay[k]], k=2)[0][:, 1]
        keep[k] = dist <= ISOLATED_KM * 1000
    dropped, act = int((~keep).sum()), act[keep]
    parts.append(pd.DataFrame(dict(source=act.report, network="", station="", lat=act.lat, lon=act.lon, first="", last="")))
    r = pd.concat(parts, ignore_index=True)
    r = r[(r.network != "SY") & r.lat.between(QUERY[0], QUERY[1]) & r.lon.between(QUERY[2], QUERY[3])]
    r["cls"] = np.where(r.source.isin(set(act.report)), "active", "passive")
    r["u_km"] = r.source.map(lambda s: UNCERTAINTY_KM["ofr" if s.startswith("ofr_") else s])
    r["key"] = r.lat.round(3).astype(str) + "," + r.lon.round(3).astype(str)
    g = r.groupby(["cls", "key"], as_index=False).agg(
        lat=("lat", "first"), lon=("lon", "first"), u_km=("u_km", "min"), sources=("source", lambda s: "+".join(sorted(set(s)))),
        codes=("station", lambda s: "+".join(sorted({f"{n}.{c}".strip(".") for n, c in zip(r.loc[s.index, "network"], s) if c}))[:80]),
        first=("first", lambda s: min([v for v in s if v] or [""])), last=("last", lambda s: max([v for v in s if v] or [""])))
    g["x"], g["y"] = TO_AK.transform(g.lon.to_numpy(), g.lat.to_numpy())
    return g.drop(columns="key"), dropped


def distances(px, py, rec: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distance (km) of every point to the nearest receiver, and the same less that receiver's position uncertainty."""
    d = np.full(len(px), np.inf, np.float32)
    d_u = d.copy()
    pts = np.c_[px, py]
    for u, g in rec.groupby("u_km"):
        dist = (cKDTree(np.c_[g.x, g.y]).query(pts, workers=4)[0] / 1000).astype(np.float32)
        d, d_u = np.minimum(d, dist), np.minimum(d_u, dist - np.float32(u))
    return d, d_u


def detection_ranges() -> pd.DataFrame:
    p = pd.read_csv(STAGE_A / "a8_production.csv")
    p = p[(p.window == "slow") & (p.component == "vertical") & (p.azimuth == "forward")]
    a = pd.read_csv(STAGE_A / "a8_attenuation.csv")
    c = clutter()
    tables = {"elastic": p.rename(columns={"numerical": "share"})[["case", "band", "range_m", "share"]],
              "attenuated": a[a.q == Q_UPPER].rename(columns={"share_attenuated": "share"})[["case", "band", "range_m", "share"]]}
    rows = []
    for name, t in tables.items():
        for (case, band), g in t.groupby(["case", "band"]):
            if band not in BANDS:
                continue
            g = g.sort_values("range_m")
            for level in ("median", "p05"):
                ok = g[g.share >= c.loc[band, level]]
                rows.append(dict(table=name, case=case, band=band, clutter_level=level, clutter=round(float(c.loc[band, level]), 4),
                                 range_m=float(ok.range_m.max()) if len(ok) else 0.0, first_range_m=float(g.range_m.min()),
                                 at_edge=bool(len(ok) and ok.range_m.max() == g.range_m.max())))
    return pd.DataFrame(rows)


def clutter() -> pd.DataFrame:
    c = pd.read_csv(STAGE_A / "a5_dprk_site_spread.csv")
    return c[(c.box == "mckinley_nome") & (c.event == "all") & (c.lag == "5-15")].set_index("band")


def sensitivity() -> tuple[pd.DataFrame, dict]:
    """The table of ranges (km) at which the extended share reaches each clutter level."""
    p = pd.read_csv(STAGE_A / "a8_production.csv")
    g = p[(p.window == "slow") & (p.component == "vertical") & (p.azimuth == "forward") & (p.case == CASE) & (p.band == BAND)]
    k = g[(g.range_m >= FIT_KM[0] * 1000) & (g.range_m <= FIT_KM[1] * 1000)]
    b, ln_a = np.polyfit(np.log(k.range_m / 1000), np.log(k.numerical), 1)
    amp = float(np.exp(ln_a))
    u = float(pd.read_csv(STAGE_A / "a8_attenuation.csv").query("case == @CASE").rayleigh_speed_ms.iloc[0])
    c = clutter().loc[BAND]
    rep = pd.read_csv(STAGE_A / "a5_lopnor_repeatable.csv").query("sta == 'COL' and cha == 'BHZ' and band == @BAND and lag == '5-15'").repeat_frac.iloc[0]
    floor = float(np.sqrt(rep))
    levels = {"cleanest_twentieth": float(c.p05), "cleanest_record": float(c["min"]),
              "cleanest_twentieth_stacked": float(c.p05) * floor, "cleanest_record_stacked": float(c["min"]) * floor}
    bodies = {"as_simulated": 1.0, "largest_base_linear": MAX_BASE_M / SIM_BASE_M, "largest_base_squared": (MAX_BASE_M / SIM_BASE_M) ** 2}
    rows = []
    for lev, cl in levels.items():
        for body, kf in bodies.items():
            for att, q in (("none", np.inf), ("generous", Q_GENEROUS), ("upper_sediment", Q_UPPER)):
                def gap(r, kf=kf, q=q, cl=cl):
                    return amp * kf * r**b * np.exp(-np.pi * F_HZ * r * 1000 / (q * u)) - cl
                rows.append(dict(clutter_level=lev, clutter=round(cl, 4), body=body, body_factor=round(kf, 3), attenuation=att,
                                 range_km=round(float(brentq(gap, 0.01, 1000.0)), 3)))
    info = dict(fit_exponent=round(float(b), 4), fit_share_at_1km=round(amp, 4), rayleigh_speed_ms=u, repeatable_share=float(rep), stacking_floor_factor=round(floor, 4))
    return pd.DataFrame(rows), info


def main() -> None:
    pr, cv = np.load(prior_dir() / "p3_prior.npz"), np.load(prior_dir() / "p3_cover.npz")
    land, ta = pr["land"], pr["track_a"]
    tf = Affine(*pr["transform"])
    cell = abs(tf.a)
    rows, cols = np.nonzero(land)
    rows, cols = rows[ta], cols[ta]
    mass = {"prior": pr["mass"][ta], "cover": cv["mass_cover"][ta], "equal": pr["mass_equal"][ta], "equal_cover": cv["mass_equal_cover"][ta]}
    comps = {k[5:]: pr[k][ta] for k in pr.files if k.startswith("comp_")}
    m, mc = mass["prior"], mass["cover"]
    cx, cy = tf * (cols + 0.5, rows + 0.5)

    # the query area must hold Track A and the margin
    lon, lat = TO_LL.transform(cx, cy)
    km_lat, km_lon = 111.2, 111.2 * np.cos(np.radians(lat.max()))
    assert lat.min() - MARGIN_KM / km_lat >= QUERY[0] and lat.max() + MARGIN_KM / km_lat <= QUERY[1], "query area too small in latitude"
    assert lon.min() - MARGIN_KM / km_lon >= QUERY[2] and lon.max() + MARGIN_KM / km_lon <= QUERY[3], "query area too small in longitude"

    off = (np.arange(SUB) + 0.5) / SUB
    ox, oy = np.meshgrid(off, off)
    px, py = tf * ((cols[:, None] + ox.ravel()[None, :]).ravel(), (rows[:, None] + oy.ravel()[None, :]).ravel())
    pmass = {k: np.repeat(v / SUB**2, SUB**2) for k, v in mass.items()}
    pcomp = {k: np.repeat(v / SUB**2, SUB**2) for k, v in comps.items()}

    def exposure(d: np.ndarray, r: float) -> dict:
        w = d <= r
        return dict(area_km2=round(float(w.sum()) * (cell / SUB / 1000) ** 2, 2), **{k: round(float(v[w].sum()), 6) for k, v in pmass.items()})

    # self-check: one receiver at a cell center
    one = pd.DataFrame(dict(x=[cx[0]], y=[cy[0]], u_km=[0.0]))
    d0 = distances(px[:SUB**2], py[:SUB**2], one)[0]
    assert (d0 <= 0.25 * cell / 1000).sum() < SUB**2 and (d0 <= 0.75 * cell / 1000).all(), "sub-cell exposure is wrong"

    rec, dropped = receivers()
    rec["km_to_track_a"] = cKDTree(np.c_[cx, cy]).query(np.c_[rec.x, rec.y])[0] / 1000
    rec["on_track_a"] = rec.km_to_track_a <= cell * np.sqrt(0.5) / 1000
    order = np.argsort(mc)[::-1]
    n_half = int(np.searchsorted(np.cumsum(mc[order]), 0.5)) + 1
    half = order[:n_half]
    rec["km_to_high_prior_half"] = cKDTree(np.c_[cx[half], cy[half]]).query(np.c_[rec.x, rec.y])[0] / 1000
    near = rec[rec.km_to_track_a <= NEAR_KM].sort_values("km_to_high_prior_half")
    near.drop(columns=["x", "y"]).round(dict(lat=3, lon=3, km_to_track_a=2, km_to_high_prior_half=2)).to_csv(HERE / "s9_receivers.csv", index=False)

    pas = rec[rec.cls == "passive"]
    dist = {"passive": distances(px, py, pas), "all": distances(px, py, rec)}

    exp = [dict(cls=cls, r_km=r, uncertainty=name, **exposure(d, r)) for cls, pair in dist.items() for name, d in zip(("no", "added"), pair) for r in RADII_KM]
    pd.DataFrame(exp).to_csv(HERE / "s9_exposure.csv", index=False)

    det = detection_ranges()
    det.to_csv(HERE / "s9_detection_range.csv", index=False)
    sens, info = sensitivity()
    out = []
    for row in sens.to_dict("records"):
        for cls, pair in dist.items():
            for name, d in zip(("no", "added"), pair):
                out.append(dict(**row, cls=cls, uncertainty=name, **exposure(d, row["range_km"])))
    sens_out = pd.DataFrame(out)
    sens_out.to_csv(HERE / "s9_sensitivity.csv", index=False)

    base = sens.query("clutter_level == 'cleanest_twentieth' and body == 'as_simulated' and attenuation == 'none'").range_km.iloc[0]
    largest_q = float(sens[sens.attenuation == "generous"].range_km.max())
    largest_any = float(sens.range_km.max())
    comp_rows = []
    for label, r in (("d83_row", base), ("largest_row_generous_attenuation", largest_q), ("largest_row_no_attenuation", largest_any)):
        w = dist["all"][1] <= r
        comp_rows += [dict(row=label, range_km=r, component=k, share_of_component=round(float(v[w].sum() / v.sum()), 6)) for k, v in pcomp.items()]
    pd.DataFrame(comp_rows).to_csv(HERE / "s9_components.csv", index=False)
    test = exposure(dist["all"][1], largest_q)
    reopened = any(test[k] > REOPEN for k in PRIORS)

    hlon, hlat = TO_LL.transform(cx[half], cy[half])
    d_half = cKDTree(np.c_[rec.x, rec.y]).query(np.c_[cx[half], cy[half]])[0] / 1000
    d_cells = cKDTree(np.c_[rec.x, rec.y]).query(np.c_[cx, cy])[0] / 1000
    hp = [dict(quantity="cells", value=n_half), dict(quantity="area_km2", value=round(n_half * (cell / 1000) ** 2, 1)),
          dict(quantity="share_of_track_a_cells", value=round(n_half / len(m), 5)),
          dict(quantity="lon_west", value=round(float(hlon.min()), 1)), dict(quantity="lon_east", value=round(float(hlon.max()), 1)),
          dict(quantity="lat_south", value=round(float(hlat.min()), 1)), dict(quantity="lat_north", value=round(float(hlat.max()), 1)),
          dict(quantity="nearest_receiver_km_min", value=round(float(d_half.min()), 2)), dict(quantity="nearest_receiver_km_median", value=round(float(np.median(d_half)), 2)),
          dict(quantity="receivers_within_10km", value=int((rec.km_to_high_prior_half <= 10).sum())), dict(quantity="receivers_within_3km", value=int((rec.km_to_high_prior_half <= 3).sum()))]
    pd.DataFrame(hp).to_csv(HERE / "s9_high_prior.csv", index=False)

    d83 = det[(det.table == "elastic") & (det.clutter_level == "p05")]
    checks = [dict(check="receivers_in_area", value=len(rec)), dict(check="passive_in_area", value=len(pas)),
              dict(check="active_in_area", value=int((rec.cls == "active").sum())), dict(check="scanned_positions_dropped_as_isolated", value=dropped),
              dict(check="passive_on_track_a", value=int(pas.on_track_a.sum())), dict(check="active_on_track_a", value=int(((rec.cls == "active") & rec.on_track_a).sum())),
              dict(check="track_a_cells", value=len(m)), dict(check="median_km_to_nearest_receiver", value=round(float(np.median(d_cells)), 2)),
              dict(check="d83_table_read_range_km", value=float(d83.range_m.max()) / 1000), dict(check="d83_row_range_km", value=base),
              dict(check="largest_range_generous_attenuation_km", value=largest_q), dict(check="largest_range_no_attenuation_km", value=largest_any),
              dict(check="typical_site_range_km", value=float(det[(det.table == "elastic") & (det.clutter_level == "median")].range_m.max()) / 1000),
              dict(check="reopen_threshold", value=REOPEN), dict(check="reopened", value=reopened)]
    checks += [dict(check=f"reopen_test_exposure_{k}", value=test[k]) for k in PRIORS]
    checks += [dict(check=k, value=v) for k, v in info.items()]
    for src, n in rec.assign(s=rec.sources.str.split("+")).explode("s").groupby("s").size().items():
        checks.append(dict(check=f"positions_from_{src}", value=int(n)))
    pd.DataFrame(checks).to_csv(HERE / "s9_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    print(sens_out.query("cls == 'all' and uncertainty == 'added'").drop(columns=["cls", "uncertainty", "body_factor"]).to_string(index=False))
    print(pd.DataFrame(hp).to_string(index=False))


if __name__ == "__main__":
    main()
