"""P3 descriptive Track A run. Every stage that reads sealed data calls p3_freeze.check().

Stages:
- inspect: for each sealed package of p3_fetch_lines.csv, print its data members, their column names
  (p3_lines.member_columns, or the CSV header) and the metadata lines naming a coordinate system.
  This informs p3_channels_trackA.csv, which is written after the freeze by the rules of
  p3_channels.py (data plumbing) and then checked with p3_channels.check.
- null, trials: p3_g1b_run.null_one and trials_one on the Track A rows (folder "sealed").
  Outputs go to $AKP_DATA/p3/trackA/g1b.
- report (the outputs; a lead is not a candidate):
  - clusters (p3_null_rates.decluster) whose cell lies in Track A (p3_prior track_a);
  - the lead threshold per system class: the median planted score of T2_high (magnetics) or
    T3_central (EM) in the open trials, pooled over geology classes (p3_lookalike.csv, q50_pooled);
  - p3_trackA_members.csv: every member and sensor, processed or skipped with the reason, with the
    height it was flown at (p3_member_heights) and the class the inventory would have given;
  - p3_trackA_leads.csv: lead counts and densities per system and geology class, beside the open-data
    look-alike density at the same threshold, the count it predicts (the transfer comparison), and the
    control line-km and cluster count that density rests on. Densities are per line-km;
  - p3_trackA_leadmap.csv: per system and RULE6_DEG cell (the cell of each 1 km grid cell's center), the
    flown km2, the leads, the look-alikes expected from the open-data density of each geology class,
    and the cell's Track A prior mass (with and without the cover likelihood);
  - p3_trackA_pd.csv: P_d by injection per member and target (class, and for magnetics the magnetization
    case: built or monolith), over all of the member's plants, both at the class's budgeted cut
    (p3_null_rates.csv, track A) and at the lead threshold. p_detect counts a hit when score >= t and
    score_bg < t (the body made the detection); p_at_or_above counts score >= t alone (the body's
    place would be flagged, with or without it);
  - p3_trackA_hide.csv: per target, threshold and prior, the searched mass sum p d, where d(cell) is the
    member's injected probability on the Track A cells it flew, capped at p3_search_map.D_CAP and
    combined over members as independent looks. At the budget cut the probability is p_detect:
    1 - searched mass is the share of the prior left hidden, and the posterior (p3_search_map) is
    saved to $AKP_DATA/p3/trackA/search_map. At the lead threshold it is p_at_or_above: the searched
    mass is the share of the prior that would sit among leads.
Run: python -m akp.stage_p3.p3_trackA_run inspect | validate | null | trials | report [--workers=N]
"""
from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer

from akp.paths import data_dir
from akp.stage_p3 import p3_g1b_run as run
from akp.stage_p3.p3_freeze import check
from akp.stage_p3.p3_lines import member_columns, open_member
from akp.stage_p3.p3_lookalike import geo
from akp.stage_p3.p3_member_heights import heights, member_class
from akp.stage_p3.p3_member_heights import load as load_heights
from akp.stage_p3.p3_null_rates import decluster, read_peaks, system_class
from akp.stage_p3.p3_prior import out_dir as prior_dir
from akp.stage_p3.p3_search_map import D_CAP

HERE = Path(__file__).parent
RULE6_DEG = 0.1
LEAD_CLASS = {"mag": "T2_high", "em": "T3_central"}
README_LINES = 120  # of each readme, printed by inspect: units, channels and projections are stated there
TARGETS = {"mag": ["T2_low", "T2_high"], "em": ["T3_opt", "T3_central"]}


def inspect() -> None:
    check()
    man = pd.read_csv(HERE / "p3_fetch_lines.csv").query("sealed").drop_duplicates("url")
    for r in man.itertuples():
        path = data_dir("p3/lines/sealed") / Path(r.path).name
        with open(path, "rb") as f:
            same = hashlib.file_digest(f, "sha256").hexdigest() == r.sha256
        print(f"=== {path.name}: sha256 {'matches the manifest' if same else 'DIFFERS from the manifest'}")
        if not same:
            raise SystemExit(f"{path.name} is not the package that was fetched")
        if not zipfile.is_zipfile(path):
            with open_member(path, "", allow_sealed=True) as f:
                print("   bare CSV header:", f.readline().decode("latin-1").strip()[:300])
            continue
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                mb = z.getinfo(n).file_size / 1e6
                if re.search(r"\.(xyz|csv|asc)$", n, re.I) and not re.search(r"grid|gxf", n, re.I):
                    if n.lower().endswith(".csv"):
                        with z.open(n) as f:
                            cols = f.readline().decode("latin-1").strip()[:300]
                    else:
                        cols = member_columns(z, n)
                    print(f"   member {n} ({mb:.0f} MB): {cols}")
                elif re.search(r"readme|read_me|_meta", n, re.I) and re.search(r"\.(txt|me)$", n, re.I):
                    print(f"   readme {n}:")
                    for line in z.read(n).decode("latin-1").splitlines()[:README_LINES]:
                        print("      " + line.rstrip()[:160])
                elif mb > 1:
                    print(f"   other {n} ({mb:.0f} MB)")
                if re.search(r"metadata/.*\.txt$", n, re.I):
                    t = z.read(n).decode("latin-1")
                    for line in t.splitlines():
                        if re.search(r"UTM_Zone_Number|Horizontal_Datum_Name|Grid_Coordinate_System_Name|Map_Projection_Name", line):
                            print("   meta:", line.strip()[:120])


def validate() -> None:
    """Check p3_channels_trackA.csv with p3_channels.check on the sealed packages; rewrite reader_ok and reason.

    Every sealed package of the manifest must have at least one row in the map (run or not, with a reason)."""
    from akp.stage_p3.p3_channels import check as check_row

    check()
    ch = pd.read_csv(HERE / "p3_channels_trackA.csv")
    want = {Path(p).name for p in pd.read_csv(HERE / "p3_fetch_lines.csv").query("sealed").path}
    missing = sorted(want - set(ch.package))
    if missing:
        raise SystemExit(f"sealed packages with no row in the Track A map: {missing}")
    # a row with no member is a placeholder for a package with nothing usable: its hand-written reason stands
    res = [("no", str(r.reason) if isinstance(getattr(r, "reason", None), str) and r.reason.strip() else "no usable member", "")
           if not (isinstance(r.member, str) and r.member.strip())
           else check_row(r, folder="p3/lines/sealed", allow_sealed=True) for r in ch.itertuples()]
    ch["reader_ok"], ch["reason"], line_col = zip(*res)
    ch["line_col"] = [lc or old for lc, old in zip(line_col, ch.get("line_col", pd.Series([""] * len(ch))).fillna(""))]
    ch.to_csv(HERE / "p3_channels_trackA.csv", index=False)
    print(ch[["package", "member", "reader_ok", "reason"]].to_string(index=False))


def rows() -> list[dict]:
    ch = pd.read_csv(HERE / "p3_channels_trackA.csv").query("reader_ok == 'yes'")
    out = [dict(r._asdict(), folder="sealed") for r in ch.itertuples()]
    dirs = [run.out_dir(SimpleNamespace(**r)) for r in out]
    if len(set(dirs)) != len(dirs):
        raise SystemExit("two Track A members share an output folder")
    return out


def not_run() -> list[dict]:
    """Map rows the reader check refused: they go into the members table with the reason."""
    ch = pd.read_csv(HERE / "p3_channels_trackA.csv").query("reader_ok != 'yes'")
    return [dict(member=f"{r.package}/{r.member}", sensor="", system=None, status=f"not run: {r.reason}", plants=0) for r in ch.itertuples()]


def report(test_rows: list[dict] | None = None, out_prefix: str = "p3_trackA") -> None:
    """The report. test_rows (open rows, no freeze check, no Track A restriction) exercise it before the freeze."""
    if test_rows is None or any(r.get("folder") == "sealed" for r in test_rows):
        check()
    pr, cv = np.load(prior_dir() / "p3_prior.npz"), np.load(prior_dir() / "p3_cover.npz")
    land, ta_mask = pr["land"], pr["track_a"]
    priors = {"W_cover": cv["mass_cover"].astype(float), "W_none": pr["mass"].astype(float)}
    land_idx = np.full(land.shape, -1)
    land_idx[land] = np.arange(land.sum())
    if test_rows is not None:  # open data lies outside Track A, where the prior is zero: a uniform prior exercises the hide table
        ta_mask, priors = np.ones_like(ta_mask), {"uniform_test": np.full(land.sum(), 1 / land.sum())}
    rr, cc = np.nonzero(land)
    to4326 = Transformer.from_crs("EPSG:3338", "EPSG:4326", always_xy=True)
    lon, lat = to4326.transform(*(Affine(*pr["transform"]) * (cc + 0.5, rr + 0.5)))
    cell01 = pd.DataFrame(dict(lon0=np.round(np.floor(lon / RULE6_DEG) * RULE6_DEG, 1), lat0=np.round(np.floor(lat / RULE6_DEG) * RULE6_DEG, 1)))

    def cells(d: pd.DataFrame) -> np.ndarray:
        """Land-cell index of each row of a table with row and col, or -1 off the land grid or outside Track A."""
        r, c = d.row.to_numpy(int), d.col.to_numpy(int)
        ok = (r >= 0) & (r < land.shape[0]) & (c >= 0) & (c < land.shape[1])
        i = np.full(len(d), -1)
        i[ok] = land_idx[r[ok], c[ok]]
        i[i >= 0] = np.where(ta_mask[i[i >= 0]], i[i >= 0], -1)
        return i

    look = pd.read_csv(HERE / "p3_lookalike.csv").query("plant_score == 'q50_pooled'")
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv").set_index("survey_id")
    cuts = pd.read_csv(HERE / "p3_null_rates.csv").query("track == 'A'").set_index("cls").c
    clusters, areas, pd_rows, member_rows, height_rows = [], [], [], [], []
    mh = load_heights()  # the open members' measured heights, for the test mode
    miss, flown = {}, {}  # (class, threshold label) -> per land cell: P(every look misses), and whether any member flew it
    for r in (test_rows if test_rows is not None else rows()):
        d = run.out_dir(SimpleNamespace(**r))
        member = f"{r['package']}/{r['member']}"
        area, tr = read_peaks(d / "area.csv.gz"), read_peaks(d / "trials.csv.gz")
        sp = run.spacing(SimpleNamespace(**r))
        if len(area):
            area = area.assign(cell=cells(area)).query("cell >= 0")
        h, unread = {}, ""
        if test_rows is None:  # a sealed member's class follows the height it was flown at
            try:
                h = heights(r)
            except Exception as e:  # one unreadable member must not stop the report
                h, unread = dict(mag_height_m=np.nan, em_height_m=np.nan), f"could not be read ({type(e).__name__}: {str(e)[:80]})"
            mh = pd.DataFrame([dict(package=r["package"], member=r["member"], **h)]).set_index(["package", "member"])
            height_rows.append(dict(member=member, **h, **{f"{s}_class_inventory": system_class(inv, r["survey_id"], s) for s in ("mag", "em")}))
        for sensor, cls_target in LEAD_CLASS.items():
            system = member_class(mh, inv, r["package"], r["member"], r["survey_id"], sensor)
            pk = read_peaks(d / f"null_{sensor}.csv.gz")
            lk = look[(look.cls == cls_target) & (look.system == system) & (look.geo == "all")]
            hm = h.get(f"{sensor}_height_m", np.nan) if test_rows is None else np.nan
            has = bool(isinstance(r.get("mag"), str) and r["mag"].strip()) if sensor == "mag" else bool(run.em_list(SimpleNamespace(**r)))
            why = (unread if unread else f"no {sensor} channels in the map" if not has
                   else f"measured height {hm:.0f} m is outside every height band" if system is None and pd.notna(hm)
                   else "no sensor height on record" if system is None
                   else f"no {sensor} peaks (the null stage did not run or found none)" if pk.empty
                   else "no flown area in Track A" if not len(area)
                   else "no open trials for this system class" if lk.empty else "")
            plants = tr[tr.cls.isin(TARGETS[sensor])] if len(tr) else tr
            member_rows.append(dict(member=member, sensor=sensor, system=system, status=f"skipped: {why}" if why else "processed",
                                    plants=len(plants), track_a_flown_km2=round(float(area.km2.sum()), 1) if len(area) else 0.0,
                                    track_a_line_km=round(float(area.km2.sum()) * 1000.0 / sp, 1) if len(area) and np.isfinite(sp) else 0.0))
            if why:
                continue
            thr = float(lk.threshold.iloc[0])
            cl = decluster(pk)
            cl = cl.assign(cell=cells(cl)).query("cell >= 0")
            clusters.append(cl.assign(system=system, lead=cl.score >= thr)[["system", "stratum", "lead", "cell"]])
            areas.append(area.assign(system=system, line_km=area.km2 * 1000.0 / run.spacing(SimpleNamespace(**r))))
            if len(plants):  # magnetic plants split by magnetization case: q = 0 is "built", q > 0 "monolith"
                q = plants.q.fillna(0) if sensor == "mag" and "q" in plants else None
                plants = plants.assign(target=plants.cls if q is None else plants.cls + np.where(q > 0, "_monolith", "_built"))
            for cname, g in (plants.groupby("target") if len(plants) else ()):
                sc, s0 = g.score.fillna(-np.inf), g.score_bg.fillna(-np.inf)
                for label, t in (("budget_cut", float(cuts.get(system, np.inf))), ("lead_threshold", thr)):
                    p, p_at = float(((sc >= t) & (s0 < t)).mean()), float((sc >= t).mean())
                    pd_rows.append(dict(member=member, system=system, cls=cname, at=label, threshold=round(t, 2), plants=len(g),
                                        p_detect=round(p, 4), p_at_or_above=round(p_at, 4)))
                    dcell = min(p if label == "budget_cut" else p_at, D_CAP)
                    miss.setdefault((cname, label), np.ones(land.sum()))[area.cell.unique()] *= 1 - dcell
                    flown.setdefault((cname, label), np.zeros(land.sum(), bool))[area.cell.unique()] = True
    dest = HERE if test_rows is None else data_dir("p3/trackA/test")
    members = pd.DataFrame(member_rows + (not_run() if test_rows is None else []))
    if height_rows:
        members = members.merge(pd.DataFrame(height_rows), on="member", how="left")
    members.to_csv(dest / f"{out_prefix}_members.csv", index=False)
    print(members.to_string(index=False))
    if not clusters:
        raise SystemExit("no Track A member was processed")
    cl, ar = pd.concat(clusters).join(cell01, on="cell"), pd.concat(areas).join(cell01, on="cell")
    for t in (cl, ar):
        t["geo"] = t.stratum.map(geo)
    dens = {}  # (system, geology class) -> open-data look-alikes per 1000 line-km at the lead threshold
    for row in look.itertuples():
        if row.cls == LEAD_CLASS[row.system.split("_")[0]]:
            dens[row.system, row.geo] = row.lookalikes_per_1000_line_km
    ar["expected"] = [km * (dens[sy, g] if pd.notna(dens.get((sy, g), np.nan)) else dens[sy, "all"]) / 1000
                      for km, sy, g in zip(ar.line_km, ar.system, ar.geo)]
    lead_rows = []
    for system in sorted(set(ar.system)):
        for g in sorted(set(ar[ar.system == system].geo)) + ["all"]:
            c, a = (t[(t.system == system) & ((t.geo == g) | (g == "all"))] for t in (cl, ar))
            km = float(a.line_km.sum())
            lk = look[(look.cls == LEAD_CLASS[system.split("_")[0]]) & (look.system == system)]
            ctl = lk[lk.geo == g]
            lead_rows.append(dict(system=system, geo=g, lead_threshold=float(lk[lk.geo == "all"].threshold.iloc[0]),
                                  track_a_flown_km2=round(float(a.km2.sum()), 1), track_a_line_km=round(km, 1),
                                  clusters=len(c), leads=int(c.lead.sum()),
                                  leads_per_1000_line_km=round(c.lead.sum() / km * 1000, 2) if km else np.nan,
                                  open_lookalikes_per_1000_line_km=dens.get((system, g), np.nan),
                                  open_control_line_km=float(ctl.control_line_km.iloc[0]) if len(ctl) else 0.0,
                                  open_control_clusters=int(ctl.control_clusters.iloc[0]) if len(ctl) else 0,
                                  expected_lookalikes=round(float(a.expected.sum()), 1)))
    pd.DataFrame(lead_rows).to_csv(dest / f"{out_prefix}_leads.csv", index=False)
    key = ["system", "lon0", "lat0"]
    mass01 = pd.DataFrame({f"prior_mass_{k}": m for k, m in priors.items()}).join(cell01).groupby(["lon0", "lat0"]).sum().round(6)
    lmap = pd.concat([ar.groupby(key).km2.sum().round(1).rename("flown_km2"), cl[cl.lead].groupby(key).size().rename("leads"),
                      ar.groupby(key).expected.sum().round(1).rename("expected_lookalikes")], axis=1)
    lmap = lmap.fillna({"leads": 0}).astype({"leads": int}).reset_index().join(mass01, on=["lon0", "lat0"])
    lmap.to_csv(dest / f"{out_prefix}_leadmap.csv", index=False)
    pd.DataFrame(pd_rows).to_csv(dest / f"{out_prefix}_pd.csv", index=False)
    hide_rows = []
    for (cname, label), m in sorted(miss.items()):
        dcell = 1 - m
        for prior, mass in priors.items():
            found, flown_mass = float((mass * dcell).sum()), float(mass[flown[cname, label]].sum())
            assert 0 <= found <= flown_mass + 1e-9
            hide_rows.append(dict(cls=cname, at=label, prior=prior, flown_prior_mass=round(flown_mass, 4), searched_mass=round(found, 4)))
            if label == "budget_cut" and test_rows is None:
                np.savez_compressed(data_dir("p3/trackA/search_map") / f"{cname}_{prior}.npz", d=dcell.astype(np.float32),
                                    posterior=(mass * m / (1 - found)).astype(np.float32))
    pd.DataFrame(hide_rows).to_csv(dest / f"{out_prefix}_hide.csv", index=False)
    for name in ("leads", "pd", "hide"):
        print(read_peaks(dest / f"{out_prefix}_{name}.csv").to_string(index=False))


def main(argv: list[str]) -> None:
    stage = argv[0]
    workers = int(argv[1].split("=")[1]) if len(argv) > 1 and argv[1].startswith("--workers=") else 2
    if stage == "inspect":
        inspect()
    elif stage == "validate":
        validate()
    elif stage == "null":
        check()
        run.run_pool(rows(), run.null_one, workers)
    elif stage == "trials":
        check()
        run.run_pool(rows(), run.trials_one, workers)
    elif stage == "report":
        report()
    elif stage == "report-test":  # before the freeze: open packages named by substring, no Track A restriction
        ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
        ch = ch[ch.package.apply(lambda p: any(s in p for s in argv[1:]))]
        report([r._asdict() for r in ch.itertuples()])
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main(sys.argv[1:])
