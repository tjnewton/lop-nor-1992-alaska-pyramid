"""P3 M5: background rates and cuts from the null peaks of the open packages (as refined).

Input: p3_g1b_run null outputs ($AKP_DATA/p3/g1b/<package>/<member>/null_{mag,em}.csv.gz, area.csv.gz),
p3_channels.csv (EPSG), p3_survey_inventory.csv (heights), and the P2a strata.
1. Declustering: peaks of one member within DECLUSTER_M of each other (any line) form one cluster,
   which keeps its largest score. A compact source seen on adjacent lines counts once.
2. System classes: a member's class is its sensor (mag or em) and its height band (HEIGHT_BANDS_M),
   from the height it was flown at as its altimeter measured it (p3_member_heights), or from
   the inventory where no measured height is on record.
3. Track A cut, per class:
   - controls are all open members of the class;
   - each cluster is weighted by w_s = (Track A line-km in stratum s for the class's Track A
     surveys) / (control line-km in s), as;
   - the tail fit and its bootstrap bound are p2e_cut.fit, expected and cut (weighted GPD,
     LEVEL upper bound), resampling BLOCK_KM spatial blocks;
   - the budget is the 0.1 times the class's share of all Track A line-km.
   The cut makes the bound on expected false candidates over the class's Track A area within its
   share.
4. Track B cut (self-calibrated): per class, all open clusters with unit weight, and the
   budget 0.1 times the class's share of all open line-km.
Track A flown area per class = land in (footprint and Track A), times the swath share is not used
here: false candidates can arise anywhere flown. Writes p3_null_rates.csv (one row per class and
track: n_clusters, k, u, sigma, xi, cut, e_point, e_bound, budget, and for Track A the share of its
flown area in strata without control) and p3_null_strata.csv (clusters
per line-km above THRESHOLDS per class and stratum).
Exposure is line-km: a peak is found where a line crosses a source, so counts scale with the
length of line flown, not with the area between lines. Flown km2 (line-km times spacing) made rates
depend on the line spacing. A member's line-km is its flown km2 over its spacing; a Track A survey's
is its footprint km2 over its inventory spacing. Members without a line spacing are left out.
Run: python -m akp.stage_p3.p3_null_rates
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import geometry_mask
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import mapping

from akp.paths import data_dir
from akp.stage_p2 import p2e_cut as pc
from akp.stage_p2.p2a_strata import label
from akp.stage_p3.p3_g0 import footprints
from akp.stage_p3.p3_g1b_run import spacing
from akp.stage_p3.p3_prior import out_dir

HERE = Path(__file__).parent
DECLUSTER_M = 300.0
# The top band ends at the largest clearance of the template bank (p3_mag_forward.CLEARANCES_M): surveys with a
# nominal height near its lower end fly well above it over rough ground. A member measured above the top
# band is in no class and is reported as such.
HEIGHT_BANDS_M = ((0.0, 60.0), (60.0, 150.0), (150.0, 300.0))
BLOCK_KM = 20.0
BUDGET = 0.1  # Family-wise budget
THRESHOLDS = (10.0, 20.0, 50.0, 100.0)
SEED = 20261113


def read_peaks(path: Path) -> pd.DataFrame:
    """A null, trials or area table, or an empty frame when the file is missing or holds no columns
    (a sensor the member lacks, or a stage that did not run for it)."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def band(h: float) -> str | None:
    for lo, hi in HEIGHT_BANDS_M:
        if lo < h <= hi or (lo == 0.0 and h == 0.0):
            return f"{lo:.0f}-{hi:.0f}"
    return None


def system_class(heights: pd.DataFrame, survey_id: str, sensor: str) -> str | None:
    """System class of a member: its sensor and the height band of the first of its survey ids that has
    a height in heights (the inventory indexed by survey_id). A package shared by several inventory
    rows lists its ids separated by ";"."""
    col = "mag_sensor_height_m" if sensor == "mag" else "em_bird_height_m"
    for sid in str(survey_id).split(";"):
        h = heights[col].get(sid.strip(), np.nan)
        if pd.notna(h) and band(float(h)):
            return f"{sensor}_{band(float(h))}"
    return None


def decluster(p: pd.DataFrame) -> pd.DataFrame:
    if len(p) < 2:
        return p.assign(cluster=np.arange(len(p)))
    xy = p[["x", "y"]].to_numpy()
    pairs = cKDTree(xy).query_pairs(DECLUSTER_M, output_type="ndarray")
    from scipy.sparse import coo_matrix
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p), len(p)))
    _, lab = connected_components(g, directed=False)
    p = p.assign(cluster=lab)
    return p.loc[p.groupby("cluster").score.idxmax()]


def load_null(ch: pd.DataFrame, sensor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters, areas = [], []
    for r in ch.itertuples():
        d = data_dir("p3/g1b") / Path(r.package).stem / Path(r.member).stem
        f, a = d / f"null_{sensor}.csv.gz", d / "area.csv.gz"
        if not f.exists() or not a.exists() or not np.isfinite(spacing(r)):
            continue
        pk = read_peaks(f)
        if len(pk):
            c = decluster(pk)
            clusters.append(c.assign(member=f"{r.package}/{r.member}", cls=r.cls))
        areas.append(pd.read_csv(a).assign(member=f"{r.package}/{r.member}", cls=r.cls, line_km=lambda x: x.km2 * 1000.0 / spacing(r)))
    return (pd.concat(clusters) if clusters else pd.DataFrame()), (pd.concat(areas) if areas else pd.DataFrame())


def track_a_area(inv: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Track A land km2 and line-km per stratum and class, inside the footprints of the Track A line surveys."""
    pr = np.load(out_dir() / "p3_prior.npz")
    land, transform, ta = pr["land"], Affine(*pr["transform"]), pr["track_a"]
    s = np.load(out_dir() / "p3_strata_grid.npz")["s"][land]
    inv = inv[(inv.intersects_track_a.astype(str) == "yes") & (inv.line_data_available == "yes")]
    fps = footprints(inv)
    rows = []
    for r in inv.itertuples():
        h = r.mag_sensor_height_m if sensor == "mag" else r.em_bird_height_m
        if r.survey_id not in fps or pd.isna(h) or pd.isna(r.line_spacing_m) or (sensor == "mag" and str(r.mag) != "1"):
            continue
        if sensor == "em" and not any(k in str(r.em_system) for k in ("DIGHEM", "RESOLVE")):
            continue
        b = band(float(h))
        if b is None:
            continue
        m = geometry_mask([mapping(fps[r.survey_id][0])], land.shape, transform, invert=True)[land] & ta
        for code, n in zip(*np.unique(s[m], return_counts=True)):
            rows.append(dict(cls=f"{sensor}_{b}", stratum=label(int(code)), km2=int(n), line_km=int(n) * 1000.0 / float(r.line_spacing_m)))
    d = pd.DataFrame(rows)
    return d.groupby(["cls", "stratum"])[["km2", "line_km"]].sum().reset_index() if len(d) else d


def main() -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv")
    hmap = inv.set_index("survey_id")
    from akp.stage_p3.p3_member_heights import load, member_class  # here: that module imports this one

    mh = load()
    rng = np.random.default_rng(SEED)
    rows, srows, uncovered = [], [], {}
    for sensor in ("mag", "em"):
        c2 = ch.assign(cls=[member_class(mh, hmap, r.package, r.member, r.survey_id, sensor) for r in ch.itertuples()]).dropna(subset=["cls"])
        clusters, areas = load_null(c2, sensor)
        if not len(clusters):
            continue
        ta = track_a_area(inv, sensor)
        ta_total = ta.line_km.sum() if len(ta) else 0.0
        open_total = areas.line_km.sum()
        for cl, cc in clusters.groupby("cls"):
            ar = areas[areas.cls == cl]
            ctl_km2 = ar.groupby("stratum").line_km.sum()  # exposure: line-km (the name is historical)
            blocks = (np.floor(cc.x / (BLOCK_KM * 1e3)).astype(int).astype(str) + "_" +
                      np.floor(cc.y / (BLOCK_KM * 1e3)).astype(int).astype(str) + "_" + cc.member)
            for t in THRESHOLDS:
                for st, km2 in ctl_km2.items():
                    srows.append(dict(cls=cl, stratum=st, threshold=t, line_km=round(float(km2), 1),
                                      clusters=int((cc[cc.stratum == st].score >= t).sum()),
                                      per_line_km=round(float((cc[cc.stratum == st].score >= t).sum() / km2), 5) if km2 else np.nan))
            tracks = {"B": (np.ones(len(cc)), BUDGET * ar.line_km.sum() / open_total)}
            if len(ta) and cl in set(ta.cls):
                tak = ta[ta.cls == cl].set_index("stratum").line_km
                w_s = (tak / ctl_km2).replace([np.inf], np.nan)
                # Track A strata with no control cannot be weighted. They take the mean rate of the strata that
                # have control: every weight is scaled up by their share of area (before, they counted as zero).
                covered = float(tak[tak.index.isin(ctl_km2[ctl_km2 > 0].index)].sum())
                uncovered[cl] = round(1 - covered / float(tak.sum()), 4)
                w = cc.stratum.map(w_s).fillna(0.0).to_numpy() * (float(tak.sum()) / covered if covered else 1.0)
                tracks["A"] = (w, BUDGET * tak.sum() / ta_total)
            for track, (w, budget) in tracks.items():
                res = pc.cut(cc.score.to_numpy(float), w, blocks.to_numpy(), rng, budget=budget)
                rows.append(dict(cls=cl, track=track, n_clusters=len(cc), budget=round(budget, 5),
                                 ta_share_without_control=uncovered.get(cl, np.nan) if track == "A" else np.nan,
                                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "boots"}))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "p3_null_rates.csv", index=False)
    pd.DataFrame(srows).to_csv(HERE / "p3_null_strata.csv", index=False)


if __name__ == "__main__":
    main()
