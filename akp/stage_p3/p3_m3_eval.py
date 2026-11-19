"""P4 M3: does a multi-line, multi-family fit cut the look-alike density at half detection?

Inputs (development packages only: p3_vault.csv rows with vault False, classes CLASSES; load() asserts it):
p3_channels.csv; the line cache of p3_g1b_run; null_mag.csv.gz and area.csv.gz of p3_g1b_run.
Stage one is the detector, unchanged (p3_mag_detect). Stage two is p3_m3_fit.

Stage run, per member:
- negatives: the member's declustered null clusters (p3_null_rates.decluster) with score >= FLOOR,
  sampled without replacement inside score bins BINS (quota NULL_QUOTA per class and bin, at least
  MIN_PER_BIN per member). weight = clusters in the member's bin / clusters sampled, so that
  sum(weight) over a set of sampled clusters estimates the number of clusters in the population.
  Every cluster's score goes to pop.csv.gz, so the detector-only density needs no weights;
- positives: PLANT_DRAWS draws per package (shared among its members by flown area, among chunks by
  samples), each drawn as p3_mag_trials.trials draws (line sample, cross-line offset, turn, remanent
  direction, base, top, and with probability one half q in p3_g1b_run.Q_RANGE, else q = 0) and
  planted at both T2 contrasts with p3_mag_inject.plant_lines into the lines within its reach. The
  detector score is the largest within HIT_R_M of the plant (score_near, equal to
  p3_mag_trials.max_score_near, which it also locates). Stage two runs at that peak position, on
  the planted lines (kind plant) and on the unplanted lines at the same place (kind plant_bg);
- control shapes (M5): CONTROL_DRAWS per package and kind (dike, plug, dipole; the alternatives of
  p3_identifiability with their parameters drawn off the fit's grid), induced, scaled to the peak of
  a pyramid drawn from the prior at the same place, through the same scoring and stage two.
Writes $AKP_DATA/p3/m3/desc/<package>/<member>.csv.gz (one row per fitted place), pop.csv.gz and
run_log.csv. A member with an output is skipped. Each member runs in a fresh worker process (two by
default; see the memory note in run_stage).

Stage report: two lead statistics per system class and case (built: q = 0; monolith: q > 0), each
made with leave-one-survey-out folds (group = package), so every row is scored by a rule that never
saw its package:
- rule: windows on the pyramid fit's top, base and contrast (the central 1 - 2 GATE_Q of the training
  plants), field angle (below their 1 - 2 GATE_Q quantile), family margin and R2 (above their
  2 GATE_Q quantile); a place inside every window keeps its detector score, any other gets none;
- gbm: sklearn HistGradientBoostingClassifier on FEATS (the detector score and every descriptor; no
  survey property), plants (each package's plants carrying its flown area) against weighted null
  clusters; the statistic is its decision function.
For each class, case, contrast (all, T2_low, T2_high) and spacing group:
(a) detector alone: null clusters per 1000 km2 with score >= the plants' median score (misses count
    as zero), as p3_lookalike;
(b) M3: null clusters per 1000 km2 with statistic >= the plants' median statistic (a plant below
    FLOOR or unfitted has none). If fewer than half of the plants have a statistic, half detection
    is out of reach and the row says so.
ratio = (a) / (b); interval: percentile bootstrap over packages (N_BOOT); verdict by success at
RATIO_SUCCESS, drop below RATIO_DROP. Nothing here raises a candidate.
The pooled ratio can be won between surveys: plants are equal in number per package and null clusters
are not, so a statistic that only ranks surveys (quiet above busy) scores well without telling a body
from a natural source. ratio_within removes that: both thresholds are set inside each package (the
median of its own plants), the passing clusters are summed over packages, and a package where the
statistic cannot keep half of its plants counts with the detector alone. verdict_within reads it.
Writes beside this file: p3_m3_eval.csv (the headline table), p3_m3_folds.csv (per held-out package,
at the pooled thresholds), p3_m3_families.csv (per class and kind of place: the share each family
fits best, and the share passing each half-detection threshold of the built case),
p3_m3_descriptors.csv (per descriptor: weighted AUC of plants against null clusters, pooled, as the
median of the AUCs inside each package, and pooled over places at or above the detector's lead
threshold), p3_m3_checks.csv.
Self-check (stage check): the light chunk reader equals p3_g1b_run.frame; score_near equals
p3_mag_trials.max_score_near; a vault package is refused.
Run: python -m akp.stage_p3.p3_m3_eval check | run [--workers=N] [--scale=F] [package-substring ...] | report
"""
from __future__ import annotations

import os
import resource
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from akp.paths import data_dir
from akp.stage_p3 import p3_g1b_run as run
from akp.stage_p3 import p3_m3_fit as m3
from akp.stage_p3 import p3_mag_detect as det
from akp.stage_p3 import p3_mag_fast as fast
from akp.stage_p3 import p3_mag_inject as inj
from akp.stage_p3 import p3_mag_trials as mtr
from akp.stage_p3.p3_mag_forward import HEIGHT_M
from akp.stage_p3.p3_null_rates import decluster, read_peaks
from akp.stage_p3.p3_vault import in_vault

HERE = Path(__file__).parent
CLASSES = ("mag_0-60", "mag_60-150")
SEED = 20261114
FLOOR = 10.0  # detector score below which no place gets a statistic; most plants score above it
BINS = (10.0, 15.0, 20.0, 30.0, 50.0, 100.0, np.inf)
NULL_QUOTA = (900, 1000, 1300, 1600, 1400, 800)  # per class and bin
MIN_PER_BIN = 4
PLANT_DRAWS = 150  # per package; each draw is planted at both contrasts
CONTROL_DRAWS = 16  # per package and control kind
CONTROLS = ("dike", "plug", "dipole")
GATE_Q = 0.025
N_BOOT = 1000
RATIO_SUCCESS, RATIO_DROP = 10.0, 3.0
FORBIDDEN = ("p3/lines/sealed", "p3/trackA", "vector_sealed")
FEATS = ["score", "r2_pyr", "r2_plug", "r2_dike", "r2_dip", "r2_pyr_ind", "margin", "margin_plug", "margin_dike", "margin_dip",
         "pyr_base", "pyr_top", "log_k", "pyr_kind", "pyr_angle", "k_ind", "ind_base", "ind_top", "n_lines", "n_lines_reach",
         "share_top_line", "share_top_line_data", "peak_nT", "log_rv_pyr", "log_tss", "pyr_off"]
CONTEXT = ["clr_med", "spacing"]  # survey properties, not descriptors: kept out of FEATS, listed in p3_m3_descriptors.csv to show why
MIN_PLANTS = 10  # a package needs this many plants for a threshold of its own


def members() -> pd.DataFrame:
    """Development members of CLASSES: the channel map joined to the vault table."""
    v = pd.read_csv(HERE / "p3_vault.csv")
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    m = ch.merge(v[["package", "member", "mag_class", "vault"]], on=["package", "member"])
    return m[(~m.vault) & m.mag_class.isin(CLASSES)].reset_index(drop=True)


def assert_open(row) -> None:
    """The vault rule: no vault package, and no path under the sealed roots, is read."""
    vault = set(pd.read_csv(HERE / "p3_vault.csv").query("vault == True").package)
    assert not in_vault(row.package) and row.package not in vault, f"{row.package} is in the vault"
    assert getattr(row, "folder", "open") != "sealed", "sealed rows are not development data"
    for d in (run.cache_dir(row), run.out_dir(row)):
        assert not any(f in d.as_posix() for f in FORBIDDEN), f"{d} is under a sealed root"


def load(path: Path, row) -> pd.DataFrame:
    """A cache chunk as p3_g1b_run.frame gives it, with only the columns the magnetics use."""
    assert_open(row)
    z = np.load(path)
    cols = {"X": row.x.strip().upper(), "Y": row.y.strip().upper(), row.mag.strip().upper(): row.mag.strip().upper(),
            row.alt.strip().upper(): row.alt.strip().upper()}
    if isinstance(row.fid, str) and row.fid:
        cols["FID"] = row.fid.strip().upper()
    df = pd.DataFrame({"line": z["line"], **{k: z[v] for k, v in cols.items()}})
    df["kind"] = pd.Categorical.from_codes(z["is_tie"].astype(np.int8), ["line", "tie"])  # compares as the strings of frame(), at a byte per row
    return df


def chunks(row) -> list[Path]:
    assert_open(row)
    assert (run.cache_dir(row) / "complete").exists(), f"{row.package}: no complete cache (this module does not build one)"
    return run.ensure_cache(row)


class Chunk:
    """One chunk by line: every line's rows in file order (ties included, for planting), their boxes,
    the traverse lines (as p3_mag_detect.survey_lines) and the plant pool."""

    def __init__(self, df: pd.DataFrame, mag: str):
        self.keep = set(det.survey_lines(df))
        self.all = {ln: g for ln, g in df.groupby("line", sort=False)}
        self.ids = list(self.all)
        self.box = np.array([[g.X.min(), g.X.max(), g.Y.min(), g.Y.max()] for g in self.all.values()]).reshape(-1, 4)
        pool = df[(df.kind == "line") & df[mag].notna() & df.line.isin(self.keep)]  # as p3_mag_trials.trials
        self.px, self.py = pool.X.to_numpy(), pool.Y.to_numpy()
        self._trav: dict = {}

    def near(self, x0: float, y0: float, r: float) -> list:
        b = self.box
        return [self.ids[i] for i in np.flatnonzero((x0 >= b[:, 0] - r) & (x0 <= b[:, 1] + r) & (y0 >= b[:, 2] - r) & (y0 <= b[:, 3] + r))]

    def trav(self, ids) -> dict:
        """The traverse rows of those of ids that are traverse lines, as df[(df.line == ln) & (df.kind == "line")]."""
        for ln in ids:
            if ln in self.keep and ln not in self._trav:
                g = self.all[ln]
                self._trav[ln] = g if (g.kind == "line").all() else g[g.kind == "line"]  # no copy where the line has no tie rows
        return {ln: self._trav[ln] for ln in ids if ln in self.keep}

    def has(self, x: float, y: float) -> bool:
        return any(np.nanmin(np.hypot(g.X.to_numpy() - x, g.Y.to_numpy() - y)) < 25.0 for g in self.trav(self.near(x, y, 25.0)).values())


def planted_lines(out: pd.DataFrame, keep: set) -> dict:
    return {ln: g for ln, g in out[out.kind == "line"].groupby("line", sort=False) if ln in keep}


def score_near(lines: dict, mag: str, alt: str, feet: bool, x0: float, y0: float) -> tuple[float, float, float]:
    """p3_mag_trials.max_score_near over per-line frames, with the position of that score."""
    best, bx, by = np.nan, np.nan, np.nan
    for line in lines.values():
        h = float(np.nanmedian(line[alt])) * (inj.FT_M if feet else 1.0)
        tbank = det.templates(h, det.line_heading(line))
        for _, v, x, y in det.segments(line, mag):
            near = np.hypot(x - x0, y - y0) <= mtr.HIT_R_M
            if not near.any():
                continue
            sc = det.trim_edges(det.score(det.regional_residual(v), tbank)[0])
            if not np.isfinite(sc[near]).any():
                continue
            i = np.flatnonzero(near)[np.argmax(sc[near])]
            if not np.isfinite(best) or sc[i] > best:
                best, bx, by = float(sc[i]), float(x[i]), float(y[i])
    return best, bx, by


def stage_two(lines: dict, mag: str, alt: str, feet: bool, x: float, y: float, spacing: float) -> dict:
    frames = [g.sort_values("FID") if "FID" in g else g for g in lines.values()
              if (np.hypot(g.X.to_numpy() - x, g.Y.to_numpy() - y) <= m3.REACH_M).any()]
    return m3.fit(m3.collect(frames, mag, alt, feet, x, y), spacing)


def control_field(kind: str, xr, yr, h, top: float, par: tuple) -> np.ndarray:
    """Induced TMI at unit strength of a control body below the origin (p3_identifiability's alternatives)."""
    if kind == "dipole":
        r = np.c_[xr, yr, h + par[0]]
        rn = np.linalg.norm(r, axis=1)
        return (3 * (r @ m3.B0) ** 2 / rn**2 - 1.0) / rn**3
    half_x, half_y, extent, turn = (m3.DIKE_LENGTH_M / 2, par[0] / 2, HEIGHT_M, par[1]) if kind == "dike" else (par[0] / 2, par[0] / 2, par[1], par[2])
    rot = m3._rot(turn)
    u = fast.prism_tensor(np.c_[xr, yr, h] @ rot.T, np.array([[-half_x, -half_y, -(top + extent)]]), np.array([[half_x, half_y, -top]]))
    return np.einsum("nij,j->ni", u, rot @ m3.B0) @ (rot @ m3.B0)


def plant_control(df: pd.DataFrame, mag: str, alt: str, feet: bool, x0: float, y0: float, kind: str, top: float, par: tuple,
                  pyramid: tuple) -> tuple[pd.DataFrame, np.ndarray, float]:
    """As p3_mag_inject.plant_lines, for a control body scaled to the peak of the pyramid (base, top, turn, dk) there."""
    near = np.flatnonzero((np.hypot(df.X.to_numpy() - x0, df.Y.to_numpy() - y0) <= inj.REACH_M)
                          & np.isfinite(df[mag].to_numpy()) & np.isfinite(df[alt].to_numpy()))
    out, amp = df.copy(), np.nan
    if near.size:
        h = inj.clean_heights(df[alt].to_numpy()[near] * (inj.FT_M if feet else 1.0))
        xr, yr = df.X.to_numpy()[near] - x0, df.Y.to_numpy()[near] - y0
        amp = float(np.abs(fast.response(xr, yr, h, *pyramid)).max())
        f = control_field(kind, xr, yr, h, top, par)
        col = out.columns.get_loc(mag)
        out.iloc[near, col] = out.iloc[near, col].to_numpy() + f * amp / np.abs(f).max()
    return out, near, amp


def member_run(job: dict) -> dict:
    t0 = time.time()
    row = SimpleNamespace(**job["row"])
    assert_open(row)
    mag, alt, feet = row.mag.strip().upper(), row.alt.strip().upper(), str(row.alt_units).lower().startswith("f")
    sp = run.spacing(row)
    nulls = pd.DataFrame(job["nulls"])
    paths = chunks(row)
    sizes = np.array([np.load(p)["line"].size for p in paths], float)
    share = sizes / sizes.sum()
    rows, done = [], np.zeros(len(nulls), bool)
    for k, path in enumerate(paths):
        ch = Chunk(load(path, row), mag)
        if not ch.ids:
            continue
        for j, c in enumerate(nulls.itertuples()):
            if done[j] or not ch.has(c.x, c.y):
                continue
            done[j] = True
            lines = ch.trav(ch.near(c.x, c.y, m3.REACH_M))
            rows.append(dict(kind="null", pid=j, score=c.score, x=c.x, y=c.y, weight=c.weight, stratum=c.stratum,
                             **stage_two(lines, mag, alt, feet, c.x, c.y, sp)))
        if len(ch.px) < 3:
            continue
        rng = np.random.default_rng([SEED, zlib.crc32(f"{row.package}/{row.member}".encode()), k])
        for kind, n in [("plant", int(round(job["plants"] * share[k])))] + [(c, int(round(job["controls"] * share[k]))) for c in CONTROLS]:
            for j in range(n):
                i = int(rng.integers(1, len(ch.px) - 1))
                hd = np.arctan2(ch.py[i + 1] - ch.py[i - 1], ch.px[i + 1] - ch.px[i - 1])
                u = float(rng.uniform(-sp / 2, sp / 2))
                x0, y0 = ch.px[i] - u * np.sin(hd), ch.py[i] + u * np.cos(hd)
                turn = float(rng.uniform(0, 90))
                rem = (float(np.degrees(np.arcsin(rng.uniform(-1, 1)))), float(rng.uniform(0, 360)))
                base, top = mtr.draw(rng, run.BASE_M), mtr.draw(rng, run.TOP_M)
                q = float(rng.uniform(*run.Q_RANGE)) if rng.random() < 0.5 else 0.0
                ids = ch.near(x0, y0, inj.REACH_M)
                truth = dict(pid=f"{k}_{kind}_{j}", x0=x0, y0=y0, u=u, turn=turn, base=base, top=top)
                if not ids:
                    rows.append(dict(kind=kind, score=np.nan, **truth))
                    continue
                sub = pd.concat([ch.all[ln] for ln in ids])  # ties included, so that the plant is exactly that of the whole chunk
                if kind == "plant":
                    todo = [(cls, dict(q=q, rem_inc=rem[0], rem_dec=rem[1], dk=dk, contrast=cls), (base, top, turn, dk, q, rem)) for cls, dk in run.MAG_CLASSES.items()]
                else:
                    dk = float(rng.choice(list(run.MAG_CLASSES.values())))
                    par = {"dike": (float(rng.uniform(20, 60)), float(rng.uniform(0, 180))),
                           "plug": (float(rng.uniform(100, 300)), float(rng.uniform(60, HEIGHT_M)), turn),
                           "dipole": (top + HEIGHT_M * float(rng.uniform(0.3, 0.9)),)}[kind]
                    todo = [(kind, dict(q=0.0, dk=dk, par=repr(par)), (kind, top, par, (base, top, turn, dk)))]
                bg = None
                for _, extra, args in todo:
                    if kind == "plant":
                        out, near = inj.plant_lines(sub, mag, alt, feet, x0, y0, *args)
                    else:
                        out, near, extra["amp_nT"] = plant_control(sub, mag, alt, feet, x0, y0, *args)
                    touched = [ln for ln in sub.line.iloc[near].unique() if ln in ch.keep]
                    if bg is None:
                        bg = score_near(ch.trav(touched), mag, alt, feet, x0, y0)[0] if touched else np.nan
                    planted = planted_lines(out, ch.keep)
                    s, x, y = score_near({ln: planted[ln] for ln in touched}, mag, alt, feet, x0, y0) if touched else (np.nan,) * 3
                    rec = dict(kind=kind, score=s, score_bg=bg, x=x, y=y, **truth, **extra)
                    if not np.isfinite(s):
                        rows.append(rec)
                        continue
                    rows.append({**rec, **stage_two(planted, mag, alt, feet, x, y, sp)})
                    if kind == "plant":
                        rows.append({**rec, "kind": "plant_bg", **stage_two(ch.trav(ids), mag, alt, feet, x, y, sp)})
        del ch
    d = pd.DataFrame(rows).assign(package=row.package, member=row.member, cls=row.mag_class, spacing=sp)
    out = desc_path(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    d.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(out)
    return dict(package=row.package, member=row.member, rows=len(d), nulls_wanted=len(nulls), nulls_found=int(done.sum()),
                seconds=round(time.time() - t0, 1), maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 3))


def desc_path(row) -> Path:
    return data_dir("p3/m3") / "desc" / Path(row.package).stem / (Path(row.member).stem + ".csv.gz")


def null_sample(ms: pd.DataFrame, scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(population, sample): every declustered null cluster with score >= FLOOR, and the stratified sample with weights."""
    pops = []
    for r in ms.itertuples():
        assert_open(r)
        pk = read_peaks(run.out_dir(r) / "null_mag.csv.gz")
        if len(pk):
            c = decluster(pk)
            pops.append(c[c.score >= FLOOR].assign(package=r.package, member=r.member, cls=r.mag_class))
    pop = pd.concat(pops, ignore_index=True)
    pop["bin"] = np.digitize(pop.score, BINS) - 1
    picks = []
    for (cls, b), g in pop.groupby(["cls", "bin"]):
        p = min(1.0, scale * NULL_QUOTA[b] / len(g))
        for (pkg, mem), gm in g.groupby(["package", "member"]):
            rng = np.random.default_rng([SEED, zlib.crc32(f"{pkg}/{mem}".encode()), int(b)])
            n = min(len(gm), max(MIN_PER_BIN, int(round(p * len(gm)))))
            picks.append(gm.loc[rng.choice(gm.index.to_numpy(), n, replace=False)].assign(weight=len(gm) / n))
    return pop, pd.concat(picks)


def run_stage(args: list[str]) -> None:
    workers, scale = 2, 1.0
    while args and args[0].startswith("--"):
        key, val = args.pop(0).split("=")
        workers, scale = (int(val), scale) if key == "--workers" else (workers, float(val))
    ms = members()
    pop, sample = null_sample(ms, scale)
    pop[["package", "member", "cls", "score", "bin"]].to_csv(data_dir("p3/m3") / "pop.csv.gz", index=False)
    km2 = {(r.package, r.member): float(pd.read_csv(run.out_dir(r) / "area.csv.gz").km2.sum()) for r in ms.itertuples()}
    pkg_km2 = pd.Series(km2).groupby(level=0).sum()
    jobs = []
    for r in ms.itertuples(index=False):
        if args and not any(a in r.package for a in args):
            continue
        if desc_path(r).exists():
            continue
        f = km2[(r.package, r.member)] / pkg_km2[r.package]
        s = sample[(sample.package == r.package) & (sample.member == r.member)]
        jobs.append(dict(row=r._asdict(), nulls=s[["x", "y", "score", "weight", "stratum"]].to_dict("list"),
                         plants=max(2, round(scale * PLANT_DRAWS * f)), controls=max(1, round(scale * CONTROL_DRAWS * f)),
                         size=sum(p.stat().st_size for p in chunks(r))))
    # Memory (measured 2026-11-13): a worker's resident size climbs to 2.3 to 2.9 GB on members of two to
    # four million rows, most of it freed blocks the allocator keeps, and a worker that goes on to a second
    # member keeps the first one's size. So every member gets a fresh process, the largest member starts
    # first and the smallest ones run beside it, and the BLAS stays single-threaded.
    jobs.sort(key=lambda j: j["size"])
    jobs = jobs[-1:] + jobs[:-1]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"{len(jobs)} members to run with {workers} workers; null sample {len(sample)} of {len(pop)} clusters", flush=True)
    m3.build_tables()  # before the pool, so that no worker builds its own
    log = data_dir("p3/m3") / "run_log.csv"
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:
        for fu in as_completed([ex.submit(member_run, j) for j in jobs]):
            res = fu.result()
            pd.DataFrame([res]).to_csv(log, mode="a", header=not log.exists(), index=False)
            print(res, flush=True)


# ---------------------------------------------------------------------------------------------- report

def read_desc() -> pd.DataFrame:
    parts = []
    for r in members().itertuples():
        assert_open(r)
        if desc_path(r).exists():
            parts.append(pd.read_csv(desc_path(r), low_memory=False))
    d = pd.concat(parts, ignore_index=True)
    d["kind"] = d.kind.fillna("null")  # read_csv takes the string "null" for a missing value
    d["score"] = np.where(d.kind == "plant_bg", d.score_bg, d.score)  # the unplanted place has the unplanted score
    d["fitted"] = (d.n.fillna(0) > 0) & (d.score >= FLOOR)
    d["case"] = np.where(d.kind.isin(["plant", "plant_bg"]), np.where(d.q > 0, "monolith", "built"), "")
    d["log_k"], d["log_rv_pyr"], d["log_tss"] = np.log10(d.pyr_k.clip(lower=1e-6)), np.log10(d.rv_pyr.clip(lower=1e-6)), np.log10(d.tss_var.clip(lower=1e-6))
    d["sgroup"] = np.where(d.spacing < 400, "lt400", np.where(d.spacing > 400, "gt400", "400"))
    return d


def gates(train: pd.DataFrame) -> dict:
    q = train[["pyr_top", "pyr_base", "log_k", "pyr_angle", "margin", "r2_pyr"]].quantile([GATE_Q, 2 * GATE_Q, 1 - 2 * GATE_Q, 1 - GATE_Q])
    lo, lo1, hi1, hi = (q.loc[v] for v in (GATE_Q, 2 * GATE_Q, 1 - 2 * GATE_Q, 1 - GATE_Q))
    return dict(pyr_top=(lo.pyr_top, hi.pyr_top), pyr_base=(lo.pyr_base, hi.pyr_base), log_k=(lo.log_k, hi.log_k),
                pyr_angle=(-np.inf, hi1.pyr_angle), margin=(lo1.margin, np.inf), r2_pyr=(lo1.r2_pyr, np.inf))


def rule_stat(d: pd.DataFrame, g: dict) -> np.ndarray:
    ok = d.fitted.to_numpy().copy()
    for col, (a, b) in g.items():
        ok &= (d[col].to_numpy() >= a) & (d[col].to_numpy() <= b)
    return np.where(ok, d.score.to_numpy(), -np.inf)


def statistics(d: pd.DataFrame, km2: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-survey-out lead statistics: d with columns rule_<case> and gbm_<case> for every row, and the rule's windows per fold."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    out, grows = d.copy(), []
    for case in ("built", "monolith"):
        out[f"rule_{case}"], out[f"gbm_{case}"] = -np.inf, -np.inf
        for cls in CLASSES:
            c = d[d.cls == cls]
            pos = c[(c.kind == "plant") & (c.case == case) & c.fitted]
            neg = c[(c.kind == "null") & c.fitted]
            for pkg in sorted(c.package.unique()):
                test = c[c.package == pkg]
                tp, tn = pos[pos.package != pkg], neg[neg.package != pkg]
                g = gates(tp)
                grows.append(dict(cls=cls, case=case, held_out=pkg, **{f"{k}_{s}": round(float(v[i]), 4) for k, v in g.items() for i, s in enumerate(("lo", "hi"))}))
                out.loc[test.index, f"rule_{case}"] = rule_stat(test, g)
                X = pd.concat([tp, tn])[FEATS]
                wp = (tp.package.map(km2) / tp.package.map(tp.package.value_counts())).to_numpy()  # each package's plants carry its flown area
                w = np.r_[wp * tn.weight.sum() / wp.sum(), tn.weight.to_numpy()]
                model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8, min_samples_leaf=20,
                                                       l2_regularization=1.0, random_state=0)
                model.fit(X, np.r_[np.ones(len(tp)), np.zeros(len(tn))], sample_weight=w)
                out.loc[test.index, f"gbm_{case}"] = np.where(test.fitted, model.decision_function(test[FEATS]), -np.inf)
            print(f"statistics {case} {cls} done", flush=True)
    return out, pd.DataFrame(grows)


def half_detection(pscore, pstat, nstat, nweight, popscore, km2: float) -> dict:
    """Densities per 1000 km2 at half detection: the detector alone (a) and the statistic (b).
    pscore, pstat: the plants' detector score (misses as zero) and statistic; nstat, nweight: the
    sampled null clusters; popscore: every null cluster's score."""
    thr_a = float(np.median(pscore))
    dens_a = float((popscore >= thr_a).sum()) / km2 * 1000
    thr_b = float(np.median(pstat))
    reached = bool(np.isfinite(thr_b))
    passing = nstat >= thr_b if reached else np.zeros(len(nstat), bool)
    dens_b = float(nweight[passing].sum()) / km2 * 1000 if reached else np.nan
    return dict(thr_detector=thr_a, per_1000km2_detector=dens_a, half_reached=reached, thr_m3=thr_b, per_1000km2_m3=dens_b,
                n_null_pass=int(passing.sum()), ratio=(dens_a / dens_b if dens_b > 0 else np.inf) if reached else np.nan)


def within(arr: dict, pkgs) -> tuple[np.ndarray, np.ndarray, int]:
    """Null clusters passing in each package, with thresholds set inside that package: the detector at
    the median score of the package's own plants, the statistic at their median statistic. Where
    fewer than half of a package's plants have a statistic the detector stands alone there (the two
    counts are equal). Returns the two count arrays and the number of such packages."""
    na, nb, alone = [], [], 0
    for p in pkgs:
        pscore, pstat, nstat, nweight, popscore, _ = arr[p]
        if len(pscore) < MIN_PLANTS:
            continue
        a = float((popscore >= np.median(pscore)).sum())
        t = np.median(pstat)
        alone += not np.isfinite(t)
        na.append(a)
        nb.append(float(nweight[nstat >= t].sum()) if np.isfinite(t) else a)
    return np.array(na), np.array(nb), alone


def verdict(ratio: float) -> str:
    return "unreached" if not np.isfinite(ratio) and ratio != np.inf else "success" if ratio >= RATIO_SUCCESS else "drop" if ratio < RATIO_DROP else "between"


def report() -> None:
    from sklearn.metrics import roc_auc_score

    ms = members()
    d = read_desc()
    pop = pd.read_csv(data_dir("p3/m3") / "pop.csv.gz")
    done = set(zip(d.package, d.member))
    ms = ms[[(r.package, r.member) in done for r in ms.itertuples()]]
    pop = pop[[k in done for k in zip(pop.package, pop.member)]]
    area = pd.DataFrame([dict(package=r.package, member=r.member, cls=r.mag_class, km2=float(pd.read_csv(run.out_dir(r) / "area.csv.gz").km2.sum()),
                              sgroup="lt400" if run.spacing(r) < 400 else "gt400" if run.spacing(r) > 400 else "400") for r in ms.itertuples()])
    pop = pop.merge(area[["package", "member", "sgroup"]], on=["package", "member"])
    d, gate_table = statistics(d, area.groupby("package").km2.sum())
    rng = np.random.default_rng(SEED)
    rows, folds = [], []
    for cls in CLASSES:
        for case in ("built", "monolith"):
            for contrast in ("all", "T2_low", "T2_high"):
                for sg in ("all", "lt400", "400", "gt400"):
                    if sg != "all" and contrast != "all":
                        continue
                    sel = lambda t: t[(t.cls == cls) & ((t.sgroup == sg) | (sg == "all"))]
                    pl = sel(d[(d.kind == "plant") & (d.case == case) & ((d.contrast == contrast) | (contrast == "all"))])
                    bgr = sel(d[(d.kind == "plant_bg") & (d.case == case) & ((d.contrast == contrast) | (contrast == "all"))])
                    nu, po, ar = sel(d[d.kind == "null"]), sel(pop), sel(area)
                    if len(pl) < 40 or not len(nu):
                        continue
                    pkgs = sorted(ar.package.unique())
                    by = {p: (pl[pl.package == p], nu[nu.package == p], po[po.package == p], float(ar[ar.package == p].km2.sum())) for p in pkgs}
                    for stat in ("rule", "gbm"):
                        col = f"{stat}_{case}"
                        arr = {p: (b[0].score.fillna(0.0).to_numpy(), b[0][col].to_numpy(), b[1][col].to_numpy(), b[1].weight.to_numpy(), b[2].score.to_numpy(), b[3])
                               for p, b in by.items()}
                        pooled = lambda pick: half_detection(*(np.concatenate([arr[p][i] for p in pick]) for i in range(5)), sum(arr[p][5] for p in pick))
                        h = pooled(pkgs)
                        boot = np.array([pooled(rng.choice(pkgs, len(pkgs)))["ratio"] for _ in range(N_BOOT if len(pkgs) > 1 else 0)], float)
                        ok = ~np.isnan(boot)
                        lo, hi = np.quantile(boot[ok], [0.025, 0.975], method="nearest") if ok.any() else (np.nan, np.nan)
                        na, nb, alone = within(arr, pkgs)
                        div = lambda a, b: float(a / b) if b > 0 else (np.inf if a > 0 else np.nan)
                        rw = div(na.sum(), nb.sum())
                        pick = rng.integers(0, len(na), (N_BOOT, len(na))) if len(na) > 1 else np.zeros((0, 0), int)
                        bw = np.array([div(na[i].sum(), nb[i].sum()) for i in pick], float)
                        wlo, whi = np.quantile(bw[~np.isnan(bw)], [0.025, 0.975], method="nearest") if (~np.isnan(bw)).any() else (np.nan, np.nan)
                        fr = []
                        for p, (ppl, pnu, ppo, pkm) in by.items():
                            if not len(ppl) or not h["half_reached"]:
                                continue
                            a = float((ppo.score >= h["thr_detector"]).sum()) / pkm * 1000
                            b = float(pnu.weight[pnu[col] >= h["thr_m3"]].sum()) / pkm * 1000
                            fr.append(dict(cls=cls, case=case, contrast=contrast, spacing=sg, statistic=stat, held_out=p, km2=round(pkm, 1), n_plants=len(ppl),
                                           plants_pass_detector=round(float((ppl.score.fillna(0) >= h["thr_detector"]).mean()), 3),
                                           plants_pass_m3=round(float((ppl[col] >= h["thr_m3"]).mean()), 3),
                                           per_1000km2_detector=round(a, 2), per_1000km2_m3=round(b, 2),
                                           ratio=round(a / b, 3) if b > 0 else (np.inf if a > 0 else np.nan),
                                           **dict(zip(("own_per_1000km2_detector", "own_per_1000km2_m3", "own_thresholds_reached"),
                                                      (lambda wa, wb, al: (round(wa[0] / pkm * 1000, 2), round(wb[0] / pkm * 1000, 2), al == 0) if len(wa) else (np.nan, np.nan, False))(*within(arr, [p]))))))
                            fr[-1]["own_ratio"] = round(div(fr[-1]["own_per_1000km2_detector"], fr[-1]["own_per_1000km2_m3"]), 3)
                        if contrast == "all" and sg == "all":
                            folds += fr
                        frr = np.array([f["ratio"] for f in fr], float)
                        both = pl[["package", "member", "pid", "contrast", col]].merge(bgr[["package", "member", "pid", "contrast", col]],
                                                                                         on=["package", "member", "pid", "contrast"], suffixes=("", "_bg"))
                        passing = both[both[col] >= h["thr_m3"]] if h["half_reached"] else both.iloc[:0]
                        rows.append(dict(cls=cls, case=case, contrast=contrast, spacing=sg, statistic=stat, n_packages=len(pkgs), n_plants=len(pl),
                                         plants_above_floor=round(float(pl.fitted.mean()), 3), n_null_sample=len(nu), km2=round(float(ar.km2.sum()), 1),
                                         **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items()},
                                         ratio_lo=round(float(lo), 3), ratio_hi=round(float(hi), 3), boot_unreached=round(float((~ok).mean()), 3) if len(boot) else np.nan,
                                         fold_ratio_median=round(float(np.nanmedian(frr)), 3) if np.isfinite(frr).any() else np.nan,
                                         fold_ratio_q10=round(float(np.nanquantile(frr, 0.1, method="nearest")), 3) if np.isfinite(frr).any() else np.nan,
                                         fold_ratio_q90=round(float(np.nanquantile(frr, 0.9, method="nearest")), 3) if np.isfinite(frr).any() else np.nan,
                                         folds_below_drop=int((frr < RATIO_DROP).sum()), folds_at_success=int((frr >= RATIO_SUCCESS).sum()),
                                         ratio_within=round(rw, 3), ratio_within_lo=round(float(wlo), 3), ratio_within_hi=round(float(whi), 3),
                                         packages_detector_alone=alone, verdict_within=verdict(rw),
                                         own_ratio_median=round(float(np.nanmedian([f["own_ratio"] for f in fr])), 3) if fr else np.nan,
                                         passing_plants_bg_also_passes=round(float((passing[col + "_bg"] >= h["thr_m3"]).mean()), 3) if len(passing) else np.nan,
                                         verdict=verdict(h["ratio"])))
                        print(rows[-1], flush=True)
    ev = pd.DataFrame(rows)
    ev.to_csv(HERE / "p3_m3_eval.csv", index=False)
    pd.DataFrame(folds).to_csv(HERE / "p3_m3_folds.csv", index=False)

    # families and control shapes: thresholds of the built case, all contrasts, all spacings
    fam, desc = [], []
    for cls in CLASSES:
        c = d[d.cls == cls]
        thr = {s: ev[(ev.cls == cls) & (ev.case == "built") & (ev.contrast == "all") & (ev.spacing == "all") & (ev.statistic == s)].iloc[0] for s in ("rule", "gbm")}
        lead = float(np.median(c[(c.kind == "plant") & (c.contrast == "T2_high")].score.fillna(0.0)))  # the lead definition on these plants
        nu = c[(c.kind == "null") & c.fitted]
        groups = {"null_above_floor": nu, "null_at_lead": nu[nu.score >= lead], "plant_built": c[(c.kind == "plant") & (c.case == "built")],
                  "plant_monolith": c[(c.kind == "plant") & (c.case == "monolith")], "plant_bg": c[c.kind == "plant_bg"],
                  **{k: c[c.kind == k] for k in CONTROLS}}
        built = groups["plant_built"]
        for name, g in groups.items():
            w = g.weight.fillna(1.0).to_numpy() if name.startswith("null") else np.ones(len(g))
            gf = g.fitted.to_numpy()
            rec = dict(cls=cls, group=name, n=len(g), n_fitted=int(gf.sum()), lead_threshold=round(lead, 2))
            for f in ("pyramid", "plug", "dike", "dipole"):
                rec[f"best_{f}"] = round(float(w[gf & (g.best_family == f).to_numpy()].sum() / max(w[gf].sum(), 1e-12)), 3)
            rec["pass_detector"] = round(float((w * (g.score.fillna(0) >= thr["gbm"].thr_detector)).sum() / w.sum()), 4)
            for s in ("rule", "gbm"):
                rec[f"pass_{s}"] = round(float((w * (g[f"{s}_built"] >= thr[s].thr_m3)).sum() / w.sum()), 4) if thr[s].half_reached else np.nan
                if name in CONTROLS:  # how the statistic ranks built pyramids against this control (0.5: no ranking)
                    both = pd.concat([built, g])
                    rec[f"auc_pyramid_over_{s}"] = round(float(roc_auc_score(np.r_[np.ones(len(built)), np.zeros(len(g))],
                                                                           np.nan_to_num(both[f"{s}_built"].to_numpy(), neginf=-1e9))), 3)
            for col in ("margin", "margin_plug", "margin_dike", "margin_dip", "r2_pyr"):
                rec[f"median_{col}"] = round(float(g[col][gf].median()), 3)
            fam.append(rec)
        for case in ("built", "monolith"):
            pos = c[(c.kind == "plant") & (c.case == case) & c.fitted]
            for f in FEATS + CONTEXT + [f"rule_{case}", f"gbm_{case}"]:
                rec = dict(cls=cls, case=case, descriptor=f)
                own = []
                for _, a in pos.groupby("package"):
                    b = nu[nu.package == a.package.iloc[0]]
                    x = np.nan_to_num(np.r_[a[f].to_numpy(float), b[f].to_numpy(float)], neginf=-1e9)
                    ok = np.isfinite(x)
                    if min(len(a), len(b)) >= 20 and len(np.unique(x[ok])) > 1:
                        own.append(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))][ok], x[ok], sample_weight=np.r_[np.ones(len(a)), b.weight.to_numpy()][ok]))
                rec["auc_within_median"], rec["packages"] = round(float(np.median(own)), 3) if own else np.nan, len(own)
                for tag, a, b in (("all", pos, nu), ("at_lead", pos[pos.score >= lead], nu[nu.score >= lead])):
                    x = np.nan_to_num(np.r_[a[f].to_numpy(float), b[f].to_numpy(float)], neginf=-1e9)
                    ok = np.isfinite(x)
                    rec[f"auc_{tag}"] = round(float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))][ok], x[ok],
                                                                  sample_weight=np.r_[np.ones(len(a)), b.weight.to_numpy()][ok])), 3)
                    rec[f"plant_median_{tag}"], rec[f"null_median_{tag}"] = round(float(np.nanmedian(a[f].replace(-np.inf, np.nan))), 4), round(float(np.nanmedian(b[f].replace(-np.inf, np.nan))), 4)
                desc.append(rec)
    pd.DataFrame(fam).to_csv(HERE / "p3_m3_families.csv", index=False)
    pd.DataFrame(desc).to_csv(HERE / "p3_m3_descriptors.csv", index=False)
    gate_table.to_csv(data_dir("p3/m3") / "rule_gates.csv", index=False)
    d.drop(columns=[c for c in d.columns if c in ("x", "y", "x0", "y0")]).to_csv(data_dir("p3/m3") / "scored.csv.gz", index=False)

    # checks
    log = pd.read_csv(data_dir("p3/m3") / "run_log.csv").drop_duplicates(["package", "member"], keep="last")
    big = log.loc[log.maxrss_gb.idxmax()]
    fit_s = d.fit_s.dropna()
    chk = [("vault_assertion", "passed for every member read", len(ms)),
           ("vault_packages_in_tables", "must be 0", int(sum(in_vault(p) for p in set(d.package) | set(pop.package)))),
           ("members_run", "development members of the two classes with an output", len(done)),
           ("members_missing", "development members without an output", len(members()) - len(done)),
           ("null_clusters_wanted", "sampled", int(log.nulls_wanted.sum())), ("null_clusters_found", "located in a chunk", int(log.nulls_found.sum())),
           ("worker_peak_rss_gb", f"largest over members ({big.member})", float(big.maxrss_gb)),
           ("fit_s_median", "stage two per place", round(float(fit_s.median()), 3)), ("fit_s_mean", "stage two per place", round(float(fit_s.mean()), 3)),
           ("fit_s_q95", "stage two per place", round(float(fit_s.quantile(0.95)), 3)),
           ("member_seconds_total", "sum over members (worker time)", round(float(log.seconds.sum()), 0))]
    for cls in CLASSES:
        c = d[d.cls == cls]
        chk += [(f"packages_{cls}", "", c.package.nunique()), (f"null_sample_{cls}", "rows", int((c.kind == "null").sum())),
                (f"null_population_{cls}", f"clusters at or above {FLOOR}", int((pop.cls == cls).sum())),
                (f"max_null_weight_{cls}", "", round(float(c.weight.max()), 1))]
        for case in ("built", "monolith"):
            for contrast in run.MAG_CLASSES:
                chk.append((f"plants_{cls}_{case}_{contrast}", "rows, misses included", int(((c.kind == "plant") & (c.case == case) & (c.contrast == contrast)).sum())))
        chk += [(f"controls_{cls}_{k}", "rows", int((c.kind == k).sum())) for k in CONTROLS]
        # the weighted sample against the full population, at the lead-like score of 30
        nu = c[c.kind == "null"]
        chk.append((f"weighted_over_population_at_30_{cls}", "sum of weights / clusters", round(float(nu.weight[nu.score >= 30].sum() / max((pop[pop.cls == cls].score >= 30).sum(), 1)), 3)))
    chk += [(f"fit_self_check_{k}", "p3_m3_fit.self_check", v) for k, v in m3.self_check().items()]
    chk += [(f"eval_self_check_{k}", "p3_m3_eval check", v) for k, v in self_check().items()]
    pd.DataFrame(chk, columns=["check", "note", "value"]).to_csv(HERE / "p3_m3_checks.csv", index=False)
    print(ev[(ev.contrast == "all") & (ev.spacing == "all")].to_string(index=False))


def self_check() -> dict:
    ms = members()
    row = ms[ms.package.str.contains("iron-creek")].iloc[0]
    path = chunks(row)[0]
    mag, alt = row.mag.strip().upper(), row.alt.strip().upper()
    df, ref = load(path, row), run.frame(path, row)
    assert all(np.array_equal(df[c].to_numpy(), ref[c].to_numpy(), equal_nan=True) for c in ("line", "X", "Y", "FID", mag, alt)) and (df.kind.astype(str) == ref.kind).all()
    ch = Chunk(df, mag)
    rng = np.random.default_rng(1)
    worst, n = 0.0, 0
    for _ in range(3):
        i = int(rng.integers(1, len(ch.px) - 1))
        x0, y0 = ch.px[i] + 60.0, ch.py[i] - 40.0
        out, near = inj.plant_lines(ref, mag, alt, True, x0, y0, 300.0, 40.0, 20.0, 0.0691, 2.0, (30.0, 100.0))
        keep = set(det.survey_lines(ref))
        touched = [ln for ln in ref.line.iloc[near].unique() if ln in keep]
        want = mtr.max_score_near(out, touched, mag, alt, True, x0, y0)
        sub = pd.concat([ch.all[ln] for ln in ch.near(x0, y0, inj.REACH_M)])
        o2, n2 = inj.plant_lines(sub, mag, alt, True, x0, y0, 300.0, 40.0, 20.0, 0.0691, 2.0, (30.0, 100.0))
        planted = planted_lines(o2, ch.keep)
        got, x, y = score_near({ln: planted[ln] for ln in sub.line.iloc[n2].unique() if ln in ch.keep}, mag, alt, True, x0, y0)
        if np.isfinite(want):
            worst, n = max(worst, abs(got - want)), n + 1
            assert np.hypot(x - x0, y - y0) <= mtr.HIT_R_M
        else:
            assert not np.isfinite(got)
    assert n >= 1 and worst < 1e-9, f"score_near differs from max_score_near by {worst}"
    vault_row = SimpleNamespace(**pd.read_csv(HERE / "p3_channels.csv").merge(pd.read_csv(HERE / "p3_vault.csv").query("vault == True")[["package", "member"]]).iloc[0].to_dict())
    try:
        assert_open(vault_row)
        raise RuntimeError("a vault package passed assert_open")
    except AssertionError:
        pass
    res = dict(reader_equals_frame=True, score_near_max_abs_diff=worst, plants_compared=n, vault_row_refused=True)
    print("p3_m3_eval self-check OK", res)
    return res


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "check"
    if stage == "check":
        self_check()
    elif stage == "run":
        run_stage(sys.argv[2:])
    elif stage == "report":
        report()
    else:
        raise SystemExit(f"unknown stage {stage}")
