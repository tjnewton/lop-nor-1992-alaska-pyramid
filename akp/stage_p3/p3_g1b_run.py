"""P3 W2 batch runner: cache, null peaks and injection trials on the open (control and Track B) packages.

Input: p3_channels.csv (one row per data member; reader_ok rows only). CSV members are cached with
p3_lines.convert_csv, "/"-header tables with p3_lines.convert_table (line id from line_col in both),
others with p3_lines.convert. Where a survey has no
EM bird altimeter, its magnetometer altimeter is used for both: for the helicopter-only altimeters
of Aniak 2000 and Koyukuk this overstates the bird height, the conservative direction. Used columns:
- package, member, epsg (or crs, a PROJ string, where no code fits), x, y, fid, mag;
- alt, alt_units, em_channels (a JSON list of {freq_hz, coil, inphase, quadrature}), em_alt, spacing_m.
Per member:
1. cache: p3_lines.convert into $AKP_DATA/p3/lines/cache/<package>/<member>/chunkNNNN.npz. Only the
   used channels are kept, and the chunks hold whole lines.
2. null: per chunk, p3_mag_detect (if mag) and p3_em_detect (if coplanar EM) on unplanted traverse
   lines. Peaks get their P2a cell and stratum. Flown area per cell is line-km x spacing.
3. trials: TRIAL_PLANTS plants per class, drawn across chunks in proportion to their samples.
   - T2_low and T2_high: base and top drawn per plant; induced, and for about half the plants also
     remanence with Q drawn from Q_RANGE (p3_mag_trials);
   - T3: 1D model (optimistic) and scaled by EM_SCALE_CENTRAL (median 3D ratio).
   Each plant records its score, cell and stratum.
Outputs, under $AKP_DATA/p3/g1b/<package>/: null_mag.csv.gz, null_em.csv.gz, area.csv.gz and
trials.csv.gz. Workers are sized from MEM_FRAC of RAM over PEAK_GB_PER_WORKER, measured on the largest
chunk. The pool holds submissions while available memory is below MIN_AVAILABLE, as in P2b.
Sealed packages are read only for rows with folder "sealed" (the Track A run), and only once
p3_freeze.check() passes; that holds for their cache too. Members with all null outputs are skipped;
a partial cache is cleared before re-conversion. Large CSV members need fewer workers (the
2026-11-13 run was stopped by mem_guard at six). Run: python -m akp.stage_p3.p3_g1b_run [null|trials] [--workers=N] [package-substring ...]
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import zlib
from types import SimpleNamespace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from affine import Affine
from pyproj import Transformer

from akp.paths import data_dir
from akp.stage_p2.p2a_strata import label
from akp.stage_p3.p3_lines import convert, convert_csv, convert_table

HERE = Path(__file__).parent
N_TRIALS = 100
SEED = 20261113
Q_RANGE = (0.0, 10.0)
EM_SCALE_CENTRAL = 0.2415  # Median 3D/1D ratio (p3_em_3d_check.csv)
MEM_FRAC, PEAK_GB_PER_WORKER, MIN_AVAILABLE, MAX_WORKERS = 0.40, 3.0, 0.25, 6
MAG_CLASSES = {"T2_low": 0.0241, "T2_high": 0.0691}  # dk SI (p3_g1a_decision.csv)
BASE_M, TOP_M = (200.0, 400.0), (30.48, 45.72)  # per-plant uniform ranges: bases, the register top depths
EM_SEP_DEFAULT_M = 8.0  # DIGHEM/RESOLVE coplanar separation when the channel map gives none (p3_PROPERTIES)


def _roots(row) -> tuple[str, str, str, bool]:
    """(data folder, cache root, output root, sealed) for a row: open by default; the Track A run sets
    row.folder = "sealed", which reads through the freeze check."""
    sealed = getattr(row, "folder", "open") == "sealed"
    return ("p3/lines/sealed", "p3/trackA/cache", "p3/trackA/g1b", True) if sealed else \
           ("p3/lines/open", "p3/lines/cache", "p3/g1b", False)


def cache_dir(row) -> Path:
    return data_dir(_roots(row)[1]) / Path(row.package).stem / Path(row.member).stem


def out_dir(row) -> Path:
    return data_dir(_roots(row)[2]) / Path(row.package).stem / Path(row.member).stem


FT_M = 0.3048
STAMP = "row.json"
USED = ("package", "member", "format", "epsg", "crs", "x", "y", "fid", "mag", "alt", "alt_units", "em_channels", "em_alt",
        "spacing_m", "line_col", "sensor_below_alt_m", "em_sep_m")  # the map fields the outputs depend on


def fresh(row, out: Path) -> None:
    """Drop a member's outputs and cache when the map row they were made from has changed.

    The fields in USED are stamped beside the outputs. Outputs made before stamps existed are taken
    as current, and are stamped."""
    def norm(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return f"{float(v):g}"
        except (TypeError, ValueError):
            return str(v)

    key = json.dumps({k: norm(getattr(row, k, None)) for k in USED}, sort_keys=True)
    f = out / STAMP
    if f.exists() and f.read_text() != key:
        for g in out.glob("*.csv.gz"):
            g.unlink()
        shutil.rmtree(cache_dir(row), ignore_errors=True)
    f.write_text(key)


def em_sep(row) -> float:
    return float(row.em_sep_m) if hasattr(row, "em_sep_m") and pd.notna(row.em_sep_m) else EM_SEP_DEFAULT_M


def em_list(row) -> list[dict]:
    if not isinstance(row.em_channels, str) or not row.em_channels.strip():
        return []
    return [c for c in json.loads(row.em_channels) if str(c.get("coil", "")).upper().startswith("CP")]


def crs(row) -> str:
    """The member's coordinate system: the channel map's PROJ string when it has one, else its EPSG code."""
    c = getattr(row, "crs", None)
    return c.strip() if isinstance(c, str) and c.strip() else f"EPSG:{int(row.epsg)}"


def ensure_cache(row) -> list[Path]:
    folder, _, _, sealed = _roots(row)
    if sealed:  # a cache of sealed data is as closed as the package it came from
        from akp.stage_p3.p3_freeze import check

        check()
    d = cache_dir(row)
    done = sorted(d.glob("chunk*.npz")) if d.exists() else []
    if done and (d / "complete").exists():
        return done
    for stale in done:  # a partial cache from an interrupted run: start clean
        stale.unlink()
    wanted = [c for c in [row.x, row.y, row.fid, row.mag, row.alt, row.em_alt] if isinstance(c, str) and c]
    for c in em_list(row):
        wanted += [c["inphase"], c["quadrature"]]
    wanted = list(dict.fromkeys(w.upper() for w in wanted))
    src = data_dir(folder) / row.package
    if row.format == "csv_header":
        paths = convert_csv(src, row.member, row.line_col, wanted, d, allow_sealed=sealed)
    elif row.format == "table_slash_header":
        paths = convert_table(src, row.member, row.line_col, wanted, d, allow_sealed=sealed)
    else:
        paths = convert(src, row.member, wanted, d, allow_sealed=sealed)
    (d / "complete").write_text("ok\n")
    return paths


def frame(path: Path, row) -> pd.DataFrame:
    z = np.load(path)
    df = pd.DataFrame({k: z[k] for k in z.files})
    ren = {row.x.strip().upper(): "X", row.y.strip().upper(): "Y"}
    if isinstance(row.fid, str) and row.fid:
        ren[row.fid.strip().upper()] = "FID"
    df = df.rename(columns=ren)
    df["kind"] = np.where(df.pop("is_tie"), "tie", "line")
    below = getattr(row, "sensor_below_alt_m", np.nan)
    if pd.notna(below) and float(below) and isinstance(row.alt, str) and row.alt.strip():  # the sensor hangs this far below the altimeter (a towed bird)
        off = float(below) / (FT_M if str(row.alt_units).lower().startswith("f") else 1.0)
        c = row.alt.strip().upper()  # only the altimeter the sensor hangs under; a separate em_alt is the bird's own
        if c in df:
            df[c] = df[c] - off
    return df


def strata():
    """(transform, P2a strata codes): from a small cached npz, built once in the main process.

    Loading p2a_strata.strata_grid itself peaks near 6 GB, too much to repeat in every worker."""
    from akp.stage_p3.p3_prior import out_dir

    cache = out_dir() / "p3_strata_grid.npz"
    if not cache.exists():
        from akp.stage_p2.p2a_strata import strata_grid

        _, s = strata_grid()
        np.savez_compressed(cache, s=s.astype(np.int16))
    pr = np.load(out_dir() / "p3_prior.npz")
    return Affine(*pr["transform"]), np.load(cache)["s"]


def assign(df: pd.DataFrame, crs: str, transform, s) -> pd.DataFrame:
    if not len(df):
        return df
    ex, ey = Transformer.from_crs(crs, "EPSG:3338", always_xy=True).transform(df.x.to_numpy(), df.y.to_numpy())
    c, r = ~transform * (np.asarray(ex), np.asarray(ey))
    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
    ok = (r >= 0) & (r < s.shape[0]) & (c >= 0) & (c < s.shape[1])
    code = np.full(len(df), -1)
    code[ok] = s[r[ok], c[ok]]
    return df.assign(row=r, col=c, stratum=[label(k) if k >= 0 else "off_land" for k in code])


def spacing(row) -> float:
    """Line spacing in m: the first number of the channel map's spacing field (some list block spacings)."""
    m = re.search(r"\d+(?:\.\d+)?", str(row.spacing_m))
    return float(m.group()) if m else np.nan


def null_one(row) -> str:
    row = SimpleNamespace(**row) if isinstance(row, dict) else row
    sp = spacing(row)
    from akp.stage_p3 import p3_em_detect as edet
    from akp.stage_p3 import p3_mag_detect as mdet

    out = out_dir(row)
    out.mkdir(parents=True, exist_ok=True)
    fresh(row, out)
    if not np.isfinite(sp):
        return f"{row.package}/{row.member}: no line spacing in the channel map; skipped"
    if all((out / f).exists() for f in ("null_mag.csv.gz", "null_em.csv.gz", "area.csv.gz")):
        return f"{row.package}: done earlier"
    transform, s = strata()
    feet = str(row.alt_units).lower().startswith("f")
    em = em_list(row)
    mags, ems, area = [], [], []
    for p in ensure_cache(row):
        df = frame(p, row)
        keep = set(mdet.survey_lines(df))
        if isinstance(row.mag, str) and row.mag:
            mags.append(mdet.detect(df, row.mag.strip().upper(), row.alt.strip().upper(), feet))
            for ln, line in df[(df.kind == "line") & df.line.isin(keep)].groupby("line"):
                for grid, _, x, y in mdet.segments(line, row.mag.strip().upper()):
                    area.append(pd.DataFrame(dict(x=x, y=y, km2=mdet.DS_M / 1e3 * sp / 1e3)))
        if em:
            chans = [ch for c in em for ch in (c["inphase"].strip().upper(), c["quadrature"].strip().upper())]
            sep = em_sep(row)
            if not (isinstance(row.mag, str) and row.mag):  # no magnetics: the flown area comes from the EM segments
                for ln, line in df[(df.kind == "line") & df.line.isin(keep)].groupby("line"):
                    for grid, _, x, y in edet.segments(line, chans):
                        area.append(pd.DataFrame(dict(x=x, y=y, km2=mdet.DS_M / 1e3 * sp / 1e3)))
            ems.append(edet.detect(df, chans, tuple(float(c["freq_hz"]) for c in em), sep,
                                   (row.em_alt if isinstance(row.em_alt, str) and row.em_alt.strip() else row.alt).strip().upper(), feet))
    for name, parts in (("null_mag", mags), ("null_em", ems)):
        d = pd.concat(parts) if parts else pd.DataFrame()
        assign(d, crs(row), transform, s).to_csv(out / f"{name}.csv.gz", index=False)
    a = assign(pd.concat(area), crs(row), transform, s) if area else pd.DataFrame()
    if len(a):
        a.groupby(["row", "col", "stratum"]).km2.sum().reset_index().to_csv(out / "area.csv.gz", index=False)
    return f"{row.package}: {sum(map(len, mags))} mag peaks, {sum(map(len, ems))} EM peaks"


TRIAL_PLANTS = 80  # per class per member; about half of the magnetic plants carry remanence
EM_CLASSES = {"T3_opt": 1.0, "T3_central": EM_SCALE_CENTRAL}  # host 1000 ohm m, body 10 ohm m


def trials_one(row) -> str:
    from akp.stage_p3 import p3_em_trials as etr
    from akp.stage_p3 import p3_mag_trials as mtr

    row = SimpleNamespace(**row) if isinstance(row, dict) else row
    out = out_dir(row)
    out.mkdir(parents=True, exist_ok=True)
    fresh(row, out)
    if (out / "trials.csv.gz").exists():
        return f"{row.package}: trials done earlier"
    sp = spacing(row)
    if not np.isfinite(sp):
        return f"{row.package}: no line spacing in the channel map; trials skipped"
    transform, s = strata()
    feet = str(row.alt_units).lower().startswith("f")
    em = em_list(row)
    paths = ensure_cache(row)
    sizes = np.array([np.load(p)["line"].size for p in paths], float)
    alloc = np.maximum(0, np.round(TRIAL_PLANTS * sizes / sizes.sum())).astype(int)
    parts = []
    for k, (p, n) in enumerate(zip(paths, alloc)):
        if n == 0:
            continue
        df = frame(p, row)
        # the member's name enters the seed: members must not share one sequence of plant draws
        seed = (SEED + 1000 * k + zlib.crc32(f"{row.package}/{row.member}".encode())) % 2**32
        if isinstance(row.mag, str) and row.mag.strip():
            for cls, dk in MAG_CLASSES.items():
                tr = mtr.trials(df, row.mag.strip().upper(), row.alt.strip().upper(), feet, sp, int(n), seed, BASE_M, TOP_M, dk, Q_RANGE)
                parts.append(tr.assign(cls=cls))
        if em:
            chans = [ch for c in em for ch in (c["inphase"].strip().upper(), c["quadrature"].strip().upper())]
            freqs = tuple(float(c["freq_hz"]) for c in em)
            alt = (row.em_alt if isinstance(row.em_alt, str) and row.em_alt.strip() else row.alt).strip().upper()
            for cls, scale in EM_CLASSES.items():
                tr = etr.trials(df, chans, freqs, em_sep(row), alt, feet, sp, int(n), seed, BASE_M, TOP_M, 10.0, 1000.0, scale)
                parts.append(tr.assign(cls=cls))
    d = pd.concat(parts) if parts else pd.DataFrame()
    if len(d):
        d = assign(d.rename(columns={"x0": "x", "y0": "y"}), crs(row), transform, s)
    d.to_csv(out / "trials.csv.gz", index=False)
    return f"{row.package}: {len(d)} trials"


def run_pool(rows, fn, max_workers: int = MAX_WORKERS) -> None:
    total = psutil.virtual_memory().total
    workers = max(1, min(max_workers, int(MEM_FRAC * total / (PEAK_GB_PER_WORKER * 1e9))))
    print(f"{workers} workers", flush=True)
    queue, pending, names = list(rows), set(), {}
    with ProcessPoolExecutor(workers) as ex:
        while queue or pending:
            low = psutil.virtual_memory().available < MIN_AVAILABLE * total
            if queue and len(pending) < workers and (not low or not pending):
                r = queue.pop(0)
                fut = ex.submit(fn, r)
                names[fut] = f"{r.get('package')}/{r.get('member')}" if isinstance(r, dict) else str(r)
                pending.add(fut)
                continue
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for d in done:
                try:
                    print(d.result(), flush=True)
                except Exception as e:  # one member's failure is logged, not fatal to the run
                    print(f"FAILED: {names.get(d)}: {type(e).__name__}: {e}", flush=True)
            if low:
                print("memory watchdog: holding submissions", flush=True)


def main(stage: str, subs: list[str]) -> None:
    workers = MAX_WORKERS
    if subs and subs[0].startswith("--workers="):
        workers, subs = int(subs[0].split("=")[1]), subs[1:]
    ch = pd.read_csv(HERE / "p3_channels.csv")
    ch = ch[ch.reader_ok == "yes"]  # p3_channels.check requires a coordinate system
    if subs:
        ch = ch[ch.package.apply(lambda p: any(s in p for s in subs))]
    strata()  # build the cache before the pool
    if stage == "null":
        run_pool([r._asdict() for r in ch.itertuples()], null_one, workers)
    elif stage == "trials":
        run_pool([r._asdict() for r in ch.itertuples()], trials_one, workers)
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
