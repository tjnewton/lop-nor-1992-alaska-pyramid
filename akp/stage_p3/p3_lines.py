"""P3 line-data reader: Geosoft-style XYZ files inside DGGS ASCII packages.

A file holds blocks headed by "LINE n" or "TIE n" (any case), each followed by whitespace-separated
rows. "*" marks a missing value, as do the null codes in NULLS. Column names come from:
- a Geosoft header comment ("/ X Y MAG ..."), when the file has one; or
- the readme's parameter table (the DIGHEM layout: NAME, sample time, UNITS, ...), when it has
  none. Names are read in the table's order.
read_package() yields one DataFrame per data member (the line or tie number in column "line",
kind "line" or "tie"). Only open packages are read: a path under p3/lines/sealed raises, because
sealed packages stay closed until the P3 freeze.
Self-check: python -m akp.stage_p3.p3_lines (on the first open package).
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NULLS = (-9999.0, -99999.0, -1e32, -1.0000000331813535e32)


SEALED = ("p3/lines/sealed", "p3/inventory/vector_sealed")


def assert_open(path: Path, allow_sealed: bool = False) -> None:
    """Raise if path lies inside a sealed folder, after resolving links and dots.

    allow_sealed admits a sealed path only once p3_freeze.check() passes: the P3 method is
    frozen and unchanged since."""
    from akp.paths import data_root

    if allow_sealed:
        from akp.stage_p3.p3_freeze import check

        check()
        return
    real = Path(path).resolve()
    for s in SEALED:
        if real.is_relative_to((data_root() / s).resolve()):
            raise PermissionError(f"{path} is sealed until the P3 freeze")
HEADER = re.compile(r"^\s*(LINE|TIE|TEST|L|T)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*$", re.I)


def open_member(path: Path, member: str, allow_sealed: bool = False):
    """A binary file object for member: inside the zip, or the file itself when path is not a zip."""
    assert_open(path, allow_sealed)
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        return z.open(member)
    return open(path, "rb")


def _token(v: str) -> float:
    """A numeric token; a yyyy/mm/dd date becomes yyyymmdd; anything else is NaN."""
    if v == "*":
        return np.nan
    try:
        return float(v)
    except ValueError:
        m = re.match(r"^(\d{4})/(\d{2})/(\d{2})$", v)
        return float(m.group(1) + m.group(2) + m.group(3)) if m else np.nan


def readme_columns(text: str) -> list[str]:
    """Names in the readme's parameter table: the lines after "PARAMETER TIME ...", up to the next "****"."""
    cols, inside = [], False
    for line in text.splitlines():
        if re.match(r"^\s*PARAMETER\s+TIME", line, re.I):
            inside = True
            continue
        if inside and "****" in line:
            break
        if inside:
            m = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s+[\d.]+", line, re.I)
            if m:
                cols.append(m.group(1).upper())
    return cols


def parse_xyz(raw: bytes, columns: list[str] | None) -> pd.DataFrame:
    rows, lines, kinds = [], [], []
    current, kind, cols = np.nan, "line", columns
    for text in io.TextIOWrapper(io.BytesIO(raw), encoding="latin-1"):
        s = text.strip()
        if not s:
            continue
        if s.startswith("/"):
            names = s.lstrip("/").split()
            if cols is None and len(names) > 2 and all(re.match(r"^[A-Za-z_][\w.\-]*$", n) for n in names):
                cols = [n.upper() for n in names]
            continue
        m = HEADER.match(s)
        if m:
            kind = {"TEST": "test", "TIE": "tie", "T": "tie"}.get(m.group(1).upper(), "line")
            current = float(m.group(2))
            continue
        vals = [_token(v) for v in s.split()]
        if all(np.isnan(v) for v in vals):
            continue  # stray control characters such as a DOS end-of-file mark
        rows.append(vals)
        lines.append(current)
        kinds.append(kind)
    width = max(len(r) for r in rows)
    if cols is None or len(cols) != width:
        raise ValueError(f"column names ({0 if cols is None else len(cols)}) do not match the data width ({width})")
    df = pd.DataFrame(rows, columns=cols)
    df = df.mask(df.isin(NULLS))
    df.insert(0, "kind", kinds)
    df.insert(0, "line", lines)
    return df


def read_package(path: Path, member_pattern: str = r"linedata.*final.*\.xyz$|\.xyz$"):
    assert_open(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        data = [n for n in names if re.search(member_pattern, n, re.I) and "raw" not in n.lower()]
        for n in data:
            stem = Path(n).stem.lower()
            readmes = [r for r in names if r.lower().endswith(".txt") and stem in r.lower() and "readme" in r.lower()]
            cols = None
            for r in readmes:
                cols = readme_columns(z.read(r).decode("latin-1")) or cols
            yield n, parse_xyz(z.read(n), cols)


if __name__ == "__main__":
    from akp.paths import data_dir, data_glob

    first = data_glob(data_dir("p3/lines/open"), "*.zip")[0]
    for name, df in read_package(first):
        print(first.name, name, df.shape, list(df.columns)[:12])
        assert df.line.notna().all() and {"X", "Y"} <= set(df.columns)
        print(df.describe().T.head(8).to_string())
    print("p3_lines self-check OK")


def member_columns(z: zipfile.ZipFile, member: str) -> list[str] | None:
    """Column names of a data member: a "/" header comment in its first lines, else its readme table."""
    with z.open(member) as f:
        for _ in range(50):
            s = f.readline().decode("latin-1").strip()
            if s.startswith("/"):
                names = s.lstrip("/").split()
                if len(names) > 2 and all(re.match(r"^[A-Za-z_][\w.\-]*$", n) for n in names):
                    return [n.upper() for n in names]
    stem = Path(member).stem.lower()
    cols = None
    for r in z.namelist():
        if r.lower().endswith(".txt") and stem in r.lower() and "readme" in r.lower():
            cols = readme_columns(z.read(r).decode("latin-1")) or cols
    return cols


def convert(path: Path, member: str, wanted: list[str], out_dir: Path, chunk_rows: int = 4_000_000,
            allow_sealed: bool = False) -> list[Path]:
    """Stream one data member into npz chunks of whole lines with only the wanted columns (float64).

    Each chunk holds arrays "line", "is_tie" and one per wanted column. Memory stays at about one
    chunk; line blocks are parsed by pandas' C reader. Returns the chunk paths.
    """
    assert_open(path, allow_sealed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        cols = member_columns(z, member)
        if cols is None:
            raise ValueError(f"no column names for {member}")
        idx = [cols.index(c) for c in wanted]
        paths, buf, block, current, tie = [], {c: [] for c in ["line", "is_tie"] + wanted}, [], np.nan, False

        def flush_block():
            if not block:
                return
            d = pd.read_csv(io.BytesIO(b"".join(block)), sep=r"\s+", header=None, na_values=["*"],
                            usecols=idx, engine="c", on_bad_lines="skip")
            if d.shape[1] != len(idx):
                return
            buf["line"].append(np.full(len(d), current)); buf["is_tie"].append(np.full(len(d), tie))
            for c, i in zip(wanted, idx):
                v = pd.to_numeric(d[i], errors="coerce").to_numpy(np.float64, copy=True)
                v[np.isin(v, NULLS)] = np.nan
                buf[c].append(v)
            block.clear()

        def flush_chunk():
            if buf["line"]:
                p = out_dir / f"chunk{len(paths):04d}.npz"
                np.savez(p, **{k: np.concatenate(v) for k, v in buf.items()})
                paths.append(p)
                for v in buf.values():
                    v.clear()

        rows = 0
        with z.open(member) as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith(b"/") or s == b"\x1a":
                    continue
                m = HEADER.match(s.decode("latin-1"))
                if m:
                    flush_block()
                    if rows >= chunk_rows:
                        flush_chunk(); rows = 0
                    tie, current = m.group(1).upper() in ("TIE", "T", "TEST"), float(m.group(2))
                    continue
                block.append(raw if raw.endswith(b"\n") else raw + b"\n")
                rows += 1
            flush_block(); flush_chunk()
    return paths

def csv_layout(path: Path, member: str, allow_sealed: bool = False) -> dict:
    """pandas.read_csv arguments for a CSV member's header.

    Most members start with their header row: {}. Some exports open with a block of "/" comment lines
    whose last line is the header ("/EASTING,NORTHING,..."): those lines are skipped and its names
    used. A "/" block that ends in no header (rules and notes only) is skipped, and the next row is
    the header."""
    k, last = 0, ""
    with open_member(path, member, allow_sealed) as f:
        for raw in f:
            s = raw.decode("latin-1")
            if not s.lstrip().startswith("/"):
                break
            k, last = k + 1, s
    if k == 0:
        return {}
    names = [c.strip() for c in last.lstrip().lstrip("/").strip().split(",")]
    if len(names) >= 3 and all(re.match(r"^[A-Za-z_][\w\[\]. \-]*$", n) for n in names):
        return dict(skiprows=k, header=None, names=names)
    return dict(skiprows=k)


def convert_csv(path: Path, member: str, line_col: str, wanted: list[str], out_dir: Path,
                chunk_rows: int = 4_000_000, allow_sealed: bool = False) -> list[Path]:
    """Stream a CSV data member (header row) into npz chunks of whole lines, like convert().

    The line id comes from line_col: a leading "T" marks a tie (is_tie); the digits give the number.
    Columns are matched case-insensitively. Ties are also caught later by heading (p3_mag_detect).
    """
    assert_open(path, allow_sealed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, carry = [], None
    layout = csv_layout(path, member, allow_sealed)
    with open_member(path, member, allow_sealed) as f:
        head = pd.read_csv(f, nrows=0, **layout)
    names = {c.strip().upper(): c for c in head.columns}
    cols = [names[line_col.strip().upper()]] + [names[w.strip().upper()] for w in wanted]
    with open_member(path, member, allow_sealed) as f:
        for chunk in pd.read_csv(f, usecols=cols, chunksize=chunk_rows, na_values=["", "*"], low_memory=False, **layout):
            d = chunk if carry is None else pd.concat([carry, chunk])
            ids = d[cols[0]].astype(str).str.strip()
            last = ids.iloc[-1]
            done, carry = d[ids != last], d[ids == last]
            if len(done):
                paths.append(_write_csv_chunk(done, cols, wanted, out_dir, len(paths)))
    if carry is not None and len(carry):
        paths.append(_write_csv_chunk(carry, cols, wanted, out_dir, len(paths)))
    return paths


def convert_table(path: Path, member: str, line_col: str, wanted: list[str], out_dir: Path,
                  allow_sealed: bool = False) -> list[Path]:
    """Read a whitespace table under a "/" header (names from member_columns) into one npz chunk, like convert_csv()."""
    assert_open(path, allow_sealed)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [line_col.strip().upper()] + [w.strip().upper() for w in wanted]
    with zipfile.ZipFile(path) as z:
        names = member_columns(z, member)
        with z.open(member) as f:
            # simplification: the whole member in memory (these are tens of MB); stream by line id if one outgrows it
            body = b"".join(ln for ln in f if not ln.lstrip().startswith(b"/"))  # header lines only; a "/" inside a row is data
        d = pd.read_csv(io.BytesIO(body), sep=r"\s+", header=None, names=names, usecols=cols, na_values=["*"])
    return [_write_csv_chunk(d, cols, wanted, out_dir, 0)]


def _write_csv_chunk(d: pd.DataFrame, cols: list[str], wanted: list[str], out_dir: Path, k: int) -> Path:
    ids = d[cols[0]].astype(str).str.strip()
    arrays = {"line": pd.to_numeric(ids.str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce").to_numpy(np.float64),
              "is_tie": ids.str.upper().str.startswith("T").to_numpy()}
    for w, c in zip(wanted, cols[1:]):
        v = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float64, copy=True)
        v[np.isin(v, NULLS)] = np.nan
        arrays[w.strip().upper()] = v
    p = out_dir / f"chunk{k:04d}.npz"
    np.savez(p, **arrays)
    return p
