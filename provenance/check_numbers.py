"""Recompute every number of the paper that is taken from a table, and compare it with the printed value.

Input: provenance/numbers.csv, one row per number: the document and section where it is printed, the printed value,
and how it is computed from a table of this repository:
- table, column: the CSV file and the column;
- reduction: cell (exactly one row), min, max, median, mean, sum, count, nunique, or ratio (sum of the column over
  numerator_rows divided by its sum over denominator_rows);
- rows: a pandas query that selects the rows (empty: every row); scale: a factor applied last (empty: 1).
A number passes when the recomputed value equals the printed one at the printed precision.
Run from the repository root: python provenance/check_numbers.py   (prints each failure and a summary; exit 1 on any)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_tables: dict[str, pd.DataFrame] = {}


def table(name: str) -> pd.DataFrame:
    if name not in _tables:
        _tables[name] = pd.read_csv(ROOT / name)
    return _tables[name]


def recompute(r: dict) -> float:
    df, col, red = table(r["table"]), r["column"], r["reduction"] or "cell"
    if red == "ratio":
        got = float(df.query(r["numerator_rows"])[col].sum()) / float(df.query(r["denominator_rows"])[col].sum())
    else:
        d = df.query(r["rows"]) if r["rows"] else df
        if red == "cell":
            if len(d) != 1:
                raise ValueError(f"the rows select {len(d)} rows, not one")
            got = float(d[col].iloc[0])
        elif red == "count":
            got = float(len(d))
        elif red == "nunique":
            got = float(d[col].nunique())
        else:
            got = float(getattr(d[col], red)())
    return got * float(r["scale"] or 1.0)


def tolerance(printed: str) -> float:
    return 0.5 * 10 ** -len(printed.split(".")[1]) if "." in printed else 0.5


def main() -> int:
    rows = list(csv.DictReader((ROOT / "provenance" / "numbers.csv").open(newline="")))
    bad = 0
    for r in rows:
        try:
            got = recompute(r)
            if abs(got - float(r["printed"])) <= tolerance(r["printed"]):
                continue
            msg = f"recomputed {got!r}"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
        bad += 1
        print(f"FAIL {r['document']} / {r['section']}: printed {r['printed']}, {msg} ({r['table']} {r['column']} {r['rows']})")
    print(f"{len(rows) - bad} of {len(rows)} numbers recomputed from the tables match the printed values")
    return 1 if bad else 0


if __name__ == "__main__":
    assert tolerance("2.00") == 0.005 and tolerance("74") == 0.5
    sys.exit(main())
