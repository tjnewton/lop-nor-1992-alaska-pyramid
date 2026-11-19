"""P3 M5 magnetics: the detector's null peaks on open packages, by P2a stratum.

Uses p3_channels.csv: one row per open package data member, with columns
package, member, epsg, mag, alt, alt_in_feet, spacing_m. For each package:
- detection runs on the unplanted traverse lines (p3_mag_detect);
- the flown area is counted per P2a land cell, as line-km x spacing, with each resampled point
  assigned to its 1 km cell;
- peaks are assigned to cells.
Rates are reported per stratum (pf|geo|lc of p2a_strata) and per survey, as peaks per km2 of flown
area above score thresholds. This is the background a Track A or Track B cut must clear. Natural
compact sources count here: they cannot be told from noise without labels.
Writes $AKP_DATA/p3/mag_null/<package>.csv.gz (peaks with cell and stratum) and p3_mag_null.csv.
Run: python -m akp.stage_p3.p3_mag_null [package-substring ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer

from akp.paths import data_dir
from akp.stage_p2.p2a_strata import label
from akp.stage_p3 import p3_mag_detect as det
from akp.stage_p3.p3_lines import read_package
from akp.stage_p3.p3_prior import out_dir as prior_dir

HERE = Path(__file__).parent
THRESHOLDS = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)


def strata_lookup():
    pr = np.load(prior_dir() / "p3_prior.npz")
    from akp.stage_p2.p2a_strata import strata_grid
    _, s = strata_grid()
    return Affine(*pr["transform"]), s


def run_package(row, transform, strata) -> tuple[pd.DataFrame, pd.DataFrame]:
    pkg = data_dir("p3/lines/open") / row.package
    df = next(d for n, d in read_package(pkg) if n == row.member)
    to3338 = Transformer.from_crs(f"EPSG:{int(row.epsg)}", "EPSG:3338", always_xy=True)
    keep = set(det.survey_lines(df))
    flown = {}
    for ln, line in df[(df.kind == "line") & df.line.isin(keep)].groupby("line"):
        for grid, v, x, y in det.segments(line, row.mag):
            ex, ey = to3338.transform(x, y)
            c, r = ~transform * (np.asarray(ex), np.asarray(ey))
            for rc in zip(np.floor(r).astype(int), np.floor(c).astype(int)):
                flown[rc] = flown.get(rc, 0.0) + det.DS_M / 1e3 * row.spacing_m / 1e3
    peaks = det.detect(df, row.mag, row.alt, bool(row.alt_in_feet))
    if len(peaks):
        ex, ey = to3338.transform(peaks.x.to_numpy(), peaks.y.to_numpy())
        c, r = ~transform * (np.asarray(ex), np.asarray(ey))
        peaks["row"], peaks["col"] = np.floor(r).astype(int), np.floor(c).astype(int)
    area = pd.DataFrame([dict(row=k[0], col=k[1], km2=v) for k, v in flown.items()])
    for d in (peaks, area):
        if len(d):
            ok = (d.row >= 0) & (d.row < strata.shape[0]) & (d.col >= 0) & (d.col < strata.shape[1])
            code = np.full(len(d), -1)
            code[ok.to_numpy()] = strata[d.row[ok], d.col[ok]]
            d["stratum"] = [label(k) if k >= 0 else "off_land" for k in code]
            d["package"] = row.package
    return peaks, area


def main(subs: list[str]) -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv")
    ch = ch[ch.package.apply(lambda p: not subs or any(s in p for s in subs))]
    transform, strata = strata_lookup()
    out_d = data_dir("p3/mag_null")
    rows = []
    for row in ch.itertuples():
        peaks, area = run_package(row, transform, strata)
        peaks.to_csv(out_d / f"{Path(row.package).stem}.csv.gz", index=False)
        area.to_csv(out_d / f"{Path(row.package).stem}_area.csv.gz", index=False)
        for st, a in area.groupby("stratum").km2.sum().items():
            p = peaks[peaks.stratum == st] if len(peaks) else peaks
            rows.append(dict(package=row.package, stratum=st, km2=round(float(a), 2),
                             **{f"n_ge_{t:.0f}": int((p.score >= t).sum()) if len(p) else 0 for t in THRESHOLDS}))
        print(row.package, f"{area.km2.sum():.0f} km2, {len(peaks)} peaks", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "p3_mag_null.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
