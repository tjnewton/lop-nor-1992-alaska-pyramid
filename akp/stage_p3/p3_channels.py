"""P3 channel map: which columns of each open line-data member the detectors use, checked with the reader.

Input: a map built beforehand (a lead: column names, EPSG, altitude units and EM channel lists per
data member, with evidence), copied to $AKP_DATA/p3/inventory/p3_channels_agent.csv. For each row:
- members whose line id is a column (format csv_header, or table_slash_header: a whitespace table
  under a "/" header) get line_col, the line-id column named in the tie_rule text;
- a row needs a coordinate system: an epsg code, or a PROJ string in crs when no code fits;
- a row needs a line spacing (the flown line-km and the plants' offsets depend on it);
- block-header members must yield a line id for nearly every sample row (LINE_ID_MIN);
- the sample's median altitude, in meters after the row's units, must lie within ALT_RATIO of the
  inventory's sensor height for the survey: a channel in feet mapped as meters, or the wrong
  altitude channel, fails here and not silently in the detector. A row whose altimeter is right and
  whose inventory height is not says why in alt_exception, and is then not tested;
- a member with EM channels whose altitude is the aircraft's (alt_kind "aircraft") must give
  sensor_below_alt_m or an alt_exception: the EM bird hangs below that altimeter;
- sensor_below_alt_m, where given, is the length by which the sensor hangs below the altimeter (a
  helicopter altimeter over a towed bird); it is taken off the altitude before the test and by the
  runners (p3_g1b_run.frame);
- the wanted columns (x, y, fid, mag, alt, em_alt and the EM in-phase/quadrature names) must exist
  in the member's columns (member_columns for XYZ, the header row for CSV);
- a sample (the first SNIFF_BYTES for XYZ, SNIFF_ROWS for CSV) must parse to the right width.
reader_ok and reason record the outcome. Rows failing are not run. Writes p3_channels.csv (the
committed map the runners use). Only open packages are touched (p3_lines.assert_open).
Run: python -m akp.stage_p3.p3_channels
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from akp.paths import data_dir
from akp.stage_p3.p3_lines import assert_open, csv_layout, member_columns, open_member, parse_xyz

HERE = Path(__file__).parent
SNIFF_BYTES = 5_000_000
SNIFF_ROWS = 2000
LINE_ID_MIN = 0.99
ALT_RATIO = (0.4, 2.5)  # feet read as meters is 3.28. Birds really fly at up to twice their nominal height over
# rough ground, so a helicopter altimeter over a towed bird (also about 2) cannot be told apart by this ratio:
# the alt_kind rule below covers it.
FT_M = 0.3048


def altitude_reason(row, sample: pd.DataFrame) -> str:
    """Empty when the sample's altitude agrees with the inventory height, else the reason."""
    if isinstance(getattr(row, "alt_exception", None), str) and row.alt_exception.strip():
        return ""
    below = getattr(row, "sensor_below_alt_m", float("nan"))
    towed_em = isinstance(row.em_channels, str) and row.em_channels.strip()
    if str(row.alt_kind).strip().lower() == "aircraft" and towed_em and pd.isna(below):
        return "an aircraft altimeter on a member with a towed EM bird: give sensor_below_alt_m, or say why not in alt_exception"
    below = float(below) if pd.notna(below) else 0.0
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv").set_index("survey_id")
    names = {c.strip().upper(): c for c in sample.columns}
    unit = FT_M if str(row.alt_units).lower().startswith("f") else 1.0
    for col, hcol in ((row.alt, "mag_sensor_height_m"), (row.em_alt, "em_bird_height_m")):
        if not (isinstance(col, str) and col.strip().upper() in names):
            continue
        heights = [inv[hcol].get(s.strip()) for s in str(row.survey_id).split(";")]
        heights = [float(h) for h in heights if h is not None and pd.notna(h)]
        own = hcol == "em_bird_height_m" and str(row.em_alt).strip().upper() != str(row.alt).strip().upper()  # the bird's own altimeter
        med = float(pd.to_numeric(sample[names[col.strip().upper()]], errors="coerce").median()) * unit - (0.0 if own else below)
        if heights and not (ALT_RATIO[0] * heights[0] <= med <= ALT_RATIO[1] * heights[0]):
            return f"altitude {col} median {med:.0f} m against inventory height {heights[0]:.0f} m: units or channel?"
    return ""


def wanted(row) -> list[str]:
    w = [c for c in (row.x, row.y, row.fid, row.mag, row.alt, row.em_alt) if isinstance(c, str) and c.strip()]
    if isinstance(row.em_channels, str) and row.em_channels.strip():
        for c in json.loads(row.em_channels):
            w += [c["inphase"], c["quadrature"]]
    return list(dict.fromkeys(x.strip() for x in w))


def check(row, folder: str = "p3/lines/open", allow_sealed: bool = False) -> tuple[str, str, str]:
    path = data_dir(folder) / row.package
    if not path.exists():
        return "no", f"package not in {folder}", ""
    assert_open(path, allow_sealed)
    want = wanted(row)
    if pd.isna(row.epsg) and not (isinstance(row.crs, str) and row.crs.strip()):
        return "no", "no coordinate system", ""
    if not re.search(r"\d", str(row.spacing_m)):
        return "no", "no line spacing", ""
    m = re.search(r"column '([^']+)'|column ([A-Za-z_][\w ]*?)[:;]", str(row.tie_rule))
    line_col = (m.group(1) or m.group(2)).strip() if m and row.format in ("csv_header", "table_slash_header") else ""
    if row.format == "csv_header":
        with open_member(path, row.member, allow_sealed) as f:
            d = pd.read_csv(f, nrows=SNIFF_ROWS, **csv_layout(path, row.member, allow_sealed))
        have = {c.strip().upper() for c in d.columns}
        miss = [w for w in [line_col] + want if w.strip().upper() not in have]
        if miss or not line_col:
            return "no", f"missing {miss}", line_col
        why = altitude_reason(row, d)
        return ("no", why, line_col) if why else ("yes", "", line_col)
    if not zipfile.is_zipfile(path):
        return "no", "not a zip and not CSV", ""
    with zipfile.ZipFile(path) as z:
        cols = member_columns(z, row.member)
        if cols is None:
            return "no", "no column names", ""
        miss = [w for w in want + ([line_col] if row.format == "table_slash_header" else []) if w.upper() not in cols]
        if miss:
            return "no", f"missing {miss}", ""
        with z.open(row.member) as f:
            raw = f.read(SNIFF_BYTES)
        raw = raw[: raw.rfind(b"\n") + 1]
        try:
            d = parse_xyz(raw, cols)
        except ValueError as e:
            return "no", str(e)[:80], ""
    if row.format != "table_slash_header" and d.line.notna().mean() < LINE_ID_MIN:
        return "no", f"line ids found for only {d.line.notna().mean():.2f} of sample rows", ""
    why = altitude_reason(row, d)
    return ("no", why, line_col) if why else ("yes", "", line_col)


def main() -> None:
    m = pd.read_csv(data_dir("p3/inventory") / "p3_channels_agent.csv")
    out = []
    for row in m.itertuples():
        ok, reason, line_col = check(row)
        out.append(dict(row._asdict(), reader_ok=ok, reason=reason, line_col=line_col))
    df = pd.DataFrame(out).drop(columns=["Index"])
    df.to_csv(HERE / "p3_channels.csv", index=False)
    print(df.reader_ok.value_counts().to_string())
    print(df[df.reader_ok == "no"][["package", "member", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
