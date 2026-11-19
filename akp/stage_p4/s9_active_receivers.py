"""P4 S9: receiver positions of the free active-source seismic surveys, read from USGS data reports.

The EarthScope assembled archive lists the sets; its files were not reachable without a browser
on 2026-11-14, so positions come from the open-file reports the archive names, fetched from
pubs.usgs.gov into $AKP_DATA/p4/seismic_census/usgs_reports and converted with `pdftotext -layout`:
- ofr_1985-0531: TACT 1984, the Chugach, Glenn Highway and Richardson Highway lines;
- ofr_1989-0321 and ofr_1992-0196: TACT 1987, Alaska Range and Fairbanks.
The Galena 2000 profiles (EarthScope assembled set 14-065) come from the archive's own spreadsheet,
$AKP_DATA/p4/galena/raw/Galena.AK.location.coord.xlsx (downloaded on 2026-11-14):
surveyed stations and shot points, latitude and longitude as degrees and decimal minutes. It is read
with the standard library only. No Galena waveform is read here.
The TACT 1985 receivers (EarthScope assembled set 00-025) come from the trace headers of the archive's
SEG-Y files, $AKP_DATA/p4/seismic_census/tact1985 (downloaded on 2026-11-14):
group coordinates in arc seconds with the header's scalar. Headers only; no sample is read.
The tables are scans read by character recognition. A row is taken when it holds a latitude and a
longitude as degrees, minutes and seconds, or as degrees and decimal minutes, inside BOX. Stray
spaces and commas around the decimal point are removed first. Rows that fail are not repaired.
MISSING names the sets with no usable table: they are reported, not guessed.
Writes s9_active_receivers.csv (report, lat, lon; one row per distinct position, to 1e-4 degree)
and prints the count per source. Self-check: the two formats parse a known line each; the Galena
sheet yields at least sixty positions.
Run: python -m akp.stage_p4.s9_active_receivers
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

from akp.paths import data_dir

HERE = Path(__file__).parent
BOX = (60.2, 65.8, 144.2, 167.2)  # the census query area: lat south, lat north, lon west positive
REPORTS = ("ofr_1985-0531", "ofr_1989-0321", "ofr_1992-0196")
MISSING = {
    "TACT 1988 and EDGE 1989 (EarthScope 00-038, 00-039)": "marine air-gun lines with a few recorders on land; their positions were not read",
    "Bering-Chukchi 1994 (EarthScope 96-001, open-file 95-650)": "its four stations on the western coast are in scans the parser does not read",
}
# Not missing, noted: the archive attaches a California report to set 00-020 (TACT 1984, Richardson Highway);
# the 1984 lines are in open-file 85-531, read above.
NUM = r"(\d{1,3})"
DMS = re.compile(rf"(?<![\d.]){NUM} +(\d{{1,2}}) +(\d{{1,2}}(?:\.\d+)?) +{NUM} +(\d{{1,2}}) +(\d{{1,2}}(?:\.\d+)?)(?![\d.])")
DM = re.compile(rf"(?<![\d.]){NUM} +(\d{{1,2}}\.\d{{2,}}) +{NUM} +(\d{{1,2}}\.\d{{2,}})(?![\d.])")


def clean(line: str) -> str:
    """Join a number split around its decimal point by the scan: '26.,4', '31 .4', '15 51. 9'."""
    line = re.sub(r"(\d)[ ,]*\.[ ,]*(\d)", r"\1.\2", line)
    return re.sub(r"(\d), (\d)", r"\1.\2", line)


def positions(text: str) -> set[tuple[float, float]]:
    out = set()
    for raw in text.splitlines():
        line = clean(raw)
        for m in DMS.finditer(line):
            a, b, c, d, e, f = (float(v) for v in m.groups())
            if b < 60 and c < 60.5 and e < 60 and f < 60.5:
                out.add((round(a + b / 60 + c / 3600, 4), round(-(d + e / 60 + f / 3600), 4)))
        for m in DM.finditer(line):
            a, b, d, e = (float(v) for v in m.groups())
            if b < 60 and e < 60:
                out.add((round(a + b / 60, 4), round(-(d + e / 60), 4)))
    return {(la, lo) for la, lo in out if BOX[0] <= la <= BOX[1] and BOX[2] <= -lo <= BOX[3]}


def galena() -> set[tuple[float, float]]:
    """Positions in the Galena spreadsheet: every cell text that holds degrees and decimal minutes, taken in pairs per row."""
    z = zipfile.ZipFile(data_dir("p4/galena") / "raw" / "Galena.AK.location.coord.xlsx")
    strings = [re.sub(r"<[^>]+>", "", s) for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), flags=re.S)]
    out = set()
    for row in re.findall(r"<row[^>]*>(.*?)</row>", z.read("xl/worksheets/sheet1.xml").decode(), flags=re.S):
        cells = {c: strings[int(v)] for c, v in re.findall(r'<c r="([A-Z]+)\d+"[^>]*t="s"[^>]*><v>(\d+)</v>', row)}
        if "C" in cells and "D" in cells:
            out |= positions(f"{cells['C']}   {cells['D']}")
    return out


def tact1985() -> set[tuple[float, float]]:
    """Receiver positions in the trace headers of the TACT 1985 SEG-Y files (headers only)."""
    from obspy.io.segy.segy import _read_segy

    out = set()
    for f in sorted((data_dir("p4/seismic_census") / "tact1985" / "00-025.TACTpws.DATA.SEGY").glob("file*.segy")):
        for tr in _read_segy(str(f), headonly=True).traces:
            h = tr.header
            assert h.coordinate_units == 2 and h.scalar_to_be_applied_to_all_coordinates < 0, "expected arc seconds with a dividing scalar"
            k = -h.scalar_to_be_applied_to_all_coordinates * 3600.0
            out.add((round(h.group_coordinate_y / k, 4), round(h.group_coordinate_x / k, 4)))
    return {(la, lo) for la, lo in out if BOX[0] <= la <= BOX[1] and BOX[2] <= -lo <= BOX[3]}


def main() -> None:
    assert positions("323   62   6 26.,4   145 48 39 .2   1910") == {(round(62 + 6 / 60 + 26.4 / 3600, 4), round(-(145 + 48 / 60 + 39.2 / 3600), 4))}
    assert positions("x  61 31.3367   145 13.6217  y") == {(round(61 + 31.3367 / 60, 4), round(-(145 + 13.6217 / 60), 4))}
    folder = data_dir("p4/seismic_census") / "usgs_reports"
    rows = []
    for rep in REPORTS:
        got = positions((folder / f"{rep}.txt").read_text(errors="replace"))
        rows += [dict(report=rep, lat=la, lon=lo) for la, lo in sorted(got)]
        print(f"{rep}: {len(got)} positions inside the area")
    got = galena()
    assert len(got) >= 60, "the Galena spreadsheet should hold at least sixty positions"
    rows += [dict(report="galena_2000", lat=la, lon=lo) for la, lo in sorted(got)]
    print(f"galena_2000: {len(got)} positions")
    got = tact1985()
    assert len(got) >= 100, "the TACT 1985 headers should hold at least a hundred receiver positions"
    rows += [dict(report="tact_1985", lat=la, lon=lo) for la, lo in sorted(got)]
    print(f"tact_1985: {len(got)} positions")
    pd.DataFrame(rows).to_csv(HERE / "s9_active_receivers.csv", index=False)
    for k, v in MISSING.items():
        print(f"missing: {k}: {v}")


if __name__ == "__main__":
    main()
