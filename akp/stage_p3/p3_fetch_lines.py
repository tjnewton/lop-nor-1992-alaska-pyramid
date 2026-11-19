"""P3 W2: download the chosen flight-line packages, with a checksum manifest.

Tier 1: helicopter frequency-domain EM with magnetics (DIGHEM, RESOLVE), statewide.
Tier 2: magnetics-only line surveys flown at most TIER2_MAX_M above ground.
Tier 3: magnetics-only line surveys flown above TIER2_MAX_M and at most TIER2B_MAX_M.
Each package is the ASCII line-data zip (line_data_url) of p3_survey_inventory.csv, fetched once per
URL (shared packages are deduplicated). Packages of surveys that intersect Track A, or whose Track A
status is unknown, go to $AKP_DATA/p3/lines/sealed/. They are not extracted or opened before the
P3 freeze: their checksum is the only thing read. The rest go to $AKP_DATA/p3/lines/open/.
Downloads stream in CHUNK-byte pieces, one at a time with a pause between files. A file whose size
already matches the server's Content-Length is skipped, so the run can resume.
Writes p3_fetch_lines.csv (survey_id, tier, sealed, url, bytes, sha256, path).
Run: python -m akp.stage_p3.p3_fetch_lines
"""
from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

import pandas as pd

from akp.paths import data_dir

HERE = Path(__file__).parent
TIER2_MAX_M = 150.0
TIER2B_MAX_M = 200.0  # Magnetics flown up to this are searchable for the largest body
CHUNK = 8 << 20
UA = {"User-Agent": "research-data-client/1.0"}


def selection() -> pd.DataFrame:
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv")
    ld = inv[(inv.line_data_available == "yes") & (inv.product_type == "line survey") & inv.line_data_url.notna()].copy()
    fdem = ld.em_system.astype(str).str.contains("DIGHEM|RESOLVE")
    tem = ld.em_system.astype(str).str.contains("TEM") & ~fdem
    h = pd.to_numeric(ld.mag_sensor_height_m, errors="coerce")
    low_mag = ~fdem & ~tem & (h <= TIER2_MAX_M)
    mid_mag = ~fdem & ~tem & (h > TIER2_MAX_M) & (h <= TIER2B_MAX_M)
    ld["tier"] = None
    ld.loc[fdem, "tier"] = 1
    ld.loc[low_mag, "tier"] = 2
    ld.loc[mid_mag, "tier"] = 3
    ld = ld[ld.tier.notna()]
    ld["sealed"] = ld.intersects_track_a.astype(str) != "no"
    # a URL shared by several rows is sealed if any of them is
    ld["sealed"] = ld.groupby("line_data_url").sealed.transform("any")
    return ld


def fetch(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        size = int(r.headers.get("Content-Length", -1))
    if dst.exists() and dst.stat().st_size == size:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
    assert size < 0 or tmp.stat().st_size == size, f"short download: {url}"
    tmp.rename(dst)
    time.sleep(2.0)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sel = selection()
    rows, done = [], {}
    for r in sel.sort_values(["tier", "line_data_url"]).itertuples():
        url = r.line_data_url
        if url not in done:
            dst = data_dir("p3/lines/sealed" if r.sealed else "p3/lines/open") / url.rsplit("/", 1)[1]
            fetch(url, dst)
            done[url] = (dst, dst.stat().st_size, sha256(dst))
            print(f"tier {r.tier} {'sealed' if r.sealed else 'open  '} {done[url][1] / 1e9:6.2f} GB {dst.name}", flush=True)
        dst, n, h = done[url]
        rows.append(dict(survey_id=r.survey_id, tier=int(r.tier), sealed=bool(r.sealed), url=url, bytes=n, sha256=h,
                         path=str(dst.relative_to(data_dir("p3").parent))))
        pd.DataFrame(rows).to_csv(HERE / "p3_fetch_lines.csv", index=False)
    print(f"done: {len(done)} packages, {sum(v[1] for v in done.values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
