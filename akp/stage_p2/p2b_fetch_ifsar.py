"""P2b data: USGS Alaska IfSAR 5 m DTM tiles for both claim boxes (bulk download).

Fetches every tile that TNM lists for each box in akp.zones.BOXES, all project vintages (the
vintage choice is made at mosaicking, so a newer vintage with gaps does not lose coverage).
Files go to $AKP_DATA/p2b/ifsar/<project>/<file>.tif. A file whose size already matches TNM's
sizeInBytes (or the server's Content-Length) is skipped, so a re-run resumes. Downloads go to
<file>.part and are renamed only when the byte count matches the response's Content-Length, which
is authoritative: TNM's sizeInBytes can be stale (2026-11-10: one tile listed 51487168 bytes and
was served as 51390164 on every attempt). Such tiles are logged "ok, catalog size stale".

This script only moves bytes. It never opens a raster, so no claim-zone imagery is viewed
before calibration. Writes p2b_fetch_ifsar.csv (box, project, file, url, publication_date,
bytes_expected, bytes, status). Run: python -m akp.stage_p2.p2b_fetch_ifsar
"""
from __future__ import annotations

import csv
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from akp.paths import data_dir
from akp.setup.pr_d_dem import box_bbox, tnm_items
from akp.zones import BOXES

OUT = Path(__file__).with_suffix(".csv")
WORKERS = 4
RETRIES = 3
UA = {"User-Agent": "akp-research/0.1"}  # some USGS hosts refuse urllib's default agent (403)


def fetch(url: str, dest: Path, listed: int) -> tuple[int, str]:
    if dest.exists() and dest.stat().st_size == listed:
        return listed, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                served = int(r.headers.get("Content-Length", listed))  # authoritative
                if dest.exists() and dest.stat().st_size == served:
                    return served, "cached"
                with part.open("wb") as f:
                    shutil.copyfileobj(r, f, 1 << 22)
            n = part.stat().st_size
            if n == served:
                part.rename(dest)
                return n, "ok" if served == listed else "ok, catalog size stale"
            status = f"size mismatch ({n} != served {served})"
        except OSError as e:  # URLError, timeouts and disk errors are all OSError
            status = f"error: {e}"
        print(f"retry {attempt}/{RETRIES} {dest.name}: {status}", flush=True)
    part.unlink(missing_ok=True)
    return 0, status


def main() -> None:
    root = data_dir("p2b/ifsar")
    jobs, seen = [], set()
    for box in BOXES:
        for it in tnm_items(box_bbox(BOXES[box])):
            url = it["downloadURL"]
            if url in seen:
                continue
            seen.add(url)
            project = url.split("/Projects/")[1].split("/")[0]
            name = url.rsplit("/", 1)[1]
            jobs.append(dict(box=box, project=project, file=name, url=url,
                             publication_date=it["publicationDate"], bytes_expected=int(it["sizeInBytes"])))
    dests = [(j["project"], j["file"]) for j in jobs]
    assert len(set(dests)) == len(dests), "two TNM items map to one local file"
    print(f"{len(jobs)} tiles, {sum(j['bytes_expected'] for j in jobs) / 1e9:.1f} GB listed", flush=True)

    def run(j):
        n, status = fetch(j["url"], root / j["project"] / j["file"], j["bytes_expected"])
        print(f"{status:>6} {j['project']}/{j['file']}", flush=True)
        return {**j, "bytes": n, "status": status}

    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(run, jobs))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["status"].startswith(("ok", "cached"))]
    print(f"done: {len(rows) - len(bad)} ok, {len(bad)} failed; wrote {OUT.name}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
