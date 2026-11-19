"""P2b data: the control IfSAR tiles drawn in p2a_controls.csv.

Files go to $AKP_DATA/p2b/ifsar_controls/<project>/<subproject>/<file>, each verified against
the served Content-Length by akp.stage_p2.p2b_fetch_ifsar.fetch (a complete file is skipped,
so a re-run resumes). Bytes only, no raster is opened. Writes p2b_fetch_controls.csv
(tile_id, project, subproject, file, bytes_expected, bytes, status).
Run: python -m akp.stage_p2.p2b_fetch_controls
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from akp.paths import data_dir
from akp.stage_p2.p2b_fetch_ifsar import WORKERS, fetch

HERE = Path(__file__).parent
OUT = Path(__file__).with_suffix(".csv")


def main() -> None:
    root = data_dir("p2b/ifsar_controls")
    with (HERE / "p2a_controls.csv").open() as f:
        tiles = list(csv.DictReader(f))

    def run(t):
        n, status = fetch(t["url"], root / t["project"] / t["subproject"] / t["file"], int(t["bytes"]))
        print(f"{status:>6} {t['project']}/{t['file']}", flush=True)
        return dict(tile_id=t["tile_id"], project=t["project"], subproject=t["subproject"],
                    file=t["file"], bytes_expected=t["bytes"], bytes=n, status=status)

    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(run, tiles))
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
