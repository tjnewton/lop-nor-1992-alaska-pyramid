"""A4: fetch public podcast episodes that carry claim versions.

Feeds are found through the public iTunes search API (feedUrl), then read as RSS. Episodes whose
title matches TITLE_RE are downloaded to $AKP_DATA/claims/audio/<show>/. Public audio only; no
account, no personal details in requests. Writes a4_claim_audio.csv (show, feed, title, published,
url, file, bytes, sha256). A re-run skips files already present.
Run: python -m akp.stage_a.a4_claim_audio
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

from akp.paths import data_dir

HERE = Path(__file__).parent
OUT = HERE / "a4_claim_audio.csv"
UA = {"User-Agent": "akp-research/0.1 (podcast archive fetch)"}
SHOWS = {"earthfiles": "Earthfiles Podcast with Linda Moulton Howe", "whyfiles": "The Why Files: Operation Podcast"}
TITLE_RE = re.compile(r"pyramid", re.I)
ALASKA_RE = re.compile(r"alaska", re.I)


def get(url: str, timeout: int = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def feed_url(name: str) -> str:
    q = urllib.parse.urlencode({"term": name, "entity": "podcast", "limit": 5})
    with get(f"https://itunes.apple.com/search?{q}") as r:
        hits = json.load(r)["results"]
    exact = [h for h in hits if h.get("collectionName", "").lower() == name.lower()]
    if not exact:
        raise RuntimeError(f"no exact iTunes match for {name!r}: {[h.get('collectionName') for h in hits]}")
    return exact[0]["feedUrl"]


def episodes(url: str):
    with get(url) as r:
        root = ET.parse(r).getroot()
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        enc = item.find("enclosure")
        if enc is None or not (TITLE_RE.search(title) and ALASKA_RE.search(title)):
            continue
        pub = item.findtext("pubDate")
        yield title, (parsedate_to_datetime(pub).date().isoformat() if pub else ""), enc.get("url")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def main() -> None:
    rows = []
    for show, name in SHOWS.items():
        feed = feed_url(name)
        out = data_dir(f"claims/audio/{show}")
        for title, published, url in episodes(feed):
            dest = out / f"{published}_{slug(title)}.mp3"
            if not dest.exists():
                part = dest.with_name(dest.name + ".part")
                with get(url, timeout=600) as r, part.open("wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                part.rename(dest)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            rows.append(dict(show=show, feed=feed, title=title, published=published, url=url,
                             file=str(dest.relative_to(data_dir("claims"))), bytes=dest.stat().st_size, sha256=sha))
            print(published, show, title, flush=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} episodes; wrote {OUT.name}")


if __name__ == "__main__":
    main()
