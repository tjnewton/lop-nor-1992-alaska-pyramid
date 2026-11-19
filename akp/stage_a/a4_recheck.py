"""A4: a second machine transcription of each quoted window, by a larger model.

The register's quotes come from faster-whisper medium.en (a4_transcribe.py) and are marked "not
yet checked by ear". This re-transcribes each quoted window with MODEL (a different, larger
model) and checks that every key phrase of the quote appears in both transcripts (normalized
text; fuzzy match at MATCH, exact for a phrase with a number). A window where both models agree on every key phrase needs no
listening for wording; a window where they disagree is marked "listen". Speaker identity (is
it the claimant's voice?) is never machine-checked. A machine agreement is a lead, not an
ear-check: the register's "not yet checked by ear" labels stay until a person listens.
Runs in the `akp-transcribe` env:
    AKP_DATA=... conda run -n akp-transcribe python -m akp.stage_a.a4_recheck
Writes a4_recheck.csv (label, priority, source, start_s, end_s, key groups, found in each
model, verdict) and, on the data drive, claims/transcripts/a4_recheck_texts.csv (both texts).
"""
from __future__ import annotations

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from akp.paths import data_dir

HERE = Path(__file__).parent
MODEL, COMPUTE = "large-v3", "int8"
MATCH = 0.85  # fuzzy partial-match ratio for a key phrase
EP = {"C3": "2025-02-20_ep-123", "C4": "2026-02-05_ep-170", "C7": "2024-05-06_the-dark-pyramid"}
NUM = {"one hundred fifty": "150", "hundred and fifty": "150", "one hundred and fifty": "150",
       "five hundred fifty": "550", "five hundred and fifty": "550", "seven hundred": "700",
       "forty five": "45", "fifty five": "55", "fifty": "50", "sixty": "60", "seventy": "70",
       "one hundred": "100", "a hundred": "100", "thirteen": "13", "twenty second": "22", "twenty two": "22"}
# (label, priority, source, start s, duration s, key groups: every group needs one alternative)
ITEMS = [
    ("C3_16m02_depth", "P1", "C3", 962, 45, [["150"], ["700"], ["550"], ["below the surface", "under the surface", "down"]]),
    ("C3_16m07_top_depth", "P1", "C3", 962, 45, [["top of the black pyramid was 150 feet down"]]),  # the register's open "550 feet down"
    ("C3_03m28_location", "P1", "C3", 208, 22, [["between nome and mount denali", "between nome and denali"]]),
    ("C3_29m07_belt", "P1", "C3", 1747, 55, [["within 50 miles of denali"], ["northwest of denali"], ["45"], ["55"]]),
    ("C4_07m45_location", "P1", "C4", 465, 20, [["60 miles west of mount mckinley"], ["bearing to nome"]]),
    ("C4_08m01_size_depth", "P1", "C4", 481, 45, [["700"], ["150"], ["550"]]),
    ("C4_05m56_distance", "P1", "C4", 356, 15, [["60 to 70 miles"]]),
    ("C7_01m56_depth_size", "P1", "C7", 116, 25, [["100 feet below the surface"], ["550 feet tall"], ["great pyramid of giza"]]),
    ("C7_16m30_base_depth", "P1", "C7", 990, 15, [["700 feet below the surface"]]),
    ("C7_00m51_location", "P1", "C7", 51, 15, [["60 miles west of mount mckinley"], ["not surveyed to date"]]),
    ("C7_19m24_location", "P1", "C7", 1164, 15, [["east of unalakleet"], ["west of mount denali"]]),
    ("C7_20m18_location", "P1", "C7", 1218, 15, [["east of unalakleet"], ["west of mount denali"]]),
    ("C3_07m35_broadcast", "P2", "C3", 455, 75, [["december"], ["byline"], ["vibrations from this explosion"], ["may 22"]]),
    ("C3_08m33_material", "P2", "C3", 513, 20, [["bigger than the one in egypt"], ["solid or hollow"]]),
    ("C3_10m09_channel13", "P2", "C3", 609, 42, [["channel 13"], ["general manager"]]),
    ("C3_06m46_map", "P2", "C3", 406, 20, [["christmas card"]]),
    ("C3_17m02_power", "P2", "C3", 1022, 12, [["power all of alaska and canada"]]),
    ("C4_09m40_stone", "P2", "C4", 580, 30, [["made out of stone"], ["basalt"]]),
    ("C4_12m58_test", "P2", "C4", 778, 25, [["one megaton", "1 megaton"], ["may 22"]]),
    ("C7_15m23_black_stone", "P2", "C7", 923, 12, [["black stone"]]),
    ("C7_26m31_dates", "P2", "C7", 1591, 25, [["late november"], ["early december"], ["egypt"]]),
    ("C7_01m30_news", "P2", "C7", 90, 12, [["late 1992"]]),
]


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("-", " "))
    s = re.sub(r"\bmt\b", "mount", re.sub(r"\bmc kinley\b", "mckinley", s))
    for w, d in sorted(NUM.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"\b{w}\b", d, s)
    return re.sub(r"\s+", " ", s).strip()


def found(phrase: str, text: str) -> bool:
    p, t = norm(phrase), norm(text)
    if p in t:
        return True
    if any(ch.isdigit() for ch in p):  # numbers match exactly: a fuzzy ratio lets 550 pass for 150
        return False
    n = len(p)
    return any(SequenceMatcher(None, p, t[i:i + n]).ratio() >= MATCH for i in range(0, max(1, len(t) - n + 1)))


def _selfcheck() -> None:
    assert norm("Mt. McKinley, sixty miles") == "mount mckinley 60 miles"
    assert found("550 feet tall", "it was five hundred and fifty feet tall")
    assert not found("bearing to nome", "the weather in juneau")
    assert not found("was 150 feet down", "the top was 550 feet down")


def main() -> None:
    from faster_whisper import WhisperModel, decode_audio

    _selfcheck()
    model = WhisperModel(MODEL, device="cpu", compute_type=COMPUTE, cpu_threads=8)
    claims = data_dir("claims")
    with (HERE / "a4_claim_audio.csv").open() as f:
        audio = {next(k for k, v in EP.items() if v in r["file"]): claims / r["file"] for r in csv.DictReader(f)}
    trans = {k: json.loads((claims / "transcripts" / ("whyfiles" if k == "C7" else "earthfiles")
                            / (p.stem + ".json")).read_text())["segments"] for k, p in audio.items()}
    wav = {k: decode_audio(str(p), sampling_rate=16000) for k, p in audio.items()}
    rows = []
    for label, prio, src, t0, dur, groups in ITEMS:
        seg = wav[src][int(t0 * 16000):int((t0 + dur) * 16000)]
        big = " ".join(s.text.strip() for s in model.transcribe(seg, language="en", beam_size=5)[0])
        med = " ".join(s["text"] for s in trans[src] if s["end"] > t0 and s["start"] < t0 + dur)
        fm = [any(found(a, med) for a in g) for g in groups]
        fl = [any(found(a, big) for a in g) for g in groups]
        rows.append(dict(label=label, priority=prio, source=src, start_s=t0, end_s=t0 + dur,
                         keys=" | ".join("/".join(g) for g in groups),
                         found_medium=sum(fm), found_large=sum(fl), n_keys=len(groups),
                         verdict="agree" if all(fm) and all(fl) else "listen", large_text=big, medium_text=med))
        print(f"{label}: medium {sum(fm)}/{len(groups)}, large {sum(fl)}/{len(groups)} -> {rows[-1]['verdict']}", flush=True)
    texts = ("large_text", "medium_text")  # transcript text stays on the drive, as the transcripts do
    for path, cols in ((HERE / "a4_recheck.csv", [k for k in rows[0] if k not in texts]),
                       (data_dir("claims/transcripts") / "a4_recheck_texts.csv", ["label", *texts])):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
