"""A4: local CPU transcription of the claim-source audio.

Runs in the separate `akp-transcribe` conda env (python 3.12, faster-whisper 1.2.1, av 13.1.0;
av 14 and later dropped the `metadata_errors` argument faster-whisper passes), not the analysis env:
    AKP_DATA=... conda run -n akp-transcribe python -m akp.stage_a.a4_transcribe
Reads a4_claim_audio.csv; writes one JSON per episode to $AKP_DATA/claims/transcripts/<show>/
(segments with start/end seconds and text) and a4_transcripts.csv (file, model, duration_s,
n_segments, sha256 of the JSON). A transcript already present is skipped. Machine transcripts:
quotes enter the register with timestamps and marked "not yet checked by ear" until a person
has listened at that timestamp.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from akp.paths import data_dir

HERE = Path(__file__).parent
MODEL, COMPUTE = "medium.en", "int8"


def main() -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL, device="cpu", compute_type=COMPUTE, cpu_threads=8)
    claims = data_dir("claims")
    rows = []
    with (HERE / "a4_claim_audio.csv").open() as f:
        episodes = list(csv.DictReader(f))
    for ep in episodes:
        audio = claims / ep["file"]
        out = data_dir(f"claims/transcripts/{ep['show']}") / (audio.stem + ".json")
        if not out.exists():
            segments, info = model.transcribe(str(audio), language="en", vad_filter=True, beam_size=5)
            segs = [dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip()) for s in segments]
            out.write_text(json.dumps(dict(source=ep, model=MODEL, compute_type=COMPUTE,
                                           duration_s=round(info.duration, 1), segments=segs), indent=1))
            print(f"{ep['published']} {ep['show']}: {len(segs)} segments", flush=True)
        d = json.loads(out.read_text())
        rows.append(dict(file=str(out.relative_to(claims)), model=d["model"], duration_s=d["duration_s"],
                         n_segments=len(d["segments"]), sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    with (HERE / "a4_transcripts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
