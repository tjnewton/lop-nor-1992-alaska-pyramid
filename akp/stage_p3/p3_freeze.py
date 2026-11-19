"""P3 freeze: record it, and check it before any sealed Track A data is read.

record() writes p3_freeze.txt: a first line with the current HEAD commit and the commit of the
installed SWE package, then one "sha256 path" line per DATA_INPUTS file and one "version name" line
per PACKAGES entry. It refuses if the tree has uncommitted changes under FROZEN, or if a freeze is
already recorded.
check() returns the freeze commit, or raises SystemExit if any of these fails:
- p3_freeze.txt is missing;
- the freeze commit is not an ancestor of HEAD;
- any FROZEN path changed since the freeze, staged, unstaged or untracked;
- the installed SWE commit differs from the recorded one;
- a DATA_INPUTS file (the $AKP_DATA inputs of the detectors and the report) differs from its recorded hash;
- an installed PACKAGES version differs from the recorded one;
- a git command fails.
FROZEN is the P3 method: the code under akp/stage_p3, with post-freeze outputs and the Track A
channel map excluded, plus the environment files. As in P2e (akp/stage_p2/p2e_claim_run.check_freeze).
Run: python -m akp.stage_p3.p3_freeze record | check
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from akp.paths import data_dir

HERE = Path(__file__).parent
REPO = HERE.parents[1]
FREEZE = HERE / "p3_freeze.txt"
FROZEN = ["akp/stage_p3", "akp/stage_p2", "akp/paths.py", "akp/zones.py", "environment-akp.yml", "env-akp.lock.txt",
          ":(exclude)akp/stage_p3/p3_freeze.txt", ":(exclude)akp/stage_p3/p3_trackA_*.csv",
          ":(exclude)akp/stage_p3/p3_channels_trackA.csv", 
          ":(exclude)**/__pycache__/**"]


DATA_INPUTS = ["p3/prior/p3_prior.npz", "p3/prior/p3_cover.npz", "p3/prior/p3_strata_grid.npz",
               "p3/mag_forward/templates_vector.npz", "p3/mag_forward/templates_vector_depths.npz", "p3/mag_forward/profiles.npz"]


PACKAGES = ["numpy", "scipy", "pandas", "simpeg", "discretize", "pyproj", "rasterio", "shapely"]


def _git(*a, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if r.returncode not in ok:
        raise SystemExit(f"git {' '.join(a[:3])} failed: {r.stderr.strip()[:200]}")
    return r


def _data_hashes() -> list[str]:
    out = []
    for rel in DATA_INPUTS:
        p = data_dir(str(Path(rel).parent)) / Path(rel).name
        out.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()} {rel}")
    return out + [f"{importlib.metadata.version(n)} {n}" for n in PACKAGES]


def _swe_commit() -> str | None:
    url = json.loads(importlib.metadata.distribution("seismic-workflow-engine").read_text("direct_url.json"))
    return url.get("vcs_info", {}).get("commit_id")


def _changed(since: str | None) -> list[str]:
    ref = [since] if since else []
    out = (_git("diff", "--name-only", *ref, "--", *FROZEN).stdout.split()
           + _git("diff", "--name-only", "--cached", *ref, "--", *FROZEN).stdout.split()
           + _git("ls-files", "--others", "--exclude-standard", "--", *FROZEN).stdout.split())
    return sorted(set(out))


def record() -> str:
    if FREEZE.exists():
        raise SystemExit(f"already frozen at {FREEZE.read_text().split()[0]}; a new freeze needs a dated deviation")
    dirty = _changed(None)
    if dirty:
        raise SystemExit(f"commit these first: {dirty[:8]}")
    head = _git("rev-parse", "HEAD").stdout.strip()
    FREEZE.write_text("\n".join([f"{head} {_swe_commit()}", *_data_hashes()]) + "\n")
    return head


def check() -> str:
    if not FREEZE.exists():
        raise SystemExit("P3 is not frozen: sealed Track A data stays closed")
    first, *hashes = FREEZE.read_text().splitlines()
    sha, swe = first.split()[:2]
    if _git("merge-base", "--is-ancestor", sha, "HEAD", ok=(0, 1)).returncode:
        raise SystemExit(f"freeze commit {sha} is not an ancestor of HEAD")
    changed = _changed(sha)
    if changed:
        raise SystemExit(f"frozen P3 paths changed since {sha}: {changed[:8]}; log a dated deviation first")
    if _swe_commit() != swe:
        raise SystemExit(f"installed SWE {_swe_commit()} is not the frozen SWE commit {swe}")
    if _data_hashes() != hashes:
        raise SystemExit(f"a P3 data input or package version changed since the freeze: {sorted(set(_data_hashes()) ^ set(hashes))[:4]}")
    return sha


if __name__ == "__main__":
    print(record() if sys.argv[1:] == ["record"] else check())
