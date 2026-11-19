#!/usr/bin/env python3
"""Run a command under a memory guard: kill its whole process tree if its resident memory passes a limit.

The limit is a fraction of physical RAM (default 0.40) or an absolute --gb.
The guard polls the tree's summed RSS every --every seconds, prints the peak at exit, and returns
the command's exit code. It returns 137 if it killed the tree.
Usage: python scripts/mem_guard.py [--frac 0.4 | --gb 19] [--every 2] -- cmd args...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import psutil


def tree_rss(p: psutil.Process) -> int:
    total = 0
    for q in [p, *p.children(recursive=True)]:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.40)
    ap.add_argument("--gb", type=float)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    limit = a.gb * 1e9 if a.gb else a.frac * psutil.virtual_memory().total
    proc = subprocess.Popen(cmd)
    p, peak = psutil.Process(proc.pid), 0
    while proc.poll() is None:
        rss = tree_rss(p)
        peak = max(peak, rss)
        if rss > limit:
            for q in [*p.children(recursive=True), p]:
                try:
                    q.kill()
                except psutil.Error:
                    pass
            print(f"mem_guard: killed at {rss / 1e9:.1f} GB (limit {limit / 1e9:.1f} GB)", file=sys.stderr, flush=True)
            proc.wait()
            return 137
        time.sleep(a.every)
    print(f"mem_guard: peak {peak / 1e9:.2f} GB (limit {limit / 1e9:.1f} GB)", file=sys.stderr, flush=True)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
