"""P4 vault check: does the look-alike density of the development surveys hold on surveys held out?

Nothing was fitted on the vault after and M3, the one method built after it, was dropped. What the Track A report still carries over from the open surveys is the lead threshold and
the look-alike density at it. This checks that transfer on the vault, before Track A is opened.
Per system class (p3_member_heights.member_class), for the lead class (T2_high for magnetics,
T3_central for EM):
- the lead threshold from the development packages alone: the pooled median planted score, a miss
  counting as zero;
- the density of declustered null clusters at or above it, per 1000 line-km, in the development
  packages and in the vault packages, and their ratio;
- the spread of that density across development packages (10th, 50th and 90th percentiles of the
  per-package values), which says how far one survey can sit from the pooled figure;
- the share of vault plants at or above the threshold (half, if detection transfers).
Writes p3_vault_check.csv. Run: python -m akp.stage_p3.p3_vault_check
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_p3.p3_lookalike import SENSOR, tables
from akp.stage_p3.p3_vault import in_vault

HERE = Path(__file__).parent
LEAD_CLASS = {"mag": "T2_high", "em": "T3_central"}


def main() -> None:
    tr, nu, ar, _ = tables()
    for d in (tr, nu, ar):
        d["vault"] = d.package.map(in_vault)
    rows = []
    for system in sorted(set(ar.system.dropna())):
        cls = LEAD_CLASS[system.split("_")[0]]
        t = tr[(tr.system == system) & (tr.cls == cls)]
        dev = t[~t.vault]
        if dev.empty:
            rows.append(dict(system=system, lead_class=cls, note="no development package with trials in this class"))
            continue
        thr = float(dev.score.fillna(0.0).quantile(0.5))
        n, a = nu[(nu.system == system) & (nu.score >= thr)], ar[ar.system == system]
        dens = {v: n[n.vault == v].shape[0] / a[a.vault == v].line_km.sum() * 1000 if a[a.vault == v].line_km.sum() > 0 else np.nan for v in (False, True)}
        per_pkg = (n[~n.vault].groupby("package").size().reindex(a[~a.vault].package.unique(), fill_value=0)
                   / a[~a.vault].groupby("package").line_km.sum() * 1000)
        vt = t[t.vault]
        rows.append(dict(system=system, lead_class=cls, lead_threshold_dev=round(thr, 2),
                         dev_packages=int(a[~a.vault].package.nunique()), dev_line_km=round(float(a[~a.vault].line_km.sum()), 1),
                         dev_per_1000_line_km=round(dens[False], 2),
                         dev_package_q10=round(float(per_pkg.quantile(0.1)), 2), dev_package_q50=round(float(per_pkg.quantile(0.5)), 2),
                         dev_package_q90=round(float(per_pkg.quantile(0.9)), 2),
                         vault_packages=int(a[a.vault].package.nunique()), vault_line_km=round(float(a[a.vault].line_km.sum()), 1),
                         vault_per_1000_line_km=round(dens[True], 2) if np.isfinite(dens[True]) else np.nan,
                         vault_over_dev=round(dens[True] / dens[False], 3) if np.isfinite(dens[True]) and dens[False] > 0 else np.nan,
                         vault_plants=len(vt),
                         vault_plant_share_at_or_above=round(float((vt.score.fillna(0.0) >= thr).mean()), 3) if len(vt) else np.nan))
    out = pd.DataFrame(rows)
    assert set(SENSOR.values()) >= {s.split("_")[0] for s in out.system}
    out.to_csv(HERE / "p3_vault_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
