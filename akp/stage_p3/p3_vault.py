"""P4 vault: open surveys held out of method development.

A package is in the vault when the first byte of the SHA-256 of its file name is divisible by
VAULT_MOD. The rule looks at no data. Whole packages are held out, so every member of a survey falls
on the same side. Method development after P4 items M3 to M6 reads only the other packages;
the vault is scored once, by the frozen method, before Track A is opened.
Prior exposure, stated: the nulls and trials ran on every open package, and the
lead threshold comes from all of them. The vault is clean only with respect to the methods
built after.
Writes p3_vault.csv: package, member, survey_id, the magnetic and EM system classes
(p3_member_heights.member_class) and vault (True or False). The self-check asserts that every system
class with at least MIN_TO_SPLIT packages has packages on both sides. A smaller class may fall wholly
on one side; the printed table shows which.
Run: python -m akp.stage_p3.p3_vault
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from akp.stage_p3.p3_member_heights import load, member_class

HERE = Path(__file__).parent
VAULT_MOD = 5
MIN_TO_SPLIT = 4


def in_vault(package: str) -> bool:
    return hashlib.sha256(Path(package).name.encode()).digest()[0] % VAULT_MOD == 0


def main() -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    inv, mh = pd.read_csv(HERE / "p3_survey_inventory.csv").set_index("survey_id"), load()
    out = ch[["package", "member", "survey_id"]].assign(
        mag_class=[member_class(mh, inv, r.package, r.member, r.survey_id, "mag") for r in ch.itertuples()],
        em_class=[member_class(mh, inv, r.package, r.member, r.survey_id, "em") for r in ch.itertuples()],
        vault=[in_vault(p) for p in ch.package])
    for col in ("mag_class", "em_class"):
        for cls, g in out.dropna(subset=[col]).groupby(col):
            pk = g.drop_duplicates("package")
            if len(pk) >= MIN_TO_SPLIT:
                assert pk.vault.any() and not pk.vault.all(), f"{cls}: the hash rule leaves one side empty"
    out.to_csv(HERE / "p3_vault.csv", index=False)
    print(out.groupby("vault").package.nunique().to_string())
    print(out.melt(id_vars=["package", "vault"], value_vars=["mag_class", "em_class"]).dropna().drop_duplicates(["package", "value"])
          .groupby(["value", "vault"]).size().unstack(fill_value=0).to_string())
    print(sorted(out[out.vault].package.unique()))


if __name__ == "__main__":
    main()
