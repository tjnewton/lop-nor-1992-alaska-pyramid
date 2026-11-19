"""P3 M2: one-hour amplitude bounds for the drape and thermal surface signatures.

Drape (upper bound). Porosity follows Athy's law, phi(z) = phi0 exp(-z / lam), with solids
conserved. A layer now at depth z has compacted by (phi0 - phi(z)) / (1 - phi0) of its thickness,
so a column of final thickness D has compacted by
    C(D) = phi0 / (1 - phi0) * [D - lam (1 - exp(-D / lam))].
The rigid body replaces the sediment between its apex and its base. The largest possible relief
over it is C(base depth) - C(apex depth): all compaction after deposition, with none leveled by
later sediment and none stopped by freezing. Real relief is a fraction of this, of unknown size.
The library holds no paper on surface drape. DRAPE_CASES are inputs:
textbook-style porosity curves for mud and for sand, uncited.

Shape of the drape (P4 item T1). If compaction is vertical, the relief copies the buried body's
plan: a square pyramid scaled to the relief. fourfold() splits a unit square pyramid into its
azimuthal mean (what a cone-like mound would also give) and the rest (the four-fold part, the only
part that marks a square plan). It returns the rest's largest size and its root-mean-square over
the base, both as shares of the relief.

Thermal (upper bound). A conductive body draws in geothermal heat. The flux through a sphere of
conductivity ratio r is 3r / (2 + r) times the far flux (Laplace solution), so the excess at the
surface is at most Q_GEO * (3r / (2 + r) - 1). Compared with:
- the mean flux that thaws the active layer each summer (latent heat of its ice over the thaw season);
- the surface-temperature change, excess / H_SURF (surface exchange coefficient).
The bounds are compared with sensor noise: INSAR_NOISE_MM and TIR_NETD_K. All are uncited inputs.
Writes p3_surface_bounds.csv (signature, case, quantity, value, units). Run: python -m akp.stage_p3.p3_surface_bounds
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_p3.p3_mag_forward import HEIGHT_M, TOPS_M

HERE = Path(__file__).parent
DRAPE_CASES = {"mud": dict(phi0=0.6, lam=500.0), "sand": dict(phi0=0.4, lam=2500.0)}
Q_GEO = 0.08  # W/m2, generous geothermal flux
K_RATIO = 2.5  # stone (about 3 W/m/K) over frozen silt (about 1.2)
ACTIVE_M, ICE_FRAC, EXCESS_ICE = 1.0, 0.5, 0.3  # generous active layer, its volumetric ice, its excess ice
THAW_DAYS = 120.0
L_ICE = 334e3 * 917.0  # J/m3 of ice
H_SURF = 10.0  # W/m2/K, a low (generous) surface exchange coefficient
INSAR_NOISE_MM = 5.0
TIR_NETD_K = 0.05


def compaction(d: float, phi0: float, lam: float) -> float:
    return phi0 / (1 - phi0) * (d - lam * (1 - np.exp(-d / lam)))


def fourfold(n: int = 801) -> tuple[float, float]:
    """(peak, rms over the base) of a unit square pyramid less its azimuthal mean, as shares of its height."""
    x = np.linspace(-1.5, 1.5, n)
    xx, yy = np.meshgrid(x, x)
    h = np.clip(1 - np.maximum(np.abs(xx), np.abs(yy)), 0, None)
    r = np.hypot(xx, yy)
    k = np.minimum((r / (x[1] - x[0])).astype(int), n)
    mean_r = np.bincount(k.ravel(), h.ravel(), n + 1) / np.maximum(np.bincount(k.ravel(), minlength=n + 1), 1)
    rest = h - mean_r[k]
    inside = np.maximum(np.abs(xx), np.abs(yy)) <= 1
    return float(np.abs(rest).max()), float(np.sqrt((rest[inside] ** 2).mean()))


def main() -> None:
    rows = []
    add = lambda sig, case, q, v, u: rows.append(dict(signature=sig, case=case, quantity=q, value=round(float(v), 4), units=u))
    for name, c in DRAPE_CASES.items():
        for top in TOPS_M:
            relief = compaction(top + HEIGHT_M, **c) - compaction(top, **c)
            add("drape", f"{name}_top{top:.0f}", "max_relief", relief, "m")
    peak, rms = fourfold()
    assert 0 < rms < peak < 0.5
    add("drape", "square_plan", "fourfold_peak_share", peak, "of relief")
    add("drape", "square_plan", "fourfold_rms_share", rms, "of relief")
    excess = Q_GEO * (3 * K_RATIO / (2 + K_RATIO) - 1)
    thaw_flux = L_ICE * ICE_FRAC * ACTIVE_M / (THAW_DAYS * 86400)
    d_thaw = ACTIVE_M * excess / thaw_flux
    add("thermal", "bound", "excess_flux", excess, "W/m2")
    add("thermal", "bound", "mean_thaw_flux", thaw_flux, "W/m2")
    add("thermal", "bound", "active_layer_change", d_thaw * 1e3, "mm")
    add("thermal", "bound", "seasonal_subsidence_change", d_thaw * EXCESS_ICE * 1e3, "mm")
    add("thermal", "bound", "insar_noise", INSAR_NOISE_MM, "mm")
    add("thermal", "bound", "surface_temperature_change", excess / H_SURF, "K")
    add("thermal", "bound", "tir_netd", TIR_NETD_K, "K")
    out = pd.DataFrame(rows)
    assert out.query("quantity == 'seasonal_subsidence_change'").value.iloc[0] < INSAR_NOISE_MM
    out.to_csv(HERE / "p3_surface_bounds.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
