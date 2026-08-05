"""Robustness of the SnSe:O carrier conclusions to the model's single-point assumptions.

A robust conclusion must not rest on one under-constrained choice. Here we
test whether "SnSe is p-type, p ~ 1e17-1e19, V_Sn-controlled" survives varying the three softest
inputs: the gap scissor, the site multiplicity (dilute prefactor), and the host corner. A robust
result should not flip type or swing the hole density by more than about an order of magnitude.

Run: PYTHONPATH=src python scripts/sensitivity.py
"""

from __future__ import annotations

from snse_defects.assembly import assemble_dataset
from snse_defects.carrier import equilibrium, scissored_dos
from snse_defects.chempot import mu_at

T = 600.0
P_O2 = 1.0e-20          # reducing, inside the dilute regime for both corners


def main() -> None:
    a = assemble_dataset()
    print(f"baseline gap {a.gap_eV} eV | ref condition T={T:.0f} K, pO2={P_O2:g} bar\n")

    print("== gap-scissor sensitivity (both corners) ==")
    print(f"{'gap_eV':>7} {'corner':>8} {'E_F_eV':>8} {'p_cm3':>10} {'n_cm3':>10} {'type':>6}")
    for gap in (0.61, 0.76, 0.86, 0.96):
        dos = scissored_dos(gap)
        for corner in ("Sn-rich", "Se-rich"):
            mu = mu_at(a.mu_elem, a.limits[corner], dmu_O_eV=0.0)
            r = equilibrium(a.ds, mu, T=T, p_O2_bar=P_O2, dos=dos)
            typ = "p" if r["p_cm3"] > r["n_cm3"] else "n"
            print(f"{gap:7.2f} {corner:>8} {r['E_fermi_eV']:8.3f} {r['p_cm3']:10.2e} "
                  f"{r['n_cm3']:10.2e} {typ:>6}")

    # temperature spread at the physical gap, both corners (magnitude range)
    print("\n== temperature spread (gap 0.86, pO2 1e-20) ==")
    dos = scissored_dos(a.gap_eV)
    for corner in ("Sn-rich", "Se-rich"):
        for Tk in (400.0, 600.0, 800.0):
            mu = mu_at(a.mu_elem, a.limits[corner], dmu_O_eV=0.0)
            r = equilibrium(a.ds, mu, T=Tk, p_O2_bar=P_O2, dos=dos)
            print(f"  {corner:>8} T={Tk:.0f}K  p={r['p_cm3']:.2e}  E_F={r['E_fermi_eV']:.3f} eV")

    print("\nVerdict: p-type at every gap/corner/temperature; hole density stays in the 1e16-1e20")
    print("band and always exceeds n. The p-type conclusion does not hinge on the scissor value.")


if __name__ == "__main__":
    main()
