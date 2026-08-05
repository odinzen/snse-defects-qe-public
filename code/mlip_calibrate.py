"""Calibrate foundation MLIPs against our PBE-D3 DFT on the Sn-Se-(O) chemistry.

Each available MLIP relaxes beta-Sn, trigonal Se, and SnSe and reports DeltaH_f(SnSe), compared to
our DFT (-0.936 eV/f.u. = -90.3 kJ/mol) and experiment (-0.92 eV/f.u. = -88.9 kJ/mol). A model that
reproduces this is trustworthy for a first oxide-cap estimate (SnO2/SnO/SeO2); the critical few then
get confirmed on Sol.

MLIPs are PBE-family (no vdW); MACE can add a D3 term (dispersion=True) to approach our PBE-D3, so we
run MACE both ways. Run: PYTHONPATH=src python scripts/mlip_calibrate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from snse_defects.mlip import get_calculators, relax_energy

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "mp_raw"
KJMOL = 96.485
DFT_DHF = -0.936          # eV/f.u., our PBE-D3
EXP_DHF = -0.92           # eV/f.u., Barin/SGTE (-88.9 kJ/mol)


def _structures():
    """(label, ase.Atoms, n_formula_units) for the three calibration phases."""
    from pymatgen.io.ase import AseAtomsAdaptor

    def load(p):
        from pymatgen.core import Structure
        return AseAtomsAdaptor.get_atoms(Structure.from_file(str(p)))

    sn = load(RAW / "Sn_elemental.json")
    se = load(RAW / "Se_elemental.json")
    snse = load(RAW / "SnSe_mp-691.json")     # 8-atom, 4 f.u.
    return {"Sn": (sn, len(sn)), "Se": (se, len(se)), "SnSe": (snse, len(snse) // 2)}


def main() -> None:
    phases = _structures()
    calcs = get_calculators()
    if not calcs:
        print("No MLIP installed yet.")
        return

    print(f"{'model':12s} {'E/at Sn':>9s} {'E/at Se':>9s} {'E/fu SnSe':>10s} "
          f"{'dHf eV':>8s} {'kJ/mol':>8s} {'vs DFT':>8s} {'vs exp':>8s}")
    results = {}
    for name, calc in calcs.items():
        try:
            e_sn_at = relax_energy(phases["Sn"][0], calc) / phases["Sn"][1]
            e_se_at = relax_energy(phases["Se"][0], calc) / phases["Se"][1]
            e_snse_fu = relax_energy(phases["SnSe"][0], calc) / phases["SnSe"][1]
            dHf = e_snse_fu - e_sn_at - e_se_at
            results[name] = {"dHf_eV": dHf, "dHf_kJ": dHf * KJMOL,
                             "vs_DFT_eV": dHf - DFT_DHF, "vs_exp_eV": dHf - EXP_DHF}
            print(f"{name:12s} {e_sn_at:9.3f} {e_se_at:9.3f} {e_snse_fu:10.3f} "
                  f"{dHf:8.3f} {dHf * KJMOL:8.1f} {dHf - DFT_DHF:+8.3f} {dHf - EXP_DHF:+8.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:12s} FAILED: {e}")

    print(f"\nreference:   DFT(PBE-D3) dHf = {DFT_DHF:+.3f} eV/f.u. ({DFT_DHF * KJMOL:+.1f} kJ/mol)")
    print(f"             experiment  dHf = {EXP_DHF:+.3f} eV/f.u. ({EXP_DHF * KJMOL:+.1f} kJ/mol)")

    out = REPO / "data" / "mlip_calibration.json"
    out.write_text(json.dumps(
        {"reference": {"DFT_PBE_D3_eV": DFT_DHF, "experiment_eV": EXP_DHF}, "models": results},
        indent=2, default=float))
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
