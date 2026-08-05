"""PBE-D3 oxidation cap on Delta_mu_O from the Sol SnO2 vc-relax.

Companion to mlip_oxide_cap.py, but the oxide energy now comes from our own PBE-D3 solid
(runs/references/SnO2_bulk.out) instead of the MLIP screen or measured enthalpies. Same boundary:

    Delta_mu_O,cap = [ DeltaH_f(SnO2) - DeltaH_f(SnSe) ] / 2      (SnSe + O2 -> SnO2 + Se, per O)

DeltaH_f(SnO2) = E(SnO2)/f.u. - mu_Sn - 2 mu_O, all PBE-D3. mu_Sn and mu_O and DeltaH_f(SnSe) are
read from the canonical harvest so this stays consistent with the rest of the dataset.

Usage (once Michael returns the harvested total energy, Ry):
    PYTHONPATH=src python scripts/oxide_cap_pbe_d3.py <E_SnO2_Ry>
The SnO2 cell is nat=6 = 2 f.u. (Sn2O4); the script divides by 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snse_defects.assembly import RY_TO_EV

REPO = Path(__file__).resolve().parents[1]
KJMOL = 96.485
EXP_OXIDE_KJ = {"SnO2": -577.6, "SnO": -280.7}   # measured formation enthalpies, kJ/mol f.u.
SNO2_FU = 2                                        # nat=6 -> 2 formula units in the cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("e_sno2_ry", type=float, help="final total energy of SnO2_bulk vc-relax, Ry")
    ap.add_argument("--write", action="store_true", help="write data/oxide_cap_pbe_d3.json")
    args = ap.parse_args()

    dataset = json.loads((REPO / "data" / "defect_formation_dataset.json").read_text())
    mu_sn = dataset["chemical_potentials_eV"]["Sn"]
    mu_o = dataset["chemical_potentials_eV"]["O"]
    dhf_snse = dataset["chemical_potential_limits"]["formation_enthalpy_SnSe_eV_per_fu"]

    e_sno2 = args.e_sno2_ry * RY_TO_EV / SNO2_FU          # eV per f.u.
    dhf_sno2 = e_sno2 - mu_sn - 2.0 * mu_o
    cap = (dhf_sno2 - dhf_snse) / 2.0

    exp_dhf_sno2 = EXP_OXIDE_KJ["SnO2"] / KJMOL
    exp_cap = (exp_dhf_sno2 - dhf_snse) / 2.0

    print(f"mu_Sn         = {mu_sn:+.4f} eV/atom")
    print(f"mu_O          = {mu_o:+.4f} eV/atom  (= E(O2)/2)")
    print(f"DeltaH_f(SnSe)= {dhf_snse:+.4f} eV/f.u.  (PBE-D3, this dataset)")
    print(f"E(SnO2)/f.u.  = {e_sno2:+.4f} eV")
    print("-" * 52)
    print(f"DeltaH_f(SnO2) PBE-D3   = {dhf_sno2:+.4f} eV/f.u. ({dhf_sno2 * KJMOL:+.1f} kJ/mol)")
    print(f"  experiment            = {exp_dhf_sno2:+.4f} eV/f.u. ({EXP_OXIDE_KJ['SnO2']:+.1f} kJ/mol)")
    print(f"  deviation             = {(dhf_sno2 - exp_dhf_sno2) * KJMOL:+.1f} kJ/mol")
    print(f"Delta_mu_O,cap PBE-D3   = {cap:+.4f} eV   (SnO2 tie-line)")
    print(f"  experiment-anchored   = {exp_cap:+.4f} eV")

    if args.write:
        out = REPO / "data" / "oxide_cap_pbe_d3.json"
        out.write_text(json.dumps({
            "e_sno2_ry": args.e_sno2_ry,
            "e_sno2_eV_per_fu": e_sno2,
            "mu_Sn_eV": mu_sn, "mu_O_eV": mu_o, "dHf_SnSe_eV_fu": dhf_snse,
            "dHf_SnO2_eV_fu": dhf_sno2, "dHf_SnO2_kJmol": dhf_sno2 * KJMOL,
            "dHf_SnO2_exp_kJmol": EXP_OXIDE_KJ["SnO2"],
            "cap_pbe_d3_eV": cap, "cap_exp_anchored_eV": exp_cap,
        }, indent=2))
        print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
