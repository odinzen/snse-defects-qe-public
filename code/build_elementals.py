"""Construct the Sn and Se elemental ground-state reference cells from crystallography.

Standard-state references for mu_Sn and mu_Se. Built from space group + Wyckoff + experimental
lattice constants (no MP API needed) and then symmetry-verified before use. They are only STARTING
geometries: both are vc-relaxed at the locked PBE-D3 / 80-640 settings on Sol, which sets the final
lattice and the reference energy per atom.

  beta-Sn (white tin): I4_1/amd (#141), a=5.8318, c=3.1819 A, Sn on 4a.  Sn-Sn nn ~3.02 A.
  gray Se (trigonal):  P3_121 (#152), a=4.3662, c=4.9536 A, Se on 3a (x,0,1/3), x=0.2254.
                       Intrachain Se-Se ~2.37 A, interchain ~3.44 A.

Dumps to data/mp_raw/{Sn,Se}_elemental.json (+ .cif) so references.py loads them like the MP cells.
"""

from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "mp_raw"

# (element, spacegroup number, lattice a,b,c / angles, species, Wyckoff generator coords, expected
#  space-group symbol, a representative nn distance in Angstrom to sanity-check)
SPECS = {
    "Sn": dict(
        sg=141,
        lattice=Lattice.tetragonal(5.8318, 3.1819),
        coords=[[0.0, 0.0, 0.0]],
        expect_sg="I4_1/amd",
        nn_A=3.02,
    ),
    "Se": dict(
        sg=152,
        lattice=Lattice.hexagonal(4.3662, 4.9536),
        coords=[[0.2254, 0.0, 1.0 / 3.0]],
        expect_sg="P3_121",
        nn_A=2.37,
    ),
}


def build(el: str) -> Structure:
    spec = SPECS[el]
    s = Structure.from_spacegroup(spec["sg"], spec["lattice"], [el], spec["coords"])
    return s


def verify(el: str, s: Structure) -> dict:
    spec = SPECS[el]
    sga = SpacegroupAnalyzer(s, symprec=1e-3)
    found = sga.get_space_group_symbol()
    number = int(sga.get_space_group_number())
    # nearest-neighbor distance from the full distance matrix (min off-diagonal), robust + simple
    dm = s.distance_matrix
    nn = float(min(dm[i][j] for i in range(len(s)) for j in range(len(s)) if i != j))
    ok_sg = bool(number == spec["sg"])
    ok_nn = bool(abs(nn - spec["nn_A"]) < 0.20)
    return {
        "element": el,
        "n_sites": len(s),
        "spacegroup_found": found,
        "spacegroup_number": number,
        "spacegroup_expected": f"{spec['expect_sg']} (#{spec['sg']})",
        "nn_distance_A": round(nn, 4),
        "nn_expected_A": spec["nn_A"],
        "lattice_abc": [round(x, 4) for x in s.lattice.abc],
        "PASS_spacegroup": ok_sg,
        "PASS_nn": ok_nn,
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for el in ("Sn", "Se"):
        s = build(el)
        rep = verify(el, s)
        all_ok = all_ok and rep["PASS_spacegroup"] and rep["PASS_nn"]
        stem = RAW_DIR / f"{el}_elemental"
        s.to(filename=str(stem.with_suffix(".json")))
        s.to(filename=str(stem.with_suffix(".cif")))
        (RAW_DIR / f"{el}_elemental_meta.json").write_text(json.dumps(rep, indent=2))
        flag = "OK" if (rep["PASS_spacegroup"] and rep["PASS_nn"]) else "FAIL"
        print(
            f"[{flag}] {el}: {rep['n_sites']} atoms  found {rep['spacegroup_found']} "
            f"(#{rep['spacegroup_number']}), expected {rep['spacegroup_expected']}  "
            f"nn={rep['nn_distance_A']} A (want ~{rep['nn_expected_A']})  abc={rep['lattice_abc']}"
        )
    if not all_ok:
        raise SystemExit("elemental structure verification FAILED - do not build QE decks")
    print("both elemental references verified.")


if __name__ == "__main__":
    main()
