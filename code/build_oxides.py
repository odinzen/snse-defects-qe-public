"""Build the Sn/Se oxide reference cells for the oxidation-cap estimate.

Standard structures, symmetry-verified, saved to data/mp_raw/. They are only starting geometries -
the MLIP (or a Sol job) relaxes them. The oxidation cap on Delta_mu_O is set by the most stable
oxide, SnO2 (cassiterite); SnO (romarchite) gives a looser cap. SeO2 (a complex chain structure) is
left for a later pass - Sn oxides dominate the cap.

  SnO2 cassiterite (rutile): P4_2/mnm (#136), a=4.7374, c=3.1864, Sn 2a (0,0,0), O 4f (x,x,0) x=0.306
  SnO  romarchite (litharge): P4/nmm (#129), a=3.8029, c=4.8382, Sn 2c, O 2a
"""

from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "mp_raw"

# SnO2 builds cleanly from its spacegroup; SnO's P4/nmm origin choice trips up from_spacegroup, so
# it uses an explicit 4-atom litharge basis (2 Sn on 2c, 2 O on 2a) instead.
SPECS = {
    "SnO2": dict(
        mode="spacegroup",
        sg=136,
        lattice=Lattice.tetragonal(4.7374, 3.1864),
        species=["Sn", "O"],
        coords=[[0.0, 0.0, 0.0], [0.3056, 0.3056, 0.0]],
        expect_sg=136,
        nn_A=2.05,
    ),
    "SnO": dict(
        mode="explicit",
        lattice=Lattice.tetragonal(3.8029, 4.8382),
        species=["Sn", "Sn", "O", "O"],
        coords=[[0.25, 0.25, 0.2372], [0.75, 0.75, 0.7628],
                [0.25, 0.75, 0.0], [0.75, 0.25, 0.0]],
        expect_sg=129,
        nn_A=2.22,
    ),
}


def build(name: str) -> Structure:
    s = SPECS[name]
    if s["mode"] == "spacegroup":
        return Structure.from_spacegroup(s["sg"], s["lattice"], s["species"], s["coords"])
    return Structure(s["lattice"], s["species"], s["coords"])


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in SPECS:
        s = build(name)
        sga = SpacegroupAnalyzer(s, symprec=1e-3)
        num = int(sga.get_space_group_number())
        dm = s.distance_matrix
        nn = float(min(dm[i][j] for i in range(len(s)) for j in range(len(s)) if i != j))
        pass_sg = num == SPECS[name]["expect_sg"]
        pass_nn = abs(nn - SPECS[name]["nn_A"]) < 0.20
        ok = ok and pass_sg and pass_nn
        s.to(filename=str(RAW / f"{name}.json"))
        s.to(filename=str(RAW / f"{name}.cif"))
        flag = "OK" if (pass_sg and pass_nn) else "FAIL"
        print(f"[{flag}] {name}: {s.composition.reduced_formula} {len(s)} atoms  "
              f"SG #{num} ({sga.get_space_group_symbol()}), want #{SPECS[name]['expect_sg']}  "
              f"nn={nn:.3f} A (want ~{SPECS[name]['nn_A']})")
        (RAW / f"{name}_meta.json").write_text(json.dumps(
            {"spacegroup": num, "n_sites": len(s), "nn_A": round(nn, 4)}, indent=2))
    if not ok:
        raise SystemExit("oxide structure verification FAILED")
    print("oxide references verified.")


if __name__ == "__main__":
    main()
