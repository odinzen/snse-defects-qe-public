"""Reference batch for PHYSICAL formation energies: the true bulk VBM + elemental mu_Sn, mu_Se.

Three cheap jobs that close the two placeholders in the formation-energy assembly:

  1. bulk_vbm  - the pristine 72-atom (3x3x1) SnSe supercell as a FIXED-occupation SCF at the exact
     production settings (PBE-D3, 80/640, k 2x2x2, nspin=1). It differs from the production bulk
     reference only in occupations='fixed', which makes QE print the highest-occupied level (the
     true VBM) instead of the smeared E_Fermi proxy currently used in assemble.py. Same cell + k as
     E_bulk so the two are on one footing.

  2. Sn_bulk   - beta-Sn (white tin) vc-relax at 80/640, PBE-D3. beta-Sn is a semimetal, so it keeps
     smearing (cold/MV). Relaxed total energy / atom = mu_Sn (standard-state reference).

  3. Se_bulk   - trigonal gray Se vc-relax at 80/640, PBE-D3. Relaxed total energy / atom = mu_Se.

mu_O = E(O2)/2 is already in hand from the settings batch, so with these three the assembly can drop
the placeholder chemical potentials and the Fermi-energy VBM proxy.

Starting structures come from scripts/build_elementals.py (data/mp_raw/{Sn,Se}_elemental.json), which
constructs and symmetry-verifies them. They are only starting geometries; vc-relax sets the final
lattice and the reference energy.

Writes runs/references/<label>.in (flat) + run_list.tsv (same format submit.sh reads).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pymatgen.core import Structure

from .qe_input import QEInput
from .structures import make_bulk_supercell

REPO = Path(__file__).resolve().parents[2]
REF_DIR = REPO / "runs" / "references"
RAW_DIR = REPO / "data" / "mp_raw"


def _pseudos() -> dict[str, str]:
    pp = yaml.safe_load((REPO / "config" / "pseudopotentials.yaml").read_text())
    return {el: d["filename"] for el, d in pp["elements"].items()}


def _elemental(el: str) -> Structure:
    path = RAW_DIR / f"{el}_elemental.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path.name}; run `PYTHONPATH=src python scripts/build_elementals.py` first"
        )
    return Structure.from_file(str(path))


def _oxide(name: str) -> Structure:
    path = RAW_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path.name}; run `PYTHONPATH=src python scripts/build_oxides.py` first"
        )
    return Structure.from_file(str(path))


def generate() -> list[dict]:
    pseudos = _pseudos()
    runlist: list[dict] = []

    def emit(group: str, label: str, qe: QEInput):
        path = qe.write(REF_DIR / f"{label}.in")
        runlist.append(
            {"group": group, "label": label, "input": str(path.relative_to(REPO)).replace("\\", "/")}
        )

    # 1. bulk VBM: production bulk supercell, fixed occupations. k = 2x2x2 to MATCH the production
    #    bulk reference (E_bulk in data/stageB_harvest.yaml), NOT the [2,2,3] in calc_settings.yaml
    #    (that was the pre-production plan; production actually ran 2x2x2).
    emit(
        "vbm",
        "bulk_vbm",
        QEInput(
            prefix="snse_bulk_vbm",
            structure=make_bulk_supercell((3, 3, 1)),
            calculation="scf",
            ecutwfc_Ry=80,
            ecutrho_Ry=640,
            kpoints=(2, 2, 2),
            pseudos={k: pseudos[k] for k in ("Sn", "Se")},
            occupations="fixed",
            vdw_corr="grimme-d3",
            nspin=1,
        ),
    )

    # 2. mu_Sn: beta-Sn vc-relax. Semimetal -> keep smearing (cold), a touch larger degauss + dense k.
    emit(
        "mu_Sn",
        "Sn_bulk",
        QEInput(
            prefix="sn_bulk",
            structure=_elemental("Sn"),
            calculation="vc-relax",
            ecutwfc_Ry=80,
            ecutrho_Ry=640,
            kpoints=(10, 10, 16),
            pseudos={"Sn": pseudos["Sn"]},
            smearing="cold",
            degauss_Ry=0.01,
            vdw_corr="grimme-d3",
            nspin=1,
        ),
    )

    # 3. mu_Se: trigonal gray Se vc-relax. Semiconductor -> gaussian smearing is fine.
    emit(
        "mu_Se",
        "Se_bulk",
        QEInput(
            prefix="se_bulk",
            structure=_elemental("Se"),
            calculation="vc-relax",
            ecutwfc_Ry=80,
            ecutrho_Ry=640,
            kpoints=(8, 8, 8),
            pseudos={"Se": pseudos["Se"]},
            smearing="gaussian",
            degauss_Ry=0.005,
            vdw_corr="grimme-d3",
            nspin=1,
        ),
    )

    # 4. Oxidation cap: SnO2 (cassiterite) vc-relax -> DeltaH_f(SnO2) at our PBE-D3 settings, to pin
    #    the Delta_mu_O cap that the MLIP screen flagged (SnO2 is the binding oxide). Wide-gap
    #    insulator, non-magnetic; O + Sn pseudos.
    emit(
        "oxide_cap",
        "SnO2_bulk",
        QEInput(
            prefix="sno2_bulk",
            structure=_oxide("SnO2"),
            calculation="vc-relax",
            ecutwfc_Ry=80,
            ecutrho_Ry=640,
            kpoints=(8, 8, 12),
            pseudos={k: pseudos[k] for k in ("Sn", "O")},
            smearing="gaussian",
            degauss_Ry=0.005,
            vdw_corr="grimme-d3",
            nspin=1,
        ),
    )

    lines = ["idx\tgroup\tlabel\tinput"]
    for i, r in enumerate(runlist, 1):
        lines.append(f"{i}\t{r['group']}\t{r['label']}\t{r['input']}")
    (REF_DIR / "run_list.tsv").write_text("\n".join(lines) + "\n", newline="\n")
    return runlist


if __name__ == "__main__":
    rl = generate()
    print(f"Wrote {len(rl)} reference inputs under runs/references/")
    for r in rl:
        print(f"  {r['group']:8s} {r['label']:12s} {r['input']}")
