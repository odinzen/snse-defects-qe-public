"""Small, fast batch that decides the last two production settings before the defect run:

  1. optB88-vdW bulk vc-relax (1 job) - compare its lattice to the PBE-D3 result and to experiment.
     Warm-started from the PBE-D3 relaxed cell. input_dft='vdW-DF-obk8', no D3 correction.
  2. O2-in-a-box ecutwfc sweep (5 jobs) - fix the plane-wave cutoff oxygen needs. Isolated O2 in a
     ~12 A box, Gamma point, spin-triplet (tot_magnetization=2). The O PAW is harder than Sn/Se, so
     this - not the bulk sweep - sets the production cutoff. Plain PBE (vdW is irrelevant for one
     isolated molecule); a cutoff trend is functional-insensitive.

Writes runs/settings/<sweep>/<label>.in and runs/settings/run_list.tsv (same format submit.sh reads).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pymatgen.core import Lattice, Structure

from .qe_input import QEInput
from .structures import load_conventional

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "runs" / "settings"

O_CUTOFF_SWEEP = [60, 70, 80, 90, 100]   # Ry
O2_BOND_A = 1.21                          # O2 equilibrium bond length (Angstrom)
O2_BOX_A = 12.0


def _cfg():
    return yaml.safe_load((REPO / "config" / "calc_settings.yaml").read_text())


def _pseudos() -> dict[str, str]:
    pp = yaml.safe_load((REPO / "config" / "pseudopotentials.yaml").read_text())
    return {el: d["filename"] for el, d in pp["elements"].items()}


def _o2_structure() -> Structure:
    half = O2_BOND_A / 2.0
    c = O2_BOX_A / 2.0
    lat = Lattice([[O2_BOX_A, 0, 0], [0, O2_BOX_A, 0], [0, 0, O2_BOX_A]])
    cart = [[c, c, c - half], [c, c, c + half]]
    return Structure(lat, ["O", "O"], cart, coords_are_cartesian=True)


def generate() -> list[dict]:
    cfg = _cfg()
    pseudos = _pseudos()
    degauss = float(cfg["electronic"]["degauss_Ry"])
    runlist: list[dict] = []

    def emit(sweep: str, label: str, qe: QEInput):
        # flat: all settings inputs live directly in runs/settings/ (small batch, trivial to upload)
        path = qe.write(SET_DIR / f"{label}.in")
        runlist.append(
            {"sweep": sweep, "label": label, "input": str(path.relative_to(REPO)).replace("\\", "/")}
        )

    # 1. optB88-vdW bulk vc-relax, warm-started from the PBE-D3 relaxed cell
    emit(
        "optb88_relax",
        "bulk_vcrelax_optb88",
        QEInput(
            prefix="snse_bulk_optb88",
            structure=load_conventional(),
            calculation="vc-relax",
            ecutwfc_Ry=60,
            ecutrho_Ry=480,
            kpoints=tuple(cfg["kpoints"]["bulk_grid"]),
            pseudos={k: pseudos[k] for k in ("Sn", "Se")},
            degauss_Ry=degauss,
            vdw_corr=None,
            input_dft="vdW-DF-obk8",
        ),
    )

    # 2. O2-in-a-box ecutwfc sweep (spin triplet), plain PBE, Gamma
    o2 = _o2_structure()
    for w in O_CUTOFF_SWEEP:
        emit(
            "o_cutoff",
            f"o2_ecutwfc_{w}",
            QEInput(
                prefix=f"o2_ecutwfc_{w}",
                structure=o2,
                calculation="scf",
                ecutwfc_Ry=w,
                ecutrho_Ry=8 * w,
                kpoints=(1, 1, 1),
                pseudos={"O": pseudos["O"]},
                degauss_Ry=degauss,
                vdw_corr=None,
                nspin=2,
                tot_magnetization=2.0,
            ),
        )

    lines = ["idx\tsweep\tlabel\tinput"]
    for i, r in enumerate(runlist, 1):
        lines.append(f"{i}\t{r['sweep']}\t{r['label']}\t{r['input']}")
    (SET_DIR / "run_list.tsv").write_text("\n".join(lines) + "\n", newline="\n")
    return runlist


if __name__ == "__main__":
    rl = generate()
    print(f"Wrote {len(rl)} settings inputs under runs/settings/")
    for r in rl:
        print(f"  {r['sweep']:14s} {r['label']:22s} {r['input']}")
