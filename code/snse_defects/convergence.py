"""Generate the convergence batch: cheap QE jobs that pin the numerical settings before production.

Three sweeps on the fixed 8-atom conventional cell (SCF single points, so geometry does not move
and we isolate the numerical error), plus one vdW cell relaxation to validate the lattice:

  1. ecutwfc  - plane-wave cutoff; total energy vs ecutwfc at fixed dual (rho = 8*wfc).
  2. dual     - density cutoff; fixed ecutwfc, vary ecutrho = dual*ecutwfc.
  3. kpoints  - Monkhorst-Pack grid; total energy vs grid density.
  4. bulk vc-relax with the vdW correction - the relaxed lattice we check against experiment.

"Converged" = the total energy per atom stops changing by more than ~1 meV/atom (a standard
threshold). We read that off the outputs Michael returns from Sol.

Writes runs/convergence/<sweep>/<label>.in and a run_list.tsv the SLURM array iterates over.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .qe_input import QEInput
from .structures import load_conventional

REPO = Path(__file__).resolve().parents[2]
CONV_DIR = REPO / "runs" / "convergence"

ECUTWFC_SWEEP = [40, 50, 60, 70, 80]        # Ry
DUAL_SWEEP = [6, 8, 10, 12]                  # ecutrho = dual * ecutwfc
KGRID_SWEEP = [(2, 2, 1), (4, 4, 2), (6, 6, 3), (8, 8, 4)]


def _cfg():
    return yaml.safe_load((REPO / "config" / "calc_settings.yaml").read_text())


def _pseudos() -> dict[str, str]:
    pp = yaml.safe_load((REPO / "config" / "pseudopotentials.yaml").read_text())
    return {el: d["filename"] for el, d in pp["elements"].items()}


def generate() -> list[dict]:
    cfg = _cfg()
    pseudos = _pseudos()
    conv = load_conventional()
    base_wfc = float(cfg["electronic"]["ecutwfc_Ry"])
    base_rho = float(cfg["electronic"]["ecutrho_Ry"])
    base_k = tuple(cfg["kpoints"]["bulk_grid"])
    vdw = cfg["functional"]["vdw_correction"]
    degauss = float(cfg["electronic"]["degauss_Ry"])

    runlist: list[dict] = []

    def emit(sweep: str, label: str, qe: QEInput):
        path = qe.write(CONV_DIR / sweep / f"{label}.in")
        runlist.append(
            {"sweep": sweep, "label": label, "input": str(path.relative_to(REPO)).replace("\\", "/")}
        )

    # 1. ecutwfc sweep, SCF, dual fixed at 8, moderate k-grid
    for wfc in ECUTWFC_SWEEP:
        emit(
            "ecutwfc",
            f"ecutwfc_{wfc}",
            QEInput(
                prefix=f"snse_ecutwfc_{wfc}",
                structure=conv,
                calculation="scf",
                ecutwfc_Ry=wfc,
                ecutrho_Ry=8 * wfc,
                kpoints=base_k,
                pseudos=pseudos,
                degauss_Ry=degauss,
                vdw_corr=vdw,
            ),
        )

    # 2. dual sweep, SCF, ecutwfc fixed at the baseline
    for dual in DUAL_SWEEP:
        emit(
            "dual",
            f"dual_{dual}",
            QEInput(
                prefix=f"snse_dual_{dual}",
                structure=conv,
                calculation="scf",
                ecutwfc_Ry=base_wfc,
                ecutrho_Ry=dual * base_wfc,
                kpoints=base_k,
                pseudos=pseudos,
                degauss_Ry=degauss,
                vdw_corr=vdw,
            ),
        )

    # 3. k-point sweep, SCF, cutoffs fixed at the baseline
    for grid in KGRID_SWEEP:
        tag = "x".join(str(g) for g in grid)
        emit(
            "kpoints",
            f"kgrid_{tag}",
            QEInput(
                prefix=f"snse_kgrid_{tag}",
                structure=conv,
                calculation="scf",
                ecutwfc_Ry=base_wfc,
                ecutrho_Ry=base_rho,
                kpoints=grid,
                pseudos=pseudos,
                degauss_Ry=degauss,
                vdw_corr=vdw,
            ),
        )

    # 4. bulk vdW vc-relax - the relaxed lattice validated against experiment
    emit(
        "bulk_relax",
        "bulk_vcrelax_pbe_d3",
        QEInput(
            prefix="snse_bulk_vcrelax",
            structure=conv,
            calculation="vc-relax",
            ecutwfc_Ry=base_wfc,
            ecutrho_Ry=base_rho,
            kpoints=base_k,
            pseudos=pseudos,
            degauss_Ry=degauss,
            vdw_corr=vdw,
        ),
    )

    _write_runlist(runlist)
    return runlist


def _write_runlist(runlist: list[dict]) -> None:
    lines = ["idx\tsweep\tlabel\tinput"]
    for i, r in enumerate(runlist, 1):
        lines.append(f"{i}\t{r['sweep']}\t{r['label']}\t{r['input']}")
    (CONV_DIR / "run_list.tsv").write_text("\n".join(lines) + "\n", newline="\n")


if __name__ == "__main__":
    rl = generate()
    print(f"Wrote {len(rl)} convergence inputs under runs/convergence/")
    for r in rl:
        print(f"  {r['sweep']:11s} {r['label']:22s} {r['input']}")
