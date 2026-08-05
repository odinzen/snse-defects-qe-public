"""Generate the production defect batch at the locked settings.

Locked (from runs/convergence + runs/settings):
  PBE-D3, PBE-D3-relaxed host cell, ecutwfc 80 / ecutrho 640, k = 2x2x3 on the 72-atom (3x3x1) cell.

Jobs:
  - one pristine bulk supercell (SCF; the E_bulk reference and, via its eigenvalues, E_VBM),
  - V_Sn, V_Se, O_Se, and O_i (best Voronoi site) across their charge states (relax, spin-polarized),
  - the two other O_i candidate sites at neutral only (site screen - confirm which O_i is lowest),
  - a size-check pair in the 144-atom (3x3x2) cell: pristine bulk + O_i(+1), to converge the
    finite-size correction.

Defects relax the ions at FIXED cell (dilute limit, host cell from the converged bulk). Charge is
QE tot_charge = q (cell charge = defect charge state). nspin=2 with a small starting moment lets QE
find the spin ground state (m=0 for closed-shell states, nonzero for odd-electron ones).

Writes runs/production/<label>.in (flat) + run_list.tsv.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from .qe_input import QEInput
from .structures import all_defects, make_bulk_supercell

REPO = Path(__file__).resolve().parents[2]
START_MOMENT = {"Sn": 0.1, "Se": 0.1, "O": 0.1}

# Two-stage strategy. A cold relax at the full settings costs 2-4 h per ionic step (measured: the
# 72-atom bulk SCF alone took 2h13m on 32 cores), so a 20-40 step relaxation never finishes. Stage A
# relaxes cheaply to get the geometry close; stage B re-relaxes at the FULL locked settings warm-
# started from it, so the final geometry and energy always come from the converged settings - stage A
# is only a head start, never an approximation we trust.
STAGES = {
    # ~20x cheaper per step: Gamma-only (vs 12 k-points) and 50/400 (vs 80/640)
    "relax": {"dir": "prod_relax", "ecutwfc": 50.0, "ecutrho": 400.0, "kgrid": (1, 1, 1)},
    # full locked settings; 2x2x2 on the near-cubic 72-atom supercell (6x6x2 unit-cell equivalent)
    "final": {"dir": "prod_final", "ecutwfc": 80.0, "ecutrho": 640.0, "kgrid": (2, 2, 2)},
}


def _cfg():
    return yaml.safe_load((REPO / "config" / "calc_settings.yaml").read_text())


def _pseudos() -> dict[str, str]:
    pp = yaml.safe_load((REPO / "config" / "pseudopotentials.yaml").read_text())
    return {el: d["filename"] for el, d in pp["elements"].items()}


def _supercell_kgrid(multiple, bulk_grid) -> tuple[int, int, int]:
    """k-grid matching the bulk density: ceil(bulk_k / supercell_multiple), min 1."""
    return tuple(max(1, math.ceil(bulk_grid[i] / multiple[i])) for i in range(3))


def generate(stage: str = "relax", only: str | None = None, out_suffix: str = "") -> list[dict]:
    """`only` keeps just labels containing that substring; `out_suffix` writes to a sibling dir.

    Used to emit a small follow-up batch (e.g. only the O_i_2 charge states) without disturbing an
    existing run directory.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    st = STAGES[stage]
    out_dir = REPO / "runs" / (st["dir"] + out_suffix)

    cfg = _cfg()
    pp = _pseudos()
    wfc = st["ecutwfc"]
    rho = st["ecutrho"]
    k_prod = st["kgrid"]
    degauss = float(cfg["electronic"]["degauss_Ry"])
    vdw = cfg["functional"]["vdw_correction"]
    prod = tuple(cfg["supercell"]["production"])     # [3,3,1]
    charge_states = cfg["charge_states"]

    runlist: list[dict] = []

    def emit(label: str, group: str, qe: QEInput):
        if only and only not in label:
            return
        path = qe.write(out_dir / f"{label}.in")
        runlist.append(
            {"group": group, "label": label, "input": str(path.relative_to(REPO)).replace("\\", "/")}
        )

    def defect_input(prefix, structure, q, species, kgrid) -> QEInput:
        return QEInput(
            prefix=prefix,
            structure=structure,
            calculation="relax",
            ecutwfc_Ry=wfc,
            ecutrho_Ry=rho,
            kpoints=kgrid,
            pseudos={s: pp[s] for s in species},
            tot_charge=float(q),
            degauss_Ry=degauss,
            vdw_corr=vdw,
            nspin=2,
            starting_magnetization={s: START_MOMENT[s] for s in species},
        )

    def species_of(structure) -> list[str]:
        seen = []
        for site in structure:
            if site.specie.symbol not in seen:
                seen.append(site.specie.symbol)
        return seen

    # --- bulk reference (72-atom pristine, SCF, non-magnetic) ---
    # Only in the final stage: the pristine supercell is a tiling of the already-relaxed cell, so it
    # needs no geometry search. It must share the final stage's exact settings with the defects, or
    # the E_defect - E_bulk difference does not cancel systematic error.
    if stage == "final":
        bulk72 = make_bulk_supercell(prod)
        emit(
            "bulk_3x3x1",
            "reference",
            QEInput(
                prefix="bulk_3x3x1",
                structure=bulk72,
                calculation="scf",
                ecutwfc_Ry=wfc,
                ecutrho_Ry=rho,
                kpoints=k_prod,
                pseudos={s: pp[s] for s in species_of(bulk72)},
                degauss_Ry=degauss,
                vdw_corr=vdw,
                nspin=1,
            ),
        )

    # --- defects in the production (72-atom) cell ---
    defects = all_defects(prod)
    by_name = {d.name: d for d in defects}

    # native + substitution: full charge-state ladders
    for name in ("V_Sn", "V_Se", "O_Se"):
        dc = by_name[name]
        species = species_of(dc.structure)
        for q in charge_states[name]:
            label = f"{name}_q{q:+d}".replace("+", "p").replace("-", "m")
            emit(label, name, defect_input(f"{label}", dc.structure, q, species, k_prod))

    # interstitials: full ladder on the GROUND-STATE site, neutral-only screen on the others.
    # The ground site is not the generator's first candidate - stage A showed O_i_2 (van der Waals
    # gap) sits 0.53 eV below O_i_0, so the site is read from config, not assumed.
    ground = cfg.get("o_i_ground_site", "O_i_0")
    for name in ("O_i_0", "O_i_1", "O_i_2"):
        dc = by_name[name]
        species = species_of(dc.structure)
        qs = charge_states["O_i"] if name == ground else [0]
        for q in qs:
            label = f"{name}_q{q:+d}".replace("+", "p").replace("-", "m")
            emit(label, "O_i", defect_input(f"{label}", dc.structure, q, species, k_prod))

    # The 144-atom (3x3x2) size-check is DEFERRED until the 72-atom results are in. Its first attempt
    # died OUT_OF_MEMORY at 64 GB (MaxRSS 67 GB); it needs mem=0 (whole node) and is the most
    # expensive job in the study, so it is not worth blocking the main batch on.

    lines = ["idx\tgroup\tlabel\tinput"]
    for i, r in enumerate(runlist, 1):
        lines.append(f"{i}\t{r['group']}\t{r['label']}\t{r['input']}")
    (out_dir / "run_list.tsv").write_text("\n".join(lines) + "\n", newline="\n")
    return runlist


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate a production stage.")
    ap.add_argument("--stage", default="relax", choices=list(STAGES))
    ap.add_argument("--only", default=None, help="keep only labels containing this substring")
    ap.add_argument("--out-suffix", default="", help="write to runs/<stagedir><suffix>/")
    a = ap.parse_args()
    rl = generate(a.stage, only=a.only, out_suffix=a.out_suffix)
    print(f"Wrote {len(rl)} inputs under runs/{STAGES[a.stage]['dir']}{a.out_suffix}/")
    grp = None
    for r in rl:
        if r["group"] != grp:
            grp = r["group"]
            print(f"  [{grp}]")
        print(f"    {r['label']:24s} {r['input']}")
