"""Parse Quantum ESPRESSO pw.x output into the quantities the defect analysis needs.

Deliberately tolerant: QE prints many blocks and a job may be a relax (many ionic steps) or an scf
(one). We always take the LAST converged values, and we report explicitly whether the run finished
and whether a relaxation actually converged, so a half-finished job can never be mistaken for a
result. That distinction bit us once already - unconverged geometries reported spurious magnetic
moments that vanished on convergence.

Energies are converted Ry -> eV at parse time; everything downstream is eV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

RY_TO_EV = 13.605693122994

_RE_ENERGY = re.compile(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", re.M)
_RE_FERMI = re.compile(r"the Fermi energy is\s+(-?\d+\.\d+)\s*ev", re.I)
_RE_HOMO_LUMO = re.compile(
    r"highest occupied,?\s*lowest unoccupied level \(ev\):\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", re.I
)
_RE_HOMO_ONLY = re.compile(r"highest occupied level \(ev\):\s+(-?\d+\.\d+)", re.I)
_RE_MAG = re.compile(r"^\s+total magnetization\s+=\s+(-?\d+\.\d+)", re.M)
_RE_NAT = re.compile(r"number of atoms/cell\s+=\s+(\d+)")
_RE_NPROC = re.compile(r"running on\s+(\d+)\s+processor")
_RE_POS_LINE = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


@dataclass
class QEResult:
    path: str
    finished: bool                 # QE printed "JOB DONE."
    relax_converged: bool | None   # True/False for a relax; None for an scf
    calculation: str               # scf | relax | vc-relax (as read from the input echo)
    energy_eV: float | None
    n_scf_cycles: int              # number of converged SCF cycles (~ ionic steps for a relax)
    fermi_eV: float | None
    vbm_eV: float | None           # highest occupied level, when QE reports it
    cbm_eV: float | None
    total_magnetization: float | None
    n_atoms: int | None
    n_procs: int | None
    species: list[str] = field(default_factory=list)
    frac_coords: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """A result we are willing to build numbers on."""
        if not self.finished or self.energy_eV is None:
            return False
        if self.relax_converged is False:
            return False
        return True


def _last_positions(text: str) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Last ATOMIC_POSITIONS block. Prefers the 'final coordinates' block when present."""
    final = re.search(
        r"Begin final coordinates(.*?)End final coordinates", text, re.S
    )
    scope = final.group(1) if final else text
    blocks = list(re.finditer(r"ATOMIC_POSITIONS[^\n]*\n", scope))
    if not blocks:
        return [], []
    tail = scope[blocks[-1].end():]
    species: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for line in tail.splitlines():
        m = _RE_POS_LINE.match(line)
        if m:
            species.append(m.group(1))
            coords.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
        elif species:
            break
    return species, coords


def parse_output(path: str | Path) -> QEResult:
    p = Path(path)
    text = p.read_text(errors="replace")

    energies = _RE_ENERGY.findall(text)
    energy = float(energies[-1]) * RY_TO_EV if energies else None

    calc_m = re.search(r"calculation\s*=\s*'([a-z\-]+)'", text)
    calculation = calc_m.group(1) if calc_m else ("relax" if "BFGS" in text else "scf")

    if calculation == "scf":
        relax_converged: bool | None = None
    else:
        relax_converged = "Begin final coordinates" in text or "bfgs converged" in text.lower()

    fermi_m = _RE_FERMI.search(text)
    hl = _RE_HOMO_LUMO.search(text)
    ho = _RE_HOMO_ONLY.search(text)
    vbm = float(hl.group(1)) if hl else (float(ho.group(1)) if ho else None)
    cbm = float(hl.group(2)) if hl else None

    mags = _RE_MAG.findall(text)
    nat_m = _RE_NAT.search(text)
    np_m = _RE_NPROC.search(text)
    species, coords = _last_positions(text)

    return QEResult(
        path=str(p),
        finished="JOB DONE." in text,
        relax_converged=relax_converged,
        calculation=calculation,
        energy_eV=energy,
        n_scf_cycles=len(energies),
        fermi_eV=float(fermi_m.group(1)) if fermi_m else None,
        vbm_eV=vbm,
        cbm_eV=cbm,
        total_magnetization=float(mags[-1]) if mags else None,
        n_atoms=int(nat_m.group(1)) if nat_m else None,
        n_procs=int(np_m.group(1)) if np_m else None,
        species=species,
        frac_coords=coords,
    )


def parse_batch(run_list: str | Path) -> dict[str, QEResult]:
    """Parse every job in a run_list.tsv, keyed by label. Missing outputs are skipped."""
    rl = Path(run_list)
    repo = rl.resolve().parents[2]
    out: dict[str, QEResult] = {}
    for line in rl.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        label, inp = parts[2], parts[3]
        outfile = repo / (inp[:-3] + ".out")
        if outfile.exists():
            out[label] = parse_output(outfile)
    return out
