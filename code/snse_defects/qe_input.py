"""Write Quantum ESPRESSO pw.x input files from a pymatgen Structure.

We write the cards explicitly (ibrav=0, CELL_PARAMETERS + ATOMIC_POSITIONS crystal) rather than
letting a converter guess, so every knob that matters for a charged-defect calc is visible and
auditable: tot_charge, the smearing, the vdW correction, and the pseudopotential filenames.

Charged cells use QE's default uniform neutralizing background (the physical content of the eFNV
correction we apply later). One Structure -> one input per charge state; positions are identical
across charge states, only tot_charge and prefix change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure

# Standard atomic masses (amu); QE wants a mass per species.
MASSES = {"Sn": 118.710, "Se": 78.971, "O": 15.999}


@dataclass
class QEInput:
    prefix: str
    structure: Structure
    calculation: str            # 'scf' | 'relax' | 'vc-relax'
    ecutwfc_Ry: float
    ecutrho_Ry: float
    kpoints: tuple[int, int, int]
    pseudos: dict[str, str]     # species -> UPF filename
    tot_charge: float = 0.0
    # 'smearing' for metals/small-gap relaxes (default); 'fixed' for a clean insulator SCF where we
    # want QE to print the true highest-occupied level (the VBM) instead of a smeared E_Fermi.
    occupations: str = "smearing"
    degauss_Ry: float = 0.005
    smearing: str = "gaussian"
    conv_thr: float = 1.0e-8
    vdw_corr: str | None = "grimme-d3"
    # input_dft overrides the functional (e.g. 'vdW-DF-obk8' = optB88-vdW). Mutually exclusive with
    # vdw_corr - a vdW-DF functional is not a D3 correction. Set vdw_corr=None when using this.
    input_dft: str | None = None
    tot_magnetization: float | None = None   # e.g. 2.0 for the O2 triplet (needs nspin=2)
    # per-species initial moment (fraction of valence), e.g. {"Sn":0.1}. Lets nspin=2 find the spin
    # ground state instead of fixing it. Emitted as starting_magnetization(i) for species i.
    starting_magnetization: dict[str, float] | None = None
    # Omit pseudo_dir so QE falls back to $ESPRESSO_PSEUDO (set by the SLURM script). This avoids
    # brittle relative paths across the different job-folder depths. Set explicitly only for a
    # local run where you want a fixed directory.
    pseudo_dir: str | None = None
    outdir: str = "./out"
    mixing_beta: float = 0.3
    etot_conv_thr_Ry: float = 1.0e-4
    forc_conv_thr: float = 1.0e-3
    nspin: int = 1

    def species(self) -> list[str]:
        # order of first appearance keeps ATOMIC_SPECIES and positions consistent
        seen: list[str] = []
        for site in self.structure:
            sym = site.specie.symbol
            if sym not in seen:
                seen.append(sym)
        return seen

    def render(self) -> str:
        syms = self.species()
        nat = len(self.structure)
        ntyp = len(syms)
        gamma_only = tuple(self.kpoints) == (1, 1, 1)

        lines: list[str] = []
        lines.append("&CONTROL")
        lines.append(f"    calculation = '{self.calculation}'")
        lines.append(f"    prefix = '{self.prefix}'")
        lines.append(f"    outdir = '{self.outdir}'")
        if self.pseudo_dir:
            lines.append(f"    pseudo_dir = '{self.pseudo_dir}'")
        lines.append("    tprnfor = .true.")
        lines.append("    tstress = .true.")
        lines.append("    verbosity = 'high'")
        if self.calculation != "scf":
            lines.append(f"    etot_conv_thr = {self.etot_conv_thr_Ry:.3e}")
            lines.append(f"    forc_conv_thr = {self.forc_conv_thr:.3e}")
        lines.append("/")

        lines.append("&SYSTEM")
        lines.append("    ibrav = 0")
        lines.append(f"    nat = {nat}")
        lines.append(f"    ntyp = {ntyp}")
        lines.append(f"    ecutwfc = {self.ecutwfc_Ry:g}")
        lines.append(f"    ecutrho = {self.ecutrho_Ry:g}")
        lines.append(f"    occupations = '{self.occupations}'")
        if self.occupations == "smearing":
            lines.append(f"    smearing = '{self.smearing}'")
            lines.append(f"    degauss = {self.degauss_Ry:g}")
        if abs(self.tot_charge) > 0:
            lines.append(f"    tot_charge = {self.tot_charge:g}")
        lines.append(f"    nspin = {self.nspin}")
        if self.tot_magnetization is not None:
            lines.append(f"    tot_magnetization = {self.tot_magnetization:g}")
        if self.starting_magnetization and self.nspin == 2:
            for i, sym in enumerate(syms, start=1):
                if sym in self.starting_magnetization:
                    lines.append(f"    starting_magnetization({i}) = {self.starting_magnetization[sym]:g}")
        if self.input_dft:
            lines.append(f"    input_dft = '{self.input_dft}'")
        if self.vdw_corr and not self.input_dft:
            lines.append(f"    vdw_corr = '{self.vdw_corr}'")
        lines.append("/")

        lines.append("&ELECTRONS")
        lines.append(f"    conv_thr = {self.conv_thr:.3e}")
        lines.append(f"    mixing_beta = {self.mixing_beta:g}")
        lines.append("/")

        if self.calculation in ("relax", "vc-relax"):
            lines.append("&IONS")
            lines.append("    ion_dynamics = 'bfgs'")
            lines.append("/")
        if self.calculation == "vc-relax":
            lines.append("&CELL")
            lines.append("    cell_dynamics = 'bfgs'")
            lines.append("/")

        lines.append("ATOMIC_SPECIES")
        for sym in syms:
            pp = self.pseudos.get(sym)
            if not pp:
                raise ValueError(f"no pseudopotential filename for {sym}")
            lines.append(f"  {sym:2s} {MASSES[sym]:10.4f}  {pp}")

        lines.append("CELL_PARAMETERS angstrom")
        for v in self.structure.lattice.matrix:
            lines.append(f"  {v[0]:18.12f} {v[1]:18.12f} {v[2]:18.12f}")

        lines.append("ATOMIC_POSITIONS crystal")
        for site in self.structure:
            f = site.frac_coords
            lines.append(f"  {site.specie.symbol:2s} {f[0]:18.12f} {f[1]:18.12f} {f[2]:18.12f}")

        lines.append("K_POINTS " + ("gamma" if gamma_only else "automatic"))
        if not gamma_only:
            k = self.kpoints
            lines.append(f"  {k[0]} {k[1]} {k[2]} 0 0 0")

        return "\n".join(lines) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # LF always - these are read by pw.x on Linux; CRLF can trip the namelist reader.
        path.write_text(self.render(), newline="\n")
        return path
