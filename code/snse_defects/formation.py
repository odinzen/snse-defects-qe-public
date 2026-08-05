"""Assemble defect formation energies, with the chemical potentials left free.

    E_f[D,q](E_F, mu) = E_def(q) - E_bulk - sum_i n_i mu_i + q (E_VBM + E_F) + E_corr(q)

n_i is the number of atoms of species i ADDED to the perfect cell (negative if removed): V_Sn has
n_Sn = -1, O_i has n_O = +1, O_Se has n_O = +1 and n_Se = -1.

mu is deliberately NOT baked in. The chemical potentials come from the CALPHAD oxygen-buffer axis in
the main workspace (Sn-rich / Se-rich bounds, and mu_O as a function of fO2 and T), so we store the
mu-independent part plus the n_i coefficients and let the carrier model sweep conditions without
touching DFT again. Until the real mu table arrives, `elemental_placeholder` gives a clearly-labelled
provisional reference so nothing silently passes for a final number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DefectEntry:
    name: str                        # e.g. "O_i_2", "V_Sn"
    charge: int
    energy_eV: float                 # E_def(q), the defect supercell total energy
    n_added: dict[str, int]          # species -> atoms added (positive)
    n_removed: dict[str, int]        # species -> atoms removed (positive)
    correction_eV: float = 0.0       # image-charge correction, already positive-to-add
    magnetization: float | None = None
    note: str = ""

    def n_i(self) -> dict[str, int]:
        """Signed stoichiometry change: added minus removed."""
        out: dict[str, int] = {}
        for s, n in self.n_added.items():
            out[s] = out.get(s, 0) + n
        for s, n in self.n_removed.items():
            out[s] = out.get(s, 0) - n
        return {s: n for s, n in out.items() if n != 0}


@dataclass
class FormationEnergy:
    """One (defect, charge) row, evaluable at any Fermi level and chemical-potential set."""

    name: str
    charge: int
    mu_independent_eV: float         # E_def - E_bulk + q*E_VBM + E_corr
    n_i: dict[str, int]
    correction_eV: float
    magnetization: float | None = None
    note: str = ""

    def value(self, e_fermi_eV: float, mu: dict[str, float]) -> float:
        """E_f at Fermi level e_fermi (referenced to the VBM) and chemical potentials mu."""
        missing = [s for s in self.n_i if s not in mu]
        if missing:
            raise KeyError(f"no chemical potential supplied for {missing}")
        mu_term = sum(n * mu[s] for s, n in self.n_i.items())
        return self.mu_independent_eV - mu_term + self.charge * e_fermi_eV


@dataclass
class DefectDataset:
    bulk_energy_eV: float
    vbm_eV: float
    band_gap_eV: float
    volume_A3: float
    entries: list[FormationEnergy] = field(default_factory=list)

    def by_defect(self) -> dict[str, list[FormationEnergy]]:
        out: dict[str, list[FormationEnergy]] = {}
        for e in self.entries:
            out.setdefault(e.name, []).append(e)
        for v in out.values():
            v.sort(key=lambda x: x.charge)
        return out

    def transition_level(self, name: str, q1: int, q2: int, mu: dict[str, float]) -> float | None:
        """Fermi level where charge states q1 and q2 cross (a thermodynamic transition level)."""
        rows = {e.charge: e for e in self.by_defect().get(name, [])}
        if q1 not in rows or q2 not in rows:
            return None
        a, b = rows[q1], rows[q2]
        if a.charge == b.charge:
            return None
        return (b.value(0.0, mu) - a.value(0.0, mu)) / (a.charge - b.charge)


def build(
    bulk_energy_eV: float,
    vbm_eV: float,
    band_gap_eV: float,
    volume_A3: float,
    entries: list[DefectEntry],
) -> DefectDataset:
    rows = []
    for d in entries:
        mu_indep = d.energy_eV - bulk_energy_eV + d.charge * vbm_eV + d.correction_eV
        rows.append(
            FormationEnergy(
                name=d.name,
                charge=d.charge,
                mu_independent_eV=mu_indep,
                n_i=d.n_i(),
                correction_eV=d.correction_eV,
                magnetization=d.magnetization,
                note=d.note,
            )
        )
    return DefectDataset(bulk_energy_eV, vbm_eV, band_gap_eV, volume_A3, rows)


def elemental_placeholder() -> dict[str, float]:
    """PROVISIONAL chemical potentials (zero reference). Not physical - a scaffold only.

    Every number produced with these is provisional. Replace with the CALPHAD oxygen-buffer values
    (Sn-rich and Se-rich bounds, mu_O(fO2, T)) before any result is reported.
    """
    return {"Sn": 0.0, "Se": 0.0, "O": 0.0}
