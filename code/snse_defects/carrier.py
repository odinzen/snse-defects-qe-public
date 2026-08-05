"""Self-consistent defect / carrier equilibrium via py-sc-fermi.

Ties together the three pieces: the defect formation energies (formation.py, at a host limit + an
oxygen chemical potential), the scissored host DOS (MP mp-691, gap opened to experiment), and the
oxygen chemical potential Delta_mu_O(T, pO2) (oxygen.py). At each (T, pO2, host-limit) it solves for
the equilibrium Fermi level that makes the whole system charge-neutral, and returns the free-carrier
(n, p) and defect concentrations.

The MP DOS is for the 8-atom (4 f.u.) Pnma cell (V = 226.26 A^3, nelect = 80 = 4*(Sn 14 + Se 6)).
The VBM is referenced to 0 and the conduction band is rigidly shifted up so the gap matches the
scissored value; formation energies stay VBM-referenced, so the two are on one axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.electronic_structure.dos import Dos
from py_sc_fermi.defect_system import DefectSystem
from py_sc_fermi.dos import DOS

from .formation import DefectDataset
from .oxygen import delta_mu_O
from .pysc_fermi import to_py_sc_fermi

REPO = Path(__file__).resolve().parents[2]

# 8-atom Pnma mp-691 cell (the DOS cell).
DOS_NELECT = 80
DOS_VOLUME_A3 = 226.257
# Defect-site multiplicity in that 8-atom cell (4 Sn + 4 Se sites; O_i vdW-gap site taken as 4,
# a logarithmic prefactor only - the concentration is set by the exponential of E_f, not this).
NSITES = {"V_Sn": 4, "V_Se": 4, "O_Se": 4, "O_i": 4}


def scissored_dos(target_gap_eV: float, dos_json: str | Path | None = None) -> DOS:
    """MP host DOS with the VBM at 0 and the conduction band rigidly scissored to target_gap."""
    import json

    path = Path(dos_json) if dos_json else REPO / "data" / "mp_raw" / "dos.json"
    dos = Dos.from_dict(json.loads(Path(path).read_text()))
    cbm, vbm = dos.get_cbm_vbm()
    pbe_gap = cbm - vbm
    edos = np.array(dos.energies, dtype=float) - vbm            # VBM -> 0
    tot = np.array(dos.get_densities(), dtype=float)
    shift = target_gap_eV - pbe_gap
    edos[edos > pbe_gap / 2.0] += shift                        # rigid CB scissor
    return DOS(dos=tot, edos=edos, bandgap=target_gap_eV, nelect=DOS_NELECT, spin_polarised=False)


def equilibrium(
    ds: DefectDataset,
    mu_host: dict[str, float],
    T: float,
    p_O2_bar: float,
    dos: DOS,
) -> dict:
    """Solve charge neutrality at (T, pO2) for a given host-limit chemical-potential set.

    mu_host already carries the Sn/Se host-corner values and the elemental O reference; the oxygen
    offset Delta_mu_O(T, pO2) is applied here so one host limit sweeps all oxygen conditions.
    """
    mu = dict(mu_host)
    mu["O"] = mu_host["O"] + delta_mu_O(T, p_O2_bar)
    species = to_py_sc_fermi(ds, mu, nsites=NSITES)
    system = DefectSystem(species, dos, volume=DOS_VOLUME_A3, temperature=T)
    # A supersaturated defect (E_f << 0 past the oxide limit) overflows the Boltzmann prefactor to
    # inf; the equilibrium E_F is still well-defined (such defects are the neutral, isovalent ones),
    # so suppress the overflow and clip the reported concentration to a finite supersaturation flag.
    cap = 1.0e30
    with np.errstate(over="ignore"):
        conc = system.concentration_dict()       # cm^-3, plus 'Fermi Energy'
    return {
        "T": T,
        "p_O2_bar": p_O2_bar,
        "delta_mu_O": delta_mu_O(T, p_O2_bar),
        "E_fermi_eV": conc.get("Fermi Energy"),
        "n_cm3": min(conc.get("n0", 0.0), cap),
        "p_cm3": min(conc.get("p0", 0.0), cap),
        "defects_cm3": {k: min(v, cap) for k, v in conc.items()
                        if k not in ("Fermi Energy", "n0", "p0")},
    }
