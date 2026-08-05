"""Chemical-potential limits for the SnSe:O defect formation energies.

The chemical potentials are bounded by the host thermodynamics, not free. In equilibrium with SnSe,
    mu_Sn + mu_Se = mu_SnSe = E(SnSe)/f.u.,   i.e.   Delta_mu_Sn + Delta_mu_Se = DeltaH_f(SnSe),
with each Delta_mu_i <= 0 (a positive value would precipitate the pure element). That fixes the two
host corners:

    Sn-rich:  Delta_mu_Sn = 0,          Delta_mu_Se = DeltaH_f
    Se-rich:  Delta_mu_Se = 0,          Delta_mu_Sn = DeltaH_f

Oxygen is the third axis. Delta_mu_O <= 0 (referenced to 1/2 E(O2)); Delta_mu_O = 0 is the nominal
O-rich reference. The realized Delta_mu_O(fO2, T) comes from the workspace CALPHAD oxygen buffer, and
the thermodynamic upper cap is actually set by the onset of Sn/Se oxide formation (SnO2, SnO, SeO2) -
neither of which is computed here, so O-rich = 0 is a labelled reference, not a hard cap.

Delta_mu here is referenced to the elemental DFT energies in `mu_elem` (eV/atom); the full potential
used in a formation energy is mu_i = mu_elem_i + Delta_mu_i.
"""

from __future__ import annotations


def formation_enthalpy_eV(e_bulk_per_fu_eV: float, mu_elem: dict[str, float]) -> float:
    """DeltaH_f(SnSe) per formula unit = E(SnSe)/f.u. - mu_Sn(elem) - mu_Se(elem)."""
    return e_bulk_per_fu_eV - mu_elem["Sn"] - mu_elem["Se"]


def host_limits(dHf_eV: float) -> dict[str, dict[str, float]]:
    """Delta_mu (referenced to the elements) for Sn and Se at the two host stoichiometry corners."""
    return {
        "Sn-rich": {"Sn": 0.0, "Se": dHf_eV},
        "Se-rich": {"Sn": dHf_eV, "Se": 0.0},
    }


def mu_at(
    mu_elem: dict[str, float],
    limit_dmu: dict[str, float],
    dmu_O_eV: float = 0.0,
) -> dict[str, float]:
    """Full chemical potentials at a host limit and a given oxygen offset.

    mu_i = mu_elem_i + Delta_mu_i. Sn/Se take the limit's Delta_mu; O takes dmu_O (0 = O-rich ref).
    """
    mu = dict(mu_elem)
    for el, d in limit_dmu.items():
        mu[el] = mu_elem[el] + d
    if "O" in mu_elem:
        mu["O"] = mu_elem["O"] + dmu_O_eV
    return mu
