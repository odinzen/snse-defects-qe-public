"""Oxygen chemical potential Delta_mu_O(T, pO2) from ideal-gas O2 thermochemistry.

Ab-initio-thermodynamics (Reuter-Scheffler) reference for the oxygen axis, referenced to 1/2 the
DFT O2 total energy so it drops straight into the defect formation energies:

    Delta_mu_O(T, p) = 1/2 [ mu~_O2(T, p0) + kB T ln(p/p0) ],
    mu~_O2(T, p0)    = [H(T) - H(0)] - T S(T),

with H(T), S(T) from the NIST-JANAF O2 Shomate fit (100-2000 K) and H(298.15)-H(0) = 8.683 kJ/mol.
Delta_mu_O = 0 is the O-rich reference (T -> 0 / infinite pressure); real conditions give < 0.

This is the standard gas-phase reference; it can be swapped for the workspace CALPHAD oxygen buffer
if a more integrated Sn-Se-O treatment is wanted. Shomate coefficients transcribed from the NIST
WebBook and verified in _selftest() against tabulated S(298), Cp(298).
"""

from __future__ import annotations

import math

KB_EV = 8.617333262e-5          # eV/K
KJMOL_PER_EV = 96.485           # 1 eV = 96.485 kJ/mol
H298_MINUS_H0_KJ = 8.683        # O2 gas, JANAF

# NIST Shomate for O2, (A,B,C,D,E,F,G) per temperature window (H term is 0 for O2).
_SHOMATE = [
    (100.0, 700.0, (31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 246.7945)),
    (700.0, 2000.0, (30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663)),
]


def _coeffs(T: float):
    for lo, hi, c in _SHOMATE:
        if lo <= T <= hi:
            return c
    if T < _SHOMATE[0][0]:
        return _SHOMATE[0][2]
    return _SHOMATE[-1][2]


def _cp_S_H(T: float) -> tuple[float, float, float]:
    """Cp [J/mol/K], S [J/mol/K], H(T)-H(298.15) [kJ/mol] from the Shomate fit."""
    A, B, C, D, E, F, G = _coeffs(T)
    t = T / 1000.0
    cp = A + B * t + C * t**2 + D * t**3 + E / t**2
    s = A * math.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G
    h = A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F
    return cp, s, h


def delta_mu_O(T: float, p_bar: float = 1.0) -> float:
    """Delta_mu_O relative to 1/2 E(O2)^DFT, in eV, at temperature T (K) and O2 pressure p (bar)."""
    _, s, h_rel298 = _cp_S_H(T)
    h_rel0_kJ = h_rel298 + H298_MINUS_H0_KJ            # H(T) - H(0)
    mu_tilde_kJ = h_rel0_kJ - T * s / 1000.0           # [H(T)-H(0)] - T S(T), kJ/mol
    mu_tilde_eV = mu_tilde_kJ / KJMOL_PER_EV
    return 0.5 * (mu_tilde_eV + KB_EV * T * math.log(p_bar))


def _selftest() -> None:
    cp298, s298, _ = _cp_S_H(298.15)
    print(f"O2 at 298.15 K:  Cp = {cp298:.2f} (JANAF 29.38)   S = {s298:.2f} (JANAF 205.15) J/mol/K")
    for T in (300, 500, 700, 1000):
        print(f"  Delta_mu_O({T} K, 1 bar) = {delta_mu_O(T):+.3f} eV   "
              f"(0.2 bar air) = {delta_mu_O(T, 0.2):+.3f} eV")


if __name__ == "__main__":
    _selftest()
