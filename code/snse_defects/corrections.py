"""Finite-size (image-charge) corrections for charged defects in a periodic supercell.

A charged defect in a periodic cell interacts with its own images and with the neutralizing
background, which spuriously stabilises the cell. The correction removes that.

What is implemented here is the screened point-charge (Makov-Payne / Lany-Zunger monopole) term:

    E_corr = -q^2 * xi / (2 * eps_eff)          [added to the DFT total energy]

where xi is the Madelung potential of a point charge in a uniform neutralizing background for THIS
supercell, obtained by Ewald summation (not a tabulated constant - our cells are not cubic), and
eps_eff is the effective static dielectric constant.

SnSe's dielectric tensor is strongly anisotropic (eps_0 = 52.8 / 46.7 / 82.5), and the standard
treatment for an anisotropic medium in this approximation is the geometric mean, which is the
determinant^(1/3) of the tensor - it is what the coordinate rescaling that maps the anisotropic
Poisson problem onto an isotropic one leaves behind.

LIMITATION, read before trusting these numbers: this is the leading monopole term only. The full
eFNV (Kumagai-Oba) scheme additionally aligns the potential far from the defect using the
electrostatic potential at atomic sites, which QE writes only through a separate pp.x run. Until
those potentials exist, treat E_corr here as a good leading estimate, not the final word. The
size-check (72 vs 144 atom cell) is what tells us how much residual is left.
"""

from __future__ import annotations

import math

import numpy as np

BOHR_PER_ANG = 1.8897261254535
HARTREE_TO_EV = 27.211386245988

try:  # scipy gives a vectorised erfc; fall back to a numpy-mapped math.erfc if absent
    from scipy.special import erfc as erfc_vec
except ImportError:  # pragma: no cover
    erfc_vec = np.vectorize(math.erfc)


def effective_dielectric(tensor: list[list[float]] | np.ndarray) -> float:
    """Geometric mean (det^(1/3)) of the static dielectric tensor."""
    eps = np.asarray(tensor, dtype=float)
    det = float(np.linalg.det(eps))
    if det <= 0:
        raise ValueError("dielectric tensor must be positive definite")
    return det ** (1.0 / 3.0)


def madelung_potential(lattice_ang: np.ndarray, eta: float | None = None,
                       r_cut: float | None = None, g_cut: float | None = None) -> float:
    """Ewald sum for xi: the potential at a point charge from its images + jellium background.

    Returns xi in Hartree/e^2 per unit charge (i.e. atomic units, 1/Bohr). Negative.

    xi = sum'_R erfc(eta R)/R  +  (4 pi / V) sum_{G!=0} exp(-G^2/4 eta^2)/G^2
         - 2 eta / sqrt(pi)                      (self term)
         - pi / (eta^2 V)                        (G=0 / neutralizing background term)
    """
    A = np.asarray(lattice_ang, dtype=float) * BOHR_PER_ANG   # rows are lattice vectors, in Bohr
    V = abs(float(np.linalg.det(A)))
    B = 2.0 * math.pi * np.linalg.inv(A).T                    # reciprocal lattice rows

    if eta is None:
        # standard balanced choice
        eta = math.sqrt(math.pi) / V ** (1.0 / 3.0)

    # Cutoffs must follow eta or neither sum converges (the real-space term decays as erfc(eta*r),
    # the reciprocal one as exp(-G^2/4eta^2)). These give ~1e-10 truncation on both sides; the test
    # for whether this is right is that xi comes out independent of eta.
    if r_cut is None:
        r_cut = 6.0 / eta          # erfc(6) ~ 2e-17
    if g_cut is None:
        g_cut = 12.0 * eta         # exp(-36) ~ 2e-16

    def _shell(basis: np.ndarray, cut: float) -> np.ndarray:
        """All lattice vectors from `basis` within `cut`, excluding the origin (vectorised)."""
        n = [int(math.ceil(cut / np.linalg.norm(basis[i]))) + 1 for i in range(3)]
        grids = np.meshgrid(
            np.arange(-n[0], n[0] + 1),
            np.arange(-n[1], n[1] + 1),
            np.arange(-n[2], n[2] + 1),
            indexing="ij",
        )
        idx = np.stack([g.ravel() for g in grids], axis=1)
        idx = idx[~np.all(idx == 0, axis=1)]
        vecs = idx @ basis
        norms = np.linalg.norm(vecs, axis=1)
        return norms[norms <= cut]

    r = _shell(A, r_cut)
    real = float(np.sum(erfc_vec(eta * r) / r))

    g = _shell(B, g_cut)
    g2 = g**2
    recip = float(np.sum(np.exp(-g2 / (4.0 * eta**2)) / g2)) * 4.0 * math.pi / V

    self_term = -2.0 * eta / math.sqrt(math.pi)
    background = -math.pi / (eta**2 * V)

    return real + recip + self_term + background


def madelung_constant(lattice_ang: np.ndarray) -> float:
    """Dimensionless alpha, defined by xi = -alpha / L with L = V^(1/3). Positive."""
    A = np.asarray(lattice_ang, dtype=float) * BOHR_PER_ANG
    L = abs(float(np.linalg.det(A))) ** (1.0 / 3.0)
    return -madelung_potential(lattice_ang) * L


def point_charge_correction(q: float, lattice_ang: np.ndarray,
                            dielectric: list[list[float]] | np.ndarray) -> float:
    """Screened monopole image-charge correction, in eV, to ADD to the DFT total energy.

    Zero for a neutral defect. Positive for any charged state (it removes a spurious stabilisation).
    """
    if q == 0:
        return 0.0
    xi = madelung_potential(lattice_ang)          # atomic units, negative
    eps = effective_dielectric(dielectric)
    e_hartree = -(q**2) * xi / (2.0 * eps)
    return e_hartree * HARTREE_TO_EV
