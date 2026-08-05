"""Foundation-MLIP calculators + relaxation, shared by the calibration and oxide-cap scripts.

Every model that imports is returned; missing ones are skipped with a note. All are PBE-family; MACE
and ORB also offer a D3-corrected variant to approach our PBE-D3. Used only for NEUTRAL structural /
formation-energy screening - charged-defect energetics stay on QE.
"""

from __future__ import annotations

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ORB_MODEL = "orb-v3-conservative-inf-omat"   # ORB v3, OMat24-trained (PBE), energy-conserving


def get_calculators(which: tuple[str, ...] | None = None) -> dict:
    """Build available MLIP ASE calculators. `which` filters by name substring if given."""
    calcs: dict = {}

    def want(cat: str) -> bool:
        # symmetric substring match so both category ("MACE") and full-name ("MACE-MP") filters work
        return which is None or any(w in cat or cat in w for w in which)

    if want("MACE"):
        try:
            from mace.calculators import mace_mp
            calcs["MACE-MP"] = mace_mp(model="medium", dispersion=False,
                                       default_dtype="float64", device="cpu")
            calcs["MACE-MP+D3"] = mace_mp(model="medium", dispersion=True,
                                          default_dtype="float64", device="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"  MACE unavailable: {e}")
    if want("CHGNet"):
        try:
            from chgnet.model.dynamics import CHGNetCalculator
            calcs["CHGNet"] = CHGNetCalculator()
        except Exception as e:  # noqa: BLE001
            print(f"  CHGNet unavailable: {e}")
    if want("ORB"):
        try:
            # ORB v3 uses torch.compile, which needs an MSVC C++ compiler on Windows; fall back to
            # eager execution when it (or the compiler) is missing rather than erroring out.
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.inference.calculator import ORBCalculator
            model, adapter = pretrained.ORB_PRETRAINED_MODELS[ORB_MODEL](
                device="cpu", precision="float32-high")
            calcs["ORB-v3"] = ORBCalculator(model, adapter, device="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"  ORB unavailable: {e}")
    return calcs


def relax_energy(atoms: Atoms, calc, *, fmax: float = 0.02, steps: int = 300, cell: bool = True) -> float:
    """Relax (cell + positions by default) and return the total energy in eV."""
    a = atoms.copy()
    a.calc = calc
    target = FrechetCellFilter(a) if cell else a
    FIRE(target, logfile=None).run(fmax=fmax, steps=steps)
    return float(a.get_potential_energy())


def o2_molecule(box: float = 12.0, bond: float = 1.21) -> Atoms:
    """Isolated O2 in a cubic box (for the oxygen reference). MLIPs are weak on isolated molecules,
    so oxide formation energies built on this are validated against experiment downstream."""
    c = box / 2.0
    return Atoms("O2", positions=[[c, c, c - bond / 2], [c, c, c + bond / 2]],
                 cell=[box, box, box], pbc=True)
