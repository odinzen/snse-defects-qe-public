"""Build the defect dataset from the harvest + config, shared by the JSON writer and carrier model.

One place that turns the harvested QE energies (data/stageB_harvest.yaml) into a DefectDataset plus
the physical extras (elemental chemical potentials, the SnSe formation enthalpy, the Sn-rich/Se-rich
host limits, the scissored gap). scripts/assemble.py renders it to JSON + diagrams; the carrier model
consumes the same object so the two can never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .chempot import formation_enthalpy_eV, host_limits
from .corrections import point_charge_correction
from .formation import DefectDataset, DefectEntry, build, elemental_placeholder
from .structures import make_bulk_supercell

REPO = Path(__file__).resolve().parents[2]
RY_TO_EV = 13.605693122994


@dataclass
class Assembled:
    ds: DefectDataset
    mu_elem: dict[str, float]          # elemental references, eV/atom (Delta_mu = 0)
    dHf_eV: float                      # DeltaH_f(SnSe) per formula unit
    limits: dict[str, dict[str, float]]  # host-corner Delta_mu for Sn / Se
    pbe_gap_eV: float
    gap_eV: float                      # scissored (experimental) gap
    volume_A3: float


def assemble_dataset() -> Assembled:
    h = yaml.safe_load((REPO / "data" / "stageB_harvest.yaml").read_text())
    cfg = yaml.safe_load((REPO / "config" / "calc_settings.yaml").read_text())
    mp = yaml.safe_load((REPO / "config" / "mp691.yaml").read_text())

    eps = mp["dielectric"]["total"]
    pbe_gap = float(cfg["gap"]["pbe_gap_eV"])
    gap = float(cfg["gap"]["experimental_gap_eV"])
    ground = cfg.get("o_i_ground_site", "O_i_2")

    prod = tuple(cfg["supercell"]["production"])
    lattice = make_bulk_supercell(prod).lattice.matrix
    volume = float(abs(make_bulk_supercell(prod).volume))
    n_fu = len(make_bulk_supercell(prod)) // 2

    e_bulk = h["bulk_3x3x1"] * RY_TO_EV
    ref = h.get("references", {})
    vbm = float(ref["vbm_eV"]) if "vbm_eV" in ref else float(h["fermi_eV"])
    mu = {el: (d["E"] / d["n"]) * RY_TO_EV for el, d in ref.get("elemental_Ry", {}).items()}
    if not mu:
        mu = elemental_placeholder()

    entries: list[DefectEntry] = []
    for name, states in h["defects"].items():
        if name.startswith("O_i_") and name != ground:
            continue
        display = "O_i" if name == ground else name
        for q, e_ry in states.items():
            entries.append(
                DefectEntry(
                    name=display,
                    charge=int(q),
                    energy_eV=e_ry * RY_TO_EV,
                    n_added={s: n for s, n in h["n_i"][name].items() if n > 0},
                    n_removed={s: -n for s, n in h["n_i"][name].items() if n < 0},
                    correction_eV=point_charge_correction(int(q), lattice, eps),
                    note="ground-state vdW-gap interstitial" if name == ground else "",
                )
            )

    ds = build(e_bulk, vbm, gap, volume, entries)
    dHf = formation_enthalpy_eV(e_bulk / n_fu, mu)
    return Assembled(ds, mu, dHf, host_limits(dHf), pbe_gap, gap, volume)
