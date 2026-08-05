"""Emit the deliverable: the formation-energy dataset, the defect diagram, and py-sc-fermi input.

The JSON dataset in data/ is the precious artefact - it is what the carrier model consumes and what
must survive. It records not just the numbers but how they were made (settings, corrections, which
chemical potentials were in force, and whether they are provisional).
"""

from __future__ import annotations

import json
from pathlib import Path

from .formation import DefectDataset

REPO = Path(__file__).resolve().parents[2]

# Line styles for the greyscale house aesthetic - defects are distinguished by dash pattern and
# marker, not colour, so the figure survives black-and-white printing.
_STYLES = [
    (0.0, (1, 0)), (0.0, (4, 2)), (0.0, (1, 1.5)),
    (0.0, (6, 2, 1, 2)), (0.0, (3, 1, 1, 1, 1, 1)), (0.0, (8, 3)),
]


def write_dataset(
    ds: DefectDataset,
    mu: dict[str, float],
    path: str | Path,
    *,
    provisional: bool = True,
    provenance: dict | None = None,
    chempot: dict | None = None,
) -> Path:
    payload = {
        "description": "Charged point-defect formation energies for O in SnSe (QE, PBE-D3).",
        "provisional": provisional,
        "provisional_reason": (
            "host bounds + true VBM + scissored gap are in; this dataset holds oxygen at the O-rich "
            "reference, and the realized Delta_mu_O(fO2,T) from gas-phase O2 (NIST-JANAF) is "
            "applied in the carrier-model sweep"
            if provisional
            else None
        ),
        "reference": {
            "bulk_energy_eV": ds.bulk_energy_eV,
            "vbm_eV": ds.vbm_eV,
            "band_gap_eV": ds.band_gap_eV,
            "volume_A3": ds.volume_A3,
        },
        "chemical_potentials_eV": mu,
        "chemical_potential_limits": chempot or {},
        "provenance": provenance or {},
        "entries": [
            {
                "defect": e.name,
                "charge": e.charge,
                # E_f at the VBM (E_F = 0) under the supplied mu - what py-sc-fermi ingests
                "formation_energy_at_vbm_eV": round(e.value(0.0, mu), 6),
                "mu_independent_eV": round(e.mu_independent_eV, 6),
                "n_i": e.n_i,
                "image_charge_correction_eV": round(e.correction_eV, 6),
                "total_magnetization": e.magnetization,
                "note": e.note,
            }
            for e in ds.entries
        ],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return p


def to_py_sc_fermi(ds: DefectDataset, mu: dict[str, float], nsites: dict[str, int] | None = None):
    """Build a py-sc-fermi DefectSystem. Requires the MP DOS; import is local so the rest of the
    module works without py-sc-fermi installed."""
    from py_sc_fermi.defect_charge_state import DefectChargeState
    from py_sc_fermi.defect_species import DefectSpecies

    nsites = nsites or {}
    species = []
    for name, rows in ds.by_defect().items():
        charge_states = {
            r.charge: DefectChargeState(charge=r.charge, energy=r.value(0.0, mu), degeneracy=1)
            for r in rows
        }
        species.append(DefectSpecies(name, nsites.get(name, 1), charge_states))
    return species


def _defect_label(name: str) -> str:
    # V_Sn -> $\mathrm{V_{Sn}}$ so the legend carries true subscripts, matching the manuscript.
    base, _, sub = name.partition("_")
    return rf"$\mathrm{{{base}_{{{sub}}}}}$" if sub else rf"$\mathrm{{{base}}}$"


def plot_formation_diagram(
    ds: DefectDataset,
    mu: dict[str, float],
    path: str | Path,
) -> Path:
    # No in-figure title: the manuscript caption names the figure (Kristina standard,
    # no baked text on figures). Greyscale, >=600 DPI.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ef = np.linspace(0.0, ds.band_gap_eV, 400)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    for i, (name, rows) in enumerate(sorted(ds.by_defect().items())):
        # the observable is the lowest-energy charge state at each Fermi level
        stacked = np.vstack([[r.value(x, mu) for x in ef] for r in rows])
        lower = stacked.min(axis=0)
        dash = _STYLES[i % len(_STYLES)][1]
        ax.plot(ef, lower, color="black", linewidth=1.4, dashes=dash, label=_defect_label(name))

    ax.set_xlabel("Fermi level above VBM (eV)")
    ax.set_ylabel("Formation energy (eV)")
    ax.set_xlim(0.0, ds.band_gap_eV)
    ax.margins(y=0.12)                       # keep the top curve off the frame
    ax.axvline(0.0, color="0.6", linewidth=0.8)
    ax.axvline(ds.band_gap_eV, color="0.6", linewidth=0.8)
    # legend outside the axes so it can never sit on top of a curve
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, color="0.9", linewidth=0.6)
    fig.tight_layout()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=600, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p
