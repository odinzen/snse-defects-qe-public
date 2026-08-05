"""Assemble the formation-energy dataset from a Stage B harvest.

Reads data/stageB_harvest.yaml (converged QE total energies) + config (dielectric, gap, host cell),
applies the image-charge correction, and writes the py-sc-fermi dataset JSON + the defect diagrams.

The DFT side is complete: true VBM (fixed-occupation SCF) and elemental chemical-potential references
are both in. Chemical potentials are bounded by the SnSe formation enthalpy (Sn-rich / Se-rich host
corners) and the gap is scissored to experiment. The one external axis still open is the realized
oxygen chemical potential Delta_mu_O(fO2, T) from the workspace CALPHAD buffer; O-rich (Delta_mu_O=0)
is used here as the labelled reference.
"""

from __future__ import annotations

from pathlib import Path

from snse_defects.assembly import assemble_dataset
from snse_defects.chempot import mu_at
from snse_defects.pysc_fermi import plot_formation_diagram, write_dataset

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    a = assemble_dataset()
    ds, mu, dHf, limits = a.ds, a.mu_elem, a.dHf_eV, a.limits
    pbe_gap, gap = a.pbe_gap_eV, a.gap_eV
    e_bulk, vbm = ds.bulk_energy_eV, ds.vbm_eV

    chempot = {
        "formation_enthalpy_SnSe_eV_per_fu": round(dHf, 4),
        "host_limits_delta_mu_eV": {k: {s: round(v, 4) for s, v in d.items()} for k, d in limits.items()},
        "oxygen_reference": "Delta_mu_O = 0 (O-rich); realized value from the CALPHAD fO2,T buffer, "
                            "hard cap set by Sn/Se oxide onset (not computed here)",
        "band_gap_eV": {"pbe": pbe_gap, "scissored": gap},
        "elemental_mu_eV_per_atom": {k: round(v, 4) for k, v in mu.items()},
    }
    prov = {
        "settings": "PBE-D3, ecutwfc 80 / ecutrho 640, k 2x2x2, spin-polarized",
        "host_cell": "PBE-D3 relaxed (data/snse_relaxed_pbe_d3.json)",
        "harvest": "data/stageB_harvest.yaml (Sol prod_final + references, 2026-07-29)",
        "vbm_note": "true VBM 7.4950 eV (fixed-occupation bulk SCF)",
        "mu_note": "elemental refs + host bounds from DeltaH_f(SnSe); O held at the O-rich reference",
        "gap_note": f"scissored to experiment {gap} eV (PBE {pbe_gap} eV); levels stay VBM-referenced",
        "open": "realized Delta_mu_O(fO2,T) from the workspace CALPHAD oxygen buffer",
    }
    out_json = write_dataset(ds, mu, REPO / "data" / "defect_formation_dataset.json",
                             provisional=True, provenance=prov, chempot=chempot)
    print(f"wrote {out_json.relative_to(REPO)} (physical; O held at the O-rich reference)")
    print(f"\nbulk E = {e_bulk:.3f} eV   VBM = {vbm:.4f} eV   gap = {gap} eV (scissored from {pbe_gap})")
    print(f"DeltaH_f(SnSe) = {dHf:+.3f} eV/f.u. ({dHf * 96.485:+.1f} kJ/mol; exp ~ -88.9)")
    print("mu_elem (eV/atom): " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(mu.items())))

    # Condition-resolved formation energies at both host corners (oxygen at the O-rich reference).
    corners = {name: mu_at(mu, dmu, dmu_O_eV=0.0) for name, dmu in limits.items()}
    for corner, mu_c in corners.items():
        print(f"\nFormation energy at the VBM (E_F=0), {corner} + O-rich, eV:")
        for e in sorted(ds.entries, key=lambda x: (x.name, x.charge)):
            print(f"  {e.name:6s} q={e.charge:+d}  E_f={e.value(0.0, mu_c):10.3f}   n_i={e.n_i}")

    print("\nCharge-state transition levels (eV above VBM) - mu-independent, scissored gap:")
    for name in sorted({e.name for e in ds.entries}):
        rows = ds.by_defect()[name]
        for a, b in zip(rows, rows[1:]):
            tl = ds.transition_level(name, a.charge, b.charge, mu)
            if tl is not None:
                inside = "in gap" if 0.0 <= tl <= gap else "outside gap"
                print(f"  {name:6s} ({a.charge:+d}/{b.charge:+d}) at {tl:+.3f}   [{inside}]")

    # Greyscale formation-energy diagrams, one per host corner (O-rich, scissored gap).
    # No baked-on title: the manuscript caption identifies each corner (Figs 1-2).
    for corner, mu_c in corners.items():
        png = plot_formation_diagram(ds, mu_c, REPO / "data" / f"defect_diagram_{corner}.png")
        print(f"wrote {png.relative_to(REPO)}")

    print("\nSTILL OPEN (external, no Sol): realized Delta_mu_O(fO2,T) from the workspace CALPHAD "
          "oxygen buffer -> the carrier-model sweep over fO2 and T.")


if __name__ == "__main__":
    main()
