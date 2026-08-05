"""Defect-controlled carrier concentrations in SnSe:O vs oxygen pressure and temperature.

Runs the self-consistent defect/carrier equilibrium (carrier.py) over a pO2 x T grid at both host
corners, writes data/carrier_summary.json, and plots hole/electron concentration vs pO2. Flags the
oxidation onset: the pO2 above which a defect concentration exceeds the site density, where the
dilute model breaks down and SnSe would oxidise rather than dope (the real Delta_mu_O upper cap).

The carrier-relevant physics: SnSe is p-type from the native Sn vacancy (V_Sn, a double acceptor);
oxygen defects are largely isovalent/neutral, so oxygen acts more as an oxidation sink than a dopant.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt

from snse_defects.assembly import assemble_dataset
from snse_defects.carrier import NSITES, equilibrium, scissored_dos
from snse_defects.chempot import mu_at

REPO = Path(__file__).resolve().parents[1]
SITE_DENSITY_CM3 = min(NSITES.values()) / (226.257e-24)   # ~1.8e22 cm^-3, dilute-limit ceiling
TEMPS = (400.0, 600.0, 800.0)
LOG_PO2 = np.arange(-30.0, 0.1, 2.0)                       # pO2 from 1e-30 to 1 bar


def _oxidised(rec: dict) -> bool:
    return max(rec["defects_cm3"].values(), default=0.0) > SITE_DENSITY_CM3


def main() -> None:
    a = assemble_dataset()
    dos = scissored_dos(a.gap_eV)
    print(f"gap {a.gap_eV} eV | DeltaH_f {a.dHf_eV:+.3f} eV/f.u. | site ceiling {SITE_DENSITY_CM3:.1e} cm^-3\n")

    summary: dict[str, list] = {}
    for corner in ("Sn-rich", "Se-rich"):
        mu_host = mu_at(a.mu_elem, a.limits[corner], dmu_O_eV=0.0)
        rows = []
        for T in TEMPS:
            for lp in LOG_PO2:
                r = equilibrium(a.ds, mu_host, T=T, p_O2_bar=float(10.0**lp), dos=dos)
                r["log10_pO2"] = float(lp)
                r["oxidised"] = _oxidised(r)
                rows.append(r)
        summary[corner] = rows
        # oxidation onset per temperature (lowest pO2 that breaks the dilute limit)
        for T in TEMPS:
            oxi = [x["log10_pO2"] for x in rows if x["T"] == T and x["oxidised"]]
            onset = f"10^{min(oxi):.0f} bar" if oxi else "none in range"
            mid = next(x for x in rows if x["T"] == T and x["log10_pO2"] == -20.0)
            print(f"{corner:8s} T={T:.0f}K  reducing (pO2=1e-20): E_F={mid['E_fermi_eV']:.3f} eV "
                  f"p={mid['p_cm3']:.2e} n={mid['n_cm3']:.2e} cm^-3 | oxidation onset {onset}")

    out = REPO / "data" / "carrier_summary.json"
    out.write_text(json.dumps(
        {"site_density_cm3": SITE_DENSITY_CM3, "temperatures_K": list(TEMPS), "conditions": summary},
        indent=2, default=float))
    print(f"\nwrote {out.relative_to(REPO)}")

    _plot(summary)


def _plot(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    T_plot = 600.0
    styles = {"p_cm3": (0.0, (1, 0)), "n_cm3": (0.0, (4, 2))}
    # No baked figure title: the manuscript caption identifies the figure and the panels.
    # Panels carry lower-case bold letters (a, b), defined in the Fig 3 caption.
    for i, (ax, corner) in enumerate(zip(axes, ("Sn-rich", "Se-rich"))):
        rows = [x for x in summary[corner] if x["T"] == T_plot]
        rows.sort(key=lambda x: x["log10_pO2"])
        lp = [x["log10_pO2"] for x in rows]
        for key, dash in styles.items():
            ax.semilogy(lp, [max(x[key], 1e-30) for x in rows], color="black",
                        dashes=dash[1], linewidth=1.5, label=("holes p" if key == "p_cm3" else "electrons n"))
        oxi = [x["log10_pO2"] for x in rows if x["oxidised"]]
        if oxi:
            ax.axvspan(min(oxi), 0.5, color="0.85", label="oxidation regime")
        ax.axhline(SITE_DENSITY_CM3, color="0.6", linewidth=0.8, dashes=(2, 2))
        ax.text(0.03, 0.97, f"({'ab'[i]})", transform=ax.transAxes,
                fontweight="bold", va="top", ha="left")
        ax.set_xlabel("log$_{10}$ pO$_2$ (bar)")
        ax.grid(True, color="0.9", linewidth=0.6)
    axes[0].set_ylabel(f"Carrier concentration (cm$^{{-3}}$), {T_plot:.0f} K")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    p = REPO / "data" / "carrier_vs_pO2.png"
    fig.savefig(p, dpi=600, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
