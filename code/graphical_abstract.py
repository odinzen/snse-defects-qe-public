"""Graphical abstract: carrier balance vs oxygen chemical potential + the SnO2 oxidation cap.

Greyscale, no baked caption or finding-line (region/axis/curve labels only), per the group figure rules.
Data-grounded: hole/electron densities from the Se-rich 600 K carrier sweep, caps from the oxide-cap
calculation. Writes graphical_abstract.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    summ = json.loads((REPO / "data" / "carrier_summary.json").read_text())
    caps = json.loads((REPO / "data" / "oxide_cap_pbe_d3.json").read_text())
    cap_pbe = caps["cap_pbe_d3_eV"]
    cap_exp = caps["cap_exp_anchored_eV"]

    rows = sorted((r for r in summ["conditions"]["Se-rich"] if r["T"] == 600.0),
                  key=lambda r: r["delta_mu_O"])
    x = [r["delta_mu_O"] for r in rows]
    p = [r["p_cm3"] for r in rows]
    n = [max(r["n_cm3"], 1e-30) for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    xlo = cap_exp - 0.15

    # thin stable strip left of the tightest cap; the rest of the accessible range is oxidising
    ax.axvspan(xlo, cap_exp, color="0.9", zorder=0)
    ax.axvline(cap_exp, color="0.4", linewidth=1.3)
    ax.axvline(cap_pbe, color="0.6", linewidth=1.1, dashes=(3, 2))

    ax.semilogy(x, p, color="black", linewidth=2.2, label="holes  $p$")
    ax.semilogy(x, n, color="black", linewidth=1.6, dashes=(4, 2), label="electrons  $n$")

    ax.set_xlim(xlo, max(x))
    ax.set_ylim(1e6, 1e21)
    ax.set_xlabel(r"oxygen chemical potential  $\Delta\mu_\mathrm{O}$  (eV)")
    ax.set_ylabel(r"carrier concentration (cm$^{-3}$)")

    # region + marker labels only (no finding sentence); placed in the clear band
    # between the p and n curves so it never overlaps a curve or the cap line
    ax.text(0.42, 0.70, r"$p$-type from $V_\mathrm{Sn}$", transform=ax.transAxes,
            ha="left", va="center", fontsize=11)
    ax.annotate(r"SnO$_2$ cap", xy=(cap_exp, 1.5e7), xytext=(cap_exp + 0.10, 6e8),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    ax.annotate("oxidation", xy=(max(x) - 0.05, 1.5e7), xytext=(max(x) - 0.55, 6e8),
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(REPO / f"graphical_abstract.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote graphical_abstract.pdf / .png")


if __name__ == "__main__":
    main()
