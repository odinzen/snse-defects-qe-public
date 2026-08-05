"""Build the bulk supercell and every defect supercell for SnSe.

The conventional Pnma cell (from data/mp_raw/) is the tiling unit. For each requested supercell
multiple we build:
  - the pristine bulk supercell (the E_bulk reference and the correction's host),
  - V_Sn, V_Se (vacancies), O_Se (substitution), O_i (Voronoi interstitials, candidate sites).

Charge is NOT a structural property here - QE applies it via tot_charge, and each charge state
relaxes from the same starting geometry. So this module returns neutral atomic arrangements plus
the defect-site coordinates (the charge correction needs the site later).

Symmetry: in Pnma SnSe both Sn and Se sit on single Wyckoff orbits, so there is one distinct
vacancy/substitution each. Interstitials are enumerated by Voronoi topology and reported per
symmetry-distinct site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Element, Structure

REPO = Path(__file__).resolve().parents[2]
# Host cell for defect supercells = the PBE-D3 vc-relaxed cell (converged bulk), NOT MP's bare-PBE
# cell. Fall back to the MP cell only if the relaxed one has not been produced yet.
RELAXED = REPO / "data" / "snse_relaxed_pbe_d3.json"
MP_CELL = REPO / "data" / "mp_raw" / "SnSe_mp-691.json"
CONVENTIONAL = RELAXED if RELAXED.exists() else MP_CELL


@dataclass
class DefectCell:
    name: str                       # e.g. "V_Sn", "O_i_0", "O_Se"
    kind: str                       # vacancy | substitution | interstitial
    structure: Structure            # the (neutral) defective supercell
    site_frac: tuple[float, float, float]  # defect site, fractional coords in the supercell
    n_added: dict[str, int]         # species -> atoms added (+) relative to bulk
    n_removed: dict[str, int]       # species -> atoms removed (+)
    supercell: tuple[int, int, int]
    multiplicity: int = 1           # symmetry multiplicity of the defect site
    note: str = ""


def load_conventional() -> Structure:
    return Structure.from_file(str(CONVENTIONAL))


def make_bulk_supercell(multiple: tuple[int, int, int], base: Structure | None = None) -> Structure:
    s = (base or load_conventional()).copy()
    s.make_supercell(list(multiple))
    return s


def min_image_distance(structure: Structure) -> float:
    """Shortest lattice-vector length: how close a point defect sits to its nearest image.

    A single point defect's nearest periodic copy is one lattice vector away, so the minimum of
    |a|, |b|, |c| is the controlling defect-defect distance for a diagonal supercell.
    """
    return round(float(min(np.linalg.norm(v) for v in structure.lattice.matrix)), 3)


def _distinct_index(structure: Structure, species: str) -> int:
    """Index of the first site of `species`. Pnma Sn/Se are single-orbit, so any works."""
    for i, site in enumerate(structure):
        if site.specie.symbol == species:
            return i
    raise ValueError(f"no {species} site in structure")


def make_vacancy(multiple, species: str, base: Structure | None = None) -> DefectCell:
    sc = make_bulk_supercell(multiple, base)
    idx = _distinct_index(sc, species)
    site_frac = tuple(float(round(x, 6)) for x in sc[idx].frac_coords)
    sc.remove_sites([idx])
    return DefectCell(
        name=f"V_{species}",
        kind="vacancy",
        structure=sc,
        site_frac=site_frac,
        n_added={},
        n_removed={species: 1},
        supercell=tuple(multiple),
    )


def make_substitution(multiple, host: str, sub: str, base: Structure | None = None) -> DefectCell:
    sc = make_bulk_supercell(multiple, base)
    idx = _distinct_index(sc, host)
    site_frac = tuple(float(round(x, 6)) for x in sc[idx].frac_coords)
    sc.replace(idx, Element(sub))
    return DefectCell(
        name=f"{sub}_{host}",
        kind="substitution",
        structure=sc,
        site_frac=site_frac,
        n_added={sub: 1},
        n_removed={host: 1},
        supercell=tuple(multiple),
    )


def make_interstitials(multiple, insert: str, max_sites: int = 3, base: Structure | None = None):
    """Voronoi-topology interstitial candidates for `insert` (e.g. O), symmetry-distinct.

    Uses pymatgen-analysis-defects. Returns up to max_sites DefectCell entries labelled
    <insert>_i_0, _1, ... ordered as the generator yields them (roughly by site openness).
    """
    from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator

    conv = base or load_conventional()
    gen = VoronoiInterstitialGenerator()
    cells: list[DefectCell] = []
    for k, defect in enumerate(gen.generate(conv, insert_species=[insert])):
        if k >= max_sites:
            break
        sc_struct = defect.get_supercell_structure(sc_mat=np.diag(multiple))
        # locate the inserted atom (the one species-matching site nearest the defect fractional site)
        site_frac = _locate_insert(sc_struct, insert, conv, defect)
        cells.append(
            DefectCell(
                name=f"{insert}_i_{k}",
                kind="interstitial",
                structure=sc_struct,
                site_frac=site_frac,
                n_added={insert: 1},
                n_removed={},
                supercell=tuple(multiple),
                multiplicity=getattr(defect, "multiplicity", 1),
                note="Voronoi candidate; screen neutral energies, carry the lowest to charge states.",
            )
        )
    return cells


def _locate_insert(sc_struct: Structure, insert: str, conv: Structure, defect) -> tuple:
    # the interstitial is the added atom; its count in the supercell exceeds the bulk tiling.
    for site in sc_struct:
        if site.specie.symbol == insert and insert not in {e.symbol for e in conv.composition}:
            return tuple(float(round(x, 6)) for x in site.frac_coords)
    # O is not in the host, so the single O site is unambiguous; fall back to defect site
    for site in sc_struct:
        if site.specie.symbol == insert:
            return tuple(float(round(x, 6)) for x in site.frac_coords)
    return tuple(float(round(x, 6)) for x in getattr(defect, "site").frac_coords)


def all_defects(multiple, base: Structure | None = None) -> list[DefectCell]:
    conv = base or load_conventional()
    out: list[DefectCell] = []
    out.append(make_vacancy(multiple, "Sn", conv))
    out.append(make_vacancy(multiple, "Se", conv))
    out.append(make_substitution(multiple, "Se", "O", conv))
    out.extend(make_interstitials(multiple, "O", max_sites=3, base=conv))
    return out


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load((REPO / "config" / "calc_settings.yaml").read_text())
    conv = load_conventional()
    print(f"Conventional cell: {conv.composition.formula}, {len(conv)} atoms")
    for label in ("small", "production", "size_check"):
        mult = tuple(cfg["supercell"][label])
        sc = make_bulk_supercell(mult, conv)
        abc = tuple(round(x, 2) for x in sc.lattice.abc)
        print(
            f"  {label:11s} {mult}  -> {len(sc):3d} atoms, "
            f"box {abc} A, min-image {min_image_distance(sc)} A"
        )
    print("\nDefects in the production supercell:")
    prod = tuple(cfg["supercell"]["production"])
    for dc in all_defects(prod, conv):
        print(
            f"  {dc.name:9s} [{dc.kind:12s}] {len(dc.structure):3d} atoms  "
            f"add={dc.n_added} remove={dc.n_removed} site={tuple(round(x,3) for x in dc.site_frac)}"
        )
