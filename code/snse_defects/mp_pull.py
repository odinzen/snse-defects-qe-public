"""Pull the SnSe bulk reference (mp-691) from Materials Project.

We need four things from MP for the defect workflow:
  - the relaxed Pnma structure (basis for every supercell),
  - the PBE band gap (~0.61 eV; underestimates the real ~0.86 eV, so a scissor is planned),
  - the static dielectric tensor (drives the image-charge correction),
  - the electronic DOS (consumed by py-sc-fermi for the carrier integral).

Raw API responses are dumped to data/mp_raw/ (gitignored) for provenance; the distilled values
that the rest of the pipeline reads go to config/mp691.yaml (tracked). Run in the odinzen-mp env,
which carries the API key. Pass the key explicitly, via MP_API_KEY, or via a configured pmgrc.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

MATERIAL_ID = "mp-691"
REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "mp_raw"
CONFIG_OUT = REPO / "config" / "mp691.yaml"


@dataclass
class BulkReference:
    material_id: str
    formula: str
    spacegroup: str
    lattice_abc: tuple[float, float, float]
    lattice_angles: tuple[float, float, float]
    volume_A3: float
    n_sites: int
    band_gap_pbe_eV: float
    is_gap_direct: bool
    dielectric_total: list[list[float]] | None
    dielectric_electronic: list[list[float]] | None
    dielectric_ionic: list[list[float]] | None
    eps_static_scalar: float | None
    eps_inf_scalar: float | None
    pull_date: str
    task_ids: dict[str, str]


def _resolve_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        try:
            from pymatgen.core import SETTINGS

            key = SETTINGS.get("PMG_MAPI_KEY")
        except Exception:
            key = None
    if not key:
        raise SystemExit(
            "No Materials Project API key found. Pass api_key=..., or set MP_API_KEY, "
            "or configure PMG_MAPI_KEY via `pmg config --add PMG_MAPI_KEY <key>`."
        )
    return key


def _tensor_scalar(tensor: list[list[float]] | None) -> float | None:
    """Isotropic average (trace/3) of a 3x3 tensor, for a quick single-number handle."""
    if not tensor:
        return None
    return round(sum(tensor[i][i] for i in range(3)) / 3.0, 4)


def pull(api_key: str | None = None, pull_date: str | None = None) -> BulkReference:
    from mp_api.client import MPRester

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    key = _resolve_key(api_key)
    # pull_date is passed in (Date.now-free); caller stamps it. Fall back to a marker.
    stamp = pull_date or "unstamped"

    task_ids: dict[str, str] = {}
    with MPRester(key) as mpr:
        summary = mpr.materials.summary.search(material_ids=[MATERIAL_ID])[0]
        structure = summary.structure
        (RAW_DIR / "summary.json").write_text(json.dumps(summary.dict(), default=str, indent=2))
        task_ids["summary"] = str(getattr(summary, "task_ids", "") or "")

        diel_total = diel_elec = diel_ionic = None
        try:
            diel = mpr.materials.dielectric.search(material_ids=[MATERIAL_ID])
            if diel:
                d = diel[0]
                (RAW_DIR / "dielectric.json").write_text(
                    json.dumps(d.dict(), default=str, indent=2)
                )
                diel_elec, diel_ionic, diel_total = _dielectric_tensors(d.dict())
                task_ids["dielectric"] = str(getattr(d, "material_id", "") or "")
        except Exception as exc:  # dielectric data may be absent for a given material
            (RAW_DIR / "dielectric_ERROR.txt").write_text(repr(exc))

        try:
            dos = mpr.get_dos_by_material_id(MATERIAL_ID)
            (RAW_DIR / "dos.json").write_text(json.dumps(dos.as_dict(), default=str))
        except Exception as exc:
            (RAW_DIR / "dos_ERROR.txt").write_text(repr(exc))

    lat = structure.lattice
    ref = BulkReference(
        material_id=MATERIAL_ID,
        formula=structure.composition.reduced_formula,
        spacegroup=structure.get_space_group_info()[0],
        lattice_abc=tuple(round(x, 6) for x in lat.abc),
        lattice_angles=tuple(round(x, 4) for x in lat.angles),
        volume_A3=round(lat.volume, 6),
        n_sites=len(structure),
        band_gap_pbe_eV=round(float(summary.band_gap), 4),
        is_gap_direct=bool(getattr(summary, "is_gap_direct", False)),
        dielectric_total=diel_total,
        dielectric_electronic=diel_elec,
        dielectric_ionic=diel_ionic,
        eps_static_scalar=_tensor_scalar(diel_total),
        eps_inf_scalar=_tensor_scalar(diel_elec),
        pull_date=stamp,
        task_ids=task_ids,
    )

    structure.to(filename=str(RAW_DIR / "SnSe_mp-691.cif"))
    structure.to(filename=str(RAW_DIR / "SnSe_mp-691.json"))
    _write_config(ref)
    return ref


def _clean(m: list[list[float]]) -> list[list[float]]:
    """Zero out numerical dust in off-diagonal entries and round for readability."""
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            v = float(m[i][j])
            if i != j and abs(v) < 1e-9:
                v = 0.0
            row.append(round(v, 6))
        out.append(row)
    return out


def _dielectric_tensors(raw: dict):
    """Prefer the full 3x3 tensors (`electronic`, `ionic`); total = electronic + ionic.

    MP also exposes isotropic scalars (`e_electronic`, `e_ionic`, `e_total`) but SnSe is strongly
    anisotropic, so the eFNV correction needs the real tensor, not the trace/3 average.
    """
    elec_raw = raw.get("electronic")
    ionic_raw = raw.get("ionic")
    if elec_raw is not None and ionic_raw is not None:
        elec = _clean(elec_raw)
        ionic = _clean(ionic_raw)
        total = _clean([[elec[i][j] + ionic[i][j] for j in range(3)] for i in range(3)])
        return elec, ionic, total
    # fallback: broadcast the isotropic scalars if the tensors are missing
    def _iso(x):
        if x is None:
            return None
        s = float(x)
        return [[s if i == j else 0.0 for j in range(3)] for i in range(3)]

    return _iso(raw.get("e_electronic")), _iso(raw.get("e_ionic")), _iso(raw.get("e_total"))


def rebuild_config_from_raw(pull_date: str) -> BulkReference:
    """Rebuild config/mp691.yaml from data/mp_raw/ without hitting the API."""
    from pymatgen.core import Structure

    structure = Structure.from_file(str(RAW_DIR / "SnSe_mp-691.json"))
    summary = json.loads((RAW_DIR / "summary.json").read_text())
    diel_path = RAW_DIR / "dielectric.json"
    diel_elec = diel_ionic = diel_total = None
    task_ids = {"summary": str(summary.get("task_ids", ""))}
    if diel_path.exists():
        raw = json.loads(diel_path.read_text())
        diel_elec, diel_ionic, diel_total = _dielectric_tensors(raw)
        task_ids["dielectric"] = str(raw.get("material_id", ""))

    lat = structure.lattice
    ref = BulkReference(
        material_id=MATERIAL_ID,
        formula=structure.composition.reduced_formula,
        spacegroup=structure.get_space_group_info()[0],
        lattice_abc=tuple(round(x, 6) for x in lat.abc),
        lattice_angles=tuple(round(x, 4) for x in lat.angles),
        volume_A3=round(lat.volume, 6),
        n_sites=len(structure),
        band_gap_pbe_eV=round(float(summary["band_gap"]), 4),
        is_gap_direct=bool(summary.get("is_gap_direct", False)),
        dielectric_total=diel_total,
        dielectric_electronic=diel_elec,
        dielectric_ionic=diel_ionic,
        eps_static_scalar=_tensor_scalar(diel_total),
        eps_inf_scalar=_tensor_scalar(diel_elec),
        pull_date=pull_date,
        task_ids=task_ids,
    )
    _write_config(ref)
    return ref


def _write_config(ref: BulkReference) -> None:
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Materials Project",
        "material_id": ref.material_id,
        "pull_date": ref.pull_date,
        "formula": ref.formula,
        "spacegroup": ref.spacegroup,
        "lattice": {"abc": list(ref.lattice_abc), "angles": list(ref.lattice_angles)},
        "volume_A3": ref.volume_A3,
        "n_sites": ref.n_sites,
        "band_gap_pbe_eV": ref.band_gap_pbe_eV,
        "is_gap_direct": ref.is_gap_direct,
        "scissor_note": "PBE gap underestimates experiment (~0.86 eV). Apply scissor downstream.",
        "dielectric": {
            "total": ref.dielectric_total,
            "electronic": ref.dielectric_electronic,
            "ionic": ref.dielectric_ionic,
            "eps_static_scalar": ref.eps_static_scalar,
            "eps_inf_scalar": ref.eps_inf_scalar,
        },
        "task_ids": ref.task_ids,
    }
    CONFIG_OUT.write_text(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    import argparse
    import datetime as _dt

    p = argparse.ArgumentParser(description="Pull mp-691 (SnSe) from Materials Project.")
    p.add_argument("--api-key", default=None, help="MP API key (else env / pmgrc).")
    args = p.parse_args()
    today = _dt.date.today().isoformat()
    r = pull(api_key=args.api_key, pull_date=today)
    print(f"Pulled {r.material_id} ({r.formula}, {r.spacegroup}) on {r.pull_date}")
    print(f"  lattice abc = {r.lattice_abc} A, angles = {r.lattice_angles} deg")
    print(f"  volume = {r.volume_A3} A^3, sites = {r.n_sites}")
    print(f"  PBE gap = {r.band_gap_pbe_eV} eV (direct={r.is_gap_direct})")
    print(f"  eps_static (avg) = {r.eps_static_scalar}, eps_inf (avg) = {r.eps_inf_scalar}")
    print(f"  wrote {CONFIG_OUT.relative_to(REPO)}")
