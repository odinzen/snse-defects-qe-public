"""Turn the harvested convergence energies into convergence deltas (meV/atom) and a lattice check.

Reads data/convergence_harvest.yaml (the raw QE total energies) and reports, for each sweep, the
change per atom as the parameter increases. "Converged" is the standard ~1 meV/atom threshold. Also
compares the PBE-D3 vc-relax lattice against the bare-PBE (MP) start and experiment.
"""

from __future__ import annotations

from pathlib import Path

import yaml

RY_TO_MEV = 13605.693
REPO = Path(__file__).resolve().parents[2]
HARVEST = REPO / "data" / "convergence_harvest.yaml"


def _deltas(pairs: list[tuple[str, float]], n_atoms: int) -> list[tuple[str, str, float]]:
    out = []
    for (label_a, ea), (label_b, eb) in zip(pairs, pairs[1:]):
        d = abs(eb - ea) * RY_TO_MEV / n_atoms
        out.append((label_a, label_b, d))
    return out


def _report_sweep(name: str, data: dict, n_atoms: int) -> None:
    pairs = [(str(k), float(v)) for k, v in data.items()]
    print(f"\n{name}")
    print(f"  {'param':<10}{'E (Ry)':>16}{'d(meV/atom) vs prev':>22}")
    prev = None
    for (label, e) in pairs:
        if prev is None:
            print(f"  {label:<10}{e:>16.6f}{'-':>22}")
        else:
            d = abs(e - prev) * RY_TO_MEV / n_atoms
            flag = "  <-- converged (<1)" if d < 1.0 else ""
            print(f"  {label:<10}{e:>16.6f}{d:>22.3f}{flag}")
        prev = e


def _lattice_check(vc: dict, refs: dict) -> None:
    relaxed = vc["lattice_angstrom"]
    mp = refs["mp_691_pbe_novdw"]
    exp = refs["experiment_300K"]
    print("\nvc-relax lattice (PBE-D3) vs bare-PBE (MP) and experiment")
    print(f"  {'axis':<6}{'relaxed':>10}{'MP(PBE)':>10}{'exp':>10}{'relaxed vs exp':>16}{'MP vs exp':>12}")
    for i, ax in enumerate(("a", "b", "c")):
        dr = 100 * (relaxed[i] - exp[i]) / exp[i]
        dm = 100 * (mp[i] - exp[i]) / exp[i]
        print(f"  {ax:<6}{relaxed[i]:>10.3f}{mp[i]:>10.3f}{exp[i]:>10.3f}{dr:>+15.1f}%{dm:>+11.1f}%")
    vexp = exp[0] * exp[1] * exp[2]
    vmp = mp[0] * mp[1] * mp[2]
    print(
        f"  {'V':<6}{vc['volume_A3']:>10.2f}{vmp:>10.2f}{vexp:>10.2f}"
        f"{100 * (vc['volume_A3'] - vexp) / vexp:>+15.1f}%{100 * (vmp - vexp) / vexp:>+11.1f}%"
    )


def main() -> None:
    d = yaml.safe_load(HARVEST.read_text())
    n = d["n_atoms"]
    _report_sweep("ecutwfc sweep (ecutrho=8x, k=6x6x3)", d["ecutwfc_sweep"], n)
    _report_sweep("dual sweep (ecutwfc=60)", d["dual_sweep"], n)
    _report_sweep("k-point sweep (ecutwfc=60, ecutrho=480)", d["kpoint_sweep"], n)
    _lattice_check(d["vc_relax"], d["reference_lattices"])


if __name__ == "__main__":
    main()
