# SnSe:O charged point defects — reproducibility package

Code and Sol-computed DFT data for the study **"Oxygen controls the carrier balance and sets the
oxidation limit of SnSe: a charged point-defect study with DFT-validated machine-learning screening"**
(M. E. Bustamante, Odinzen LLC / Arizona State University; target: *npj Computational Materials*).

This repository lets anyone reproduce the formation energies, the self-consistent carrier model, the
oxidation cap, and the machine-learning-potential calibration from the harvested first-principles
energies. It contains **no proprietary software** — only Quantum ESPRESSO, py-sc-fermi, and open
foundation potentials are used.

## What's here
```
code/                 analysis + figure scripts, and the snse_defects package they import
  assemble.py         SINGLE SOURCE OF TRUTH pipeline: harvested QE energies + true VBM + elemental
                      references -> eFNV image-charge correction + gap scissor -> the defect dataset
                      and figures 1-2
  carrier_model.py    self-consistent defect/carrier equilibrium (py-sc-fermi) over pO2 x T; figure 3
  oxide_cap_pbe_d3.py PBE-D3 oxidation cap from the SnO2 vc-relax energy
  mlip_calibrate.py   foundation-MLIP validation against our dHf(SnSe)
  mlip_oxide_cap.py   MLIP oxide screen (which oxide binds)
  build_elementals.py / build_oxides.py   reference cells
  sensitivity.py      robustness of the p-type conclusion to the gap scissor
data/
  stageB_harvest.yaml THE DFT INPUTS: converged QE total energies (Ry) for the 72-atom defect
                      supercells + true VBM + beta-Sn/Se/O2 references, computed on ASU's Sol
                      supercomputer. Everything downstream derives from this.
  defect_formation_dataset.json, carrier_summary.json, oxide_cap_pbe_d3.json,
  mlip_calibration.json, mlip_oxide_cap.json, snse_relaxed_pbe_d3.json,
  settings_harvest.yaml, convergence_harvest.yaml
config/               locked settings, distilled Materials Project mp-691 values, SSSP pseudos (names + checksums)
figures/              the three main-text figures (PDF, greyscale)
```

## Reproduce
QE ran on ASU's Sol supercomputer; the converged energies enter the analysis as constants in
`data/stageB_harvest.yaml`. Nothing here re-runs DFT.
```
PYTHONPATH=code python code/assemble.py                                   # dataset + figs 1-2
PYTHONPATH=code python code/carrier_model.py                             # carrier sweep + fig 3
PYTHONPATH=code python code/oxide_cap_pbe_d3.py -485.36553929 --write    # PBE-D3 oxidation cap
```
Core analysis needs numpy, scipy, pymatgen, ase, py-sc-fermi. The MLIP scripts additionally need
torch + mace-torch + orb-models. Temperatures are in K throughout; figures are greyscale, 600 DPI.

## License
- **Code** (`code/`): MIT — see [LICENSE](LICENSE).
- **Data and figures** (`data/`, `figures/`, `config/`): Creative Commons Attribution 4.0
  International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.

## Citation
Please cite the paper (details to be added on acceptance) and this repository. The reference list and
the first-principles provenance are given in the manuscript.

## Acknowledgement
Calculations used the Sol supercomputer at Arizona State University.
