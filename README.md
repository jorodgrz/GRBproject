# GRB Classification from Compact Binary Mergers
### Using COMPAS Population Synthesis to Predict Short and Long cbGRB Rates

This project applies the [Gottlieb et al. (2023)](https://arxiv.org/abs/2309.00038) GRB classification scheme to COMPAS binary population synthesis simulations. The analysis predicts the rates, mass distributions, and redshift evolution of short and long compact binary GRBs (cbGRBs) produced by BNS and BHNS mergers, including uncertainty estimates across binary physics models, NS equation of state, and BH spin.

---

## Scientific Background

Gravitational wave detections (GW170817, GW211211A) suggest that both short and long GRBs can be produced by compact binary mergers. Gottlieb et al. (2023) propose a unified hybrid scenario where the GRB class is determined by the merger remnant:

| Outcome | GRB Class | Condition (BNS) |
|---|---|---|
| HMNS collapse + jet | Short cbGRB | M_tot < M_crit (~2.8 M_sun) |
| BH + light disk + jet | Short cbGRB | M_tot >= M_crit, q < 1.2 |
| BH + massive disk + extended jet | Long cbGRB | M_tot >= M_crit, q >= 1.2 |

For BHNS, the class depends on whether the NS is tidally disrupted (via the Foucart 2012 fitting formula) and the resulting disk mass. A pre-disruption check (Roche lobe vs ISCO radius) is applied before evaluating the disk mass formula.

---

## Analysis Pipeline

### Phase 1 - Data and Validation

- **Data source:** COMPAS BNS fiducial simulation from [Zenodo 5189849](https://zenodo.org/records/5189849); BHNS from [Zenodo 5178777](https://zenodo.org/records/5178777)
- **Validation:** Formation efficiency vs metallicity for BNS (roughly flat, as expected) and BHNS
- **Mass plane:** M_1 vs M_2 scatter plot for all merging systems
- **Notebooks:** `GRB_BNS.ipynb`, `GRB_BHNS.ipynb`

### Phase 2 - GRB Classification

**BNS** (`GRB_BNS.ipynb`): Gottlieb et al. (2023) hybrid scenario applied to each merging system:

```
M_tot < 2.8 M_sun                   -> short cbGRB  (HMNS-powered)
M_tot >= 2.8 M_sun  and  q < 1.2   -> short cbGRB  (BH + light disk)
M_tot >= 2.8 M_sun  and  q >= 1.2  -> long cbGRB   (BH + massive disk)
```

**BHNS** (`GRB_BHNS.ipynb`): Foucart (2012) disk mass formula with physical disruption pre-check:

```
No tidal disruption (NS plunges)    -> no GRB
M_disk >= 0.1 M_sun (massive disk) -> long cbGRB
M_disk < 0.1 M_sun  (light disk)   -> short cbGRB
```

Spin is treated as a free parameter: `a = 0.0`, `0.5`, `0.7`.

### Phase 3 - Formation Efficiency and Class Fractions

For each metallicity grid point (weighted by STROOPWAFEL sampling):

- Total formation efficiency vs metallicity
- Formation efficiency split by GRB class
- Class fraction vs metallicity

Sensitivity analyses included in the BNS notebook:
- M_crit sweep from 2.6 to 3.0 M_sun
- Mass ratio threshold sweep

### Phase 4 - Cosmic Integration

**Notebook:** `GRB_CosmicRate.ipynb`

Convolves formation efficiencies and delay-time distributions with the Neijssel et al. (2019) metallicity-specific star formation rate density. A custom `compute_merger_rate` function accumulates merger rates in O(n_z) memory to avoid kernel crashes from the full 2D allocation.

Key outputs:
- Merger rate density vs redshift for each GRB class (short cbGRB, long cbGRB, all mergers)
- Separate curves for BNS and BHNS
- Combined BNS + BHNS rate plot
- Exported arrays: `results/rates_BNS.npy`, `results/rates_BHNS.npy`

### Phase 5 - Uncertainties

Three uncertainty axes explored in `GRB_CosmicRate.ipynb`:

| Axis | Parameter range | Output |
|---|---|---|
| BH spin (BHNS) | a = 0.0, 0.5, 0.7 | Long cbGRB rate vs redshift for each spin |
| NS EOS (BNS) | M_crit = 2.6, 2.8, 3.0 M_sun | Short and long cbGRB rates vs redshift |
| Binary physics (BNS) | Model A (fiducial) vs Model K | Rate comparison with uncertainty band |

---

## Repository Structure

```
GRBproject/
├── GRB_BNS.ipynb            # BNS classification, efficiency, and sensitivity analysis
├── GRB_BHNS.ipynb           # BHNS classification, spin and EOS sensitivity
├── GRB_CosmicRate.ipynb     # Cosmic integration and uncertainty analysis (Phase 4-5)
├── GRB_comparsion.ipynb     # Summary comparison: BNS vs BHNS class fractions and rates
├── results/
│   ├── eff_BNS.npy          # Formation efficiency arrays (BNS)
│   ├── eff_BHNS.npy         # Formation efficiency arrays (BHNS)
│   ├── rates_BNS.npy        # Merger rate density arrays (BNS)
│   └── rates_BHNS.npy       # Merger rate density arrays (BHNS)
├── Demos/                   # Reference notebooks from Broekgaarden et al.
├── Papers/                  # Gottlieb et al. 2023 and related literature
├── requirements.txt         # Python dependencies
└── README.md
```

Data files (`.h5`, `.hdf5`) are excluded from this repo. Download from the Zenodo links above.

---

## Setup

```bash
# Create and activate the conda environment
conda create -n grb-env python=3.10
conda activate grb-env
python -m pip install -r requirements.txt

# Register kernel with Jupyter
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

Run notebooks in order: `GRB_BNS.ipynb` -> `GRB_BHNS.ipynb` -> `GRB_CosmicRate.ipynb` -> `GRB_comparsion.ipynb`. The comparison and cosmic rate notebooks depend on `.npy` files exported by the first two.

---

## Key References

- Gottlieb et al. (2023) - GRB classification scheme: [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Broekgaarden et al. (2021) - COMPAS BHNS population: [arXiv:2112.05763](https://arxiv.org/abs/2112.05763)
- Foucart (2012) - NS tidal disruption fitting formula for BHNS remnant disk mass
- Neijssel et al. (2019) - Metallicity-specific star formation rate density model

---

## Glossary

| Term | Definition |
|---|---|
| BNS | Binary Neutron Star |
| BHNS | Black Hole - Neutron Star binary |
| cbGRB | Compact Binary Gamma-Ray Burst |
| HMNS | Hypermassive Neutron Star |
| DCO | Double Compact Object |
| ZAMS | Zero Age Main Sequence |
| SFRD | Star Formation Rate Density |
| LVK | LIGO-Virgo-KAGRA collaboration |
| M_crit | Critical total mass threshold for HMNS collapse |
| q | Mass ratio = max(M1, M2) / min(M1, M2) |
| ISCO | Innermost Stable Circular Orbit |
