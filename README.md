# GRB Classification from Compact Binary Mergers
### Using COMPAS Population Synthesis to Predict Short and Long cbGRB Rates

This project applies the [Gottlieb et al. (2023)](https://arxiv.org/abs/2309.00038) classification scheme to COMPAS binary population synthesis simulations. It predicts the rates, mass distributions, and redshift evolution of all GRB types produced by BNS and BHNS mergers, with uncertainty estimates across binary physics models, NS equation of state, and BH spin.

---

## Results

### 1. GRB Class Fractions: BNS vs BHNS

![GRB class fractions](Plots/comparison_grb_fractions.png)

About 92% of BHNS mergers are GRB-dark: the neutron star plunges directly into the black hole without forming a disk. BNS mergers are predominantly GRB-capable. This is the headline result of the analysis.

---

### 2. BHNS GRB Classification by BH Spin

![BHNS mass plane by spin](Plots/bhns_mass_plane_spin.png)

BH mass vs NS mass colored by GRB class at spin values a = 0.0, 0.5, 0.7. Higher spin dramatically expands the tidally-disrupted region, converting GRB-dark systems into long cbGRB producers. The NS remnant mass gap from the rapid supernova engine is visible as a horizontal gap across all panels.

---

### 3. EOS Sensitivity: Short/Long Fraction vs M_crit

![BNS M_crit sensitivity](Plots/bns_mcrit_sensitivity.png)

GRB class fractions as a function of the prompt-collapse threshold M_crit, mapped to NS radius R_1.4 on the upper axis. As the EOS stiffens (higher M_crit), more mergers produce HMNS remnants (Short cbGRB Type I) rather than prompt collapse. Vertical lines mark four published EOS models; the green dashed line is the GW170817-consistent fiducial at 2.8 M_sun.

---

### 4. Progenitor Stars of Each GRB Class (ZAMS Masses)

![ZAMS progenitor plane](Plots/bns_progenitor_zams_plane.png)

ZAMS primary vs secondary mass for each BNS GRB class. Long cbGRB progenitors originate from highly asymmetric binaries scattered at high mass ratios. Short cbGRB progenitors cluster tightly on the equal-mass diagonal, reflecting the different tidal history required for massive disk formation.

---

### 5. All Five cbGRB Classes: Rate Density vs Redshift

![Unified rate figure](Plots/rate_figure2_unified.png)

Merger rate density vs redshift for all five Figure 2 classes from Gottlieb et al. (2023), computed via cosmic integration with the Neijssel et al. (2019) metallicity-specific SFRD. BHNS long cbGRBs dominate at low redshift; BNS channels take over at z > 1-2. The BHNS short cbGRB channel is suppressed at a = 0.5 because all disrupting systems form massive disks.

---

### 6. BNS Mass Plane and M_tot Distribution at Redshift Slices

![Mass distribution at redshift slices](Plots/bns_mass_dist_redshift_slices.png)

Rate-weighted M1 vs M2 scatter (top) and M_tot histograms (bottom) at z = 0, 0.5, 1.0, 2.0. Only the top 40% rate contributors are shown in the scatter panels. The M_tot distribution shifts toward lower masses at higher redshift as lower-metallicity, lower-mass progenitors begin to dominate the merger rate.

---

## Classification Scheme

Gottlieb et al. (2023) propose a unified picture where GRB class is set by the merger remnant. Five physically distinct outcomes are identified.

**BNS mergers:**

| Type | Class | Condition | Engine |
|---|---|---|---|
| Type I sGRB | Short cbGRB | M_tot < M_crit (~2.8 M_sun) | HMNS remnant powers jet before collapse |
| Type II sGRB | Short cbGRB | M_tot >= M_crit, q < 1.2 | Immediate BH + light accretion disk |
| lGRB | Long cbGRB | M_tot >= M_crit, q >= 1.2 | BH + massive disk from asymmetric merger |

**BHNS mergers** (outcome set by Foucart 2012 disk mass formula):

| Type | Class | Condition |
|---|---|---|
| No disruption | No GRB | NS plunges into BH without disk |
| sGRB | Short cbGRB | M_disk < 0.1 M_sun |
| lGRB | Long cbGRB | M_disk >= 0.1 M_sun |

---

## Analysis Pipeline

Run notebooks in order: `GRB_BNS.ipynb` -> `GRB_BHNS.ipynb` -> `GRB_CosmicRate.ipynb` -> `GRB_comparsion.ipynb`. The cosmic rate notebook depends on `.npy` files exported by the first two.

| Notebook | Contents |
|---|---|
| `GRB_BNS.ipynb` | BNS classification, efficiency, M_crit and q sensitivity, Model A vs K |
| `GRB_BHNS.ipynb` | BHNS classification, spin sensitivity, EOS sensitivity |
| `GRB_CosmicRate.ipynb` | Cosmic integration, rate vs redshift for all five classes, uncertainty bands |
| `GRB_comparsion.ipynb` | Summary: BNS vs BHNS class fractions and formation efficiency |

**Data sources:**
- BNS: [Zenodo 5189849](https://zenodo.org/records/5189849)
- BHNS: [Zenodo 5178777](https://zenodo.org/records/5178777)

Data files (`.h5`, `.hdf5`) are not included. Download from the Zenodo links above before running.

---

## Setup

```bash
conda create -n grb-env python=3.10
conda activate grb-env
python -m pip install -r requirements.txt
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

---

## References

- Gottlieb et al. (2023): [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Broekgaarden et al. (2021): [arXiv:2112.05763](https://arxiv.org/abs/2112.05763)
- Foucart (2012): NS tidal disruption fitting formula
- Neijssel et al. (2019): Metallicity-specific star formation rate density

---

## License

MIT License. See [LICENSE](LICENSE) for details.
