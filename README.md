# GRB Classification from Compact Binary Mergers
### Using COMPAS Population Synthesis to Predict Short and Long cbGRB Rates

This project applies the [Gottlieb et al. (2023)](https://arxiv.org/abs/2309.00038) GRB classification scheme to COMPAS binary population synthesis simulations. The goal is to predict the rates, mass distributions, and redshift evolution of short and long compact binary GRBs (cbGRBs) produced by BNS and BHNS mergers.

---

## Scientific Background

Gravitational wave detections (GW170817, GW211211A) suggest that both short and long GRBs can be produced by compact binary mergers. Gottlieb et al. (2023) propose a unified hybrid scenario where the GRB class is determined by the merger remnant:

| Outcome | GRB Class | Condition (BNS) |
|---|---|---|
| HMNS collapse → jet | Short cbGRB | M_tot < M_crit (~2.8 M☉) |
| BH + light disk → jet | Short cbGRB | M_tot ≥ M_crit, q < 1.2 |
| BH + massive disk → extended jet | Long cbGRB | M_tot ≥ M_crit, q ≥ 1.2 |

For BHNS, the class depends on whether the NS is tidally disrupted (via the Foucart 2012 fitting formula) and the resulting disk mass.

---

## Project Roadmap

### Phase 1 — Data & Validation (BNS)

**Step 1: Get the data**
- Download `fiducial.zip` from [Zenodo 5189849](https://zenodo.org/records/5189849) (BNS) and [Zenodo 5178777](https://zenodo.org/records/5178777) (BHNS)
- Clone [FloorBroekgaarden/Double-Compact-Object-Mergers](https://github.com/FloorBroekgaarden/Double-Compact-Object-Mergers) for demo notebooks and plotting infrastructure
- Clone [TeamCOMPAS/COMPAS](https://github.com/TeamCOMPAS/COMPAS) for the cosmic integration post-processing code

**Step 2: Explore HDF5 structure and reproduce the formation efficiency plot**
- Open the BNS HDF5 file; key group is `BSE_Double_Compact_Objects`
- Fields needed: `Mass(1)`, `Mass(2)`, `Metallicity@ZAMS(1)`, `Merges_Hubble_Time`
- Sampling weights: `BSE_System_Parameters` (STROOPWAFEL adaptive weights)
- Reproduce formation efficiency vs metallicity — BNS should come out roughly flat. This validates the data pipeline.

**Step 3: M₁ vs M₂ mass plane for merging BNS**
- Filter to `Merges_Hubble_Time == 1`
- 2D scatter or density plot, optionally color-coded by metallicity
- This is the raw mass landscape before classification

---

### Phase 2 — GRB Classification (BNS)

**Step 4: Implement Gottlieb Figure 2 classification for BNS**

For each merging BNS system compute:
- `M_tot = M₁ + M₂`
- `q = max(M₁, M₂) / min(M₁, M₂)`

Then classify (hybrid scenario):
```
M_tot < 2.8 M☉                        → short cbGRB  (HMNS-powered)
M_tot ≥ 2.8 M☉  and  q < 1.2         → short cbGRB  (BH + light disk)
M_tot ≥ 2.8 M☉  and  q ≥ 1.2         → long cbGRB   (BH + massive disk, GW211211A-like)
```

Figures:
- M₁ vs M₂ colored by GRB class (novel result)
- Mass ratio distribution with classification boundaries

**Step 5: Formation efficiency and class fractions vs metallicity**

At each of the ~30 metallicity grid points (weighted):
- Total BNS formation efficiency vs metallicity
- Formation efficiency split by GRB class (short vs long cbGRB)
- Fraction of mergers in each class vs metallicity

This shows whether the GRB type mix changes with birth metallicity, which feeds directly into the redshift evolution.

---

### Phase 3 — BHNS Classification

**Step 6: Add BHNS and classify using Gottlieb Figure 2**
- Open BHNS fiducial HDF5 from [Zenodo 5178777](https://zenodo.org/records/5178777)
- COMPAS provides `M_BH`, `M_NS`, giving `q = M_BH / M_NS`
- BH spin is not simulated — test three assumptions: `a = 0`, `a = 0.5`, `a = 0.7`
- Use the **Foucart (2012)** fitting formula to compute disk mass `M_d`:

```
No disruption (NS plunges)       → no GRB
M_d ≳ 0.1 M☉ (massive disk)    → long cbGRB
M_d ≲ 0.01 M☉ (light disk)     → short cbGRB
```

Figures (same set as Phase 2, plus spin sensitivity):
- M₁ vs M₂ colored by GRB class for each spin assumption
- Formation efficiency vs metallicity split by class
- Class fractions vs metallicity
- Classification change across `a = 0 / 0.5 / 0.7`

---

### Phase 4 — Cosmic Integration

**Step 7: Merger rate density and mass distributions across redshift**

Convolve formation efficiencies and delay-time distributions with the star formation rate density (Neijssel et al. 2019 model, standard COMPAS). Use the `ClassCOMPAS` infrastructure, adding GRB class as a filter before integration.

Key figures:
- **Merger rate density vs redshift** for each GRB class (short cbGRB, long cbGRB, no EM), for BNS and BHNS separately and combined — the headline result
- **Mass distributions at redshift slices** (z = 0, 0.5, 1, 2) with Gottlieb boundaries overlaid — shows how GRB class fractions shift with cosmic time
- **GRB class fraction vs redshift** — do long cbGRBs become more common at high z?

---

### Phase 5 — Uncertainties

**Step 8: Explore uncertainties across three axes**

| Axis | What to vary | Output |
|---|---|---|
| Binary physics | CE efficiency α, mass transfer β, SN kicks (additional Zenodo model zips) | Uncertainty bands on all rate plots |
| EOS / thresholds | M_crit from 2.6–3.0 M☉; q boundary for disk mass | Local (z=0) rates vs M_crit |
| BH spin (BHNS) | a = 0, 0.5, 0.7 propagated through cosmic integration | Spin sensitivity on BHNS GRB rates |

---

## Repository Structure

```
GRBproject/
├── Demo/                  # Reference notebooks from Broekgaarden et al.
├── Papers/                # Gottlieb et al. 2023 and related literature
├── requirements.txt       # Python dependencies
└── README.md
```

Data files (`.h5`, `.hdf5`) are excluded from this repo — download from Zenodo links above.


## Key References

- Gottlieb et al. (2023) — GRB classification scheme: [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Broekgaarden et al. (2021) — COMPAS BHNS population: [arXiv:2112.05763](https://arxiv.org/abs/2112.05763)
- Foucart (2012) — NS tidal disruption fitting formula
- Neijssel et al. (2019) — Metallicity-specific SFRD model

---

## Glossary

| Term | Definition |
|---|---|
| BNS | Binary Neutron Star |
| BHNS | Black Hole – Neutron Star binary |
| cbGRB | Compact Binary Gamma-Ray Burst |
| HMNS | Hypermassive Neutron Star |
| DCO | Double Compact Object |
| ZAMS | Zero Age Main Sequence |
| SFRD | Star Formation Rate Density |
| LVK | LIGO–Virgo–KAGRA collaboration |
| M_crit | Critical total mass threshold for HMNS collapse |
| q | Mass ratio = max(M₁, M₂) / min(M₁, M₂) |
