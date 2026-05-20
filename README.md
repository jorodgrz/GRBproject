# GRB Classification from Compact Binary Mergers

[![tests](https://github.com/jorodgrz/GRBproject/actions/workflows/pytest.yml/badge.svg)](https://github.com/jorodgrz/GRBproject/actions/workflows/pytest.yml)

Population-level predictions for merger-driven short and long GRBs. The pipeline applies the Gottlieb et al. (2023, 2024) classification frameworks to COMPAS binary population synthesis (Broekgaarden et al. 2021, Models A, F, G, J, K).

## Setup

```bash
conda env create -f environment.yml
conda activate grb-env
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

## Data

```bash
python tools/download_compas_data.py --tier 1 --confirm    # 5 core models A, F, G, J, K
python tools/download_compas_data.py --confirm             # full 20-model grid, ~45 GB
```

Files land in `Data/COMPASCompactOutput_<KIND>_<SUFFIX>.h5`. BNS catalogues from [Zenodo 5189849](https://zenodo.org/records/5189849), BHNS from [Zenodo 5178777](https://zenodo.org/records/5178777). The observational comparison in [comparison.ipynb](comparison.ipynb) reads `Data/rastinejad_2024.csv` (Rastinejad et al. 2024 component decomposition).

The downloader chains `tools/embed_model_metadata.py`, which writes `model` and `ns_max` as HDF5 root attributes. Loaders validate these against `expected_model` / `expected_ns_max` and fail loudly on filename or metadata drift.

## Layout

| File | Purpose |
|---|---|
| [grb_main.ipynb](grb_main.ipynb) | Main figures, Sections 1 to 13 (with sub-sections 4b TNG-resolution sweep and 7b detected vs intrinsic) |
| [comparison.ipynb](comparison.ipynb) | BH-engine vs HMNS-engine prediction against the Rastinejad et al. (2024) sample, using Gottlieb et al. (2025) Eq. 11 |
| [grb_physics.py](grb_physics.py) | Remnant mass, ejecta, EOS, Gottlieb thresholds, ISCO |
| [grb_classify.py](grb_classify.py) | BNS, BHNS, unified grid, formation channels, observed-merger classifier |
| [grb_rates.py](grb_rates.py) | Levina+ 2026 MSSFR, cosmic integration, BH-spin marginalization, beaming, detected rates |
| [grb_io.py](grb_io.py) | COMPAS HDF5 loading, STROOPWAFEL weights, metadata validation |
| [grb_offsets.py](grb_offsets.py) | Hernquist orbits, projected offset CDFs |
| [grb_plot_style.py](grb_plot_style.py) | Project palette and ApJ rcParams (`apply_apj_rcparams`) |

`Plots/` is tracked; `Data/`, `COMPAS/`, `Papers/`, `Demos/` are not.

## Classification

**BNS, Gottlieb (2024) four-class hybrid** (`classify_bns_2024`). $M_\mathrm{TOV} = 2.2\,M_\odot$, $M_\mathrm{thresh} = 1.27\,M_\mathrm{TOV}$ (Gottlieb 2023 fiducial; Bauswein et al. 2013 give an EOS-dependent band $k \in [1.30, 1.70]$), mass ratio $q \equiv M_1 / M_2$, $q_\mathrm{thresh} = 1.2$.

| Class | Condition |
|---|---|
| sbGRB + blue KN | $M_\mathrm{tot} < 1.2\,M_\mathrm{TOV}$ |
| lbGRB + red KN (HMNS) | $1.2\,M_\mathrm{TOV} \leq M_\mathrm{tot} < M_\mathrm{thresh}$ |
| lbGRB + red KN (disk) | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q \geq q_\mathrm{thresh}$ |
| Faint lbGRB | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q < q_\mathrm{thresh}$ |

**BHNS, disk mass** (`classify_bhns`). $M_\mathrm{disk} = M_\mathrm{rem}^\mathrm{Foucart\,2018} - M_\mathrm{dyn}^\mathrm{KF\,2020}$.

| Class | Condition |
|---|---|
| No GRB | $M_\mathrm{disk} < 0.01\,M_\odot$ |
| Short cbGRB | $0.01 \leq M_\mathrm{disk} < 0.1\,M_\odot$ |
| Long cbGRB | $M_\mathrm{disk} \geq 0.1\,M_\odot$ |

All disk-mass-based GRB rates are upper bounds: 100 percent jet launching above threshold (Gottlieb 2023).

## Assumptions

- **Cosmology**: Planck 2015, matching COMPAS `FastCosmicIntegration`. $H_0 = 67.74$ km/s/Mpc, $\Omega_m = 0.3089$, $\Omega_\Lambda = 0.6911$.
- **SN engine**: Fryer et al. (2012) rapid mechanism, with a global Alsing, Silva, Berti (2018) double-Gaussian NS-mass remap (closes the artificial 1.7 $M_\odot$ deficit, Mandel and Muller 2020).
- **MSSFR**: Levina et al. (2026) Azzalini skew log-normal best-fit to IllustrisTNG TNG100-1 (`MSSFR_PARAMS_LEVINA26_TNG100`, `SFR_PARAMS_LEVINA26_TNG100`); TNG50-1 and TNG300-1 are exposed for the Section 4b resolution sweep.

## Testing

```bash
make ci         # lint + typecheck + smoke; what CI runs on every push and PR
make smoke      # fast subset (no Data/, no compas), under ~15 s
make coverage   # 70 percent coverage floor on grb_*.py over unit + anchors
make test       # full suite; data-bound tests auto-skip if Data/ is empty
```

See [tests/README.md](tests/README.md) for the per-folder, per-section layout and the marker reference.

## References

- Abbott et al. (2019), [arXiv:1805.11579](https://arxiv.org/abs/1805.11579), PRX 9, 011001 (GW170817 properties)
- Abbott et al. (2020), [arXiv:2001.01761](https://arxiv.org/abs/2001.01761), ApJL 892, L3 (GW190425)
- Abbott et al. (2023), [arXiv:2111.03606](https://arxiv.org/abs/2111.03606) (GWTC-3 population)
- Alsing, Silva, and Berti (2018), [arXiv:1709.07889](https://arxiv.org/abs/1709.07889)
- Antoniadis et al. (2016), [arXiv:1605.01665](https://arxiv.org/abs/1605.01665)
- Bardeen, Press, and Teukolsky (1972), [ADS 1972ApJ...178..347B](https://ui.adsabs.harvard.edu/abs/1972ApJ...178..347B)
- Bauswein, Baumgarte, and Janka (2013), [arXiv:1302.6530](https://arxiv.org/abs/1302.6530)
- Bauswein et al. (2020), [arXiv:2004.00846](https://arxiv.org/abs/2004.00846)
- Bavera et al. (2020), [arXiv:1906.12257](https://arxiv.org/abs/1906.12257)
- Beniamini, Nava, and Piran (2016), [arXiv:1606.00311](https://arxiv.org/abs/1606.00311) (radiative-efficiency anchor for `GOTTLIEB25_F_RANGE`)
- Beniamini and Nakar (2019), [arXiv:1808.05076](https://arxiv.org/abs/1808.05076)
- Berger (2014), [arXiv:1311.2603](https://arxiv.org/abs/1311.2603), ARA&A 52, 43
- Binney and Tremaine (2008), *Galactic Dynamics*, 2nd ed., Princeton University Press
- Bloom, Sigurdsson, and Pols (1999), [arXiv:astro-ph/9904338](https://arxiv.org/abs/astro-ph/9904338), MNRAS 305, 763
- Broekgaarden et al. (2021), [arXiv:2103.02608](https://arxiv.org/abs/2103.02608)
- Colombo et al. (2022), [arXiv:2204.07592](https://arxiv.org/abs/2204.07592)
- Crameri, Shephard, and Heron (2020), Nat. Commun. 11, 5444
- Dietrich and Ujevic (2017), [arXiv:1612.03665](https://arxiv.org/abs/1612.03665)
- Finn and Chernoff (1993), [ADS 1993PhRvD..47.2198F](https://ui.adsabs.harvard.edu/abs/1993PhRvD..47.2198F)
- Fong and Berger (2013), [arXiv:1307.0819](https://arxiv.org/abs/1307.0819)
- Fong et al. (2015), [arXiv:1509.02922](https://arxiv.org/abs/1509.02922)
- Fong et al. (2022), [arXiv:2206.01763](https://arxiv.org/abs/2206.01763) (sGRB hosts I/II)
- Foucart, Hinderer, and Nissanke (2018), [arXiv:1807.00011](https://arxiv.org/abs/1807.00011)
- Foucart et al. (2019), [arXiv:1903.09166](https://arxiv.org/abs/1903.09166)
- Fragos et al. (2010), [arXiv:1001.1107](https://arxiv.org/abs/1001.1107)
- Fryer et al. (2012), [arXiv:1110.1726](https://arxiv.org/abs/1110.1726)
- Fujibayashi et al. (2018), [arXiv:1710.07579](https://arxiv.org/abs/1710.07579), ApJ 860, 64
- Fuller and Ma (2019), [arXiv:1905.08793](https://arxiv.org/abs/1905.08793)
- Gerosa et al. (2018), [arXiv:1808.02491](https://arxiv.org/abs/1808.02491)
- Ghirlanda et al. (2016), [arXiv:1607.07875](https://arxiv.org/abs/1607.07875), A&A 594, A84
- Goldstein et al. (2017), [arXiv:1710.05446](https://arxiv.org/abs/1710.05446), ApJL 848, L14
- Gottlieb et al. (2023), [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Gottlieb et al. (2024, 2025), [arXiv:2411.13657](https://arxiv.org/abs/2411.13657)
- Hernquist (1990), [ADS 1990ApJ...356..359H](https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H)
- Kasen et al. (2017), [arXiv:1710.05463](https://arxiv.org/abs/1710.05463), Nature 551, 80
- Kawaguchi et al. (2015), [arXiv:1601.07711](https://arxiv.org/abs/1601.07711)
- Koppel, Bovard, and Rezzolla (2019), [arXiv:1901.09977](https://arxiv.org/abs/1901.09977), ApJL 872, L16
- Kroupa (2001), [arXiv:astro-ph/0009005](https://arxiv.org/abs/astro-ph/0009005)
- Kruger and Foucart (2020), [arXiv:2002.07728](https://arxiv.org/abs/2002.07728)
- Lattimer and Prakash (2001), [arXiv:astro-ph/0002232](https://arxiv.org/abs/astro-ph/0002232), ApJ 550, 426
- Levan et al. (2024), [arXiv:2307.02098](https://arxiv.org/abs/2307.02098), Nature 626, 737
- Levina et al. (2026), [arXiv:2601.20202](https://arxiv.org/abs/2601.20202)
- Lippuner et al. (2017), [arXiv:1703.06216](https://arxiv.org/abs/1703.06216), MNRAS 472, 904
- Madau and Dickinson (2014), [arXiv:1403.0007](https://arxiv.org/abs/1403.0007)
- Mandel and Muller (2020), [arXiv:2006.08360](https://arxiv.org/abs/2006.08360)
- Margalit and Metzger (2017), [arXiv:1710.05938](https://arxiv.org/abs/1710.05938), ApJL 850, L19
- Metzger (2019), [arXiv:1910.01617](https://arxiv.org/abs/1910.01617), Living Rev. Rel. 23, 1
- Mooley et al. (2018), [arXiv:1806.09693](https://arxiv.org/abs/1806.09693), Nature 561, 355
- Neijssel et al. (2019), [arXiv:1906.08136](https://arxiv.org/abs/1906.08136)
- Patton and Sukhbold (2020), [arXiv:2005.03055](https://arxiv.org/abs/2005.03055), MNRAS 499, 2803
- Planck Collaboration / Ade et al. (2016), [arXiv:1502.01589](https://arxiv.org/abs/1502.01589), A&A 594, A13
- Raaijmakers et al. (2021), [arXiv:2105.06981](https://arxiv.org/abs/2105.06981)
- Radice et al. (2018), [arXiv:1809.11163](https://arxiv.org/abs/1809.11163)
- Rastinejad et al. (2022), [arXiv:2204.10864](https://arxiv.org/abs/2204.10864), Nature 612, 223 (GRB 211211A)
- Rastinejad et al. (2024), [arXiv:2306.14947](https://arxiv.org/abs/2306.14947)
- Read et al. (2009), [arXiv:0812.2163](https://arxiv.org/abs/0812.2163)
- Salpeter (1955), [ADS 1955ApJ...121..161S](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S)
- Villasenor et al. (2005), [ADS 2005Natur.437..855V](https://ui.adsabs.harvard.edu/abs/2005Natur.437..855V) (GRB 050709)
- Wanderman and Piran (2015), [arXiv:1405.5878](https://arxiv.org/abs/1405.5878)

## License

MIT. See [LICENSE](LICENSE).
