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

- Alsing, Silva, and Berti (2018), [arXiv:1709.07889](https://arxiv.org/abs/1709.07889)
- Bardeen, Press, and Teukolsky (1972), [ADS 1972ApJ...178..347B](https://ui.adsabs.harvard.edu/abs/1972ApJ...178..347B)
- Bauswein et al. (2013), [arXiv:1302.6530](https://arxiv.org/abs/1302.6530)
- Beniamini and Nakar (2019), [arXiv:1808.05076](https://arxiv.org/abs/1808.05076)
- Broekgaarden et al. (2021), [arXiv:2103.02608](https://arxiv.org/abs/2103.02608)
- Fong et al. (2015), [arXiv:1509.02922](https://arxiv.org/abs/1509.02922)
- Fong and Berger (2013), [arXiv:1307.0819](https://arxiv.org/abs/1307.0819)
- Foucart, Hinderer, and Nissanke (2018), [arXiv:1807.00011](https://arxiv.org/abs/1807.00011)
- Fryer et al. (2012), [arXiv:1110.1726](https://arxiv.org/abs/1110.1726)
- Gottlieb et al. (2023), [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Gottlieb et al. (2024, 2025), [arXiv:2411.13657](https://arxiv.org/abs/2411.13657)
- Hernquist (1990), [ADS 1990ApJ...356..359H](https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H)
- Kawaguchi et al. (2015), [arXiv:1601.07711](https://arxiv.org/abs/1601.07711)
- Kroupa (2001), [arXiv:astro-ph/0009005](https://arxiv.org/abs/astro-ph/0009005)
- Kruger and Foucart (2020), [arXiv:2002.07728](https://arxiv.org/abs/2002.07728)
- Levina et al. (2026), [arXiv:2601.20202](https://arxiv.org/abs/2601.20202)
- Madau and Dickinson (2014), [arXiv:1403.0007](https://arxiv.org/abs/1403.0007)
- Mandel and Muller (2020), [arXiv:2006.08360](https://arxiv.org/abs/2006.08360)
- Neijssel et al. (2019), [arXiv:1906.08136](https://arxiv.org/abs/1906.08136)
- Raaijmakers et al. (2021), [arXiv:2105.06981](https://arxiv.org/abs/2105.06981)
- Rastinejad et al. (2024), [arXiv:2306.14947](https://arxiv.org/abs/2306.14947)
- Read et al. (2009), [arXiv:0812.2163](https://arxiv.org/abs/0812.2163)
- Wanderman and Piran (2015), [arXiv:1405.5878](https://arxiv.org/abs/1405.5878)

## License

MIT. See [LICENSE](LICENSE).
