# GRB Classification from Compact Binary Mergers

[![tests](https://github.com/jorodgrz/GRBproject/actions/workflows/pytest.yml/badge.svg)](https://github.com/jorodgrz/GRBproject/actions/workflows/pytest.yml)

Population-level predictions for merger-driven short and long GRBs. The pipeline applies the Gottlieb et al. (2023, 2024) classification frameworks to COMPAS binary population synthesis (Broekgaarden et al. 2021, Models A, F, G, J, K). 

## Setup

```bash
conda env create -f environment.yml
conda activate grb-env
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

[environment.yml](environment.yml) pins `compas_python_utils` to upstream commit `81722d4`.

## Data

```bash
python tools/download_compas_data.py --tier 1 --confirm    # 5 core models A, F, G, J, K (manuscript figures)
python tools/download_compas_data.py --confirm    # full 20-model grid, ~45 GB
```

Files land in `Data/COMPASCompactOutput_<KIND>_<SUFFIX>.h5`. BNS catalogues from [Zenodo 5189849](https://zenodo.org/records/5189849), BHNS from [Zenodo 5178777](https://zenodo.org/records/5178777). The observational comparison in [comparison.ipynb](comparison.ipynb) reads `Data/rastinejad_2024.csv` (Rastinejad et al. 2024 component decomposition).

## Layout

| File | Purpose |
|---|---|
| [grb_main.ipynb](grb_main.ipynb) | Main figures, Sections 1 to 12 |
| [comparison.ipynb](comparison.ipynb) | Observational comparison; canonical ApJ rcParams block |
| [grb_physics.py](grb_physics.py) | Remnant mass, ejecta, EOS, Gottlieb thresholds |
| [grb_classify.py](grb_classify.py) | BNS, BHNS, unified grid, formation channels |
| [grb_rates.py](grb_rates.py) | MSSFR convolution, BH-spin marginalization, beaming |
| [grb_io.py](grb_io.py) | COMPAS HDF5 loading, STROOPWAFEL weights |
| [grb_offsets.py](grb_offsets.py) | Hernquist orbits, projected offset CDFs |

`Plots/` is tracked; `Data/`, `COMPAS/`, `Papers/`, `Demos/` are not.

## Classification

**BNS, Gottlieb (2024) four-class hybrid** (`classify_bns_2024`). $M_\mathrm{TOV} = 2.2\,M_\odot$, $M_\mathrm{thresh} = 1.27\*M_\mathrm{TOV}$, $q_\mathrm{thresh} = 1.2$.

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



## References

- Alsing, Silva, and Berti (2018), [arXiv:1709.07889](https://arxiv.org/abs/1709.07889)
- Bauswein et al. (2013), [arXiv:1302.6530](https://arxiv.org/abs/1302.6530)
- Broekgaarden et al. (2021), [arXiv:2103.02608](https://arxiv.org/abs/2103.02608)
- Fong and Berger (2013), [arXiv:1307.0819](https://arxiv.org/abs/1307.0819)
- Foucart, Hinderer, and Nissanke (2018), [arXiv:1807.00011](https://arxiv.org/abs/1807.00011)
- Fryer et al. (2012), [arXiv:1110.1726](https://arxiv.org/abs/1110.1726)
- Gottlieb et al. (2023), [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Gottlieb et al. (2024), [arXiv:2411.13657](https://arxiv.org/abs/2411.13657)
- Hernquist (1990), [ADS 1990ApJ...356..359H](https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H)
- Kroupa (2001), [arXiv:astro-ph/0009005](https://arxiv.org/abs/astro-ph/0009005)
- Kruger and Foucart (2020), [arXiv:2002.07728](https://arxiv.org/abs/2002.07728)
- Madau and Dickinson (2014), [arXiv:1403.0007](https://arxiv.org/abs/1403.0007)
- Mandel and Muller (2020), [arXiv:2006.08360](https://arxiv.org/abs/2006.08360)
- Neijssel et al. (2019), [arXiv:1906.08136](https://arxiv.org/abs/1906.08136)
- Raaijmakers et al. (2021), [arXiv:2105.06981](https://arxiv.org/abs/2105.06981)
- Read et al. (2009), [arXiv:0812.2163](https://arxiv.org/abs/0812.2163)
- Wanderman and Piran (2015), [arXiv:1405.5878](https://arxiv.org/abs/1405.5878)


## License

MIT. See [LICENSE](LICENSE).
