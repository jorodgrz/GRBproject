# GRB Classification from Compact Binary Mergers

Population-level predictions for merger-driven short and long GRBs. Applies the Gottlieb et al. (2023, 2024) frameworks to COMPAS population synthesis (Broekgaarden et al. 2021; Models A and K).

## Structure

| File | Purpose |
|---|---|
| `grb_main.ipynb` | Main research notebook |
| `comparison.ipynb` | Observational comparison |
| `grb_physics.py` | Remnant mass, ejecta, EOS, thresholds |
| `grb_classify.py` | BNS, BHNS, grid, formation channels |
| `grb_rates.py` | MSSFR convolution, beaming, BH-spin |
| `grb_io.py` | COMPAS HDF5 loading and weighting |
| `grb_offsets.py` | Hernquist orbits, offset CDFs |

`Plots/` holds generated figures. `Data/`, `COMPAS/`, `Papers/`, `Demos/` are not tracked.

## Setup

```bash
conda env create -f environment.yml
conda activate grb-env
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

`environment.yml` installs `compas_python_utils` from upstream at the pinned commit `81722d4`.

## Classification

**BNS, Gottlieb (2024) four-class.** $M_\mathrm{TOV} = 2.2\,M_\odot$, $M_\mathrm{thresh} = 1.27\,M_\mathrm{TOV}$, $q_\mathrm{thresh} = 1.2$.

| Class | Condition |
|---|---|
| sbGRB + blue KN | $M_\mathrm{tot} < 1.2\,M_\mathrm{TOV}$ |
| lbGRB + red KN (HMNS) | $1.2\,M_\mathrm{TOV} \leq M_\mathrm{tot} < M_\mathrm{thresh}$ |
| lbGRB + red KN (disk) | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q \geq q_\mathrm{thresh}$ |
| Faint lbGRB | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q < q_\mathrm{thresh}$ |

**BHNS, disk-mass three-class.** $M_\mathrm{disk} = M_\mathrm{rem}^\mathrm{Foucart\,2018} - M_\mathrm{dyn}^\mathrm{KF\,2020}$.

| Class | Condition |
|---|---|
| No GRB | $M_\mathrm{disk} < 0.01\,M_\odot$ |
| Short cbGRB | $0.01 \leq M_\mathrm{disk} < 0.1\,M_\odot$ |
| Long cbGRB | $M_\mathrm{disk} \geq 0.1\,M_\odot$ |

## Cosmology

Planck 2015 (matching COMPAS `FastCosmicIntegration`): $H_0 = 67.74$ km/s/Mpc, $\Omega_m = 0.3089$, $\Omega_\Lambda = 0.6911$. MSSFR from Neijssel et al. (2019).

## Data

COMPAS HDF5 files (Broekgaarden+ 2021): BNS [Zenodo 5189849](https://zenodo.org/records/5189849), BHNS [Zenodo 5178777](https://zenodo.org/records/5178777). Use the bundled downloader:

```bash
python tools/download_compas_data.py --dry-run                  # plan, no download
python tools/download_compas_data.py --tier 1 --confirm         # core 5 models (A, F, G, J, K)
python tools/download_compas_data.py --kind BNS --models J F G  # explicit subset
python tools/download_compas_data.py --confirm                  # full grid (~45 GB)
```

Tier 1 covers manuscript figures. Tier 2 adds kick variants (N, O). Files land at `Data/COMPASCompactOutput_<KIND>_<SUFFIX>.h5`.

## Key References

- Broekgaarden et al. (2021). [arXiv:2103.02608](https://arxiv.org/abs/2103.02608)
- Foucart, Hinderer, Nissanke (2018). [arXiv:1807.00011](https://arxiv.org/abs/1807.00011)
- Gottlieb et al. (2023). [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Gottlieb et al. (2024). [arXiv:2411.13657](https://arxiv.org/abs/2411.13657)
- Kruger & Foucart (2020). [arXiv:2002.07728](https://arxiv.org/abs/2002.07728)
- Neijssel et al. (2019). MSSFR for DCO mergers.
- Raaijmakers et al. (2021). [arXiv:2105.06981](https://arxiv.org/abs/2105.06981)
- Wanderman & Piran (2015). [arXiv:1405.5878](https://arxiv.org/abs/1405.5878)

## License

MIT. See [LICENSE](LICENSE).
