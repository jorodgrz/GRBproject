# GRB Classification from Compact Binary Mergers

Applies Gottlieb et al. (2023, 2024) GRB-class frameworks to COMPAS population synthesis (Broekgaarden et al. 2021, Models A and K).

## Setup

```bash
conda env create -f environment.yml
conda activate grb-env
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

`environment.yml` pins `compas_python_utils` to upstream commit `81722d4`.

## Data

```bash
python tools/download_compas_data.py --tier 1 --confirm    # 5 core models, manuscript figures
python tools/download_compas_data.py --confirm             # full grid, ~45 GB
```

Files land in `Data/COMPASCompactOutput_<KIND>_<SUFFIX>.h5`. BNS [Zenodo 5189849](https://zenodo.org/records/5189849), BHNS [Zenodo 5178777](https://zenodo.org/records/5178777).

## Layout

| File | Purpose |
|---|---|
| `grb_main.ipynb` | Main figures |
| `comparison.ipynb` | Observational comparison |
| `grb_physics.py` | Remnant mass, ejecta, EOS, thresholds |
| `grb_classify.py` | BNS, BHNS, grid, formation channels |
| `grb_rates.py` | MSSFR convolution, beaming, BH-spin |
| `grb_io.py` | COMPAS HDF5 loading, STROOPWAFEL weights |
| `grb_offsets.py` | Hernquist orbits, offset CDFs |

`Plots/` is tracked; `Data/`, `COMPAS/`, `Papers/`, `Demos/` are not.

## Classification

**BNS, Gottlieb (2024).** $M_\mathrm{TOV} = 2.2\,M_\odot$, $M_\mathrm{thresh} = 1.27\,M_\mathrm{TOV}$, $q_\mathrm{thresh} = 1.2$.

| Class | Condition |
|---|---|
| sbGRB + blue KN | $M_\mathrm{tot} < 1.2\,M_\mathrm{TOV}$ |
| lbGRB + red KN (HMNS) | $1.2\,M_\mathrm{TOV} \leq M_\mathrm{tot} < M_\mathrm{thresh}$ |
| lbGRB + red KN (disk) | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q \geq q_\mathrm{thresh}$ |
| Faint lbGRB | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q < q_\mathrm{thresh}$ |

**BHNS, disk mass.** $M_\mathrm{disk} = M_\mathrm{rem}^\mathrm{Foucart\,2018} - M_\mathrm{dyn}^\mathrm{KF\,2020}$.

| Class | Condition |
|---|---|
| No GRB | $M_\mathrm{disk} < 0.01\,M_\odot$ |
| Short cbGRB | $0.01 \leq M_\mathrm{disk} < 0.1\,M_\odot$ |
| Long cbGRB | $M_\mathrm{disk} \geq 0.1\,M_\odot$ |

## Cosmology

Planck 2015 (matches COMPAS `FastCosmicIntegration`): $H_0 = 67.74$ km/s/Mpc, $\Omega_m = 0.3089$, $\Omega_\Lambda = 0.6911$. MSSFR from Neijssel et al. (2019). Disk-mass GRB rates assume 100 percent jet launching above threshold and are upper bounds.

## References

- Broekgaarden et al. (2021), [arXiv:2103.02608](https://arxiv.org/abs/2103.02608)
- Foucart, Hinderer, Nissanke (2018), [arXiv:1807.00011](https://arxiv.org/abs/1807.00011)
- Gottlieb et al. (2023), [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Gottlieb et al. (2024), [arXiv:2411.13657](https://arxiv.org/abs/2411.13657)
- Kruger and Foucart (2020), [arXiv:2002.07728](https://arxiv.org/abs/2002.07728)
- Neijssel et al. (2019), MSSFR for DCO mergers
- Raaijmakers et al. (2021), [arXiv:2105.06981](https://arxiv.org/abs/2105.06981)
- Wanderman and Piran (2015), [arXiv:1405.5878](https://arxiv.org/abs/1405.5878)

## License

MIT. See [LICENSE](LICENSE).
