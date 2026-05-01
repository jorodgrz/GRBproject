# GRB Classification from Compact Binary Mergers

Population-level predictions for merger-driven short and long GRBs. Applies the Gottlieb et al. (2023, 2024) classification frameworks to COMPAS binary population synthesis (Broekgaarden et al. 2021, 2022; Models A and K).

## Structure

| File | Purpose |
|---|---|
| `grb_main.ipynb` | Main research notebook |
| `comparison.ipynb` | Observational comparison (BH-engine vs HMNS-engine) |
| `grb_physics.py` | Foucart remnant mass, Kruger & Foucart ejecta, NS compactness, EOS, Gottlieb thresholds |
| `grb_classify.py` | BNS three- and four-class, BHNS disk-mass, unified grid, formation channels |
| `grb_rates.py` | MSSFR convolution, rate weights, BH-spin marginalization, beaming, observed sGRB rates |
| `grb_io.py` | COMPAS HDF5 loading, metallicity grid, weighted subsampling |
| `grb_offsets.py` | Hernquist orbit integration, projected offset CDFs |

| Folder | Contents |
|---|---|
| `Plots/` | Generated figures |
| `Data/` | COMPAS HDF5 files and observational CSVs (not tracked) |

External, not tracked in this repo:

- `COMPAS/`: post-processing utilities cloned from upstream (see Setup).
- `Papers/`: local reference PDFs.
- `Demos/`: introductory notebooks.

## Setup

One-command setup with conda (recommended):

```bash
conda env create -f environment.yml
conda activate grb-env
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

Manual setup (equivalent):

```bash
conda create -n grb-env python=3.10
conda activate grb-env
conda install -c conda-forge numpy scipy matplotlib h5py pandas jupyterlab ipykernel ipython
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

### COMPAS post-processing

`grb_main.ipynb` Section 11 imports `compas_python_utils.cosmic_integration.FastCosmicIntegration`. Install COMPAS at the pinned commit used for this project:

```bash
git clone https://github.com/TeamCOMPAS/COMPAS.git
cd COMPAS && git checkout 81722d4483609446a2a40cce4028e2743544c187
cd .. && pip install -e ./COMPAS
```

The `environment.yml` route handles this automatically.

## Classification

**BNS, Gottlieb (2024) four-class hybrid.** $M_\mathrm{TOV} = 2.2\,M_\odot$, $M_\mathrm{thresh} = 1.27\,M_\mathrm{TOV}$, $q_\mathrm{thresh} = 1.2$.

| Class | Condition |
|---|---|
| sbGRB + blue KN | Long-lived HMNS, $M_\mathrm{tot} < 1.2\,M_\mathrm{TOV}$ |
| lbGRB + red KN (HMNS) | Short-lived HMNS, $1.2\,M_\mathrm{TOV} \leq M_\mathrm{tot} < M_\mathrm{thresh}$ |
| lbGRB + red KN (disk) | Prompt collapse, massive disk: $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q \geq q_\mathrm{thresh}$ |
| Faint lbGRB | Prompt collapse, small disk: $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q < q_\mathrm{thresh}$ |

**BHNS, disk-mass three-class.** $M_\mathrm{disk} = M_\mathrm{rem}^\mathrm{Foucart\,2018} - M_\mathrm{dyn}^\mathrm{KF\,2020}$.

| Class | Condition |
|---|---|
| No GRB | $M_\mathrm{disk} < 0.01\,M_\odot$ |
| Short cbGRB | $0.01 \leq M_\mathrm{disk} < 0.1\,M_\odot$ |
| Long cbGRB | $M_\mathrm{disk} \geq 0.1\,M_\odot$ |

## Cosmology

Planck 2015, matching COMPAS `FastCosmicIntegration`: $H_0 = 67.74$ km/s/Mpc, $\Omega_m = 0.3089$, $\Omega_\Lambda = 0.6911$. MSSFR from Neijssel et al. (2019).

## Data

COMPAS HDF5 files are not git-tracked. Download from Zenodo into `Data/`:

| Dataset | Zenodo |
|---|---|
| BNS Models A, K | [5189849](https://zenodo.org/records/5189849) |
| BHNS Model A | [5178777](https://zenodo.org/records/5178777) |

## Key References

- Broekgaarden et al. (2021, 2022). COMPAS BHNS and BNS populations. [arXiv:2103.02608](https://arxiv.org/abs/2103.02608), [arXiv:2112.05763](https://arxiv.org/abs/2112.05763)
- Foucart, Hinderer, Nissanke (2018). BHNS remnant baryon mass. [arXiv:1807.00011](https://arxiv.org/abs/1807.00011)
- Fryer et al. (2012). Rapid SN remnant masses. [arXiv:1110.1726](https://arxiv.org/abs/1110.1726)
- Gottlieb et al. (2023). Unified short and long GRB picture. [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Gottlieb et al. (2024). Hybrid kilonova plus GRB model. [arXiv:2411.13657](https://arxiv.org/abs/2411.13657)
- Hernquist (1990). Spherical galaxy potential. [ADS](https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H)
- Kruger & Foucart (2020). Disk and ejecta mass fits. [arXiv:2002.07728](https://arxiv.org/abs/2002.07728)
- Neijssel et al. (2019). MSSFR for double compact object mergers.
- Raaijmakers et al. (2021). NICER constraints on $M_\mathrm{TOV}$. [arXiv:2105.06981](https://arxiv.org/abs/2105.06981)
- Wanderman & Piran (2015). Short GRB rate and luminosity function. [arXiv:1405.5878](https://arxiv.org/abs/1405.5878)

Reference PDFs are not redistributed in this repository; the arXiv and ADS links above are the canonical sources.

## License

MIT. See [LICENSE](LICENSE).
