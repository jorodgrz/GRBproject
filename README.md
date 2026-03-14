# GRB Project — Compact Binary Population Visualizations

Jupyter notebooks for visualizing gravitational wave source formation channels using COMPAS binary population synthesis simulation data.

Based on [Broekgaarden et al. 2022](https://arxiv.org/abs/2112.05763).

---

## Notebooks

| Notebook | Description |
|---|---|
| `Demo/demo_high_school_project-Copy1.ipynb` | Tutorial notebook — computes and plots BHNS formation efficiencies vs. metallicity. Recreates Figure 1 from Broekgaarden et al. 2022. |
| `Demo/demo_plotting_BHNS_distributions_for_astrophysical_and_LVK-population.ipynb` | Advanced notebook — plots astrophysical and LVK-detectable population distributions (masses, delay times, merger rates) for BHNS, BBH, and BNS systems. |

---

## Data

The notebooks require COMPAS HDF5 output files from Zenodo:

- **BHNS/BBH/BNS (AllDCO):** [https://zenodo.org/record/5178777](https://zenodo.org/record/5178777)
- **BBH (fiducial):** [https://zenodo.org/records/5651073](https://zenodo.org/records/5651073)
- **BNS (fiducial):** [https://zenodo.org/records/5189849](https://zenodo.org/records/5189849)

Download the relevant `.h5` files and update the `path` variable at the top of each notebook to point to your local copy. Data files are excluded from this repo via `.gitignore`.

---

## Setup

### Requirements

- Python 3.11+
- See `requirements.txt` for all dependencies

### Install

```bash
# Clone the repo
git clone https://github.com/jorodgrz/GRBproject.git
cd GRBproject

# Create and activate a virtual environment
python3 -m venv grb-env
source grb-env/bin/activate   # On Windows: grb-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
jupyter lab
```

Then open a notebook from the `Demo/` folder and update the data `path` variable to point to your downloaded `.h5` file.

---

## Glossary

| Term | Definition |
|---|---|
| BBH | Binary Black Hole |
| BNS | Binary Neutron Star |
| BHNS | Black Hole – Neutron Star binary |
| DCO | Double Compact Object |
| GW | Gravitational Wave |
| ZAMS | Zero Age Main Sequence |
| LVK | LIGO–Virgo–KAGRA collaboration |
