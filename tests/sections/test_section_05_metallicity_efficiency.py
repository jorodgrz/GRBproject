"""Section 5 of grb_main.ipynb: Metallicity Dependence of GRB Formation Efficiency.

NEW. Smoke-level invariants on ``formation_efficiency`` that the
per-class plot relies on:

1. Per-class arrays partition the total at every Z bin (synthetic data).
2. On real Model A BNS, every per-class efficiency is finite and
   non-negative across the COMPAS metallicity grid (skips without data).

These pin the data plumbing the figure consumes; the absolute amplitudes
are exercised in ``tests/integration/test_notebook_output_vs_literature.py``.

Reference: Broekgaarden et al. (2021), arXiv:2103.02608 Sec. 4
(formation efficiency definition); Gottlieb et al. (2024),
arXiv:2411.13657 (four-class scheme used here).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import classify_bns_2024
from grb_io import METALLICITY_GRID, load_bns
from grb_rates import formation_efficiency


def test_per_class_efficiency_partitions_total_synthetic():
    rng = np.random.default_rng(7)
    n = 2000
    Z_all = rng.choice(METALLICITY_GRID, size=n)
    w_all = rng.uniform(0.0, 1.0, n)
    m1 = rng.uniform(1.1, 3.5, n)
    m2 = rng.uniform(1.1, 3.5, n)
    masks = classify_bns_2024(m1, m2)

    eff = formation_efficiency(
        METALLICITY_GRID,
        Z_all,
        w_all,
        masks=masks,
        mean_mass_evolved=1.0,
    )

    assert "total" in eff
    summed = np.zeros_like(eff["total"])
    for label in masks:
        summed += eff[label]
    np.testing.assert_allclose(summed, eff["total"], rtol=1e-9, atol=1e-12)


@pytest.mark.requires_data
def test_per_class_efficiency_finite_on_modelA(bns_a_path):
    """Real-data smoke: every per-class efficiency entry is finite and
    non-negative.  Uses ``mean_mass_evolved=1.0`` because the partition
    invariant is independent of the calibration anchor."""
    bns = load_bns(bns_a_path, expected_model="A", expected_ns_max=2.5)
    masks = classify_bns_2024(bns["m1"], bns["m2"])
    eff = formation_efficiency(
        METALLICITY_GRID,
        bns["metallicity"],
        bns["weights"],
        masks=masks,
        mean_mass_evolved=1.0,
    )
    for label, arr in eff.items():
        assert np.all(np.isfinite(arr)), f"non-finite efficiency in {label}"
        assert np.all(arr >= 0.0), f"negative efficiency in {label}"
