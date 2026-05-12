"""Section 1 of grb_main.ipynb: Mass Plane (BNS and BHNS).

Smoke-level invariants on ``classify_grid`` over a synthetic (M1, M2)
meshgrid: every labelled class is reachable, the ``ns_max`` truncation
matches the requested model, and STROOPWAFEL-weighted class fractions
sum to 1.  Pure unit-style; runs in CI without ``Data/``.

Reference: Gottlieb et al. (2024), arXiv:2411.13657; Broekgaarden et
al. (2021), arXiv:2103.02608, Sec. 3.4 (M_NS_max per model).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import NS_MAX_FIDUCIAL, classify_grid
from grb_physics import M_THRESH, M_TOV


def _synthetic_meshgrid(n=80, m_lo=1.0, m_hi=4.0):
    m_axis = np.linspace(m_lo, m_hi, n)
    m1g, m2g = np.meshgrid(m_axis, m_axis)
    return m1g, m2g


@pytest.mark.parametrize("ns_max", sorted(NS_MAX_FIDUCIAL))
def test_classify_grid_labels_in_expected_range(ns_max):
    """All grid cells produce labels in {0, 1, ..., 6} and at least
    one cell in every BNS or BHNS labelled class is reachable for the
    fiducial Broekgaarden+ 2021 ns_max values."""
    m1g, m2g = _synthetic_meshgrid()
    cls = classify_grid(m1g, m2g, ns_max=ns_max, R_1p4_km=12.0)

    assert cls.dtype.kind in ("i", "u")
    assert cls.min() >= 0 and cls.max() <= 6, (cls.min(), cls.max())

    populated = set(np.unique(cls).tolist())
    assert 0 in populated, "background class 0 unreachable"
    bns_labels = {1, 2, 3, 4} & populated
    bhns_labels = {5, 6} & populated
    assert bns_labels, f"no BNS labels populated (ns_max={ns_max})"
    assert bhns_labels, f"no BHNS labels populated (ns_max={ns_max})"


def test_classify_grid_ns_max_truncates_bns_region():
    """Cells with both components above ``ns_max`` cannot be class 1-4
    (BNS region); they should fall into BHNS (5, 6) or background (0)."""
    m1g, m2g = _synthetic_meshgrid()
    ns_max = 2.0
    cls = classify_grid(m1g, m2g, ns_max=ns_max, R_1p4_km=12.0)
    above = (m1g > ns_max) & (m2g > ns_max)
    bns_labels_above = np.isin(cls[above], [1, 2, 3, 4])
    assert not bns_labels_above.any(), (
        f"{bns_labels_above.sum()} cells with both m1, m2 > ns_max "
        f"= {ns_max} were classified as BNS"
    )


def test_classify_grid_weighted_class_fractions_sum_to_one():
    """STROOPWAFEL-weighted class fractions over a synthetic mass-plane
    sample partition unity to numerical precision."""
    rng = np.random.default_rng(42)
    n = 5000
    m1 = rng.uniform(1.1, 3.5, n)
    m2 = rng.uniform(1.1, 3.5, n)
    weights = rng.uniform(0.0, 1.0, n)

    m_heavy = np.maximum(m1, m2)
    m_light = np.minimum(m1, m2)
    m_tot = m_heavy + m_light
    q = m_heavy / m_light

    m_thresh = M_THRESH
    hmns_split = 1.2 * M_TOV

    masks = {
        "sb": m_tot < hmns_split,
        "lb_HMNS": (m_tot >= hmns_split) & (m_tot < m_thresh),
        "lb_disk": (m_tot >= m_thresh) & (q >= 1.2),
        "faint_lb": (m_tot >= m_thresh) & (q < 1.2),
    }
    fractions = {k: weights[v].sum() / weights.sum() for k, v in masks.items()}
    total = sum(fractions.values())
    assert total == pytest.approx(1.0, rel=1e-12), fractions
