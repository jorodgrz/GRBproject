"""Section 2 of grb_main.ipynb: Component Mass Distributions by GRB Class.

Smoke-level invariants on the Alsing, Silva and Berti (2018) double-Gaussian
NS-mass remap that produces the per-class component-mass histograms.  The
remap closes the Fryer 2012 baryonic-to-gravitational artifact near 1.7 Msun
(Eq. 12-13; present in both delayed and rapid engines, per Broekgaarden+
2021 footnote 3) and preserves the ``m1 >= m2`` invariant.

Reference: Alsing, Silva and Berti (2018), MNRAS 478, 1377 (Table 3 fit);
Mandel and Muller (2020), MNRAS 499, 3214 (gap motivation).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_physics import (
    M_TOV,
    NS_REMAP_M_MIN,
    NS_REMAP_W1,
    NS_REMAP_W2,
    remap_ns_masses_double_gaussian,
)


@pytest.fixture(scope="module")
def synthetic_pre_remap_ns_masses():
    """Two-population sample with an artificial deficit near 1.7 Msun.

    Mimics the Fryer 2012 Eq. 12-13 baryonic-to-gravitational mass
    artifact on a small N sample so the test is fast in CI; real-data
    behaviour is verified end-to-end in ``tests/anchors/test_literature_anchors.py``.
    """
    rng = np.random.default_rng(123)
    n = 5000
    a = rng.normal(1.30, 0.06, n // 2)
    b = rng.normal(1.95, 0.10, n // 2)
    pool = np.concatenate([a, b])
    pool = pool[(pool > NS_REMAP_M_MIN) & (pool < M_TOV)]
    rng.shuffle(pool)
    half = pool.size // 2
    m1 = pool[:half]
    m2 = pool[half : 2 * half]
    m_heavy = np.maximum(m1, m2)
    m_light = np.minimum(m1, m2)
    return m_heavy, m_light


def test_remap_preserves_m1_geq_m2(synthetic_pre_remap_ns_masses):
    m1, m2 = synthetic_pre_remap_ns_masses
    m1_new, m2_new = remap_ns_masses_double_gaussian(m1, m2)
    assert (m1_new >= m2_new).all(), "m1 >= m2 invariant broken by remap"


def test_remap_stays_within_truncation_window(synthetic_pre_remap_ns_masses):
    m1, m2 = synthetic_pre_remap_ns_masses
    m1_new, m2_new = remap_ns_masses_double_gaussian(m1, m2)
    for arr in (m1_new, m2_new):
        assert (arr >= NS_REMAP_M_MIN - 1e-9).all()
        assert (arr <= M_TOV + 1e-9).all()


def test_remap_closes_artificial_gap_near_1p7(synthetic_pre_remap_ns_masses):
    """Pre-remap pool has a deficit in [1.65, 1.80]; post-remap fills it
    with non-zero density (the whole point of the Alsing remap)."""
    m1, m2 = synthetic_pre_remap_ns_masses
    pool_pre = np.concatenate([m1, m2])
    pool_pre_in_gap = ((pool_pre >= 1.65) & (pool_pre <= 1.80)).sum()

    m1_new, m2_new = remap_ns_masses_double_gaussian(m1, m2)
    pool_post = np.concatenate([m1_new, m2_new])
    pool_post_in_gap = ((pool_post >= 1.65) & (pool_post <= 1.80)).sum()

    assert pool_post_in_gap > pool_pre_in_gap, (
        f"remap did not close the Fryer 2012 gap; pre={pool_pre_in_gap}, post={pool_post_in_gap}"
    )


def test_alsing_double_gauss_weights_sum_to_one():
    assert NS_REMAP_W1 + NS_REMAP_W2 == pytest.approx(1.0, rel=1e-6), (NS_REMAP_W1, NS_REMAP_W2)
