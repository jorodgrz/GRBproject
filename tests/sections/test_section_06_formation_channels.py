"""Section 6 of grb_main.ipynb: Formation Channel Breakdown per GRB Class.

NEW data path.  ``load_bns_with_channels`` plus
``classify_formation_channels`` plus ``channel_class_crosstab`` on real
Model A BNS.  Asserts the joint contingency table is well-formed and
the row / column / total normalisations agree with the helper contract.

Marked ``requires_data``: skips cleanly without the BNS-A HDF5.
The helper-only invariants (synthetic inputs) are exercised by
``tests/unit/test_phase4_helpers.py::test_channel_class_crosstab_*``.

Reference: Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 5
(channels I to V); Gottlieb et al. (2024), arXiv:2411.13657
(four-class scheme).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import (
    channel_class_crosstab,
    classify_bns_2024,
    classify_formation_channels,
)
from grb_io import load_bns_with_channels


@pytest.mark.requires_data
def test_channel_class_crosstab_modelA_real_data(bns_a_path):
    bns = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)

    channels = classify_formation_channels(
        dblCE=bns["dblCE"],
        fc_CEE=bns["fc_CEE"],
        fc_mt_p1=bns["fc_mt_p1"],
        fc_mt_s1=bns["fc_mt_s1"],
        fc_mt_p1_K1=bns["fc_mt_p1_K1"],
        fc_mt_s1_K2=bns["fc_mt_s1_K2"],
    )
    classes = classify_bns_2024(bns["m1"], bns["m2"])

    raw = channel_class_crosstab(channels, classes, bns["weights"])
    assert raw.shape == (5, 4), raw.shape
    assert (raw.values >= 0.0).all(), "negative entry in raw crosstab"
    assert raw.values.sum() > 0.0, "crosstab is empty"

    by_channel = channel_class_crosstab(channels, classes, bns["weights"], normalise="channel")
    row_sums = by_channel.sum(axis=1).values
    nonzero_rows = row_sums > 0
    np.testing.assert_allclose(row_sums[nonzero_rows], 1.0, rtol=1e-9)

    by_class = channel_class_crosstab(channels, classes, bns["weights"], normalise="class")
    col_sums = by_class.sum(axis=0).values
    nonzero_cols = col_sums > 0
    np.testing.assert_allclose(col_sums[nonzero_cols], 1.0, rtol=1e-9)

    by_total = channel_class_crosstab(channels, classes, bns["weights"], normalise="total")
    assert by_total.values.sum() == pytest.approx(1.0, rel=1e-9)
