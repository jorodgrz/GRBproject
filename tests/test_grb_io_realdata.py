"""Real-data audit for ``grb_io.py``.

The chairman's "one thing to do first" from the council transcript
2026-05-06: a 30-line audit that pins the COMPAS HDF5 contract
against the actual Broekgaarden et al. (2021) Zenodo files in
``Data/``.  Each test maps 1:1 to a chairman bullet:

1. No NaN in ``w[mask]`` for both Model A files.
2. ``_match_sn_to_dco`` returns NaN for at most 1 percent of the
   merging subset (or logs the count if the threshold trips).
3. ``np.unique(Z[mask])`` is a subset of ``np.unique(METALLICITY_GRID)``
   to within float tolerance.
4. ``f['doubleCompactObjects'].keys()`` contains every column the
   loaders depend on.
5. ``np.all(out['m1'] >= out['m2'])`` after ``load_bns(...)``.

Tests skip cleanly via the ``compas_data_available`` fixture in
``tests/conftest.py`` so CI without the multi-GB Zenodo archives
runs them as no-ops.

Run once per Zenodo download:

    pytest tests/test_grb_io_realdata.py -v
"""

from __future__ import annotations

import sys

import h5py as h5
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. Weight integrity
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.parametrize("fixture_name", ["bns_a_path", "bhns_a_path"])
def test_no_nan_weights_after_mask(request, fixture_name):
    """STROOPWAFEL ``weight`` must be NaN-free on every merging row.

    Council Contrarian #1 / Chairman fix #1.  The downstream rates
    use ``np.average(x, weights=w)`` and ``np.histogram(weights=w)``,
    both of which silently propagate NaN to the science output.
    """
    path = request.getfixturevalue(fixture_name)
    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        w = dco["weight"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()

    w_masked = w[mh == 1]
    n_nan = int(np.isnan(w_masked).sum())
    assert n_nan == 0, (
        f"{path}: {n_nan} of {len(w_masked)} merging rows have NaN "
        f"STROOPWAFEL weight; downstream rates would be poisoned."
    )


# ─────────────────────────────────────────────────────────────────────
# 2. SN-to-DCO match completeness
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_match_sn_to_dco_no_nan_on_merging_subset(bns_a_path, capsys):
    """Less than 1 percent of merging BNS systems may have unmatched seeds.

    Council Contrarian #3 / Chairman fix #2.  The chairman accepts a
    small unmatched fraction (Hernquist offsets drop NaN rows via
    ``np.isfinite`` in ``compute_offsets_population``), but a >1
    percent unmatched fraction means a paper-killer in the SN join
    rather than a benign edge-case.
    """
    from grb_io import load_bns_with_kicks

    out = load_bns_with_kicks(path=bns_a_path)
    v_sys = out["v_sys"]
    n = out["n_merging"]
    n_nan = int(np.isnan(v_sys).sum())
    frac = n_nan / max(n, 1)

    print(
        f"_match_sn_to_dco: {n_nan} of {n} merging BNS rows have NaN "
        f"v_sys ({100 * frac:.3f} percent)",
        file=sys.stderr,
    )
    assert frac < 0.01, (
        f"_match_sn_to_dco unmatched fraction {100 * frac:.2f} percent "
        f"exceeds the 1 percent threshold; the lexsort/searchsorted "
        f"join is suspect."
    )


# ─────────────────────────────────────────────────────────────────────
# 3. METALLICITY_GRID provenance vs actual data
# ─────────────────────────────────────────────────────────────────────
def _grid_contains(values, grid, atol=1e-8):
    """True iff every entry of ``values`` matches a ``grid`` entry within ``atol``."""
    return all(np.any(np.isclose(grid, v, atol=atol)) for v in values)


@pytest.mark.requires_data
@pytest.mark.parametrize("fixture_name", ["bns_a_path", "bhns_a_path"])
def test_metallicity_grid_matches_data(request, fixture_name):
    """Birth metallicities in the HDF5 must be a subset of ``METALLICITY_GRID``.

    Council Contrarian #2 / Chairman fix #5.  The duplicate ``0.03``
    in the literal grid is harmless when callers ``np.unique`` the
    result, but the audit also catches a far more dangerous failure
    mode: a Zenodo file whose metallicity discretisation has drifted
    away from the 53-element grid burned into ``grb_io.py``.
    """
    from grb_io import METALLICITY_GRID

    path = request.getfixturevalue(fixture_name)
    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()

    Z_unique = np.unique(Z[mh == 1])
    grid_unique = np.unique(METALLICITY_GRID)
    missing = [v for v in Z_unique
               if not np.any(np.isclose(grid_unique, v, atol=1e-8))]
    assert _grid_contains(Z_unique, grid_unique), (
        f"{path}: data carries metallicities not present in "
        f"METALLICITY_GRID; first 5 missing values = "
        f"{missing[:5]}.  This means the literal grid in grb_io.py "
        f"has drifted from the actual Broekgaarden+ 2021 Zenodo file."
    )


# ─────────────────────────────────────────────────────────────────────
# 4. HDF5 schema audit
# ─────────────────────────────────────────────────────────────────────
_REQUIRED_DCO_COLS = {
    "M1", "M2", "weight", "Metallicity1",
    "mergesInHubbleTimeFlag", "tc", "tform", "stellarType1",
}


@pytest.mark.requires_data
@pytest.mark.parametrize("fixture_name", ["bns_a_path", "bhns_a_path"])
def test_hdf5_schema_pinned(request, fixture_name):
    """``doubleCompactObjects`` must contain the columns every loader uses.

    Council blind spot identified by every peer reviewer / Chairman
    fix #6.  Catches Zenodo schema drift on the next download
    (column rename, dropped field, group restructure).
    """
    path = request.getfixturevalue(fixture_name)
    with h5.File(path, "r") as f:
        keys = set(f["doubleCompactObjects"].keys())

    missing = _REQUIRED_DCO_COLS - keys
    assert not missing, (
        f"{path}: doubleCompactObjects schema drift detected. "
        f"Missing columns: {sorted(missing)}.  Review the Zenodo "
        f"changelog (5189849 / 5178777) and update the loaders."
    )


# ─────────────────────────────────────────────────────────────────────
# 5. m1 >= m2 invariant after default load
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_m1_ge_m2_after_load_default(bns_a_path):
    """Sanity-check that ``sort_masses=True`` is actually the default.

    Reviewer 3 blind spot: nobody actually verified the default in
    the council round.  Redundant with
    ``tests/test_physics.py::test_load_bns_mass_ordering`` but kept
    here so the audit file stands alone per the chairman's verdict.
    """
    from grb_io import load_bns

    out = load_bns(path=bns_a_path)
    assert (out["m1"] >= out["m2"]).all(), (
        "load_bns default did not enforce m1 >= m2; downstream "
        "classifiers in grb_classify.py rely on this invariant."
    )
    assert out["population"] == "BNS"
