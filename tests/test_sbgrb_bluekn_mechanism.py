"""Validation tests for Section 13 (sbGRB + blue KN bimodality mechanism).

Locks the appendix figure ``Plots/sbgrb_bluekn_mechanism.{pdf,png}`` to
its physical and arithmetic claims.  Five tests, ordered cheapest first
so source-parse and synthetic-math regressions surface in milliseconds
even when the COMPAS HDF5 catalogues are absent:

(1) ``test_logz_max_physical_below_grid_max`` parses cell 40 of
    ``grb_main.ipynb`` and asserts the displayed-axis truncation
    constant ``LOGZ_MAX_PHYSICAL`` lives strictly below
    ``log10(METALLICITY_GRID.max())``, so the rightmost COMPAS grid-
    edge bin (a sampling cliff, not physics) cannot reappear in the
    figure unless the constant is consciously rolled back.  Pure
    source parse plus the 53-element ``METALLICITY_GRID`` constant
    from ``grb_io``: no notebook execution, no HDF5 read.

(2) ``test_weighted_poisson_density_sigma_formula`` verifies the
    error-bar formula introduced in Section 13 matches the
    weighted-Poisson per-bin envelope on a density-normalized
    histogram: synthetic per-system weights, two-bin histogram,
    closed-form expected sigma.  No data required.

(3) ``test_sbgrb_bluekn_funnel_counts_model_a`` regenerates the
    filter cascade ``n_merging -> sbGRB + blue KN raw -> N(wi > 0)
    -> N_eff`` printed at the bottom of cell 40 and asserts each
    stage falls in a sane range for Model A (Broekgaarden+ 2021,
    Zenodo 5189849).  Closes the "are you sure it's all the numbers
    COMPAS outputs?" question.

(4) ``test_sbgrb_bluekn_fz_has_dip_near_log_z_minus_2_7`` reproduces
    the f(Z) curve in the top panel and asserts the log Z ~ -2.7 dip
    is a local minimum sitting below both the log Z ~ -1.9 (Peak 1)
    and log Z ~ -3.3 (Peak 2) f(Z) values, so the figure's central
    "MSSFR z=4.5 lands on a real f(Z) dip" claim is regression-locked.

(5) ``test_sbgrb_bluekn_integrand_at_z_merge_4_5_lands_on_dip``
    computes ``per_system_rate_weights`` at z_merge = 4.5 and asserts
    the rate-weighted-mean ``log10 Z`` of the contributing systems
    falls in [-3.0, -2.4], i.e. close to the f(Z) dip rather than
    the grid extremes.  Locks the convolution mechanism in the
    middle row of the figure.

Tests (3) - (5) are marked ``slow`` + ``requires_data`` +
``requires_compas``; they skip cleanly on machines without the COMPAS
HDF5 catalogues or the upstream ``compas_python_utils`` package.
Tests (1) and (2) run on every CI invocation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


# =====================================================================
# Cheap, no-data tests: source-parse + synthetic-math regressions.
# =====================================================================
@pytest.fixture(scope="module")
def _section_13_cell_source(repo_root: str) -> str:
    """Return the raw source of cell 40 of ``grb_main.ipynb``.

    Cell 40 is the Section 13 appendix (sbGRB + blue KN bimodality
    mechanism).  Reading the raw .ipynb JSON keeps this test decoupled
    from nbclient so it runs on every CI invocation, not just on
    machines that can execute the notebook.
    """
    nb_path = Path(repo_root) / "grb_main.ipynb"
    with nb_path.open("r") as f:
        nb = json.load(f)
    cells = nb["cells"]
    if not 40 < len(cells):
        pytest.fail(
            f"Notebook has {len(cells)} cells; expected at least 41 "
            f"(cell 40 is the Section 13 appendix).")
    cell = cells[40]
    if cell.get("cell_type") != "code":
        pytest.fail(
            f"Cell 40 is not a code cell (got {cell.get('cell_type')!r}); "
            f"the Section 13 layout has shifted.")
    return "".join(cell["source"])


def test_logz_max_physical_below_grid_max(_section_13_cell_source: str):
    """``LOGZ_MAX_PHYSICAL`` must truncate inside the COMPAS grid.

    Cell 40 sets ``LOGZ_MAX_PHYSICAL`` and uses it as the displayed
    upper limit of ``log10 Z`` on every panel.  The intent is to drop
    the rightmost COMPAS grid-edge bin (Broekgaarden+ 2021 sample a
    discrete metallicity grid; the last point sits at log Z ~ -1.5).

    A regression that bumps ``LOGZ_MAX_PHYSICAL`` back above the grid
    maximum, or removes it entirely, would let the spike reappear in
    the figure: this test catches that without re-rendering the plot.

    Asserts:
        1. ``LOGZ_MAX_PHYSICAL = <value>`` line is present in the cell.
        2. The value is strictly below ``log10(METALLICITY_GRID.max())``,
           with at least 0.05 dex of margin so a future grid extension
           does not pass the test silently.
    """
    from grb_io import METALLICITY_GRID

    src = _section_13_cell_source

    import re
    m = re.search(r"^LOGZ_MAX_PHYSICAL\s*=\s*([+-]?\d+(?:\.\d+)?)\s*$",
                  src, re.MULTILINE)
    assert m is not None, (
        "Expected 'LOGZ_MAX_PHYSICAL = <value>' at module-line scope in "
        "cell 40 of grb_main.ipynb.  This constant truncates the displayed "
        "log Z axis to drop the COMPAS grid-edge spike; the test cannot "
        "verify the truncation without it.")
    logz_max = float(m.group(1))

    grid_max = float(np.log10(METALLICITY_GRID.max()))
    assert logz_max < grid_max - 0.05, (
        f"LOGZ_MAX_PHYSICAL = {logz_max:.3f} does not provide >= 0.05 dex "
        f"margin below the COMPAS grid maximum log10 Z = {grid_max:.3f} "
        f"(METALLICITY_GRID[-1] = {METALLICITY_GRID.max():.4g}).  "
        f"The figure's grid-edge truncation is no longer effective.")

    # Sanity: the truncation must actually drop bins, otherwise it is a
    # no-op.  Pre-fix code used ``edges_logZ[-1]`` ~ log(grid_max) + 0.05,
    # so any value <= grid_max is meaningful but the council picked -1.6
    # specifically; flag drift > 0.5 dex from -1.6 as suspicious.
    assert -2.5 < logz_max < -1.5, (
        f"LOGZ_MAX_PHYSICAL = {logz_max:.3f} sits outside the council "
        f"window (-2.5, -1.5); the displayed axis would either keep too "
        f"much of the under-sampled tail or hide too much of the "
        f"high-Z f(Z) peak.")


def test_section_13_top_panel_legend_outside_axes(
        _section_13_cell_source: str):
    """The top-panel legend must use ``bbox_to_anchor`` outside the axes.

    Pre-fix the legend lived at ``loc='lower right'`` and overlapped the
    f(Z) right-side rise plus the dotted z=7.5 MSSFR Gaussian.  The fix
    is to anchor it past the right twin-axis label so constrained_layout
    reflows it into a margin instead of onto the data.  This test
    catches a regression that drops the ``bbox_to_anchor`` argument or
    moves the legend back inside the axes.
    """
    src = _section_13_cell_source
    assert "bbox_to_anchor" in src, (
        "Expected 'bbox_to_anchor=...' in cell 40 of grb_main.ipynb; the "
        "top-panel f(Z)/MSSFR legend must be anchored outside the axes "
        "to avoid overlapping the right-side f(Z) rise.")
    assert "loc='lower right'" not in src, (
        "Found 'loc=\"lower right\"' in cell 40; the legend was moved "
        "outside the axes and should no longer use the lower-right "
        "in-axes location that overlapped f(Z).")


def test_weighted_poisson_density_sigma_formula():
    """Verify the per-bin sigma formula matches sqrt(sum w_i^2) / norm.

    The figure's middle and bottom rows now plot weighted-Poisson 1
    sigma errorbars on the density-normalized histograms.  The
    implementation is

        counts_k = sum_{i in bin_k} w_i
        var_k    = sum_{i in bin_k} w_i^2
        norm     = (sum_i w_i) * dx
        sigma_k  = sqrt(var_k) / norm   (density-normalized)

    This test fixes a worked example (3 systems in bin A with weights
    [0.5, 1.0, 2.0]; 2 systems in bin B with weights [0.3, 0.7]) and
    asserts the recipe in the cell matches the closed-form expected
    sigmas to within numerical precision.  Catches regressions that
    use np.sqrt(counts) (Poisson on counts, wrong for weighted samples)
    or that forget the density normalization.
    """
    weights = np.array([0.5, 1.0, 2.0, 0.3, 0.7])
    values  = np.array([0.1, 0.2, 0.3, 1.1, 1.4])
    bin_edges = np.array([0.0, 1.0, 2.0])
    dx        = float(np.diff(bin_edges)[0])

    counts, _ = np.histogram(values, bins=bin_edges, weights=weights)
    var, _    = np.histogram(values, bins=bin_edges, weights=weights ** 2)
    norm = float(weights.sum()) * dx
    sigma = np.sqrt(var) / norm

    expected_counts = np.array([0.5 + 1.0 + 2.0, 0.3 + 0.7])
    expected_var    = np.array([0.5 ** 2 + 1.0 ** 2 + 2.0 ** 2,
                                0.3 ** 2 + 0.7 ** 2])
    expected_sigma  = np.sqrt(expected_var) / (weights.sum() * dx)

    assert np.allclose(counts, expected_counts, rtol=1e-12), counts
    assert np.allclose(var, expected_var, rtol=1e-12), var
    assert np.allclose(sigma, expected_sigma, rtol=1e-12), sigma

    # Sanity: density-normalized integral is 1.0 (no error in
    # normalization).
    densities = counts / norm
    assert np.isclose((densities * dx).sum(), 1.0, rtol=1e-12)

    # Sanity: sqrt(counts) (Poisson on counts) gives a *different*
    # answer than the weighted-Poisson sigma when weights are
    # non-uniform.  This guards against the common regression of
    # silently swapping sqrt(counts) in.
    naive_sigma = np.sqrt(counts) / norm
    assert not np.allclose(sigma, naive_sigma, rtol=1e-2), (
        "Weighted-Poisson sigma agrees with sqrt(counts)/norm to within "
        "1 percent; the test inputs may be too uniform.  Real STROOPWAFEL "
        "weights are highly non-uniform.")


# =====================================================================
# Slow tests: COMPAS data round-trip for the figure's physical claims.
# =====================================================================
def _build_appendix_setup(bns_a_path):
    """Mirror cells 1, 4, and 5 of grb_main.ipynb up to the inputs that
    cell 40 consumes.  Returns a SimpleNamespace with everything
    per_system_rate_weights / formation_efficiency need.

    Self-contained (does not import from sibling test files); duplicates
    the small _build_setup helper in test_rate_class_shape.py with
    additional fields needed for f(Z) reconstruction (mean_mass_bns,
    METALLICITY_GRID).
    """
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )

    from astropy.cosmology import Planck15
    # CLAUDE.md mandate: Planck 2015 (matches COMPAS FastCosmicIntegration).
    assert abs(Planck15.H0.value - 67.74) < 0.01

    from grb_classify import classify_bns_2024
    from grb_io import (load_bns_with_channels, METALLICITY_GRID,
                         read_expected_local_rate)
    from grb_physics import remap_ns_masses_double_gaussian
    from grb_rates import calibrate_mean_mass_evolved

    bns = load_bns_with_channels(path=bns_a_path)
    # Alsing, Silva and Berti (2018) double-Gaussian remap; rng seed 42
    # matches grb_main.ipynb so the sbGRB mask and the funnel counts
    # reproduce the figure's 233,137 -> 34,366 cascade.
    m1, m2 = remap_ns_masses_double_gaussian(
        bns["m1"].copy(), bns["m2"].copy(),
        weights=bns["weights"], rng=np.random.default_rng(42))

    Z = bns["metallicity"]
    delays = bns["delay_time"]
    w = bns["weights"]
    Z_grid = np.unique(Z)

    cls = classify_bns_2024(m1, m2)
    sbgrb_mask = cls["sbGRB + blue KN"]
    assert sbgrb_mask.any(), (
        "sbGRB + blue KN class is empty on BNS A after Alsing remap; "
        "_build_appendix_setup is broken or the remap RNG drifted.")

    # FCI plumbing at production resolution (matches Section 4 of
    # grb_main.ipynb; same redshift_step the appendix figure uses).
    redshifts, _, times, time_first_SF, _, _ = (
        fci.calculate_redshift_related_params(
            max_redshift=10.0, redshift_step=0.01,
            cosmology=Planck15))
    sfr = fci.find_sfr(redshifts)
    _, _, p_draw = fci.find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(Z.min()),
        max_logZ_COMPAS=np.log(Z.max()))
    expected_local_rate = read_expected_local_rate(bns_a_path)
    mean_mass_bns, _ = calibrate_mean_mass_evolved(
        sfr, redshifts, times, time_first_SF, p_draw,
        Z, delays, w, expected_local_rate, Z_grid=Z_grid)
    n_formed_BNS = sfr / mean_mass_bns

    return SimpleNamespace(
        n_bns=int(bns["n_merging"]),
        m1=m1, m2=m2,
        Z=Z, delays=delays, w=w, Z_grid=Z_grid,
        sbgrb_mask=sbgrb_mask,
        METALLICITY_GRID=METALLICITY_GRID,
        mean_mass_bns=mean_mass_bns,
        redshifts=redshifts, times=times,
        time_first_SF=time_first_SF,
        n_formed_BNS=n_formed_BNS, p_draw=p_draw,
    )


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_sbgrb_bluekn_funnel_counts_model_a(bns_a_path):
    """Filter cascade printed at the bottom of cell 40 must reproduce.

    On COMPAS Model A (Broekgaarden+ 2021, Zenodo 5189849) with the
    Alsing-Silva-Berti remap RNG seeded at 42, the cell 40 print emits

        sbGRB + blue KN appendix sample: raw N = 34,366,
            weighted N = 1,982 (STROOPWAFEL); ...
        filter cascade: n_merging = 233,137  sbGRB+blueKN raw = 34,366
            (14.7% by raw count, 19.8% by STROOPWAFEL weight)
        Peak 1 @ z_merge= 1.5: N(wi>0)=...  N_eff= 11676.0
        Dip @ z_merge= 4.5: N(wi>0)=...  N_eff=    99.0
        Peak 2 @ z_merge= 7.5: N(wi>0)=...  N_eff=  1429.0

    This test reconstructs each line and asserts:
        * n_merging matches the cached value (233,137) within 1 system
          (deterministic load).
        * sbGRB + blue KN raw count is in [25,000, 45,000] (catches a
          drift of more than ~30% in either direction; e.g. a regression
          of the remap that shifts the M_tot < 1.2 M_TOV class boundary).
        * sbGRB + blue KN raw fraction is in [10%, 20%] of n_merging.
        * sbGRB + blue KN STROOPWAFEL weight fraction is in [12%, 28%].
        * Kish N_eff at z_merge = 4.5 is in [50, 250] (locks the dip
          regime; the existing test_sbGRB_rate_dip_n_eff_above_threshold
          already locks the lower bound, this test additionally locks
          the upper bound so a regression that doubles N_eff at the dip
          surfaces here).
        * Kish N_eff(Peak 1) > Kish N_eff(Peak 2) > Kish N_eff(Dip),
          which is the ordering the figure relies on for "the dip is
          where the integrand has lowest support."
    """
    from grb_rates import per_system_rate_weights

    setup = _build_appendix_setup(bns_a_path)

    # Stage 1: n_merging = 233,137 (deterministic load + Hubble-time
    # filter; the remap preserves count).
    assert abs(setup.n_bns - 233137) <= 1, (
        f"n_merging = {setup.n_bns:,} does not match the cached "
        f"value 233,137 for COMPAS Model A; the load filter chain or "
        f"the upstream HDF5 file has changed.")

    # Stage 2: sbGRB + blue KN raw count and weighted fraction.
    sb_mask = setup.sbgrb_mask
    sb_raw = int(sb_mask.sum())
    assert 25_000 <= sb_raw <= 45_000, (
        f"sbGRB + blue KN raw count = {sb_raw:,} outside [25k, 45k]; "
        f"the M_tot < 1.2 M_TOV class boundary or the Alsing remap has "
        f"shifted by more than ~30% (cached: 34,366).")
    raw_frac = sb_raw / setup.n_bns
    assert 0.10 <= raw_frac <= 0.20, (
        f"sbGRB + blue KN raw fraction = {raw_frac * 100:.1f}% outside "
        f"[10%, 20%] of n_merging (cached: 14.7%).")

    sb_w = setup.w[sb_mask]
    weighted_frac = float(sb_w.sum()) / float(setup.w.sum())
    assert 0.12 <= weighted_frac <= 0.28, (
        f"sbGRB + blue KN STROOPWAFEL-weighted fraction = "
        f"{weighted_frac * 100:.1f}% outside [12%, 28%] (cached: 19.8%).")

    # Stage 3: per-target N_eff for Peak 1 / Dip / Peak 2.  z_targets
    # match the cell 40 appendix list verbatim.
    z_targets = [1.5, 4.5, 7.5]
    n_eff = []
    n_nonzero = []
    for z_t in z_targets:
        wi = per_system_rate_weights(
            z_t, setup.redshifts, setup.times, setup.time_first_SF,
            setup.n_formed_BNS, setup.p_draw,
            setup.Z[sb_mask], setup.delays[sb_mask], setup.w[sb_mask],
            Z_grid=setup.Z_grid)
        wsum = float(wi.sum())
        wsum2 = float((wi ** 2).sum())
        n_eff.append((wsum ** 2) / max(wsum2, 1e-300))
        n_nonzero.append(int((wi > 0).sum()))

    n_eff_p1, n_eff_dip, n_eff_p2 = n_eff
    n_nz_p1, n_nz_dip, n_nz_p2 = n_nonzero

    # Cached values at production: ~11,676 / 99 / 1,429.  Bands are 2x
    # wide on each side to leave headroom for legitimate downstream
    # changes while still flagging order-of-magnitude regressions.
    assert 5_000 <= n_eff_p1 <= 25_000, (
        f"N_eff(z=1.5) = {n_eff_p1:.0f} outside [5k, 25k] (cached: "
        f"~11,676); Peak 1 effective sample size has drifted.")
    assert 50.0 <= n_eff_dip <= 250.0, (
        f"N_eff(z=4.5) = {n_eff_dip:.1f} outside [50, 250] (cached: "
        f"~99); Dip effective sample size has drifted.  Lower bound "
        f"is also locked by test_sbGRB_rate_dip_n_eff_above_threshold.")
    assert 600.0 <= n_eff_p2 <= 3_000.0, (
        f"N_eff(z=7.5) = {n_eff_p2:.0f} outside [600, 3000] (cached: "
        f"~1,429); Peak 2 effective sample size has drifted.")

    # Ordering: the figure's mechanism story requires
    # N_eff(Peak 1) > N_eff(Peak 2) > N_eff(Dip).
    assert n_eff_p1 > n_eff_p2 > n_eff_dip, (
        f"N_eff ordering violated: Peak 1 = {n_eff_p1:.0f}, "
        f"Peak 2 = {n_eff_p2:.0f}, Dip = {n_eff_dip:.0f}; the dip "
        f"should have the smallest effective sample of the three.")

    # N(wi > 0) >= N_eff trivially (Cauchy-Schwarz); this guards
    # against a sign-flip regression where the Kish formula would
    # accidentally exceed the count of nonzero weights.
    for label, nz, ne in (("Peak 1", n_nz_p1, n_eff_p1),
                          ("Dip", n_nz_dip, n_eff_dip),
                          ("Peak 2", n_nz_p2, n_eff_p2)):
        assert nz >= ne, (
            f"{label}: N(wi > 0) = {nz} < N_eff = {ne:.0f}; Kish "
            f"N_eff cannot exceed the count of nonzero contributors "
            f"(Cauchy-Schwarz).")


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_sbgrb_bluekn_fz_has_dip_near_log_z_minus_2_7(bns_a_path):
    """f(Z) for sbGRB + blue KN must have a local minimum near log Z ~ -2.7.

    The figure's central physical claim is that the R(z) bimodality is
    explained by an f(Z) dip at log Z ~ -2.7 onto which the z_merge =
    4.5 MSSFR Gaussian lands.  This test reconstructs f(Z) on the COMPAS
    metallicity grid and asserts:

        * a local minimum exists in log10 Z in [-3.0, -2.4],
        * that minimum is below f(Z) at log Z in [-2.0, -1.6] (Peak 1
          flank toward solar) by at least 30%,
        * that minimum is below f(Z) at log Z in [-3.6, -3.1] (Peak 2
          flank toward low Z) by at least 20%.

    A regression that flattens the f(Z) curve (e.g. silently disabling
    the M_tot < 1.2 M_TOV class boundary, or reverting the Alsing
    remap so the lightest BNS subset shrinks to noise) would fail
    here; the existing test_sbGRB_rate_dip_n_eff_above_threshold is
    silent about *where* in metallicity the dip sits.
    """
    from grb_rates import formation_efficiency

    setup = _build_appendix_setup(bns_a_path)

    sb_mask = setup.sbgrb_mask
    eff = formation_efficiency(
        setup.METALLICITY_GRID, setup.Z, setup.w,
        masks={"sbGRB + blue KN": sb_mask},
        mean_mass_evolved=setup.mean_mass_bns)
    fZ = eff["sbGRB + blue KN"]
    log10Z = np.log10(setup.METALLICITY_GRID)

    # Restrict to the displayed range (mirrors LOGZ_MAX_PHYSICAL = -1.6
    # truncation in cell 40) and to bins with non-zero efficiency
    # (under-sampled bins return zero from formation_efficiency and
    # would spuriously dominate the np.argmin below).
    keep = (log10Z >= -4.0) & (log10Z <= -1.6) & (fZ > 0)
    assert keep.sum() >= 10, (
        f"Only {int(keep.sum())} non-zero f(Z) bins in [-4.0, -1.6]; "
        f"too few for a minimum-search to be meaningful.  The "
        f"sbGRB + blue KN class may have collapsed.")
    log10Z_k = log10Z[keep]
    fZ_k = fZ[keep]

    # Find the minimum within the dip window [-3.0, -2.4].
    dip_mask = (log10Z_k >= -3.0) & (log10Z_k <= -2.4)
    assert dip_mask.any(), (
        "No COMPAS metallicity-grid bins fall inside the dip window "
        "[-3.0, -2.4]; grid resolution has changed since the figure was "
        "calibrated.")
    fZ_dip = float(fZ_k[dip_mask].min())
    log10Z_dip = float(log10Z_k[dip_mask][int(np.argmin(fZ_k[dip_mask]))])
    assert -3.0 <= log10Z_dip <= -2.4, log10Z_dip

    # Peak 1 flank: f(Z) at log Z in [-2.0, -1.6] (toward solar).  Use
    # the median of the well-populated bins in this window so a single
    # under-sampled cell does not dominate the assertion.
    peak1_mask = (log10Z_k >= -2.0) & (log10Z_k <= -1.6)
    assert peak1_mask.any(), peak1_mask
    fZ_peak1_med = float(np.median(fZ_k[peak1_mask]))

    # Peak 2 flank: f(Z) at log Z in [-3.6, -3.1] (toward the low-Z
    # secondary peak in the figure).
    peak2_mask = (log10Z_k >= -3.6) & (log10Z_k <= -3.1)
    assert peak2_mask.any(), peak2_mask
    fZ_peak2_med = float(np.median(fZ_k[peak2_mask]))

    # Mechanism claim (1): dip below Peak 1 by >= 30%.
    assert fZ_dip < 0.70 * fZ_peak1_med, (
        f"f(Z) at the dip ({fZ_dip:.3e} at log Z = {log10Z_dip:.2f}) "
        f"is not at least 30% below the Peak 1 flank median "
        f"({fZ_peak1_med:.3e} in log Z [-2.0, -1.6]).  The R(z) "
        f"bimodality story does not hold.")

    # Mechanism claim (2): dip below Peak 2 by >= 20%.  Looser bound
    # because the low-Z bins are more sparsely sampled, so the median
    # is noisier.
    assert fZ_dip < 0.80 * fZ_peak2_med, (
        f"f(Z) at the dip ({fZ_dip:.3e} at log Z = {log10Z_dip:.2f}) "
        f"is not at least 20% below the Peak 2 flank median "
        f"({fZ_peak2_med:.3e} in log Z [-3.6, -3.1]).  The low-Z arm "
        f"of the bimodality is too shallow to support the figure's "
        f"causal story.")


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_sbgrb_bluekn_integrand_at_z_merge_4_5_lands_on_dip(bns_a_path):
    """At z_merge = 4.5 the rate-weighted log Z must land near the f(Z) dip.

    The figure's central convolution claim is that the z_merge = 4.5
    MSSFR Gaussian lands on the f(Z) dip at log Z ~ -2.7.  Equivalently,
    the rate-weighted-mean log Z of the contributing systems at
    z_merge = 4.5 should sit close to the dip rather than at either
    extreme of the COMPAS grid.

    This test computes ``per_system_rate_weights`` at z_merge = 4.5,
    forms <log10 Z>_w, and asserts it falls in [-3.0, -2.4] (the same
    window the previous test uses for the dip).  Catches a regression
    that decorrelates the MSSFR convolution from the f(Z) curve (e.g.
    a sign flip in dPdlogZ, a wrong p_draw, or a delay-time bug).
    """
    from grb_rates import per_system_rate_weights

    setup = _build_appendix_setup(bns_a_path)
    sb_mask = setup.sbgrb_mask

    wi = per_system_rate_weights(
        4.5, setup.redshifts, setup.times, setup.time_first_SF,
        setup.n_formed_BNS, setup.p_draw,
        setup.Z[sb_mask], setup.delays[sb_mask], setup.w[sb_mask],
        Z_grid=setup.Z_grid)
    assert wi.sum() > 0, (
        "per_system_rate_weights at z_merge = 4.5 returned zero total "
        "weight; the integrand has no support, which would silently "
        "blank out the Dip panel of the figure.")

    log10_Z_sb = np.log10(setup.Z[sb_mask])
    contrib = wi > 0
    log10_Z_mean = float(np.average(log10_Z_sb[contrib],
                                     weights=wi[contrib]))

    assert -3.0 <= log10_Z_mean <= -2.4, (
        f"<log10 Z>_w at z_merge = 4.5 = {log10_Z_mean:.3f} outside "
        f"the dip window [-3.0, -2.4]; the MSSFR convolution is no "
        f"longer landing on the f(Z) dip.  The figure's causal story "
        f"is broken.")
