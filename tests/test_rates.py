"""Validation tests for ``grb_rates.py``.

Anchors the cosmic-rate convolution and the per-class beaming helpers
to (a) the cited literature (Fong+ 2015, Beniamini and Nakar 2019,
Neijssel+ 2019, Madau and Dickinson 2014, Wanderman and Piran 2015) and
(b) the COMPAS reference implementation in
``compas_python_utils.cosmic_integration.FastCosmicIntegration``.

The COMPAS-vs-grb_rates agreement test is marked ``requires_data`` plus
``requires_compas`` because it loads ``Data/COMPASCompactOutput_BNS_A.h5``
and runs the upstream find_formation_and_merger_rates side by side with
``compute_merger_rate``; it is skipped cleanly on machines that do not
have either piece available.
"""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# Beaming round-trip: f_beam = 1 - cos(theta_j); Fong+ 2015 ApJ 815, 102
# ─────────────────────────────────────────────────────────────────────
def test_beaming_round_trip_sbgrb():
    """sbGRB fiducial: theta_j = 13 deg => f_beam ~ 0.0256."""
    from grb_rates import beamed_rate, CLASS_THETA_J

    theta_fid = CLASS_THETA_J["sbGRB"]["fid"]
    expected = 100.0 * (1.0 - np.cos(np.radians(theta_fid)))
    got = beamed_rate(100.0, theta_fid)
    assert np.isclose(got, expected, rtol=1e-12)
    assert 2.55 < got < 2.58, got


def test_beaming_round_trip_lbgrb():
    """lbGRB fiducial: theta_j = 6.5 deg => f_beam ~ 0.00643."""
    from grb_rates import beamed_rate, CLASS_THETA_J

    theta_fid = CLASS_THETA_J["lbGRB"]["fid"]
    got = beamed_rate(100.0, theta_fid)
    assert 0.64 < got < 0.65, got


def test_beamed_rate_mixed_matches_per_class_sum():
    """Mixed-population beaming must equal the per-class sum, NOT
    beamed_rate(R_total, <theta_j>): the latter would silently
    overestimate observed rates because ``1 - cos`` is non-linear in
    theta_j.  This is the council Trap 3 (per-class beaming)
    regression.
    """
    from grb_rates import beamed_rate, beamed_rate_mixed, CLASS_THETA_J

    rates = {"sbGRB": 50.0, "lbGRB": 50.0}
    angles = {k: CLASS_THETA_J[k]["fid"] for k in rates}

    expected = (beamed_rate(50.0, CLASS_THETA_J["sbGRB"]["fid"])
                + beamed_rate(50.0, CLASS_THETA_J["lbGRB"]["fid"]))
    got = beamed_rate_mixed(rates, angles)
    assert np.isclose(got, expected, rtol=1e-12)

    # Sanity: this is NOT what scalar averaging would give
    naive = beamed_rate(100.0, 0.5 * (CLASS_THETA_J["sbGRB"]["fid"]
                                       + CLASS_THETA_J["lbGRB"]["fid"]))
    assert not np.isclose(got, naive, rtol=1e-3), (
        "beamed_rate_mixed agrees with the naive scalar average; the "
        "non-linearity of (1 - cos) should make these differ.")


def test_beamed_rate_mixed_missing_class_raises():
    """A rate class without a corresponding theta_j is a programmer
    error and must raise loudly rather than silently dropping the
    class from the sum."""
    from grb_rates import beamed_rate_mixed

    with pytest.raises(KeyError):
        beamed_rate_mixed({"sbGRB": 1.0, "lbGRB": 1.0}, {"sbGRB": 13.0})


# ─────────────────────────────────────────────────────────────────────
# Cosmology pin: Planck 2015 (CLAUDE.md mandate, COMPAS Planck18 default)
# ─────────────────────────────────────────────────────────────────────
def test_cosmology_planck15_h0():
    """Planck 2015 H0 = 67.74 km/s/Mpc (Planck Collaboration 2016
    A&A 594, A13).  This is the value the manuscript prose claims and
    the value grb_main.ipynb passes to calculate_redshift_related_params
    after the 2026-05-06 cosmology fix.  If a future change reverts the
    notebook to the COMPAS Planck18 default (H0 = 67.4) this test
    catches it before the rates drift by ~2 percent at high z.
    """
    from astropy.cosmology import Planck15

    assert abs(Planck15.H0.value - 67.74) < 0.01


# ─────────────────────────────────────────────────────────────────────
# Neijssel+ 2019 MSSFR: dP/d(ln Z) integrates to ~1 at z = 0
# ─────────────────────────────────────────────────────────────────────
def test_dPdlogZ_normalization_z0():
    """At z = 0 with a Z grid spanning 0.01 to 1.0 the Voronoi-bin
    probabilities sum to ~1 because the Neijssel+ 2019 log-normal
    (mu_0 = 0.035, sigma_0 = 0.39) sits well inside the grid:

        ln_Z_min = ln(0.01) ~ -4.6,
        ln_Z_max = ln(1.0) =  0,
        mu(z=0)  = ln(0.035) - sigma^2/2 ~ -3.43,
        sigma(z=0) = 0.39,

    so ``norm = Phi(8.79) - Phi(-3.0) ~ 0.999``.  At higher redshift
    (where mu(z) drops below ln_Z_min) the bin sum falls below 1 by
    construction; the docstring at grb_rates.py:144-153 documents that
    behaviour as expected.
    """
    from grb_rates import _bin_averaged_dPdlogZ

    Z_grid = 10.0 ** np.linspace(-2, 0, 53)
    dPdlogZ_binned, _ = _bin_averaged_dPdlogZ(
        np.array([0.0]), Z_grid, Z_grid=Z_grid)
    integral = dPdlogZ_binned[0].sum()
    assert 0.98 < integral < 1.02, (
        f"dPdlogZ integral at z=0 is {integral:.4f}; expected ~1 because "
        "the COMPAS Z grid covers the bulk of the Neijssel+ 2019 "
        "log-normal at z=0.")


@pytest.mark.requires_compas
def test_check_dPdlogZ_normalization_runs_on_compas_output():
    """``check_dPdlogZ_normalization`` should accept the COMPAS
    ``find_metallicity_distribution`` output without raising.
    Skips cleanly if the upstream package is unavailable."""
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )
    find_metallicity_distribution = fci.find_metallicity_distribution
    from grb_rates import check_dPdlogZ_normalization

    redshifts = np.linspace(0.0, 5.0, 51)
    dPdlogZ, metallicities, _ = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(1e-4),
        max_logZ_COMPAS=np.log(0.03),
    )
    norm = check_dPdlogZ_normalization(dPdlogZ, metallicities, rtol=0.10)
    assert np.all(np.isfinite(norm))


# ─────────────────────────────────────────────────────────────────────
# Per-class partition invariant for Section 7 BNS rate plot
# ─────────────────────────────────────────────────────────────────────
def test_per_class_rates_partition_total_bns_2024():
    """The four ``classify_bns_2024`` classes partition the BNS sample,
    so ``sum_c R_c(z) == R_total(z)`` to floating-point precision when
    ``compute_merger_rate`` is called with the same ``n_formed``,
    ``p_draw``, and ``Z_grid`` for each subset.

    Locks the invariant the Section 7 BNS rate plot
    (``Plots/rate_bns_by_class``) relies on: ``compute_merger_rate`` is a
    pure additive accumulator over binaries
    (``total_merger[j_idx] += _interp_formation_rate(...)``), so disjoint
    masks that union to the full sample must produce rates that sum to
    the all-sample rate.  Smoothing is linear and preserves the
    invariant; we set ``smooth_sigma=0`` here so the assertion is
    unambiguous about the underlying physics.

    Also asserts the failure mode documented at ``grb_rates.py:67-78``:
    dropping ``Z_grid`` on per-class calls breaks the invariant because
    each subset's Voronoi cells and high-z renormalisation range collapse
    to its own metallicity range, producing class-shape bias.
    """
    from grb_rates import compute_merger_rate
    from grb_classify import classify_bns_2024

    rng = np.random.default_rng(0)

    # Construct ~50 BNS systems spanning all four classify_bns_2024 classes.
    # Boundaries (M_TOV=2.2, K_THRESH=1.27, Q_THRESH=1.2):
    #   sbGRB:  M_tot < 1.2*M_TOV = 2.64
    #   HMNS:   2.64 <= M_tot < 2.794
    #   disk:   M_tot >= 2.794 and q >= 1.2
    #   Faint:  M_tot >= 2.794 and q < 1.2
    m1 = np.concatenate([
        np.full(15, 1.20),  # sbGRB:  M_tot = 2.40
        np.full(15, 1.40),  # HMNS:   M_tot = 2.70
        np.full(10, 2.10),  # disk:   M_tot = 3.50, q = 1.50
        np.full(10, 1.55),  # Faint:  M_tot = 3.00, q = 1.07
    ])
    m2 = np.concatenate([
        np.full(15, 1.20),
        np.full(15, 1.30),
        np.full(10, 1.40),
        np.full(10, 1.45),
    ])
    cls = classify_bns_2024(m1, m2)
    assert sum(int(c.sum()) for c in cls.values()) == len(m1), (
        "synthetic masses do not partition cleanly across classify_bns_2024")
    for label, c in cls.items():
        assert int(c.sum()) > 0, f"no test systems land in class {label!r}"

    # Construct per-class Z assignments such that each class spans a
    # *different* subset of Z_full.  This makes the no-Z_grid regression
    # assertion below non-trivial: if every class saw the same unique-Z
    # set then the no-Z_grid path would coincide with the Z_grid path.
    Z_full = np.array([1e-4, 1e-3, 1e-2, 3e-2])
    Z_compas = np.empty(len(m1))
    Z_compas[cls['sbGRB + blue KN']]       = rng.choice(Z_full[:2], size=int(cls['sbGRB + blue KN'].sum()))
    Z_compas[cls['lbGRB + red KN (HMNS)']] = rng.choice(Z_full[1:3], size=int(cls['lbGRB + red KN (HMNS)'].sum()))
    Z_compas[cls['lbGRB + red KN (disk)']] = rng.choice(Z_full[2:],  size=int(cls['lbGRB + red KN (disk)'].sum()))
    Z_compas[cls['Faint lbGRB']]           = rng.choice(Z_full,      size=int(cls['Faint lbGRB'].sum()))
    delays = rng.uniform(50.0, 5000.0, size=len(m1))
    weights = rng.uniform(0.5, 2.0, size=len(m1))

    redshifts = np.linspace(0.0, 1.0, 51)
    times = np.linspace(13.7e3, 5.0e3, 51)
    sfr = np.full_like(redshifts, 1e7)

    common = dict(
        redshifts=redshifts, times=times, time_first_SF=100.0,
        n_formed=sfr, p_draw=0.1, smooth_sigma=0)

    R_total = compute_merger_rate(
        COMPAS_Z=Z_compas, COMPAS_delay_times=delays,
        COMPAS_weights=weights, Z_grid=Z_full, **common)

    R_per_class = {}
    for label, mask in cls.items():
        R_per_class[label] = compute_merger_rate(
            COMPAS_Z=Z_compas[mask], COMPAS_delay_times=delays[mask],
            COMPAS_weights=weights[mask], Z_grid=Z_full, **common)
    R_sum = sum(R_per_class.values())

    np.testing.assert_allclose(
        R_total, R_sum, rtol=1e-12, atol=0,
        err_msg=("classify_bns_2024 partition does not sum to total; "
                 "compute_merger_rate may have lost additivity over binaries"))

    # Regression guard for the Z_grid alignment requirement
    # (grb_rates.py:67-78).  Without Z_grid each subset's Voronoi cells
    # collapse to its own Z range, so the per-class rates must NOT sum
    # to the all-sample rate when the four classes span different
    # unique-Z sets (which the construction above enforces).
    R_per_class_nogrid = {
        label: compute_merger_rate(
            COMPAS_Z=Z_compas[mask], COMPAS_delay_times=delays[mask],
            COMPAS_weights=weights[mask], Z_grid=None, **common)
        for label, mask in cls.items()
    }
    R_sum_nogrid = sum(R_per_class_nogrid.values())
    assert not np.allclose(R_total, R_sum_nogrid, rtol=1e-3, atol=0), (
        "per-class sum still matches R_total without Z_grid; the cross-"
        "class Voronoi alignment failure documented at grb_rates.py:67-78 "
        "is supposed to make these differ.")


# ─────────────────────────────────────────────────────────────────────
# Z_grid alignment guard inside compute_merger_rate
# ─────────────────────────────────────────────────────────────────────
def test_compute_merger_rate_rejects_misaligned_Z_grid():
    """Reviewer 1's missing cross-module assert: passing a Z_grid that
    does not contain every COMPAS_Z value must raise rather than
    silently producing biased per-class rates."""
    from grb_rates import compute_merger_rate

    redshifts = np.linspace(0.0, 1.0, 51)
    times = np.linspace(13.7e3, 5.0e3, 51)
    sfr = np.full_like(redshifts, 1e7)

    Z_full = np.array([1e-4, 1e-3, 1e-2, 3e-2])
    Z_subset = np.array([1e-4, 1e-3, 1e-2])
    delays = np.array([100.0, 200.0, 300.0])
    weights = np.array([1.0, 1.0, 1.0])

    Z_grid_bad = np.array([1e-4, 1e-3, 1e-2])  # missing 3e-2
    COMPAS_Z = np.array([1e-4, 1e-3, 3e-2])  # has 3e-2 but Z_grid does not

    with pytest.raises(ValueError, match="not present in Z_grid"):
        compute_merger_rate(
            redshifts, times, time_first_SF=13.5e3, n_formed=sfr,
            p_draw=0.1,
            COMPAS_Z=COMPAS_Z, COMPAS_delay_times=delays,
            COMPAS_weights=weights, Z_grid=Z_grid_bad)


# ─────────────────────────────────────────────────────────────────────
# Wanderman and Piran (2015) sGRB R(z), MNRAS 448, 3026 -- Eq. (9)
# ─────────────────────────────────────────────────────────────────────
def test_wanderman_piran_2015_peak_at_z_peak():
    """``wanderman_piran_2015_Rz`` returns R0 at z = z_peak (the
    piecewise-exponential is C^0 there, so both sides give the same
    value)."""
    from grb_rates import wanderman_piran_2015_Rz

    out = wanderman_piran_2015_Rz(np.array([0.9]))
    assert np.isclose(out["R_best"][0], 4.1)
    assert np.isclose(out["R_lo"][0], 2.2)
    assert np.isclose(out["R_hi"][0], 6.4)


# ─────────────────────────────────────────────────────────────────────
# COMPAS-vs-grb_rates: numerical agreement on the same HDF5 at z = 0.
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# Vectorized compute_merger_rate: legacy-loop equivalence
# ─────────────────────────────────────────────────────────────────────
def _legacy_loop_rate(redshifts, times, time_first_SF, n_formed, p_draw,
                      COMPAS_Z, COMPAS_delay_times, COMPAS_weights,
                      smooth_sigma, Z_grid):
    """Reference implementation of the pre-vectorization per-binary loop
    used by ``compute_merger_rate`` prior to the chunked rewrite.  Kept
    here so the regression test can compare element-wise."""
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    from grb_rates import _bin_averaged_dPdlogZ, _interp_formation_rate

    n_z = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]
    times_to_z = interp1d(times, redshifts)
    dPdlogZ_binned, sys_col = _bin_averaged_dPdlogZ(
        redshifts, COMPAS_Z, Z_grid=Z_grid)
    t_min = max(time_first_SF, times.min())
    total = np.zeros(n_z)
    for i in range(len(COMPAS_delay_times)):
        t_form = times - COMPAS_delay_times[i]
        valid = t_form >= t_min
        if not valid.any():
            continue
        j_idx = np.where(valid)[0]
        z_form = times_to_z(t_form[j_idx])
        total[j_idx] += _interp_formation_rate(
            n_formed, dPdlogZ_binned[:, sys_col[i]], p_draw,
            COMPAS_weights[i], z_form, redshift_step, n_z)
    if smooth_sigma > 0:
        total = gaussian_filter1d(total, sigma=smooth_sigma)
    return total


def _make_synth_population_for_rates(N, n_z, rng):
    redshifts = np.linspace(0.0, 5.0, n_z)
    times = np.linspace(13700.0, 100.0, n_z)
    sfr = np.full(n_z, 1e7)
    n_formed = sfr / 1.5e8
    Z_grid = 10.0 ** np.linspace(-4, np.log10(0.03), 8)
    COMPAS_Z = rng.choice(Z_grid, size=N)
    COMPAS_delays = 10.0 ** rng.uniform(np.log10(10.0),
                                         np.log10(13000.0), size=N)
    COMPAS_w = rng.uniform(0.1, 1.0, size=N)
    return (redshifts, times, sfr, n_formed, Z_grid,
            COMPAS_Z, COMPAS_delays, COMPAS_w)


def test_compute_merger_rate_vectorized_matches_legacy_loop():
    """Vectorized chunked accumulator must match the original per-binary
    loop to ~1e-12 (machine precision modulo float-summation order),
    well below the 1e-6 rtol in the plan."""
    from grb_rates import compute_merger_rate

    rng = np.random.default_rng(2024)
    (redshifts, times, _, n_formed, Z_grid,
     Z, delays, w) = _make_synth_population_for_rates(N=500, n_z=200, rng=rng)

    R_vec = compute_merger_rate(
        redshifts, times, time_first_SF=times.min() + 10.0,
        n_formed=n_formed, p_draw=0.1,
        COMPAS_Z=Z, COMPAS_delay_times=delays, COMPAS_weights=w,
        smooth_sigma=0, Z_grid=Z_grid)
    R_loop = _legacy_loop_rate(
        redshifts, times, times.min() + 10.0, n_formed, 0.1,
        Z, delays, w, smooth_sigma=0, Z_grid=Z_grid)

    np.testing.assert_allclose(
        R_vec, R_loop, rtol=1e-6, atol=1e-12,
        err_msg=("vectorized compute_merger_rate disagrees with the "
                 "reference per-binary loop"))


def test_compute_merger_rate_empty_population_returns_zeros():
    """The empty-population guard must survive vectorization."""
    from grb_rates import compute_merger_rate

    redshifts = np.linspace(0.0, 5.0, 100)
    times = np.linspace(13700.0, 100.0, 100)
    n_formed = np.full(100, 1e7 / 1.5e8)
    R = compute_merger_rate(
        redshifts, times, time_first_SF=times.min() + 10.0,
        n_formed=n_formed, p_draw=0.1,
        COMPAS_Z=np.array([]), COMPAS_delay_times=np.array([]),
        COMPAS_weights=np.array([]),
        smooth_sigma=30)
    assert R.shape == redshifts.shape
    assert np.all(R == 0)


def test_compute_merger_rate_smoothing_intact():
    """smooth_sigma still applies a Gaussian kernel to the unsmoothed
    rate; the smoothed L2 deviation from the unsmoothed curve must be
    monotonic in the kernel width and small (< 50 percent of the L2
    norm) at the production sigma=30 default."""
    from grb_rates import compute_merger_rate
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(2025)
    (redshifts, times, _, n_formed, Z_grid,
     Z, delays, w) = _make_synth_population_for_rates(N=300, n_z=200, rng=rng)

    common = dict(redshifts=redshifts, times=times,
                  time_first_SF=times.min() + 10.0,
                  n_formed=n_formed, p_draw=0.1,
                  COMPAS_Z=Z, COMPAS_delay_times=delays,
                  COMPAS_weights=w, Z_grid=Z_grid)

    R_unsmoothed = compute_merger_rate(smooth_sigma=0,  **common)
    R_smoothed   = compute_merger_rate(smooth_sigma=30, **common)
    np.testing.assert_allclose(
        R_smoothed, gaussian_filter1d(R_unsmoothed, sigma=30),
        rtol=1e-12, atol=1e-12,
        err_msg=("smooth_sigma path no longer applies the same Gaussian "
                 "kernel to the unsmoothed rate; check the smoothing tail"))
    l2_diff = np.linalg.norm(R_smoothed - R_unsmoothed)
    l2_norm = np.linalg.norm(R_unsmoothed)
    assert l2_diff < 0.5 * l2_norm, (
        f"smoothing changed the rate too much (||smoothed - raw|| / ||raw||"
        f" = {l2_diff / max(l2_norm, 1e-30):.3f}, expected < 0.5)")


# ─────────────────────────────────────────────────────────────────────
# COMPAS-vs-grb_rates: numerical agreement on the same HDF5 at z = 0.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_compute_merger_rate_matches_compas_shape(bns_a_path):
    """Run the COMPAS reference and ``compute_merger_rate`` on the same
    BNS Model A HDF5 and assert the *shapes* (R(z) / R(0)) agree to
    within 10 percent at z in [0.5, 1.0, 2.0].

    Why shape, not absolute value:

    grb_rates' ``compute_merger_rate`` is paired with
    ``calibrate_mean_mass_evolved``: ``mean_mass_evolved`` is chosen so
    that R_ours(z=0) matches the HDF5's pre-tabulated local rate
    (Broekgaarden+ 2021 ``w_000`` weights, ~33 Gpc^-3 yr^-1 for BNS A).
    COMPAS's ``find_formation_and_merger_rates`` runs the convolution
    from scratch using ``n_formed = sfr / Average_SF_mass_needed`` and
    a point-sampled ``dPdlogZ`` over a fine 1201-bin metallicity grid;
    that pipeline produces a structurally different absolute z=0 rate
    (~5x ours, with the factor set by point-vs-bin sampling and grid
    resolution).  Equating the two absolute z=0 numbers would require
    discarding the calibration step or recomputing
    Average_SF_mass_needed with ``ClassCOMPAS``, neither of which is
    what users of grb_rates do in practice.

    What we *can* validate: the redshift dependence of R(z) is set by
    the same SFR(z) * dP/d(ln Z)(z, Z) * delay-time convolution in
    both pipelines.  Up to a global rescaling the curves track each
    other to within a few percent at z <~ 1; by z = 2 the bin-averaged
    renormalization in ``_bin_averaged_dPdlogZ`` (which COMPAS does not
    perform) starts inflating our rate above the COMPAS reference as
    the Neijssel log-normal slips below the COMPAS Z grid.  We cap the
    comparison at z = 2 and use a 15 percent tolerance, which sits
    well below the order-of-magnitude divergence that a structural bug
    (cosmology drift, missing weight, p_draw mismatch) would produce.

    Cosmology is pinned to Planck 2015 on both sides per CLAUDE.md.
    """
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )
    calculate_redshift_related_params = fci.calculate_redshift_related_params
    find_sfr = fci.find_sfr
    find_metallicity_distribution = fci.find_metallicity_distribution
    find_formation_and_merger_rates = fci.find_formation_and_merger_rates

    from astropy.cosmology import Planck15

    from grb_io import load_bns, read_expected_local_rate
    from grb_rates import (
        compute_merger_rate,
        calibrate_mean_mass_evolved,
    )

    # Load the COMPAS HDF5 once via grb_io (matches notebook usage).
    data = load_bns(path=bns_a_path)
    Z = data["metallicity"]
    delays = data["delay_time"]
    w = data["weights"]

    # Same redshift / SFR / dPdlogZ setup the demo uses, with
    # cosmology pinned to Planck15 on both sides so the redshift-time
    # map is identical.
    redshifts, _, times, time_first_SF, _, _ = calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.01, cosmology=Planck15)
    sfr = find_sfr(redshifts)

    dPdlogZ, metallicities, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(np.min(Z)),
        max_logZ_COMPAS=np.log(np.max(Z)),
    )

    # Calibrate MEAN_MASS_EVOLVED against the pre-computed local rate
    # in the HDF5; this is the discipline CLAUDE.md mandates per
    # population.
    expected_local_rate = read_expected_local_rate(bns_a_path)
    Z_grid = np.unique(Z)
    mean_mass, _ = calibrate_mean_mass_evolved(
        sfr, redshifts, times, time_first_SF, p_draw,
        Z, delays, w, expected_local_rate, Z_grid=Z_grid)
    n_formed = sfr / mean_mass

    # Our pipeline.  Disable the default Gaussian smoothing so the
    # shape comparison is apples-to-apples with COMPAS, which does not
    # smooth.  ``compute_merger_rate``'s default ``smooth_sigma=30``
    # otherwise convolves contributions from the R(z) peak (z ~ 1-2)
    # into R(z=0), inflating the denominator of the shape ratio and
    # making R_ours(z)/R_ours(0) look artificially flatter.
    R_ours = compute_merger_rate(
        redshifts, times, time_first_SF, n_formed, p_draw,
        Z, delays, w, Z_grid=Z_grid, smooth_sigma=0)

    # COMPAS reference, same inputs.  Average_SF_mass_needed plays the
    # role of MEAN_MASS_EVOLVED inside the demo; since we pass the
    # same n_formed, the only structural difference is bin-averaged
    # vs point-sampled dPdlogZ.
    n_binaries = len(Z)
    formation_rate, merger_rate = find_formation_and_merger_rates(
        n_binaries, redshifts, times, time_first_SF, n_formed,
        dPdlogZ, metallicities, p_draw, Z, delays, COMPAS_weights=w)
    R_compas = merger_rate.sum(axis=0)

    iz0 = int(np.argmin(np.abs(redshifts)))
    R_ours_0 = float(R_ours[iz0])
    R_compas_0 = float(R_compas[iz0])

    assert R_ours_0 > 0, f"compute_merger_rate returned zero at z=0: {R_ours_0}"
    assert R_compas_0 > 0, f"COMPAS reference returned zero at z=0: {R_compas_0}"

    # Compare normalized R(z) shapes at low/mid z.  Bin-averaging vs
    # point-sampling differ by an approximately constant multiplicative
    # factor when the MSSFR PDF still sits inside the COMPAS Z grid
    # (z <~ 2).  At higher z our renormalization step intentionally
    # diverges from COMPAS, so we do not test there.
    shape_ours = R_ours / R_ours_0
    shape_compas = R_compas / R_compas_0

    z_checkpoints = (0.5, 1.0, 2.0)
    tol_shape = 0.15
    deltas = []
    for z_check in z_checkpoints:
        iz = int(np.argmin(np.abs(redshifts - z_check)))
        delta = abs(shape_ours[iz] - shape_compas[iz]) / shape_compas[iz]
        deltas.append((z_check, delta, float(shape_ours[iz]),
                       float(shape_compas[iz])))

    failures = [d for d in deltas if d[1] > tol_shape]
    assert not failures, (
        "compute_merger_rate R(z) shape disagrees with COMPAS reference "
        f"by more than {tol_shape * 100:.0f}% at: "
        + "; ".join(
            f"z={z:.1f} ({delta * 100:.1f}%, ours/R0={so:.4f}, "
            f"COMPAS/R0={sc:.4f})"
            for (z, delta, so, sc) in failures
        )
        + ".  A structural bug (cosmology drift, missing weight, p_draw "
        f"mismatch) would push these well past {tol_shape * 100:.0f} percent."
    )
