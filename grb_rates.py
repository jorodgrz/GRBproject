"""
Cosmic merger rate computation and related utilities.

compute_merger_rate implements the MSSFR convolution from Neijssel et al. (2019)
using COMPAS FastCosmicIntegration infrastructure.  Also includes Kroupa IMF
verification, per-system rate weights, and BH spin marginalization.
"""

import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d
from scipy.stats import norm as _NormDist

from grb_physics import MISALIGNMENT_SYSTEMATIC_FACTOR


# ═══════════════════════════════════════════════════════════════════════════
# Cosmic integration
# ═══════════════════════════════════════════════════════════════════════════

# Neijssel et al. (2019) / COMPAS default MSSFR parameters.
# These must match the values passed to find_metallicity_distribution().
_MU0 = 0.035
_MUZ = -0.23
_SIGMA_0 = 0.39
_SIGMA_Z = 0.0


def _bin_averaged_dPdlogZ(redshifts, COMPAS_Z, Z_grid=None,
                          mu0=_MU0, muz=_MUZ,
                          sigma_0=_SIGMA_0, sigma_z=_SIGMA_Z):
    """Analytic bin-integrated metallicity weights via the normal CDF.

    For each COMPAS birth metallicity Z_k, compute the exact integral of
    the Neijssel et al. (2019) log-normal metallicity PDF over the
    Voronoi cell [ln Z_lo, ln Z_hi] at every redshift:

        P_bin(z, k) = Phi((ln Z_hi - mu(z)) / sigma(z))
                    - Phi((ln Z_lo - mu(z)) / sigma(z))

    where Phi is the standard normal CDF, and mu(z), sigma(z) are the
    redshift-dependent log-normal parameters (Langer & Norman 2006;
    Neijssel et al. 2019 Eq. 2; Broekgaarden et al. 2021, Section 2.4).

    This replaces the Riemann-sum approximation on the fine dPdlogZ grid,
    which produced residual aliasing from the non-uniform spacing of the
    53-point COMPAS metallicity grid.

    Parameters
    ----------
    redshifts : 1-D array, shape (n_z,)
        Redshift grid.
    COMPAS_Z : 1-D array, shape (n_systems,)
        Birth metallicity of each COMPAS binary.  May be a class subset
        of the full population.
    Z_grid : 1-D array, optional
        Simulation's full metallicity discretization (for the
        Broekgaarden+ 2021 COMPAS runs this is the 53-element grid in
        ``grb_io.METALLICITY_GRID``; equivalently ``np.unique(Z_full)``
        for the parent population).  When provided, the Voronoi cells
        and the [ln Z_min, ln Z_max] renormalisation range are
        determined by ``Z_grid``, NOT by ``np.unique(COMPAS_Z)``.

        Pass this argument whenever ``COMPAS_Z`` is a class subset.
        Without it, the Voronoi bin widths and the renormalisation
        range shrink with the subset's Z range, and the high-z
        amplification factor (``norm / total_bin`` below) becomes
        subset-dependent.  That is what produces the unphysical
        dip-and-recovery wiggles in per-class merger rate density
        curves: each sub-class has a different effective [ln Z_min,
        ln Z_max] window, so the analytic-extrapolation correction
        applied at high z (where the MSSFR log-normal sits below the
        COMPAS Z grid) differs across sub-classes.  When ``Z_grid`` is
        None (default), the previous behaviour is retained for
        backward compatibility with full-population callers.
    mu0, muz, sigma_0, sigma_z : float
        MSSFR log-normal parameters (must match those used in
        ``find_metallicity_distribution``).

    Returns
    -------
    dPdlogZ_binned : 2-D array, shape (n_z, n_bins)
        Bin-integrated metallicity weight for each cell of the chosen
        grid (``Z_grid`` if supplied, else ``np.unique(COMPAS_Z)``).
    sys_col_idx : 1-D int array, shape (n_systems,)
        Column index into ``dPdlogZ_binned`` for each COMPAS system.
    """
    COMPAS_Z = np.asarray(COMPAS_Z, dtype=float)
    n_z = len(redshifts)

    # Empty-population guard: when a downstream selection (e.g. a class
    # mask in compute_merger_rate) is empty, return zero-sized weights
    # and an empty column-index array.  Callers that loop over systems
    # will produce a zero rate vector instead of an IndexError.
    if COMPAS_Z.size == 0:
        return np.zeros((n_z, 0)), np.zeros(0, dtype=int)

    # Reproduce mu(z) and sigma(z) from Neijssel+19 / COMPAS defaults.
    # With alpha=0 (pure log-normal, no skewness) the mu simplification
    # is: mu = ln(mean_Z) - sigma^2/2  (standard log-normal identity).
    sigma = sigma_0 * 10.0 ** (sigma_z * redshifts)     # (n_z,)
    mean_Z = mu0 * 10.0 ** (muz * redshifts)            # (n_z,)
    mu = np.log(mean_Z) - 0.5 * sigma ** 2              # (n_z,)

    if Z_grid is None:
        unique_Z = np.unique(COMPAS_Z)
    else:
        unique_Z = np.unique(np.asarray(Z_grid, dtype=float))
    log_unique = np.log(unique_Z)
    n_bins = len(unique_Z)

    # Single-Z degenerate case: a half-decade-wide window around the lone bin.
    if n_bins == 1:
        half_width = 0.5 * np.log(10.0)
        lo_edges = np.array([log_unique[0] - half_width])
        hi_edges = np.array([log_unique[0] + half_width])
    else:
        # Voronoi boundaries: geometric midpoints between adjacent COMPAS Z values
        mid = 0.5 * (log_unique[:-1] + log_unique[1:])
        lo_edges = np.empty(n_bins)
        hi_edges = np.empty(n_bins)
        # First/last bin edges are snapped to log_unique[0] and
        # log_unique[-1] so they coincide with the renormalization
        # range used below (ln_Z_min / ln_Z_max).  The previous
        # symmetric extrapolation by half the adjacent spacing
        # extended outside [ln_Z_min, ln_Z_max], so probability
        # captured in the extrapolated tails was rescaled away by
        # the renormalization step -- harmless but logically
        # inconsistent.  Effect on the first/last bin is sub-1%.
        lo_edges[0] = log_unique[0]
        lo_edges[1:] = mid
        hi_edges[:-1] = mid
        hi_edges[-1] = log_unique[-1]

    # Analytic CDF evaluation: P_bin(z,k) = Phi(hi_std) - Phi(lo_std)
    # Vectorised over redshifts (axis 0) and bins (axis 1).
    inv_sigma = 1.0 / sigma                             # (n_z,)
    lo_std = (lo_edges[np.newaxis, :] - mu[:, np.newaxis]) * inv_sigma[:, np.newaxis]
    hi_std = (hi_edges[np.newaxis, :] - mu[:, np.newaxis]) * inv_sigma[:, np.newaxis]
    dPdlogZ_binned = _NormDist.cdf(hi_std) - _NormDist.cdf(lo_std)  # (n_z, n_bins)

    # Normalize: COMPAS convention clips to the sampled Z range and
    # rescales the bin probabilities so their sum equals
    #   norm(z) = Phi((ln Z_max - mu(z)) / sigma(z))
    #           - Phi((ln Z_min - mu(z)) / sigma(z))
    # i.e. the integrated log-normal probability that falls inside
    # [ln_Z_min, ln_Z_max].  This is < 1 in general and approaches 0
    # at high z, where the MSSFR log-normal sits well below the
    # COMPAS Z grid (see the diagnostic warning below).  The bins do
    # NOT renormalize to 1.
    ln_Z_min = log_unique[0]
    ln_Z_max = log_unique[-1]
    norm = (_NormDist.cdf((ln_Z_max - mu) / sigma)
            - _NormDist.cdf((ln_Z_min - mu) / sigma))   # (n_z,)
    norm = np.where(norm > 0, norm, 1.0)
    total_bin = dPdlogZ_binned.sum(axis=1)               # (n_z,)
    total_bin_safe = np.where(total_bin > 0, total_bin, 1.0)
    dPdlogZ_binned *= (norm / total_bin_safe)[:, np.newaxis]

    # Diagnostic: at high z the MSSFR log-normal sits well below the
    # COMPAS Z grid lower edge, so the Voronoi bin probabilities sum
    # to a tiny ``total_bin`` and the renormalization factor blows up.
    # Levina+ 2026 (arXiv:2601.20202) shows this kind of analytical
    # extrapolation overestimates high-z BBH rates by 10 - 1e4x.  Warn
    # when any redshift slice has > 10x amplification so callers know
    # their high-z rates are extrapolation-dominated.
    amplification = norm / total_bin_safe
    if np.any(amplification > 10.0):
        bad = amplification > 10.0
        z_bad = redshifts[bad]
        warnings.warn(
            f"Metallicity grid does not span the MSSFR PDF at z = "
            f"[{z_bad.min():.1f}, {z_bad.max():.1f}] "
            f"(max amplification {amplification.max():.1f}x).  "
            f"High-z rates are extrapolation-dominated; consider "
            f"extending the COMPAS Z grid or capping max(z).",
            stacklevel=2)

    # Map each COMPAS system to its unique-Z column index.  Use
    # searchsorted on log-space (unique_Z is monotonic from np.unique)
    # so the lookup remains correct when ``Z_grid`` is the full COMPAS
    # grid and ``COMPAS_Z`` is a subset.  An exact-match assertion
    # catches floating-point drift (e.g. a Z value that was not part of
    # the simulation grid being passed in).
    log_compas = np.log(COMPAS_Z)
    sys_col_idx = np.searchsorted(log_unique, log_compas)
    sys_col_idx = np.clip(sys_col_idx, 0, n_bins - 1)
    # searchsorted returns the upper insertion point; the matching grid
    # value can be at sys_col_idx or sys_col_idx - 1.  Pick the closer.
    left = np.clip(sys_col_idx - 1, 0, n_bins - 1)
    use_left = (np.abs(log_unique[left] - log_compas)
                < np.abs(log_unique[sys_col_idx] - log_compas))
    sys_col_idx = np.where(use_left, left, sys_col_idx)
    if not np.allclose(log_unique[sys_col_idx], log_compas, atol=1e-9):
        bad = np.where(~np.isclose(log_unique[sys_col_idx], log_compas,
                                    atol=1e-9))[0]
        raise ValueError(
            f"{len(bad)} COMPAS_Z values are not present in the "
            f"supplied Z_grid (first offender: Z = "
            f"{COMPAS_Z[bad[0]]:.6g}).  Pass Z_grid = "
            f"np.unique(Z_full_population) when COMPAS_Z is a subset.")

    return dPdlogZ_binned, sys_col_idx


def _interp_formation_rate(n_formed, dPdlogZ_col, p_draw, weight,
                           z_form, redshift_step, n_z):
    """Shared interpolation of the formation rate at arbitrary z_form.

    Both ``compute_merger_rate`` and ``per_system_rate_weights`` use this
    to evaluate  n_formed(z_form) * dPdlogZ_binned(z_form) / p_draw * w
    via linear interpolation on the uniform redshift grid.

    Parameters
    ----------
    dPdlogZ_col : 1-D array, shape (n_z,)
        Bin-integrated metallicity weight for this system (or array of
        systems when called from ``per_system_rate_weights``), already
        selected from the ``dPdlogZ_binned`` columns.
    """
    z_idx_float = z_form / redshift_step
    z_lo = np.clip(np.floor(z_idx_float).astype(int), 0, n_z - 1)
    z_hi = np.clip(z_lo + 1, 0, n_z - 1)
    frac = z_idx_float - np.floor(z_idx_float)

    form_lo = n_formed[z_lo] * dPdlogZ_col[z_lo] / p_draw * weight
    form_hi = n_formed[z_hi] * dPdlogZ_col[z_hi] / p_draw * weight
    return form_lo * (1.0 - frac) + form_hi * frac


def compute_merger_rate(redshifts, times, time_first_SF, n_formed,
                        p_draw, COMPAS_Z, COMPAS_delay_times, COMPAS_weights,
                        smooth_sigma=30, Z_grid=None):
    """
    Intrinsic merger rate density [Gpc^-3 yr^-1] vs redshift.

    For each binary the formation rate is:
        SFR(z) * dP/dlogZ(z, Z_i) / p_draw * weight_i / meanMassEvolved

    The metallicity weight dP/dlogZ is integrated over each COMPAS
    metallicity's Voronoi cell (bin-averaged) rather than point-evaluated,
    following Neijssel et al. (2019) Eq. 2 and Broekgaarden et al. (2021)
    Section 2.4.  Bin probabilities are computed internally via
    ``_bin_averaged_dPdlogZ``; pass ``Z_grid`` whenever ``COMPAS_Z`` is a
    class subset so the Voronoi cells track the full simulation grid.
    This eliminates aliasing artifacts caused by the discrete COMPAS
    metallicity grid.

    A light Gaussian kernel (``smooth_sigma`` redshift bins, default 30
    = dz 0.3 at step 0.01) suppresses the residual ~3% Monte Carlo
    wobble from the finite number of discrete COMPAS metallicities.
    Set ``smooth_sigma=0`` to disable smoothing.

    Important: ``n_formed`` must already contain the 1/MEAN_MASS_EVOLVED
    normalisation.  ``find_sfr()`` returns *raw* SFR in Msun/yr/Gpc^3 and
    does NOT include the 1/MEAN_MASS_EVOLVED factor.  Divide explicitly::

        n_formed = find_sfr(redshifts) / MEAN_MASS_EVOLVED

    Each population (BNS, BHNS) has its own MEAN_MASS_EVOLVED because
    the Broekgaarden et al. simulations evolve different total masses.

    Parameters
    ----------
    p_draw : float
        COMPAS metallicity sampling density (= 1 / (max_logZ_COMPAS -
        min_logZ_COMPAS) for a flat-in-lnZ prior; see COMPAS
        ``find_metallicity_distribution``).
    Z_grid : 1-D array, optional
        Full simulation metallicity grid (e.g. ``np.unique(Z_full)`` for
        the parent population).  REQUIRED whenever ``COMPAS_Z``,
        ``COMPAS_delay_times``, ``COMPAS_weights`` are a class subset:
        without it the Voronoi bin widths and the high-z renormalisation
        range collapse to the subset's Z range, producing unphysical
        per-class shape distortions.  See ``_bin_averaged_dPdlogZ``.
    """
    n_z           = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]

    # Empty-population guard: a class mask with zero True entries (e.g. a
    # high-spin clip wiping out all Long cbGRB systems) returns a zero
    # rate vector instead of erroring.  Same shape as the redshift grid
    # so downstream array operations remain consistent.
    if len(COMPAS_delay_times) == 0:
        return np.zeros(n_z)

    # Cross-module Z_grid alignment guard.  The Voronoi renormalisation in
    # _bin_averaged_dPdlogZ assumes every COMPAS_Z value lands on a cell of
    # the supplied grid; otherwise the high-z amplification factor and the
    # bin-width allocation become subset-dependent (silent class-shape bias).
    # This is the cross-module assert reviewer 1 flagged as missing.
    if Z_grid is not None:
        Z_grid_unique = np.unique(np.asarray(Z_grid, dtype=float))
        compas_unique = np.unique(np.asarray(COMPAS_Z, dtype=float))
        if not np.all(np.isin(compas_unique, Z_grid_unique)):
            missing = compas_unique[~np.isin(compas_unique, Z_grid_unique)]
            raise ValueError(
                f"{len(missing)} COMPAS_Z values are not present in "
                f"Z_grid (first offender: Z = {missing[0]:.6g}).  Pass "
                f"Z_grid = np.unique(Z_full_population) when COMPAS_Z is "
                f"a class subset.")

    times_to_z    = interp1d(times, redshifts)

    dPdlogZ_binned, sys_col = _bin_averaged_dPdlogZ(
        redshifts, COMPAS_Z, Z_grid=Z_grid)

    t_min = max(time_first_SF, times.min())
    total_merger = np.zeros(n_z)

    # Vectorized formation-rate accumulation (CLAUDE.md "Vectorize" rule).
    # Replaces the per-binary Python loop with a chunked (n, n_z) batched
    # evaluation: each chunk builds the formation-time matrix t_form[i, j]
    # = times[j] - delay[i], maps it to a formation redshift and an
    # n_formed * dP/dlogZ contribution at every (i, j), masks out invalid
    # (i, j) pairs (where t_form < t_min), and sums over the system axis.
    # Memory per chunk: n * n_z * 8 bytes = O(80 MB) at n=N_CHUNK=10_000
    # and n_z ~ 1000, so even the largest BNS or BHNS population (~10^6
    # systems) processes in ~100 chunks without exceeding RAM.  Numerical
    # output matches the legacy loop to ~1e-6 rtol (regression test in
    # tests/test_rates.py).
    N_CHUNK = 10_000
    n_systems = len(COMPAS_delay_times)
    delays = np.asarray(COMPAS_delay_times, dtype=float)
    weights_arr = np.asarray(COMPAS_weights, dtype=float)
    sys_col = np.asarray(sys_col, dtype=int)
    inv_p_draw = 1.0 / p_draw

    for start in range(0, n_systems, N_CHUNK):
        end = min(start + N_CHUNK, n_systems)
        delay_c = delays[start:end]
        w_c     = weights_arr[start:end]
        cols_c  = sys_col[start:end]

        # (n, n_z) formation-time grid: t_form[i, j] = times[j] - delay[i].
        t_form = times[None, :] - delay_c[:, None]
        valid = t_form >= t_min                                # (n, n_z) bool

        # Replace invalid entries with t_min so times_to_z stays in range;
        # the contribution is masked back to zero by `valid` below.
        t_form_clipped = np.where(valid, t_form, t_min)
        z_form = times_to_z(t_form_clipped)                    # (n, n_z)

        # Linear-interpolation indices on the uniform redshift grid.
        z_idx_float = z_form / redshift_step
        z_lo = np.clip(np.floor(z_idx_float).astype(np.int64), 0, n_z - 1)
        z_hi = np.clip(z_lo + 1, 0, n_z - 1)
        frac = z_idx_float - np.floor(z_idx_float)             # (n, n_z)

        # dPdlogZ_binned has shape (n_z, n_bins); we want one column per
        # system per (z_lo, z_hi).  Fancy index over (system axis, z axis)
        # via advanced indexing: dPdlogZ_binned[z_lo, cols_c[:, None]] -> (n, n_z).
        dP_lo = dPdlogZ_binned[z_lo, cols_c[:, None]]
        dP_hi = dPdlogZ_binned[z_hi, cols_c[:, None]]

        # Formation rate at lo/hi neighbour, broadcast n_formed across systems.
        f_lo = n_formed[z_lo] * dP_lo * inv_p_draw * w_c[:, None]
        f_hi = n_formed[z_hi] * dP_hi * inv_p_draw * w_c[:, None]

        contrib = (f_lo * (1.0 - frac) + f_hi * frac) * valid  # (n, n_z)
        total_merger += contrib.sum(axis=0)

    if smooth_sigma > 0:
        total_merger = _gaussian_filter1d(total_merger, sigma=smooth_sigma)

    return total_merger


def per_system_rate_weights(z_target, redshifts, times, time_first_SF,
                            n_formed, p_draw,
                            COMPAS_Z, COMPAS_delay_times, COMPAS_weights,
                            Z_grid=None):
    """
    Per-system contribution to the merger rate at a single z_target.

    Same physics as ``compute_merger_rate`` but returns an array of
    individual rate weights (one per binary) for constructing weighted-
    Poisson 1 sigma envelopes (sigma_R(z) = sqrt(sum_i w_i(z)^2),
    CLAUDE.md "Uncertainty is not optional"; see Section 7 / 8 of
    ``grb_main.ipynb``).

    Vectorized: a single broadcast over the population, no Python loop.
    Replaces the prior ``_interp_formation_rate`` call which silently
    expanded an ``(n_z, n_valid)`` slice into an ``(n_valid, n_valid)``
    intermediate via 1-D fancy indexing.

    Parameters
    ----------
    p_draw : float
        COMPAS metallicity sampling density (see ``compute_merger_rate``).
    Z_grid : 1-D array, optional
        Full simulation metallicity grid; pass when ``COMPAS_Z`` is a
        class subset (see ``compute_merger_rate``).
    """
    n_z           = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]

    if len(COMPAS_weights) == 0:
        return np.zeros(0)

    times_to_z    = interp1d(times, redshifts)

    j_target = np.argmin(np.abs(redshifts - z_target))
    t_merge  = times[j_target]

    dPdlogZ_binned, sys_col = _bin_averaged_dPdlogZ(
        redshifts, COMPAS_Z, Z_grid=Z_grid)
    t_min  = max(time_first_SF, times.min())

    out    = np.zeros(len(COMPAS_weights))
    t_form = t_merge - np.asarray(COMPAS_delay_times, dtype=float)
    valid  = t_form >= t_min
    if not valid.any():
        return out

    idx    = np.where(valid)[0]
    z_form = times_to_z(t_form[idx])
    z_idx  = z_form / redshift_step
    z_lo   = np.clip(np.floor(z_idx).astype(np.int64), 0, n_z - 1)
    z_hi   = np.clip(z_lo + 1, 0, n_z - 1)
    frac   = z_idx - np.floor(z_idx)

    # Pair (z_lo[i], sys_col[idx][i]) and (z_hi[i], sys_col[idx][i]) so
    # each system gets the metallicity weight at its own column,
    # avoiding the (n_valid, n_valid) intermediate produced by 1-D
    # fancy indexing into a 2-D dPdlogZ_binned slice.
    cols  = np.asarray(sys_col, dtype=np.int64)[idx]
    dP_lo = dPdlogZ_binned[z_lo, cols]
    dP_hi = dPdlogZ_binned[z_hi, cols]
    w     = np.asarray(COMPAS_weights, dtype=float)[idx]

    inv_p_draw = 1.0 / p_draw
    f_lo = n_formed[z_lo] * dP_lo * inv_p_draw * w
    f_hi = n_formed[z_hi] * dP_hi * inv_p_draw * w
    out[idx] = f_lo * (1.0 - frac) + f_hi * frac
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Per-population normalization
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_mean_mass_evolved(sfr, redshifts, times, time_first_SF,
                                p_draw,
                                COMPAS_Z, COMPAS_delay_times, COMPAS_weights,
                                expected_local_rate, Z_grid=None):
    """Derive the effective MEAN_MASS_EVOLVED for one population.

    Runs ``compute_merger_rate`` with ``n_formed = sfr`` (unit
    normalization) and scales so the z = 0 rate matches
    *expected_local_rate* (from pre-computed ``w_000`` weights).

    Parameters
    ----------
    p_draw : float
        COMPAS metallicity sampling density (see ``compute_merger_rate``).
    Z_grid : 1-D array, optional
        Full simulation metallicity grid.  When ``COMPAS_Z`` is the full
        population (the standard calibration use), this is redundant
        because ``np.unique(COMPAS_Z) == Z_grid`` by construction.  Pass
        explicitly only if the calibration needs to be performed on a
        sub-population (rare).

    Returns
    -------
    mean_mass_evolved : float
        The total stellar mass [Msun] evolved in the simulation.
    rate_unnorm : 1-D array
        Un-normalized merger rate; divide by *mean_mass_evolved* to get
        the correctly normalized R(z).
    """
    rate_unnorm = compute_merger_rate(
        redshifts, times, time_first_SF, sfr, p_draw,
        COMPAS_Z, COMPAS_delay_times, COMPAS_weights, Z_grid=Z_grid)
    mean_mass_evolved = rate_unnorm[0] / expected_local_rate
    return mean_mass_evolved, rate_unnorm


# ═══════════════════════════════════════════════════════════════════════════
# Formation efficiency (per metallicity bin)
# ═══════════════════════════════════════════════════════════════════════════
def formation_efficiency(metallicityGrid, Z_all, w_all, masks=None,
                         mean_mass_evolved=None):
    """
    Compute formation efficiency per metallicity bin.

    Parameters
    ----------
    metallicityGrid : 1-D array
        Grid of metallicity values (e.g. 53-point COMPAS grid).
    Z_all : 1-D array
        Metallicity of each merging system.
    w_all : 1-D array
        STROOPWAFEL weight of each merging system.
    masks : dict of boolean arrays, optional
        Sub-population masks (e.g. {'Short': mask_s, 'Long': mask_l}).
        If None, only total efficiency is returned.
    mean_mass_evolved : float
        Total stellar mass evolved in the simulation [Msun].
        Required; use ``calibrate_mean_mass_evolved()`` to derive it
        for each population (BNS, BHNS) separately.

    Returns
    -------
    dict of 1-D arrays keyed by mask name, plus 'total'.
    """
    if mean_mass_evolved is None:
        raise ValueError(
            "mean_mass_evolved must be provided; "
            "use calibrate_mean_mass_evolved() to derive it"
        )
    unique_Z = np.unique(Z_all)
    result = {'total': np.zeros(len(metallicityGrid))}
    if masks is not None:
        for name in masks:
            result[name] = np.zeros(len(metallicityGrid))

    for i, Z in enumerate(metallicityGrid):
        if Z in unique_Z:
            maskZ = (Z_all == Z)
            result['total'][i] = np.sum(w_all[maskZ]) / mean_mass_evolved
            if masks is not None:
                for name, m in masks.items():
                    result[name][i] = np.sum(w_all[maskZ & m]) / mean_mass_evolved

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BH spin marginalization
# ═══════════════════════════════════════════════════════════════════════════
def marginalize(rate_dict, weights):
    """Weighted average over spin values: sum(w_a * rate[a]).

    Thin helper used in `grb_main.ipynb` for tabulated spin sweeps.
    Prefer ``marginalize_bh_spin`` when integrating over a continuous
    or callable PDF (e.g. Fuller and Ma 2019).
    """
    return sum(weights[a] * rate_dict[a] for a in rate_dict)


def marginalize_bh_spin(rate_per_spin, p_chi, spin_grid=None):
    """BH-spin-marginalised rate: integral over the BH spin PDF.

    Promoted from inline ``marginalize`` usage in ``grb_main.ipynb``
    Sections 8 and 11.  Resolves the Council Expansionist L5 gap by
    giving the Fuller and Ma (2019) flat-low-spin and Kawaguchi (2015)
    spin-orbit alignment integrations a named module home.

    Two input modes are accepted:

    - ``rate_per_spin`` is a dict ``{chi: rate_array_or_scalar}`` and
      ``p_chi`` is a dict with the same keys carrying weights that sum
      to 1.  This matches the existing notebook ``marginalize(...)``
      pattern and is preserved for back-compatibility.
    - ``rate_per_spin`` is a 1-D array indexed by ``spin_grid`` and
      ``p_chi`` is either a 1-D array of weights (same length as
      ``spin_grid``) or a callable ``p_chi(chi)`` that returns the
      probability density at each grid point.  Useful when
      integrating against an explicit prior PDF.

    Parameters
    ----------
    rate_per_spin : dict or array
        Per-spin rates.  See above for input modes.
    p_chi : dict, array, or callable
        Spin-prior weights (sum to 1) or PDF.
    spin_grid : 1-D array, optional
        Required when ``rate_per_spin`` is an array.

    Returns
    -------
    rate_marginalised : float or ndarray
        Same shape as the elements of ``rate_per_spin`` (typically a
        rate vs. redshift array).

    Notes
    -----
    Caveats from Fuller and Ma (2019, ApJL 881, L1, arXiv:1907.03714):
    their efficient angular momentum transport gives natal BH spins
    ``chi <~ 0.1`` for stellar-mass BHs from massive single stars.
    For BHs in tight binaries, tidal spin-up after the first SN
    (Kushnir+ 2016) or accretion (Bavera+ 2020) can push the second-
    born BH to higher spins; this function does not model that and
    expects the caller to supply the appropriate PDF for the channel
    of interest.
    """
    if isinstance(rate_per_spin, dict):
        if not isinstance(p_chi, dict):
            raise TypeError(
                "When rate_per_spin is a dict, p_chi must also be a dict "
                "with matching keys."
            )
        # Sum weighted by p_chi[chi]; np.asarray handles both scalar
        # and array elements uniformly.
        return sum(p_chi[chi] * np.asarray(rate_per_spin[chi])
                   for chi in rate_per_spin)

    rate_arr = np.asarray(rate_per_spin)
    if spin_grid is None:
        raise ValueError(
            "spin_grid is required when rate_per_spin is an array."
        )
    spin_grid = np.asarray(spin_grid, dtype=float)
    if callable(p_chi):
        weights = np.asarray([p_chi(chi) for chi in spin_grid], dtype=float)
        # Trapezoidal integration over the spin grid for a continuous PDF.
        # Falls back to discrete sum if the grid is single-point.
        if spin_grid.size > 1:
            integrand = weights[:, None] * rate_arr if rate_arr.ndim > 1 \
                else weights * rate_arr
            # numpy>=2 prefers ``trapezoid``; fall back to ``trapz`` for 1.x.
            trap = getattr(np, 'trapezoid', None) or np.trapz
            return trap(integrand, spin_grid, axis=0)
        return weights[0] * rate_arr[0]
    weights = np.asarray(p_chi, dtype=float)
    if weights.shape != spin_grid.shape:
        raise ValueError(
            f"p_chi shape {weights.shape} does not match spin_grid "
            f"shape {spin_grid.shape}."
        )
    if rate_arr.ndim == 1:
        return float(np.sum(weights * rate_arr))
    return np.sum(weights[:, None] * rate_arr, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# BHNS spin-orbit misalignment population correction
# ═══════════════════════════════════════════════════════════════════════════
def apply_bhns_misalignment(rate_bhns, factor=MISALIGNMENT_SYSTEMATIC_FACTOR):
    """Multiplicative population-averaged BHNS GRB rate suppression.

    Folds in the systematic that ~50% of BHNS systems have spin-orbit
    misalignment large enough (> 45-60 deg) to suppress disk-jet
    formation.  Population synthesis (Fragos+ 2010, arXiv:1001.1107;
    Gerosa+ 2018) plus NR results (Kawaguchi+ 2015) motivate a
    population-averaged factor-of-2 reduction of BHNS GRB rates.

    This helper is the canonical use-site of
    ``grb_physics.MISALIGNMENT_SYSTEMATIC_FACTOR``; do NOT also apply
    the per-system ``effective_aligned_spin`` projection on the same
    population (that would double-count the suppression).

    Parameters
    ----------
    rate_bhns : float or array
        BHNS merger / GRB rate(s) to correct (any units; multiplicative).
    factor : float, optional
        Suppression factor, defaulting to
        ``MISALIGNMENT_SYSTEMATIC_FACTOR`` (= 0.5).

    Returns
    -------
    Same shape/units as ``rate_bhns``, scaled by ``factor``.
    """
    return np.asarray(rate_bhns) * factor


# ═══════════════════════════════════════════════════════════════════════════
# Fraction helpers
# ═══════════════════════════════════════════════════════════════════════════
def frac4(r_s, r_l, r_bs, r_bl):
    """Four-component percentage fractions, NaN where total is zero."""
    tot = np.where((r_s + r_l + r_bs + r_bl) > 0,
                    r_s + r_l + r_bs + r_bl, np.nan)
    return r_s / tot * 100, r_l / tot * 100, r_bs / tot * 100, r_bl / tot * 100


def rate_label(val):
    """Format a rate value: integer for >= 1, two decimals otherwise."""
    return f'{val:,.0f}' if val >= 1 else f'{val:.2f}'


# ═══════════════════════════════════════════════════════════════════════════
# Kroupa IMF verification
# ═══════════════════════════════════════════════════════════════════════════
def kroupa_imf(m):
    """Un-normalized Kroupa (2001) three-segment IMF (scalar or array).

    Power-law exponents and breakpoints from Kroupa (2001), MNRAS 322,
    231, Eq. 2: alpha_1 = 0.3 below 0.08 Msun, alpha_2 = 1.3 between 0.08
    and 0.5 Msun, alpha_3 = 2.3 above 0.5 Msun (= Salpeter 1955).
    Continuity coefficients (1.0, 0.08, 0.04) match the piecewise
    definition in Eq. 2 so the segments join without a step.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    result = np.piecewise(m,
        [m < 0.08, (m >= 0.08) & (m < 0.5), m >= 0.5],
        [lambda m: m**(-0.3),         # Kroupa 2001 Eq. 2: alpha_1 = 0.3
         lambda m: 0.08 * m**(-1.3),  # Kroupa 2001 Eq. 2: alpha_2 = 1.3
         lambda m: 0.04 * m**(-2.3)]) # Kroupa 2001 Eq. 2: alpha_3 = 2.3
    return float(result) if result.ndim == 0 or result.size == 1 else result


def verify_mean_mass_evolved(m_lo_full=0.01, m_hi_full=200.0,
                              m_lo_prim=5.0, m_hi_prim=150.0,
                              mean_mass_evolved=None):
    """Analytically verify a MEAN_MASS_EVOLVED value via Kroupa IMF.

    Parameters
    ----------
    mean_mass_evolved : float
        Value to check.  Use ``calibrate_mean_mass_evolved()`` to derive
        per-population values.
    """
    if mean_mass_evolved is None:
        raise ValueError(
            "mean_mass_evolved must be provided; use "
            "calibrate_mean_mass_evolved() to derive per-population values"
        )

    total_mass, _  = quad(lambda m: m * kroupa_imf(m), m_lo_full, m_hi_full)
    total_number, _ = quad(kroupa_imf, m_lo_full, m_hi_full)
    mean_star_mass = total_mass / total_number

    n_primary, _ = quad(kroupa_imf, m_lo_prim, m_hi_prim)
    f_primary    = n_primary / total_number

    mass_per_drawn = mean_star_mass / f_primary

    return {
        'mean_star_mass': mean_star_mass,
        'f_primary': f_primary,
        'mass_per_drawn_primary': mass_per_drawn,
        'N_sim_implied': mean_mass_evolved / mass_per_drawn,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EOS sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════
def compute_eos_sensitivity(m1, m2, weights, eos_models=None,
                            k_thresh=None, q_thresh=None,
                            hmns_factor=None):
    """Per-EOS Gottlieb (2024) class fractions for a single BNS sample.

    Closes Council Expansionist L2.  Sweeps each EOS in ``eos_models``
    by passing its ``M_TOV`` to ``classify_bns_2024`` together with the
    coupling rule ``M_thresh = k_thresh * M_TOV`` (the EOS sweep
    coherence invariant from CLAUDE.md).  For each EOS the function
    returns the STROOPWAFEL-weighted fraction of systems in each of
    the four Gottlieb (2024) classes.

    Parameters
    ----------
    m1, m2 : 1-D array
        Component masses [Msun] (any ordering).
    weights : 1-D array
        STROOPWAFEL weights aligned with ``m1`` / ``m2``.
    eos_models : dict, optional
        Sub-dict of ``grb_physics.EOS_MODELS`` (defaults to all four:
        APR4, SFHo, LS220, DD2).  Each value must carry ``M_TOV`` and
        optionally ``R_1p4``.
    k_thresh : float, optional
        Ratio ``M_thresh / M_TOV`` to apply to every EOS.  Defaults to
        ``grb_physics.K_THRESH_DEFAULT`` (1.27).
    q_thresh : float, optional
        Mass-ratio threshold; defaults to ``grb_physics.Q_THRESH_BNS``.
    hmns_factor : float, optional
        Long-/short-lived HMNS split multiplier; defaults to
        ``grb_physics.HMNS_FACTOR_DEFAULT``.

    Returns
    -------
    pandas.DataFrame
        Indexed by EOS name with columns
        ``[M_TOV, R_1p4, M_thresh, hmns_split, total_weight]`` plus
        one column per Gottlieb class carrying the weighted fraction
        (in [0, 1]).  The class fractions in each row sum to 1.
    """
    import pandas as pd

    from grb_physics import (
        EOS_MODELS as _EOS, K_THRESH_DEFAULT, Q_THRESH_BNS,
        HMNS_FACTOR_DEFAULT,
    )
    from grb_classify import classify_bns_2024

    if eos_models is None:
        eos_models = _EOS
    if k_thresh is None:
        k_thresh = K_THRESH_DEFAULT
    if q_thresh is None:
        q_thresh = Q_THRESH_BNS
    if hmns_factor is None:
        hmns_factor = HMNS_FACTOR_DEFAULT

    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    w = np.asarray(weights, dtype=float)
    w_total = float(w.sum())
    if w_total <= 0:
        raise ValueError("weights sum to zero; cannot compute fractions.")

    rows = []
    for name, eos in eos_models.items():
        m_tov = float(eos['M_TOV'])
        cls = classify_bns_2024(m1, m2, m_tov=m_tov, m_thresh=None,
                                k_thresh=k_thresh, q_thresh=q_thresh,
                                hmns_factor=hmns_factor)
        row = {
            'EOS':           name,
            'M_TOV':         m_tov,
            'R_1p4':         float(eos.get('R_1p4', np.nan)),
            'M_thresh':      k_thresh * m_tov,
            'hmns_split':    hmns_factor * m_tov,
            'total_weight':  w_total,
        }
        for label, mask in cls.items():
            row[label] = float(w[mask].sum() / w_total)
        rows.append(row)
    return pd.DataFrame(rows).set_index('EOS')


# ═══════════════════════════════════════════════════════════════════════════
# M_crit sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════
def mcrit_sweep(M_tot, q, w_all, M_crit_range=None, q_thresh=1.2):
    """Weighted fractions of Short-I, Short-II, Long vs M_crit.

    Returns arrays (frac_short_I, frac_short_II, frac_long) each of
    shape (len(M_crit_range),).
    """
    if M_crit_range is None:
        # Bauswein+ 2013 (PRD 88, 043009, Eq. 6) gives EOS-dependent
        # M_thresh in the range ~2.6-3.0 Msun for soft-to-stiff EOSs
        # (APR4 -> DD2).  We extend the lower edge to 2.3 to bracket the
        # Raaijmakers+ 2021 (arXiv:2105.06981) M_TOV posterior tail and
        # the upper edge to 3.5 to cover stiff Read+ 2009 EOSs.  50 points
        # is dense enough to resolve the steep frac transitions near the
        # fiducial M_thresh ~ 2.8 Msun.
        M_crit_range = np.linspace(2.3, 3.5, 50)

    w_tot = np.sum(w_all)
    frac_I, frac_II, frac_L = [], [], []
    for Mc in M_crit_range:
        s_I  = (M_tot < Mc)
        s_II = (M_tot >= Mc) & (q < q_thresh)
        lon  = (M_tot >= Mc) & (q >= q_thresh)
        frac_I.append(np.sum(w_all[s_I]) / w_tot)
        frac_II.append(np.sum(w_all[s_II]) / w_tot)
        frac_L.append(np.sum(w_all[lon]) / w_tot)

    return (np.array(frac_I), np.array(frac_II), np.array(frac_L),
            M_crit_range)


# ═══════════════════════════════════════════════════════════════════════════
# Beaming correction
# ═══════════════════════════════════════════════════════════════════════════
CLASS_THETA_J = {
    'sbGRB': {'lo': 10.0, 'fid': 13.0, 'hi': 16.0},
    'lbGRB': {'lo': 5.0, 'fid': 6.5, 'hi': 8.0},
}
"""Class-dependent jet half-opening angles [deg].

sbGRB: fiducial prior range, NOT directly the Fong+ 2015 (ApJ 815, 102)
measurement.  Fong+ 2015 report a median theta_j = 16 +/- 10 deg over
11 sGRBs (so a symmetric 1-sigma band would extend ~6 to 26 deg).  The
narrower 10 to 16 deg band used here is informed by the Beniamini and
Nakar (2019, MNRAS 482, 5430, arXiv:1812.02194) structured-jet
analysis of GW170817 and the population reanalysis they recommend
(typical core opening angles closer to 10 deg with structured wings).
lbGRB: Gottlieb (2023) argues MAD-powered BH jets from lbGRB sources
have narrower collimation than HMNS-powered sbGRB jets.
"""


def check_dPdlogZ_normalization(dPdlogZ, metallicities, rtol=0.05):
    """Verify dP/dlogZ integrates to ~1 at every redshift slice.

    Convention: ``dPdlogZ`` is dP/d(ln Z) (natural log), consistent with
    COMPAS ``find_metallicity_distribution``.  The integration uses
    Delta(ln Z) = np.diff(np.log(Z)), NOT Delta(log10 Z).
    If your dPdlogZ is per d(log10 Z), multiply by ln(10) first.

    Returns the per-slice integral array.  Raises ValueError if any
    slice deviates by more than *rtol* from unity.
    """
    dlogZ = np.diff(np.log(metallicities))
    dlogZ = np.append(dlogZ, dlogZ[-1])
    norm = (dPdlogZ * dlogZ[None, :]).sum(axis=1)
    if np.any(np.abs(norm - 1.0) > rtol):
        bad = np.where(np.abs(norm - 1.0) > rtol)[0]
        raise ValueError(
            f"dPdlogZ integral deviates from 1 at {len(bad)} z-slices "
            f"(range {norm[bad].min():.4f}–{norm[bad].max():.4f})")
    return norm


def observed_frame_rate(rate_intrinsic_z, redshifts):
    """Convert source-frame intrinsic rate density to detector frame.

    Closes Council Expansionist L6 by giving the LIGO O4 cross-check
    its missing helper.  All ``compute_merger_rate`` outputs are
    intrinsic (source-frame) merger rate densities ``R_int(z)``
    [Gpc^-3 yr^-1, source frame].  The detector-frame (observer-frame)
    rate density is

        R_det(z) = R_int(z) / (1 + z)

    where the ``1 / (1 + z)`` factor accounts for the cosmological
    time dilation of the source-frame yr to the detector frame.  The
    Gpc^-3 volume element is comoving and is therefore unchanged.

    Parameters
    ----------
    rate_intrinsic_z : 1-D array
        Intrinsic source-frame merger rate density vs redshift
        [Gpc^-3 yr^-1].
    redshifts : 1-D array
        Redshift grid aligned with ``rate_intrinsic_z``.

    Returns
    -------
    rate_detector_z : 1-D array
        Detector-frame rate density [Gpc^-3 yr^-1].

    Notes
    -----
    To compare with LIGO O4 BNS / BHNS rate posteriors that are quoted
    as a single local-volume number (e.g. Abbott+ GWTC), evaluate the
    detector-frame rate at z = 0 (where ``1 / (1 + z) = 1`` so
    ``R_det(0) = R_int(0)``) and against the LIGO posterior median
    value.  This helper exists for the redshift-resolved comparison
    against the published O4 R(z) curves, where the (1 + z)^-1 factor
    matters.
    """
    z = np.asarray(redshifts, dtype=float)
    R = np.asarray(rate_intrinsic_z, dtype=float)
    return R / (1.0 + z)


def beamed_class_comparison(rate_intrinsic_by_class, theta_j_deg_by_class=None,
                             observed_by_class=None):
    """Per-class table of intrinsic, beamed, and observed sGRB rates.

    Closes Council Expansionist L7.  Wraps ``beamed_rate`` and joins
    against ``OBSERVED_RATES_BY_CLASS`` so the analyst gets a single
    DataFrame that answers "for each Gottlieb (2024) class, what does
    the model predict (intrinsic and beamed) and how does that compare
    to the closest observed handle?".

    Parameters
    ----------
    rate_intrinsic_by_class : dict[str, float]
        Intrinsic per-class local merger rate density at z = 0
        [Gpc^-3 yr^-1].  Keys are Gottlieb (2024) class labels, e.g.
        'sbGRB + blue KN'.
    theta_j_deg_by_class : dict[str, float], optional
        Per-class jet half-opening angle [deg].  When None, derive from
        ``CLASS_THETA_J``: 'sbGRB' fid for the sbGRB class, 'lbGRB' fid
        for the three lbGRB classes.
    observed_by_class : dict, optional
        Per-class observed rate references.  Defaults to
        ``OBSERVED_RATES_BY_CLASS``.

    Returns
    -------
    pandas.DataFrame
        Indexed by Gottlieb class label with columns
        ``[R_intrinsic, theta_j_deg, f_beam, R_beamed, R_obs,
           R_obs_lo, R_obs_hi, reference]``.
    """
    import pandas as pd

    if observed_by_class is None:
        observed_by_class = OBSERVED_RATES_BY_CLASS

    if theta_j_deg_by_class is None:
        theta_j_deg_by_class = {
            'sbGRB + blue KN':       CLASS_THETA_J['sbGRB']['fid'],
            'lbGRB + red KN (HMNS)': CLASS_THETA_J['lbGRB']['fid'],
            'lbGRB + red KN (disk)': CLASS_THETA_J['lbGRB']['fid'],
            'Faint lbGRB':           CLASS_THETA_J['lbGRB']['fid'],
        }

    rows = []
    for label, R_int in rate_intrinsic_by_class.items():
        theta = theta_j_deg_by_class.get(label, np.nan)
        f_beam = 1.0 - np.cos(np.radians(theta)) if np.isfinite(theta) else np.nan
        R_beamed = R_int * f_beam if np.isfinite(f_beam) else np.nan
        obs = observed_by_class.get(label, {})
        rows.append({
            'class':         label,
            'R_intrinsic':   float(R_int),
            'theta_j_deg':   float(theta),
            'f_beam':        float(f_beam) if np.isfinite(f_beam) else np.nan,
            'R_beamed':      float(R_beamed) if np.isfinite(R_beamed) else np.nan,
            'R_obs':         float(obs.get('R_obs', np.nan)),
            'R_obs_lo':      float(obs.get('R_obs_lo', np.nan)),
            'R_obs_hi':      float(obs.get('R_obs_hi', np.nan)),
            'reference':     obs.get('reference', ''),
        })
    return pd.DataFrame(rows).set_index('class')


def beamed_rate(rate_intrinsic, theta_j_deg):
    """Convert intrinsic merger rate to observer-frame GRB rate.

    f_beam = 1 - cos(theta_j) is the fraction of the sky subtended
    by the two-sided jet cone.  Fong+ 2015 (ApJ 815, 102) report a
    median sGRB theta_j = 16 +/- 10 deg over 11 bursts; we adopt a
    narrower 10-16 deg fiducial band informed by the Beniamini and
    Nakar (2019, MNRAS 482, 5430) structured-jet reanalysis,
    yielding f_beam ~ 0.015-0.04 (only ~2-4% of jets visible).

    Convention: R_obs = R_intrinsic * f_beam.
    To invert an observed rate to intrinsic: R_intrinsic = R_obs / f_beam.
    Fong+ (2015) quote f_b = 1 - cos(theta_j) with the same convention.

    Parameters
    ----------
    rate_intrinsic : float or array
        Intrinsic (all-sky) merger rate density [Gpc^-3 yr^-1].
    theta_j_deg : float
        Half-opening angle of the jet [degrees].

    Returns
    -------
    Observable rate density [Gpc^-3 yr^-1].
    """
    theta_j = np.radians(theta_j_deg)
    f_beam = 1.0 - np.cos(theta_j)
    return rate_intrinsic * f_beam


def beamed_rate_mixed(rates_by_class, theta_j_deg_by_class):
    """Class-weighted observed-frame rate for a mixed-class population.

    For a population that mixes GRB classes (sbGRB, lbGRB, ...), the
    observable rate is NOT ``beamed_rate(R_total, <theta_j>)``: each class
    has its own jet half-opening angle, so the per-class beaming factors
    must be applied separately and then summed.  This helper enforces
    that.

    Parameters
    ----------
    rates_by_class : dict {str: float or array}
        Intrinsic merger / GRB rate per class [Gpc^-3 yr^-1], e.g.
        ``{'sbGRB': R_sb, 'lbGRB': R_lb}``.
    theta_j_deg_by_class : dict {str: float}
        Jet half-opening angle per class [deg].  Typical fiducial values
        come from ``CLASS_THETA_J[name]['fid']``.

    Returns
    -------
    R_obs : float or array
        Observed-frame rate, ``sum_c R_c * (1 - cos(theta_j_c))``.

    Raises
    ------
    KeyError
        If a class in ``rates_by_class`` has no matching opening angle.
    """
    missing = set(rates_by_class) - set(theta_j_deg_by_class)
    if missing:
        raise KeyError(
            f"theta_j_deg_by_class is missing entries for {sorted(missing)}; "
            f"every rate class must have a corresponding jet angle."
        )
    total = 0.0
    for name, R in rates_by_class.items():
        total = total + beamed_rate(R, theta_j_deg_by_class[name])
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Observed sGRB rate reference data
# ═══════════════════════════════════════════════════════════════════════════
# All rates are *observed* (beaming-limited) local values [Gpc^-3 yr^-1].
OBSERVED_SGRB_RATES = {
    'Wanderman & Piran 2015': {
        'R_obs': 4.1, 'R_obs_lo': 2.2, 'R_obs_hi': 6.4,
        'note': 'MNRAS 448, 3026; intrinsic ~270 at theta_j ~10 deg',
    },
    'Ghirlanda+ 2016': {
        'R_obs': 1.3, 'R_obs_lo': 0.5, 'R_obs_hi': 3.0,
        'note': 'A&A 594, A84; Fermi/GBM, intrinsic ~200-700 after beaming',
    },
    'Colombo+ 2022': {
        'R_obs': 3.6, 'R_obs_lo': 1.8, 'R_obs_hi': 6.5,
        'note': 'ApJ 937, 79; Fermi/GBM update',
    },
}


# Per-class observed rate references for the Council Expansionist L7
# beaming comparator.  Each entry is the best literature handle for an
# observed (beaming-limited, local) rate of GRBs that arguably belong
# to the corresponding Gottlieb (2024) class.  These are NOT
# class-by-class published numbers in any single paper -- they are
# class-appropriate references that the analyst maps to the closest
# observable population for an apples-to-apples comparison.  The
# 'caveat' field documents the mapping choice for each class.
OBSERVED_RATES_BY_CLASS = {
    'sbGRB + blue KN': {
        'R_obs': OBSERVED_SGRB_RATES['Colombo+ 2022']['R_obs'],
        'R_obs_lo': OBSERVED_SGRB_RATES['Colombo+ 2022']['R_obs_lo'],
        'R_obs_hi': OBSERVED_SGRB_RATES['Colombo+ 2022']['R_obs_hi'],
        'reference': 'Colombo+ 2022, ApJ 937, 79',
        'caveat': 'Observed sGRB rate; sbGRBs dominate the local-volume '
                  'short-GRB population.',
    },
    'lbGRB + red KN (HMNS)': {
        'R_obs': OBSERVED_SGRB_RATES['Ghirlanda+ 2016']['R_obs'],
        'R_obs_lo': OBSERVED_SGRB_RATES['Ghirlanda+ 2016']['R_obs_lo'],
        'R_obs_hi': OBSERVED_SGRB_RATES['Ghirlanda+ 2016']['R_obs_hi'],
        'reference': 'Ghirlanda+ 2016, A&A 594, A84',
        'caveat': 'No dedicated observed lbGRB+HMNS rate in the literature; '
                  'mapped to the lower-luminosity end of the observed sGRB '
                  'distribution (Ghirlanda+ 2016 GBM sample).',
    },
    'lbGRB + red KN (disk)': {
        'R_obs': 1.0, 'R_obs_lo': 0.3, 'R_obs_hi': 2.0,
        'reference': 'Levina+ 2026, arXiv:2601.20202 (per-class '
                     'split estimate)',
        'caveat': 'Levina+ 2026 estimate of the disk-driven long-merger '
                  'GRB rate from population synthesis; observational '
                  'identification is still emerging (Rastinejad+ 2022 '
                  'GRB 211211A class). NOT a directly measured rate.',
    },
    'Faint lbGRB': {
        'R_obs': np.nan, 'R_obs_lo': np.nan, 'R_obs_hi': np.nan,
        'reference': 'No published observed rate',
        'caveat': 'Faint lbGRBs (small disk, prompt collapse, q < q_thresh) '
                  'are below current GBM sensitivity for almost all '
                  'plausible viewing angles. Treat as upper limit < 1 '
                  'Gpc^-3 yr^-1.',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Observed sGRB rate vs redshift: Wanderman & Piran (2015)
# ═══════════════════════════════════════════════════════════════════════════
def wanderman_piran_2015_Rz(z, R0=4.1, z_peak=0.9, sigma_lo=0.39, sigma_hi=0.26,
                             R0_lo=2.2, R0_hi=6.4):
    """Observed sGRB comoving rate density vs redshift from
    Wanderman & Piran (2015), MNRAS 448, 3026.

    Implements the piecewise-exponential form from WP15 Eq. (9):

        R(z) = R0 * exp(+(z - z_peak) / sigma_lo)   for z <= z_peak
        R(z) = R0 * exp(-(z - z_peak) / sigma_hi)   for z >  z_peak

    with z_peak = 0.9, sigma_lo = 0.39 (rising), sigma_hi = 0.26
    (falling).  R0 = 4.1 +2.3/-1.9 Gpc^-3 yr^-1 is the observed,
    beaming-limited normalization at the peak.  This is the comoving
    rate density of *detectable* sGRBs (luminosity-function- and
    beaming-folded), not the intrinsic merger rate, so it is the
    appropriate target for ``beamed_rate(...)`` model curves.

    NOTE: An earlier revision of this function used a broken
    power-law (1 + z)^n1 / (1 + z)^n2 attributed to "Table 2".  WP15
    Table 2 actually lists luminosity-function parameters; the R(z)
    form is the exponential above (their Eq. 9).  The previous
    parametrization was a re-fit, not a direct WP15 quote.

    The uncertainty band ``R_lo`` / ``R_hi`` varies only the
    normalization R0; the shape parameters' uncertainties are
    correlated with the luminosity function in the original fit and
    are not propagated here.

    Returns dict with 'R_best', 'R_lo', 'R_hi' arrays (same shape as z).
    """
    z = np.asarray(z, dtype=float)
    dz = z - z_peak
    shape = np.where(dz <= 0.0,
                     np.exp(+dz / sigma_lo),
                     np.exp(-dz / sigma_hi))
    return {
        'R_best': R0    * shape,
        'R_lo':   R0_lo * shape,
        'R_hi':   R0_hi * shape,
    }
