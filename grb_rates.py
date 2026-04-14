"""
Cosmic merger rate computation and related utilities.

compute_merger_rate implements the MSSFR convolution from Neijssel et al. (2019)
using COMPAS FastCosmicIntegration infrastructure.  Also includes Kroupa IMF
verification, per-system rate weights, and BH spin marginalization.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


# ═══════════════════════════════════════════════════════════════════════════
# Cosmic integration
# ═══════════════════════════════════════════════════════════════════════════
def compute_merger_rate(redshifts, times, time_first_SF, n_formed,
                        dPdlogZ, metallicities, p_draw,
                        COMPAS_Z, COMPAS_delay_times, COMPAS_weights):
    """
    Intrinsic merger rate density [Gpc^-3 yr^-1] vs redshift.

    For each binary the formation rate is:
        SFR(z) * dP/dlogZ(z, Z_i) / p_draw * weight_i / meanMassEvolved

    The merger rate at z_merge equals the formation rate evaluated at the
    redshift when the binary was born (z_form corresponding to
    t = t(z_merge) - delay_time).

    Important: ``n_formed`` must already contain the 1/MEAN_MASS_EVOLVED
    normalisation.  ``find_sfr()`` returns *raw* SFR in Msun/yr/Gpc^3 and
    does NOT include the 1/MEAN_MASS_EVOLVED factor.  Divide explicitly::

        n_formed = find_sfr(redshifts) / MEAN_MASS_EVOLVED

    Each population (BNS, BHNS) has its own MEAN_MASS_EVOLVED because
    the Broekgaarden et al. simulations evolve different total masses.
    """
    n_z           = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]
    times_to_z    = interp1d(times, redshifts)

    Z_bins = np.clip(np.digitize(COMPAS_Z, metallicities),
                     0, len(metallicities) - 1)

    t_min = max(time_first_SF, times.min())
    total_merger = np.zeros(n_z)

    for i in range(len(COMPAS_delay_times)):
        form_i = n_formed * dPdlogZ[:, Z_bins[i]] / p_draw * COMPAS_weights[i]
        t_form = times - COMPAS_delay_times[i]

        valid = (t_form >= t_min)
        if not valid.any():
            continue

        j_idx      = np.where(valid)[0]
        z_form     = times_to_z(t_form[j_idx])

        z_idx_float = z_form / redshift_step
        z_lo = np.clip(np.floor(z_idx_float).astype(int), 0, n_z - 1)
        z_hi = np.clip(z_lo + 1, 0, n_z - 1)
        frac = z_idx_float - np.floor(z_idx_float)
        total_merger[j_idx] += form_i[z_lo] * (1.0 - frac) + form_i[z_hi] * frac

    return total_merger


def per_system_rate_weights(z_target, redshifts, times, time_first_SF,
                            n_formed, dPdlogZ, metallicities, p_draw,
                            COMPAS_Z, COMPAS_delay_times, COMPAS_weights):
    """
    Per-system contribution to the merger rate at a single z_target.

    Same physics as compute_merger_rate but returns an array of individual
    weights (one per binary) for constructing rate-weighted histograms.
    """
    n_z           = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]
    times_to_z    = interp1d(times, redshifts)

    j_target = np.argmin(np.abs(redshifts - z_target))
    t_merge  = times[j_target]

    Z_bins = np.clip(np.digitize(COMPAS_Z, metallicities),
                     0, len(metallicities) - 1)
    t_min  = max(time_first_SF, times.min())

    out    = np.zeros(len(COMPAS_weights))
    t_form = t_merge - COMPAS_delay_times
    valid  = t_form >= t_min

    if valid.any():
        z_form      = times_to_z(t_form[valid])
        z_idx_float = z_form / redshift_step
        z_lo = np.clip(np.floor(z_idx_float).astype(int), 0, n_z - 1)
        z_hi = np.clip(z_lo + 1, 0, n_z - 1)
        frac = z_idx_float - np.floor(z_idx_float)

        val_lo = (n_formed[z_lo] * dPdlogZ[z_lo, Z_bins[valid]]
                  / p_draw * COMPAS_weights[valid])
        val_hi = (n_formed[z_hi] * dPdlogZ[z_hi, Z_bins[valid]]
                  / p_draw * COMPAS_weights[valid])
        out[valid] = val_lo * (1.0 - frac) + val_hi * frac
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Per-population normalization
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_mean_mass_evolved(sfr, redshifts, times, time_first_SF,
                                dPdlogZ, metallicities, p_draw,
                                COMPAS_Z, COMPAS_delay_times, COMPAS_weights,
                                expected_local_rate):
    """Derive the effective MEAN_MASS_EVOLVED for one population.

    Runs ``compute_merger_rate`` with ``n_formed = sfr`` (unit
    normalization) and scales so the z = 0 rate matches
    *expected_local_rate* (from pre-computed ``w_000`` weights).

    Returns
    -------
    mean_mass_evolved : float
        The total stellar mass [Msun] evolved in the simulation.
    rate_unnorm : 1-D array
        Un-normalized merger rate; divide by *mean_mass_evolved* to get
        the correctly normalized R(z).
    """
    rate_unnorm = compute_merger_rate(
        redshifts, times, time_first_SF, sfr,
        dPdlogZ, metallicities, p_draw,
        COMPAS_Z, COMPAS_delay_times, COMPAS_weights)
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
    """Weighted average over spin values: sum(w_a * rate[a])."""
    return sum(weights[a] * rate_dict[a] for a in rate_dict)


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
    """Un-normalized Kroupa (2001) three-segment IMF (scalar or array)."""
    m = np.atleast_1d(np.asarray(m, dtype=float))
    result = np.piecewise(m,
        [m < 0.08, (m >= 0.08) & (m < 0.5), m >= 0.5],
        [lambda m: m**(-0.3),
         lambda m: 0.08 * m**(-1.3),
         lambda m: 0.04 * m**(-2.3)])
    return float(result) if result.ndim == 0 or result.size == 1 else result


def verify_mean_mass_evolved(m_lo_full=0.01, m_hi_full=200.0,
                              m_lo_prim=5.0, m_hi_prim=150.0,
                              mean_mass_evolved=None):
    """Analytically verify a MEAN_MASS_EVOLVED value via Kroupa IMF.

    Parameters
    ----------
    mean_mass_evolved : float, optional
        Value to check.  If None, uses the deprecated legacy constant
        from grb_physics (for back-compat diagnostics only).
    """
    if mean_mass_evolved is None:
        from grb_physics import MEAN_MASS_EVOLVED
        mean_mass_evolved = MEAN_MASS_EVOLVED

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
# M_crit sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════
def mcrit_sweep(M_tot, q, w_all, M_crit_range=None, q_thresh=1.2):
    """Weighted fractions of Short-I, Short-II, Long vs M_crit.

    Returns arrays (frac_short_I, frac_short_II, frac_long) each of
    shape (len(M_crit_range),).
    """
    if M_crit_range is None:
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

sbGRB: Fong+ 2015 (ApJ 815, 102), Beniamini & Nakar 2019 (MNRAS 482, 5430).
lbGRB: Gottlieb (2023) argues MAD-powered BH jets from lbGRB sources have
narrower collimation than HMNS-powered sbGRB jets.
"""


def check_dPdlogZ_normalization(dPdlogZ, metallicities, rtol=0.05):
    """Verify dP/dlogZ integrates to ~1 at every redshift slice.

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


def beamed_rate(rate_intrinsic, theta_j_deg):
    """Convert intrinsic merger rate to observer-frame GRB rate.

    f_beam = 1 - cos(theta_j) is the fraction of the sky subtended
    by the two-sided jet cone.  Typical sGRB jets have theta_j ~ 10-16 deg
    (Fong+ 2015, ApJ 815, 102; Beniamini & Nakar 2019, MNRAS 482, 5430),
    giving f_beam ~ 0.015-0.04 (i.e. only ~2-4% of jets are visible).

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


# ═══════════════════════════════════════════════════════════════════════════
# Observed sGRB rate vs redshift: Wanderman & Piran (2015)
# ═══════════════════════════════════════════════════════════════════════════
def wanderman_piran_2015_Rz(z, R0=4.1, n1=1.7, n2=-0.6, z_peak=0.9,
                             R0_lo=2.2, R0_hi=6.4,
                             n1_lo=1.3, n1_hi=2.2):
    """Observed sGRB rate density vs redshift from Wanderman & Piran (2015).

    Broken power-law fit from MNRAS 448, 3026, Table 2:
        R(z) = R0 * (1+z)^n1          for z <= z_peak
        R(z) = R0 * A * (1+z)^n2      for z > z_peak
    where A = (1+z_peak)^(n1-n2) ensures continuity at z_peak.

    Best-fit parameters: R0 ~ 4.1 +2.3/-1.9 Gpc^-3 yr^-1 (observed,
    beaming-limited), n1 = 1.7 +0.5/-0.4, z_peak ~ 0.9, n2 ~ -0.6.

    The uncertainty band varies *both* R0 and n1 so the envelope
    shape changes with redshift (not just the normalization).

    Returns dict with 'R_best', 'R_lo', 'R_hi' arrays (same shape as z).
    """
    z = np.asarray(z, dtype=float)

    def _shape(n1_val):
        A = (1.0 + z_peak) ** (n1_val - n2)
        return np.where(z <= z_peak,
                        (1.0 + z) ** n1_val,
                        A * (1.0 + z) ** n2)

    return {
        'R_best': R0    * _shape(n1),
        'R_lo':   R0_lo * _shape(n1_lo),
        'R_hi':   R0_hi * _shape(n1_hi),
    }
