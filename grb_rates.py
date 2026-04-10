"""
Cosmic merger rate computation and related utilities.

compute_merger_rate implements the MSSFR convolution from Neijssel et al. (2019)
using COMPAS FastCosmicIntegration infrastructure.  Also includes Kroupa IMF
verification, per-system rate weights, and BH spin marginalization.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

from grb_physics import MEAN_MASS_EVOLVED


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
    normalisation.  Callers typically pass the output of
    ``FastCosmicIntegration.find_sfrd()``, which includes this factor.
    If you pass raw SFR values, divide by MEAN_MASS_EVOLVED first.
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
        z_form_idx = np.clip(np.ceil(z_form / redshift_step).astype(int),
                             0, n_z - 1)
        total_merger[j_idx] += form_i[z_form_idx]

    return total_merger


def per_system_rate_weights(z_target, redshifts, times, time_first_SF,
                            n_formed, dPdlogZ, metallicities, p_draw,
                            COMPAS_Z, COMPAS_delay_times, COMPAS_weights):
    """
    Per-system contribution to the merger rate at a single z_target.

    Same physics as compute_merger_rate but returns an array of individual
    weights (one per binary) for constructing rate-weighted histograms.
    """
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
        z_form     = times_to_z(t_form[valid])
        z_form_idx = np.clip(np.ceil(z_form / redshift_step).astype(int),
                             0, len(redshifts) - 1)
        out[valid] = (n_formed[z_form_idx]
                      * dPdlogZ[z_form_idx, Z_bins[valid]]
                      / p_draw
                      * COMPAS_weights[valid])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Formation efficiency (per metallicity bin)
# ═══════════════════════════════════════════════════════════════════════════
def formation_efficiency(metallicityGrid, Z_all, w_all, masks=None,
                         mean_mass_evolved=MEAN_MASS_EVOLVED):
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

    Returns
    -------
    dict of 1-D arrays keyed by mask name, plus 'total'.
    """
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
    """Un-normalized Kroupa (2001) three-segment IMF."""
    if m < 0.08:
        return m**(-0.3)
    elif m < 0.5:
        return 0.08 * m**(-1.3)
    else:
        return 0.08 * 0.5 * m**(-2.3)


def verify_mean_mass_evolved(m_lo_full=0.01, m_hi_full=200.0,
                              m_lo_prim=5.0, m_hi_prim=150.0):
    """Analytically verify the MEAN_MASS_EVOLVED constant via Kroupa IMF."""
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
        'N_sim_implied': MEAN_MASS_EVOLVED / mass_per_drawn,
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
