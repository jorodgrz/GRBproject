"""
Data loading and export helpers for COMPAS HDF5 compact-object outputs.

Standardises how BNS and BHNS populations are read from the COMPAS
doubleCompactObjects/formationChannels groups. All loaders filter to
merging systems (mergesInHubbleTimeFlag == 1) and return plain numpy arrays.
"""

import os
import numpy as np
import h5py as h5

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

DEFAULT_BNS_PATH  = os.path.join(_DATA_DIR, 'COMPASCompactOutput_BNS_A.h5')
DEFAULT_BHNS_PATH = os.path.join(_DATA_DIR, 'COMPASCompactOutput_BHNS_A.h5')
DEFAULT_BNS_K_PATH = os.path.join(_DATA_DIR, 'COMPASCompactOutput_BNS_K.h5')


# ═══════════════════════════════════════════════════════════════════════════
# Simulation metadata
# ═══════════════════════════════════════════════════════════════════════════
def read_expected_local_rate(path):
    """Read the pre-computed local (z=0) intrinsic merger rate from a
    Broekgaarden et al. COMPAS HDF5 file.

    Returns the sum of the ``weights_intrinsic/w_000`` column, which
    represents the fiducial-model merger rate density at z = 0 in
    Gpc^-3 yr^-1.  This value is used to calibrate the per-population
    ``MEAN_MASS_EVOLVED`` normalization constant.
    """
    with h5.File(path, 'r') as f:
        return float(f['weights_intrinsic']['w_000'][...].sum())


def read_metallicity_range(path):
    """Return (Z_min, Z_max) of birth metallicities in the HDF5 file."""
    with h5.File(path, 'r') as f:
        Z = f['systems']['Metallicity1'][...].squeeze()
    return float(Z.min()), float(Z.max())


# ═══════════════════════════════════════════════════════════════════════════
# BNS data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_bns(path=None, sort_masses=True):
    """Load merging BNS population from COMPAS HDF5 file.

    Parameters
    ----------
    path : str, optional
        Path to HDF5 file. Defaults to Data/COMPASCompactOutput_BNS_A.h5.
    sort_masses : bool
        If True, m1 >= m2 is enforced (heavier/lighter).

    Returns
    -------
    dict with keys:
        'm1', 'm2'         : component masses [Msun]
        'weights'          : STROOPWAFEL weights
        'metallicity'      : birth metallicity (Metallicity1)
        'delay_time'       : tform + tc [Myr]
        'n_merging'        : number of merging systems
        'mask_merging'     : boolean mask over the full catalogue
    """
    if path is None:
        path = DEFAULT_BNS_PATH

    with h5.File(path, 'r') as f:
        dco = f['doubleCompactObjects']
        m1    = dco['M1'][...].squeeze()
        m2    = dco['M2'][...].squeeze()
        w     = dco['weight'][...].squeeze()
        Z     = dco['Metallicity1'][...].squeeze()
        mh    = dco['mergesInHubbleTimeFlag'][...].squeeze()
        tc    = dco['tc'][...].squeeze()
        tform = dco['tform'][...].squeeze()

    mask = (mh == 1)

    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    return {
        'm1':           m1_out,
        'm2':           m2_out,
        'weights':      w[mask],
        'metallicity':  Z[mask],
        'delay_time':   (tform + tc)[mask],
        'n_merging':    int(mask.sum()),
        'mask_merging': mask,
    }


def load_bns_with_channels(path=None, sort_masses=True):
    """Load merging BNS with formation-channel columns.

    Returns the same dict as load_bns, plus additional keys for the
    formation-channel classification:
        'dblCE', 'fc_CEE', 'fc_mt_p1', 'fc_mt_s1',
        'fc_mt_p1_K1', 'fc_mt_s1_K2',
        'm1zams', 'm2zams', 'sep_preCE', 'sep_postCE'
    """
    if path is None:
        path = DEFAULT_BNS_PATH

    with h5.File(path, 'r') as f:
        dco = f['doubleCompactObjects']
        m1    = dco['M1'][...].squeeze()
        m2    = dco['M2'][...].squeeze()
        w     = dco['weight'][...].squeeze()
        Z     = dco['Metallicity1'][...].squeeze()
        mh    = dco['mergesInHubbleTimeFlag'][...].squeeze()
        tc    = dco['tc'][...].squeeze()
        tform = dco['tform'][...].squeeze()
        m1z   = dco['M1ZAMS'][...].squeeze()
        m2z   = dco['M2ZAMS'][...].squeeze()
        dblCE = dco['doubleCommonEnvelopeFlag'][...].squeeze()
        sep_pre  = dco['SemiMajorAxisPreCEE'][...].squeeze()
        sep_post = dco['SemiMajorAxisPostCEE'][...].squeeze()

        fc = f['formationChannels']
        fc_mt_p1    = fc['mt_primary_ep1'][...].squeeze()
        fc_mt_p1_K1 = fc['mt_primary_ep1_K1'][...].squeeze()
        fc_mt_s1    = fc['mt_secondary_ep1'][...].squeeze()
        fc_mt_s1_K2 = fc['mt_secondary_ep1_K2'][...].squeeze()
        fc_CEE      = fc['CEE'][...].squeeze()

    mask = (mh == 1)

    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    # NOTE: sort_masses swaps m1/m2 so that m1 >= m2, but it does NOT
    # reorder the formation-channel columns (dblCE, fc_*, sep_*, ZAMS
    # masses).  Those columns still reference the *original* COMPAS
    # primary/secondary labelling, not the heavier/lighter compact object.
    # This is correct for formation-channel classification (which follows
    # COMPAS primary/secondary), but callers must not assume that
    # fc_mt_p1 corresponds to the heavier compact remnant.
    return {
        'm1':           m1_out,
        'm2':           m2_out,
        'weights':      w[mask],
        'metallicity':  Z[mask],
        'delay_time':   (tform + tc)[mask],
        'n_merging':    int(mask.sum()),
        'mask_merging': mask,
        'm1zams':       m1z[mask],
        'm2zams':       m2z[mask],
        'dblCE':        dblCE[mask],
        'sep_preCE':    sep_pre[mask],
        'sep_postCE':   sep_post[mask],
        'fc_mt_p1':     fc_mt_p1[mask],
        'fc_mt_p1_K1':  fc_mt_p1_K1[mask],
        'fc_mt_s1':     fc_mt_s1[mask],
        'fc_mt_s1_K2':  fc_mt_s1_K2[mask],
        'fc_CEE':       fc_CEE[mask],
    }


# ═══════════════════════════════════════════════════════════════════════════
# BHNS data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_bhns(path=None):
    """Load merging BHNS population from COMPAS HDF5 file.

    BH/NS identity is resolved using stellarType1 (14 = BH).

    Returns
    -------
    dict with keys:
        'M_BH', 'M_NS'    : component masses [Msun]
        'weights'          : STROOPWAFEL weights
        'metallicity'      : birth metallicity (Metallicity1)
        'delay_time'       : tform + tc [Myr]
        'n_merging'        : number of merging systems
        'mask_merging'     : boolean mask over the full catalogue
    """
    if path is None:
        path = DEFAULT_BHNS_PATH

    with h5.File(path, 'r') as f:
        dco = f['doubleCompactObjects']
        m1    = dco['M1'][...].squeeze()
        m2    = dco['M2'][...].squeeze()
        w     = dco['weight'][...].squeeze()
        Z     = dco['Metallicity1'][...].squeeze()
        mh    = dco['mergesInHubbleTimeFlag'][...].squeeze()
        tc    = dco['tc'][...].squeeze()
        tform = dco['tform'][...].squeeze()
        st1   = dco['stellarType1'][...].squeeze()

    is_BH1 = (st1 == 14)
    M_BH = np.where(is_BH1, m1, m2)
    M_NS = np.where(is_BH1, m2, m1)

    mask = (mh == 1)

    return {
        'M_BH':         M_BH[mask],
        'M_NS':         M_NS[mask],
        'weights':      w[mask],
        'metallicity':  Z[mask],
        'delay_time':   (tform + tc)[mask],
        'n_merging':    int(mask.sum()),
        'mask_merging': mask,
    }


def load_bhns_with_channels(path=None):
    """Load merging BHNS with formation-channel columns.

    Same as load_bhns plus formation-channel keys for classify_formation_channels.
    """
    if path is None:
        path = DEFAULT_BHNS_PATH

    with h5.File(path, 'r') as f:
        dco = f['doubleCompactObjects']
        m1    = dco['M1'][...].squeeze()
        m2    = dco['M2'][...].squeeze()
        w     = dco['weight'][...].squeeze()
        Z     = dco['Metallicity1'][...].squeeze()
        mh    = dco['mergesInHubbleTimeFlag'][...].squeeze()
        tc    = dco['tc'][...].squeeze()
        tform = dco['tform'][...].squeeze()
        st1   = dco['stellarType1'][...].squeeze()
        dblCE = dco['doubleCommonEnvelopeFlag'][...].squeeze()
        sep_pre = dco['SemiMajorAxisPreCEE'][...].squeeze()

        fc = f['formationChannels']
        fc_mt_p1    = fc['mt_primary_ep1'][...].squeeze()
        fc_mt_p1_K1 = fc['mt_primary_ep1_K1'][...].squeeze()
        fc_mt_s1    = fc['mt_secondary_ep1'][...].squeeze()
        fc_mt_s1_K2 = fc['mt_secondary_ep1_K2'][...].squeeze()
        fc_CEE      = fc['CEE'][...].squeeze()

    is_BH1 = (st1 == 14)
    M_BH = np.where(is_BH1, m1, m2)
    M_NS = np.where(is_BH1, m2, m1)

    mask = (mh == 1)

    return {
        'M_BH':         M_BH[mask],
        'M_NS':         M_NS[mask],
        'weights':      w[mask],
        'metallicity':  Z[mask],
        'delay_time':   (tform + tc)[mask],
        'n_merging':    int(mask.sum()),
        'mask_merging': mask,
        'dblCE':        dblCE[mask],
        'sep_preCE':    sep_pre[mask],
        'fc_mt_p1':     fc_mt_p1[mask],
        'fc_mt_p1_K1':  fc_mt_p1_K1[mask],
        'fc_mt_s1':     fc_mt_s1[mask],
        'fc_mt_s1_K2':  fc_mt_s1_K2[mask],
        'fc_CEE':       fc_CEE[mask],
    }


# ═══════════════════════════════════════════════════════════════════════════
# COMPAS metallicity grid
# ═══════════════════════════════════════════════════════════════════════════
# 53-element grid from COMPAS default settings.  The final value (0.03)
# appears twice: this matches the original COMPAS configuration where the
# last bin edge is duplicated.  np.digitize handles it correctly (both map
# to the same bin), but callers doing exact-equality checks against
# np.unique(METALLICITY_GRID) will see only 52 unique values.
METALLICITY_GRID = np.array([
    0.0001, 0.00011, 0.00012, 0.00014, 0.00016, 0.00017,
    0.00019, 0.00022, 0.00024, 0.00027, 0.0003,  0.00034,
    0.00037, 0.00042, 0.00047, 0.00052, 0.00058, 0.00065,
    0.00073, 0.00081, 0.00091, 0.00102, 0.00114, 0.00128,
    0.00143, 0.0016,  0.00179, 0.002,   0.00224, 0.00251,
    0.00281, 0.00315, 0.00353, 0.00395, 0.00443, 0.00496,
    0.00556, 0.00623, 0.00698, 0.00782, 0.00876, 0.00981,
    0.01098, 0.0123,  0.01378, 0.01545, 0.01731, 0.01939,
    0.02172, 0.02433, 0.02726, 0.03,    0.03,
])


# ═══════════════════════════════════════════════════════════════════════════
# Export helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_efficiencies(filepath, arrays, labels=None):
    """Save efficiency arrays as a single .npy stack."""
    stack = np.array(arrays)
    np.save(filepath, stack)
    if labels:
        print(f"Saved {filepath}: rows = {labels}")


def save_rates(filepath, redshifts, rate_dict):
    """Save merger rate arrays (redshifts + named rates) as .npy stack."""
    rows = [redshifts] + [rate_dict[k] for k in rate_dict]
    labels = ['redshifts'] + list(rate_dict.keys())
    np.save(filepath, np.array(rows))
    print(f"Saved {filepath}: rows = {labels}")


# ═══════════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════
def weighted_sample(mask, weight, n_target=12000, rng=None):
    """Subsample indices weighted by STROOPWAFEL weight for scatter plots."""
    if rng is None:
        rng = np.random.default_rng(42)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return idx
    w = weight[idx]
    if w.sum() == 0:
        return idx[:min(n_target, len(idx))]
    w = w / w.sum()
    n = min(n_target, len(idx))
    return rng.choice(idx, size=n, replace=False, p=w)


def log_jitter(Z, scale=0.04, rng=None):
    """Add small log-uniform jitter to metallicity values for scatter plots."""
    if rng is None:
        rng = np.random.default_rng(42)
    return Z * 10 ** rng.uniform(-scale, scale, size=len(Z))


# ═══════════════════════════════════════════════════════════════════════════
# Kick / velocity loading (for kinematic offset analysis)
# ═══════════════════════════════════════════════════════════════════════════
def _match_sn_to_dco(f):
    """Extract the post-2nd-SN systemic velocity matched to DCO rows.

    The supernovae group has two rows per system (one per SN).
    We take the LAST SN event per seed (the 2nd SN that forms the DCO).

    Vectorized with numpy for performance on large populations.
    """
    sn = f['supernovae']
    sn_seed = sn['randomSeed'][...].squeeze()
    sn_vsys = sn['systemicVelocity'][...].squeeze()
    sn_time = sn['time'][...].squeeze()

    dco = f['doubleCompactObjects']
    dco_seed_key = 'seed' if 'seed' in dco else 'm_randomSeed'
    dco_seed = dco[dco_seed_key][...].squeeze()

    order = np.lexsort((sn_time, sn_seed))
    sn_seed_s = sn_seed[order]
    sn_vsys_s = sn_vsys[order]

    u_seeds, first_rev = np.unique(sn_seed_s[::-1], return_index=True)
    last_idx = len(sn_seed_s) - 1 - first_rev
    last_vsys = sn_vsys_s[last_idx]

    vsys_out = np.full(len(dco_seed), np.nan)
    match = np.searchsorted(u_seeds, dco_seed)
    match_clipped = np.clip(match, 0, len(u_seeds) - 1)
    found = (match < len(u_seeds)) & (u_seeds[match_clipped] == dco_seed)
    vsys_out[found] = last_vsys[match_clipped[found]]

    return vsys_out


def load_bns_with_kicks(path=None, sort_masses=True):
    """Load merging BNS with kick/velocity columns for offset analysis.

    Returns the standard load_bns dict plus:
        'drawnKick1', 'drawnKick2', 'v_sys', 'sep_DCO', 'ecc_DCO'
    """
    if path is None:
        path = DEFAULT_BNS_PATH

    with h5.File(path, 'r') as f:
        dco = f['doubleCompactObjects']
        m1    = dco['M1'][...].squeeze()
        m2    = dco['M2'][...].squeeze()
        w     = dco['weight'][...].squeeze()
        Z     = dco['Metallicity1'][...].squeeze()
        mh    = dco['mergesInHubbleTimeFlag'][...].squeeze()
        tc    = dco['tc'][...].squeeze()
        tform = dco['tform'][...].squeeze()
        dk1   = dco['drawnKick1'][...].squeeze()
        dk2   = dco['drawnKick2'][...].squeeze()
        sep   = dco['separationDCOFormation'][...].squeeze()
        ecc   = dco['eccentricityDCOFormation'][...].squeeze()

        vsys_all = _match_sn_to_dco(f)

    mask = (mh == 1)
    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    return {
        'm1': m1_out, 'm2': m2_out,
        'weights': w[mask], 'metallicity': Z[mask],
        'delay_time': (tform + tc)[mask],
        'n_merging': int(mask.sum()), 'mask_merging': mask,
        'drawnKick1': dk1[mask], 'drawnKick2': dk2[mask],
        'v_sys': vsys_all[mask], 'sep_DCO': sep[mask], 'ecc_DCO': ecc[mask],
    }


def load_bhns_with_kicks(path=None):
    """Load merging BHNS with kick/velocity columns for offset analysis."""
    if path is None:
        path = DEFAULT_BHNS_PATH

    with h5.File(path, 'r') as f:
        dco = f['doubleCompactObjects']
        m1    = dco['M1'][...].squeeze()
        m2    = dco['M2'][...].squeeze()
        w     = dco['weight'][...].squeeze()
        Z     = dco['Metallicity1'][...].squeeze()
        mh    = dco['mergesInHubbleTimeFlag'][...].squeeze()
        tc    = dco['tc'][...].squeeze()
        tform = dco['tform'][...].squeeze()
        st1   = dco['stellarType1'][...].squeeze()
        dk1   = dco['drawnKick1'][...].squeeze()
        dk2   = dco['drawnKick2'][...].squeeze()
        sep   = dco['separationDCOFormation'][...].squeeze()
        ecc   = dco['eccentricityDCOFormation'][...].squeeze()

        vsys_all = _match_sn_to_dco(f)

    is_BH1 = (st1 == 14)
    M_BH = np.where(is_BH1, m1, m2)
    M_NS = np.where(is_BH1, m2, m1)
    mask = (mh == 1)

    return {
        'M_BH': M_BH[mask], 'M_NS': M_NS[mask],
        'weights': w[mask], 'metallicity': Z[mask],
        'delay_time': (tform + tc)[mask],
        'n_merging': int(mask.sum()), 'mask_merging': mask,
        'drawnKick1': dk1[mask], 'drawnKick2': dk2[mask],
        'v_sys': vsys_all[mask], 'sep_DCO': sep[mask], 'ecc_DCO': ecc[mask],
    }
