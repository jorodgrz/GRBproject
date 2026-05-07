"""
Data loading and export helpers for COMPAS HDF5 compact-object outputs.

Standardises how BNS and BHNS populations are read from the COMPAS
doubleCompactObjects/formationChannels groups. All loaders filter to
merging systems (mergesInHubbleTimeFlag == 1) and return plain numpy arrays.
"""

import os
import warnings
import numpy as np
import h5py as h5

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

DEFAULT_BNS_PATH  = os.path.join(_DATA_DIR, 'COMPASCompactOutput_BNS_A.h5')
DEFAULT_BHNS_PATH = os.path.join(_DATA_DIR, 'COMPASCompactOutput_BHNS_A.h5')
DEFAULT_BNS_K_PATH = os.path.join(_DATA_DIR, 'COMPASCompactOutput_BNS_K.h5')

# Track which paths we've already complained about so the "no metadata"
# warning fires once per file per process instead of on every loader call.
_METADATA_WARN_CACHE: set[str] = set()


def _validate_hdf5_metadata(path, expected_kind=None,
                            expected_model=None, expected_ns_max=None):
    """Validate Broekgaarden+ 2021 model metadata embedded by
    ``tools/embed_model_metadata.py``.

    If the attributes are absent, emit a one-time UserWarning per path
    and return without raising; this preserves backward compatibility
    with un-annotated archives.  If they are present and contradict
    the caller's expectation, raise ``ValueError`` -- this is the
    Reviewer-3 defense against silent file mislabeling.
    """
    with h5.File(path, 'r') as f:
        attrs = dict(f.attrs)

    if not attrs.get('model') and not attrs.get('ns_max'):
        if path not in _METADATA_WARN_CACHE:
            _METADATA_WARN_CACHE.add(path)
            warnings.warn(
                f"COMPAS HDF5 {os.path.basename(path)} has no embedded "
                f"model identifier; cannot validate against expected "
                f"({expected_kind}, model={expected_model}, "
                f"ns_max={expected_ns_max}).  Run "
                f"`python tools/embed_model_metadata.py` once per "
                f"download to enable validation.",
                stacklevel=3)
        return

    actual_kind = str(attrs.get('kind', '')) or None
    actual_model = str(attrs.get('model', '')) or None
    actual_ns_max = float(attrs['ns_max']) if 'ns_max' in attrs else None

    if expected_kind and actual_kind and actual_kind != expected_kind:
        raise ValueError(
            f"COMPAS HDF5 {os.path.basename(path)} has embedded "
            f"kind={actual_kind!r}, expected {expected_kind!r}.  "
            f"Either the file is mislabeled or the wrong loader was "
            f"called."
        )
    if expected_model and actual_model and actual_model != expected_model:
        raise ValueError(
            f"COMPAS HDF5 {os.path.basename(path)} is model "
            f"{actual_model!r}, expected {expected_model!r}."
        )
    if (expected_ns_max is not None and actual_ns_max is not None
            and not np.isclose(actual_ns_max, expected_ns_max)):
        raise ValueError(
            f"COMPAS HDF5 {os.path.basename(path)} has ns_max="
            f"{actual_ns_max}, expected {expected_ns_max}."
        )


def _validate_delay_times(dt, label=""):
    """Sanity-check that delay times are in Myr (expected range ~1-14000).

    COMPAS stores tform and tc in Myr.  If someone passes data in Gyr
    the values would be 1e3x too small; in yr they'd be 1e6x too large.
    """
    if dt.max() > 1e5 or dt.min() < 0:
        raise ValueError(
            f"delay_time range [{dt.min():.1f}, {dt.max():.1f}]{label} "
            "looks wrong; expected Myr (typical range 1-14000)")


def _check_weights_no_nan(w_masked, path):
    """Hard fail if STROOPWAFEL weights contain NaN after the merging mask.

    CLAUDE.md treats weights as mandatory for every reduction over the
    sample.  A silent NaN here propagates into ``np.average`` and
    weighted histograms as ``nan`` outputs, so we refuse to return
    poisoned weights rather than warn (council Contrarian #1).
    """
    if np.any(np.isnan(w_masked)):
        raise ValueError(
            f"STROOPWAFEL weight has NaN in {os.path.basename(path)} "
            f"after mergesInHubbleTimeFlag mask; refusing to return "
            f"poisoned weights."
        )


def _validate_loader_dict(out, n_merging, path):
    """Assert all per-system arrays in a loader dict have length ``n_merging``.

    ``mask_merging`` is excluded because it spans the full pre-mask
    catalogue.  Non-array values (the ``population`` string tag, the
    ``n_merging`` int) are skipped via the ``isinstance`` filter.

    Catches the copy-paste typo where one column is returned without
    ``[mask]`` (council Contrarian #4 / Chairman fix #3).
    """
    lengths = {
        k: len(v) for k, v in out.items()
        if isinstance(v, np.ndarray) and k != 'mask_merging'
    }
    bad = {k: L for k, L in lengths.items() if L != n_merging}
    assert not bad, (
        f"Loader dict shape mismatch in {os.path.basename(path)}: "
        f"keys {bad} have wrong length, expected n_merging={n_merging}."
    )


def verify_shared_metallicity_prior(path_a, path_b):
    """Assert two COMPAS files use the same birth-metallicity prior.

    Both populations must span the same Z range for a single p_draw
    to be valid.  Returns the shared (Z_min, Z_max) tuple.
    """
    r_a = read_metallicity_range(path_a)
    r_b = read_metallicity_range(path_b)
    if r_a != r_b:
        raise ValueError(
            f"Metallicity ranges differ: {r_a} vs {r_b}; "
            "need separate dPdlogZ / p_draw per population")
    return r_a


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
def load_bns(path=None, sort_masses=True, expected_model=None,
             expected_ns_max=None):
    """Load merging BNS population from COMPAS HDF5 file.

    Parameters
    ----------
    path : str, optional
        Path to HDF5 file. Defaults to Data/COMPASCompactOutput_BNS_A.h5.
    sort_masses : bool
        If True, m1 >= m2 is enforced (heavier/lighter).
    expected_model : str, optional
        Broekgaarden+ 2021 model letter ('A', 'J', 'K').  When the HDF5
        carries the embedded ``model`` attribute (see
        ``tools/embed_model_metadata.py``), the loader validates and
        raises if the file is mislabeled.  When the attribute is
        absent, a one-time UserWarning is emitted and the value is
        ignored (backward compatible with un-annotated archives).
    expected_ns_max : float, optional
        Validates the ``ns_max`` attribute the same way as
        ``expected_model`` (Models J / A / K = 2.0 / 2.5 / 3.0 Msun).

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

    _validate_hdf5_metadata(path, expected_kind='BNS',
                            expected_model=expected_model,
                            expected_ns_max=expected_ns_max)

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
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bns")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    out = {
        'm1':           m1_out,
        'm2':           m2_out,
        'weights':      w_out,
        'metallicity':  Z[mask],
        'delay_time':   delay,
        'n_merging':    int(mask.sum()),
        'mask_merging': mask,
        'population':   'BNS',
    }
    _validate_loader_dict(out, out['n_merging'], path)
    return out


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
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bns_with_channels")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

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
    out = {
        'm1':           m1_out,
        'm2':           m2_out,
        'weights':      w_out,
        'metallicity':  Z[mask],
        'delay_time':   delay,
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
        'population':   'BNS',
    }
    _validate_loader_dict(out, out['n_merging'], path)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# BHNS data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_bhns(path=None, expected_model=None, expected_ns_max=None):
    """Load merging BHNS population from COMPAS HDF5 file.

    BH/NS identity is resolved using stellarType1 (14 = BH).

    Parameters
    ----------
    path : str, optional
        HDF5 path; defaults to Data/COMPASCompactOutput_BHNS_A.h5.
    expected_model : str, optional
        See ``load_bns``.  When the embedded ``model`` attribute is
        present and differs, raises ``ValueError`` (Reviewer-3 defense
        against silent file mislabeling).
    expected_ns_max : float, optional
        See ``load_bns``.

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

    _validate_hdf5_metadata(path, expected_kind='BHNS',
                            expected_model=expected_model,
                            expected_ns_max=expected_ns_max)

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
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bhns")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    out = {
        'M_BH':         M_BH[mask],
        'M_NS':         M_NS[mask],
        'weights':      w_out,
        'metallicity':  Z[mask],
        'delay_time':   delay,
        'n_merging':    int(mask.sum()),
        'mask_merging': mask,
        'population':   'BHNS',
    }
    _validate_loader_dict(out, out['n_merging'], path)
    return out


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
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bhns_with_channels")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    out = {
        'M_BH':         M_BH[mask],
        'M_NS':         M_NS[mask],
        'weights':      w_out,
        'metallicity':  Z[mask],
        'delay_time':   delay,
        'n_merging':    int(mask.sum()),
        'mask_merging': mask,
        'dblCE':        dblCE[mask],
        'sep_preCE':    sep_pre[mask],
        'fc_mt_p1':     fc_mt_p1[mask],
        'fc_mt_p1_K1':  fc_mt_p1_K1[mask],
        'fc_mt_s1':     fc_mt_s1[mask],
        'fc_mt_s1_K2':  fc_mt_s1_K2[mask],
        'fc_CEE':       fc_CEE[mask],
        'population':   'BHNS',
    }
    _validate_loader_dict(out, out['n_merging'], path)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# COMPAS metallicity grid
# ═══════════════════════════════════════════════════════════════════════════
# 53-element grid extracted from the Broekgaarden et al. (2021) Zenodo
# files (5189849 / 5178777, Models A and K, BNS and BHNS).  All four
# archives share this identical grid; verified by
# ``tests/test_grb_io_realdata.py::test_metallicity_grid_matches_data``.
#
# The previous literal carried two errors caught by the council audit
# (chairman fix #5): a slow drift away from the data starting near
# index 20 (e.g. 0.00091 vs the true 0.0009, 0.00102 vs 0.00101) and a
# spurious duplicate ``0.03`` at the tail.  The actual data has 53
# unique values; do not re-introduce the duplicate.
METALLICITY_GRID = np.array([
    0.0001,  0.00011, 0.00012, 0.00014, 0.00016, 0.00017,
    0.00019, 0.00022, 0.00024, 0.00027, 0.0003,  0.00034,
    0.00037, 0.00042, 0.00047, 0.00052, 0.00058, 0.00065,
    0.00073, 0.00081, 0.0009,  0.00101, 0.00113, 0.00126,
    0.0014,  0.00157, 0.00175, 0.00195, 0.00218, 0.00243,
    0.00272, 0.00303, 0.00339, 0.00378, 0.00422, 0.00471,
    0.00526, 0.00587, 0.00655, 0.00732, 0.00817, 0.00912,
    0.01018, 0.01137, 0.01269, 0.01416, 0.01581, 0.01765,
    0.01971, 0.022,   0.0244,  0.02705, 0.03,
])


# ═══════════════════════════════════════════════════════════════════════════
# Observed BNS gravitational-wave events
# ═══════════════════════════════════════════════════════════════════════════
# Component-mass medians under the *low-spin* prior from the LIGO/Virgo
# discovery papers.  These are the published medians, intended for figure
# annotations and quick sanity checks against classification thresholds;
# use the full posteriors (GWOSC) for any quantitative population work.
# Each entry pairs the mass values with a single citation so plotting
# code never needs to inline a hard-coded mass without a source.
OBSERVED_GW_EVENTS = {
    'GW170817': {
        'M1': 1.46,
        'M2': 1.27,
        'reference': 'Abbott+ 2019, PRX 9, 011001 (Table I, low-spin prior)',
    },
    'GW190425': {
        'M1': 1.61,
        'M2': 1.50,
        'reference': 'Abbott+ 2020, ApJL 892, L3 (Table 2, low-spin prior)',
    },
}


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

    n_unmatched = int((~found).sum())
    if n_unmatched > 0:
        # Council Contrarian #3: NaN in v_sys silently disappears via
        # np.isfinite(v_sys_km) in compute_offsets_population
        # (grb_offsets.py); surface it at load time so the count is
        # auditable in the notebook log.
        warnings.warn(
            f"_match_sn_to_dco ({os.path.basename(f.filename)}): "
            f"{n_unmatched} of {len(dco_seed)} DCO seeds had no matching "
            f"SN entry; v_sys is NaN for those rows.  Downstream offset "
            f"analysis will silently drop them via np.isfinite() in "
            f"compute_offsets_population.",
            stacklevel=2,
        )

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
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bns_with_kicks")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    out = {
        'm1': m1_out, 'm2': m2_out,
        'weights': w_out, 'metallicity': Z[mask],
        'delay_time': delay,
        'n_merging': int(mask.sum()), 'mask_merging': mask,
        'drawnKick1': dk1[mask], 'drawnKick2': dk2[mask],
        'v_sys': vsys_all[mask], 'sep_DCO': sep[mask], 'ecc_DCO': ecc[mask],
        'population': 'BNS',
    }
    _validate_loader_dict(out, out['n_merging'], path)
    return out


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
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bhns_with_kicks")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    out = {
        'M_BH': M_BH[mask], 'M_NS': M_NS[mask],
        'weights': w_out, 'metallicity': Z[mask],
        'delay_time': delay,
        'n_merging': int(mask.sum()), 'mask_merging': mask,
        'drawnKick1': dk1[mask], 'drawnKick2': dk2[mask],
        'v_sys': vsys_all[mask], 'sep_DCO': sep[mask], 'ecc_DCO': ecc[mask],
        'population': 'BHNS',
    }
    _validate_loader_dict(out, out['n_merging'], path)
    return out
