"""
GRB and kilonova classification for compact binary mergers.

Implements Gottlieb et al. (2023) three-class and (2024) five-class BNS schemes,
BHNS disk-mass classification, and a unified mass-plane grid classifier.
Also includes Broekgaarden-style formation channel classification.
"""

import warnings

import numpy as np
from grb_physics import (
    foucart_disk_mass,
    M_CRIT_BNS, Q_THRESH_BNS,
    M_TOV, M_THRESH, K_THRESH_DEFAULT, MDISK_SHORT, MDISK_LONG,
)


def _resolve_m_thresh(m_tov, m_thresh, k_thresh):
    """Reconcile the (m_tov, m_thresh, k_thresh) kwarg triple.

    If ``k_thresh`` is given, the prompt-collapse threshold is taken as
    ``k_thresh * m_tov`` so EOS sweeps that change ``m_tov`` move both
    thresholds together.  Passing both ``k_thresh`` and an explicit
    ``m_thresh`` is ambiguous; ``k_thresh`` wins and a warning is
    emitted.
    """
    if k_thresh is None:
        return m_thresh
    derived = k_thresh * m_tov
    if m_thresh is not None and not np.isclose(m_thresh, derived):
        warnings.warn(
            f"Both k_thresh ({k_thresh}) and m_thresh ({m_thresh}) "
            f"were provided; using k_thresh * m_tov = {derived:.3f} "
            f"and ignoring the explicit m_thresh.",
            stacklevel=3)
    return derived


# ═══════════════════════════════════════════════════════════════════════════
# BNS classification: Gottlieb et al. (2023)
# ═══════════════════════════════════════════════════════════════════════════
def classify_bns_2023(m1, m2, M_crit=M_CRIT_BNS, q_thresh=Q_THRESH_BNS):
    """Three-class BNS scheme (Gottlieb 2023).

    Parameters
    ----------
    m1, m2 : array-like
        Component masses (either ordering; heavier/lighter detected internally).

    Returns
    -------
    dict of boolean arrays keyed by class name:
        'Short Type-I'  : HMNS remnant (M_tot < M_crit)
        'Short Type-II' : prompt collapse, light disk (M_tot >= M_crit, q < q_thresh)
        'Long cbGRB'    : prompt collapse, massive disk (M_tot >= M_crit, q >= q_thresh)
    """
    m1, m2 = np.asarray(m1), np.asarray(m2)
    M_tot = m1 + m2
    q = np.maximum(m1, m2) / np.minimum(m1, m2)

    return {
        'Short Type-I':  (M_tot < M_crit),
        'Short Type-II': (M_tot >= M_crit) & (q < q_thresh),
        'Long cbGRB':    (M_tot >= M_crit) & (q >= q_thresh),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BNS classification: Gottlieb et al. (2024) hybrid model
# ═══════════════════════════════════════════════════════════════════════════
def classify_bns_2024(m1, m2, m_tov=M_TOV, m_thresh=M_THRESH,
                      q_thresh=Q_THRESH_BNS, hmns_factor=1.2,
                      k_thresh=None):
    """Four-class BNS scheme (Gottlieb 2024).

    Classes
    -------
    'sbGRB + blue KN' : Long-lived HMNS engine
        (M_tot < hmns_factor * M_TOV).
    'lbGRB + red KN (HMNS)' : Short-lived HMNS + BH engine
        (hmns_factor * M_TOV <= M_tot < M_thresh).
    'lbGRB + red KN (disk)' : Prompt collapse, massive disk
        (M_tot >= M_thresh and q >= q_thresh).
    'Faint lbGRB' : Prompt collapse, small disk
        (M_tot >= M_thresh and q < q_thresh).

    Parameters
    ----------
    m1, m2 : array-like
        Component masses (either ordering; heavier/lighter detected
        internally).
    m_tov : float, optional
        Maximum non-rotating NS mass [Msun]; default ``M_TOV`` (2.2).
    m_thresh : float, optional
        Prompt-collapse total-mass threshold [Msun]; default
        ``M_THRESH`` (~ 2.794).  Ignored if ``k_thresh`` is given.
    q_thresh : float, optional
        Mass ratio (>=1) above which the prompt-collapse remnant
        forms a massive disk; default ``Q_THRESH_BNS`` (1.2).
    hmns_factor : float, optional
        Multiplier on ``m_tov`` setting the boundary between long-lived
        HMNS (sbGRB + blue KN) and short-lived HMNS (lbGRB + red KN).
        Default 1.2 is heuristic; Gottlieb (2024) discusses this
        boundary near ~2.7 Msun (close to ``M_THRESH`` for typical
        ``M_TOV``).  Pass ``hmns_factor=1.0`` to fold the short-lived
        HMNS class into prompt collapse entirely.
    k_thresh : float, optional
        If given, ``m_thresh`` is overridden to ``k_thresh * m_tov`` so
        that EOS sweeps move both thresholds together.  See
        ``grb_physics.K_THRESH_DEFAULT`` for the fiducial 1.27.

    Notes
    -----
    The ``Q_NO_DISK = 1.05`` cut from earlier revisions of this code is
    intentionally absent: there is no separate near-equal-mass
    "no-disk" boundary in Gottlieb (2023, 2024), and routing
    GW170817-like systems (q ~ 1.0) to a "no GRB" bin contradicts
    observations.  All prompt-collapse mergers below ``q_thresh`` now
    fall into 'Faint lbGRB'.
    """
    m1, m2 = np.asarray(m1), np.asarray(m2)
    m_heavy = np.maximum(m1, m2)
    m_light = np.minimum(m1, m2)
    M_tot = m_heavy + m_light
    q = m_heavy / m_light

    m_thresh = _resolve_m_thresh(m_tov, m_thresh, k_thresh)
    hmns_split = hmns_factor * m_tov

    return {
        'sbGRB + blue KN':       (M_tot < hmns_split),
        'lbGRB + red KN (HMNS)': (M_tot >= hmns_split) & (M_tot < m_thresh),
        'lbGRB + red KN (disk)': (M_tot >= m_thresh) & (q >= q_thresh),
        'Faint lbGRB':           (M_tot >= m_thresh) & (q < q_thresh),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BHNS classification from disk mass
# ═══════════════════════════════════════════════════════════════════════════
def classify_bhns(M_BH, M_NS, a_BH=0.5, md_short=MDISK_SHORT, md_long=MDISK_LONG,
                  **foucart_kw):
    """Classify BHNS mergers by Foucart disk mass.

    Returns
    -------
    dict of boolean arrays:
        'No GRB'       : M_disk < md_short
        'Short cbGRB'  : md_short <= M_disk < md_long
        'Long cbGRB'   : M_disk >= md_long
    Also includes 'M_disk' key with the raw disk mass array.
    """
    M_BH, M_NS = np.asarray(M_BH), np.asarray(M_NS)
    M_disk = foucart_disk_mass(M_BH, M_NS, a_BH=a_BH, **foucart_kw)
    return {
        'No GRB':      (M_disk < md_short),
        'Short cbGRB': (M_disk >= md_short) & (M_disk < md_long),
        'Long cbGRB':  (M_disk >= md_long),
        'M_disk':      M_disk,
    }


def classify_bhns_spins(M_BH, M_NS, spins=(0.0, 0.3, 0.5, 0.7, 0.9), **kw):
    """Run classify_bhns for each spin value. Returns {spin: result_dict}."""
    return {a: classify_bhns(M_BH, M_NS, a_BH=a, **kw) for a in spins}


# ═══════════════════════════════════════════════════════════════════════════
# Unified mass-plane grid classifier (Gottlieb 2024 hybrid)
# ═══════════════════════════════════════════════════════════════════════════
GRID_CLASS_LABELS = {
    1: 'Faint lbGRB (BNS)',
    2: 'lbGRB + red KN (HMNS)',
    3: 'sbGRB + blue KN',
    4: 'lbGRB + red KN (BNS disk)',
    5: 'Faint lbGRB (BHNS)',
    6: 'lbGRB + red KN (BHNS disk)',
}
"""Integer class -> label map for ``classify_grid``.

Integer ``0`` is reserved for cells *outside* any classified region:
``m_light <= 0`` cells (grid edges where the lighter component has
no mass), BHNS cells with ``m_light < ns_min``, and the non-BNS /
non-BHNS region (e.g. m_heavy > ns_max with m_light also > ns_max,
i.e. BBH).  ``0`` is intentionally absent from the label map so
plotting code can mask it as background."""


def classify_grid(m1g, m2g, m_tov=M_TOV, m_thresh=M_THRESH,
                  q_thresh=Q_THRESH_BNS, a_bh=0.5,
                  hmns_factor=1.2, k_thresh=None,
                  ns_max=None, ns_min=1.1):
    """Return an integer class map on a (M1, M2) grid.

    2024 Gottlieb hybrid model: all BH-powered jets produce lbGRBs;
    only long-lived HMNS (NS engine) produces sbGRBs.

    Classes (grouped by physical hierarchy; see ``GRID_CLASS_LABELS``):

        Background (default-fill):
        0 = unclassified          (outside BNS/BHNS regions, m_light < ns_min,
                                   or m_light <= 0)

        BNS region:
        1 = Faint lbGRB + red KN  (prompt collapse, small disk q < q_thresh)
        2 = lbGRB + red KN        (short-lived HMNS, BNS)
        3 = sbGRB + blue KN       (long-lived HMNS, BNS)
        4 = lbGRB + red KN        (BNS prompt collapse, massive disk q >= q_thresh)

        BHNS region:
        5 = Faint lbGRB + red KN  (BHNS, 0.01 <= Md < 0.1)
        6 = lbGRB + red KN        (BHNS, massive disk Md >= 0.1)

    Parameters
    ----------
    m_thresh : float, optional
        Prompt-collapse total-mass threshold [Msun]; default
        ``M_THRESH``.  Ignored if ``k_thresh`` is given.
    hmns_factor : float, optional
        Multiplier on ``m_tov`` for the long-/short-lived HMNS split.
        See ``classify_bns_2024`` for details.  Default 1.2.
    k_thresh : float, optional
        If given, ``m_thresh`` is overridden to ``k_thresh * m_tov`` so
        that EOS sweeps move both thresholds together.
    ns_max : float, optional
        Upper mass edge [Msun] of the BNS region on the grid: any
        component with mass > ns_max is treated as a BH.  This MUST
        match the ``M_NS_max`` configuration of the COMPAS run that
        produced your population (Broekgaarden+ 2021, arXiv:2103.02608:
        Models A/J/K use M_NS_max = 2.5 / 2.0 / 3.0 Msun).  If left as
        None, defaults to ``m_tov + 0.15`` and emits a UserWarning.
    ns_min : float, optional
        Lower mass [Msun] of the NS in BHNS systems.  Cells with
        ``m_light < ns_min`` are excluded from BHNS classification (and
        a warning is emitted reporting the count).  Default 1.1 Msun
        matches the COMPAS rapid-mechanism NS minimum (Fryer+ 2012);
        pass ``ns_min=0.8`` to be inclusive of ECSN/USSN NS, which
        reproduces the pre-patch behaviour.
    """
    # Symmetrize the q convention so callers can pass either ordering
    # (or a rectangular meshgrid that includes both triangles).  This
    # matches classify_bns_2024's convention q = m_heavy / m_light and
    # avoids silently dropping cells where m2 > m1 into the default
    # "no GRB" class.  Cells with m2 > m1 are masked off downstream
    # in the visualization in the typical mass-plane use case.
    m1g = np.asarray(m1g, dtype=float)
    m2g = np.asarray(m2g, dtype=float)
    m_heavy = np.maximum(m1g, m2g)
    m_light = np.minimum(m1g, m2g)

    cls = np.full_like(m1g, 0, dtype=int)
    m_tot = m1g + m2g
    q = np.where(m_light > 0, m_heavy / m_light, 999.0)

    m_thresh = _resolve_m_thresh(m_tov, m_thresh, k_thresh)

    if ns_max is None:
        # Default is a pragmatic buffer ~0.15 Msun above M_TOV; with the
        # default m_tov=2.2 this gives ns_max ~ 2.35 Msun.  This is NOT
        # a physical NS upper mass: COMPAS runs typically impose a
        # tighter or different M_NS_max (Broekgaarden+ 2021 fiducial
        # 2.5 Msun, models J/K = 2.0/3.0 Msun).  Pass ns_max explicitly
        # to match the run configuration.
        ns_max = m_tov + 0.15
        warnings.warn(
            f"classify_grid: ns_max not specified; using default "
            f"m_tov + 0.15 = {ns_max:.2f} Msun.  This must match the "
            f"COMPAS run's M_NS_max (Broekgaarden+ 2021: 2.5 Msun "
            f"fiducial; models J/K = 2.0/3.0).  Pass ns_max explicitly "
            f"to suppress this warning.",
            stacklevel=2)

    # BNS region: both components below the NS-mass cap, and the
    # lighter component must have positive mass.  Without the
    # ``m_light > 0`` guard, grid edges where the lighter component
    # has no mass (e.g. m_light = 0) would fall into the
    # ``M_tot < hmns_split`` branch and be silently labeled as
    # 'sbGRB + blue KN' (class 3); the q guard at the top of this
    # function never fires because q is not consulted on that branch.
    # The BNS class is symmetric in (m_heavy, m_light) so we do not
    # require any ordering on the input; mirror cells across the
    # m1=m2 diagonal receive identical classifications.
    is_bns = (m_heavy <= ns_max) & (m_light <= ns_max) & (m_light > 0)
    hmns_split = hmns_factor * m_tov

    cls[is_bns & (m_tot < hmns_split)] = 3
    cls[is_bns & (m_tot >= hmns_split) & (m_tot < m_thresh)] = 2
    cls[is_bns & (m_tot >= m_thresh) & (q >= q_thresh)] = 4
    cls[is_bns & (m_tot >= m_thresh) & (q < q_thresh)] = 1

    # BHNS region: heavier component above the NS-mass cap (the BH),
    # lighter component at or above the NS-mass floor ``ns_min``.
    # Cells with 0 < m_light < ns_min are excluded (left at class 0)
    # and a warning is emitted; pass ns_min lower (e.g. 0.8 for ECSN/
    # USSN) to include them.
    bhns_candidate = (m_heavy > ns_max) & (m_light <= ns_max)
    excluded = bhns_candidate & (m_light > 0) & (m_light < ns_min)
    if np.any(excluded):
        n_excl = int(np.sum(excluded))
        warnings.warn(
            f"classify_grid: excluded {n_excl} cells with "
            f"m_light < {ns_min} Msun from BHNS region "
            f"(set ns_min lower to include them; 0.8 reproduces "
            f"the pre-patch default).",
            stacklevel=2)

    is_bhns = bhns_candidate & (m_light >= ns_min)
    if np.any(is_bhns):
        md = foucart_disk_mass(m_heavy[is_bhns], m_light[is_bhns], a_BH=a_bh)
        bhns_cls = np.zeros_like(md, dtype=int)
        bhns_cls[md >= MDISK_LONG] = 6
        bhns_cls[(md >= MDISK_SHORT) & (md < MDISK_LONG)] = 5
        cls[is_bhns] = bhns_cls

    return cls


# ═══════════════════════════════════════════════════════════════════════════
# Formation channel classification (Broekgaarden et al.)
# ═══════════════════════════════════════════════════════════════════════════
def classify_formation_channels(*, dblCE, fc_CEE, fc_mt_p1, fc_mt_s1,
                                fc_mt_p1_K1, fc_mt_s1_K2):
    """Event-sequence formation channel classification.

    Parameters are 1-D arrays over the merging subset.

    Returns
    -------
    dict of boolean arrays:
        'I  Classic CE', 'II  Stable MT only', 'III Single-core CE',
        'IV  Double-core CE', 'V   Other'
    """
    has_p1 = fc_mt_p1 > 0
    has_s1 = fc_mt_s1 > 0
    first_is_primary   = has_p1 & (~has_s1 | (fc_mt_p1 < fc_mt_s1))
    first_is_secondary = has_s1 & (~has_p1 | (fc_mt_s1 < fc_mt_p1))
    no_stable_mt       = ~has_p1 & ~has_s1

    donor_type = np.where(first_is_primary, fc_mt_p1_K1,
                 np.where(first_is_secondary, fc_mt_s1_K2, -1))

    # Case-A (donor_type 0 or 1) mass transfer routes straight to "Other".
    # Broekgaarden et al. treat Case-A donors as a separate evolutionary
    # pathway that does not fit cleanly into the CE/stable-MT hierarchy.
    # This is a deliberate classification choice, not an omission.
    is_case_A = (donor_type == 0) | (donor_type == 1)
    had_CE    = (fc_CEE > 0)

    first_rlof_time = np.where(first_is_primary, fc_mt_p1,
                      np.where(first_is_secondary, fc_mt_s1, 999))
    ce_first = had_CE & (no_stable_mt | (fc_CEE < first_rlof_time))

    ch_IV    = (dblCE == 1)
    ch_other = ~ch_IV & (is_case_A | (no_stable_mt & ~had_CE))
    ch_II    = ~ch_IV & ~ch_other & ~had_CE
    ch_III   = ~ch_IV & ~ch_other & ce_first
    ch_I     = ~ch_IV & ~ch_other & had_CE & ~ce_first
    ch_other = ch_other | ~(ch_I | ch_II | ch_III | ch_IV)

    return {
        'I  Classic CE':      ch_I,
        'II  Stable MT only': ch_II,
        'III Single-core CE': ch_III,
        'IV  Double-core CE': ch_IV,
        'V   Other':          ch_other,
    }
