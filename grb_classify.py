"""
GRB and kilonova classification for compact binary mergers.

Implements Gottlieb et al. (2023) three-class and (2024) five-class BNS schemes,
BHNS disk-mass classification, and a unified mass-plane grid classifier.
Also includes Broekgaarden-style formation channel classification.
"""

import numpy as np
from grb_physics import (
    foucart_disk_mass,
    M_CRIT_BNS, Q_THRESH_BNS, Q_NO_DISK,
    M_TOV, M_THRESH, MDISK_SHORT, MDISK_LONG,
)


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
                      q_thresh=Q_THRESH_BNS, q_no_disk=Q_NO_DISK):
    """Five-class BNS scheme (Gottlieb 2024).

    Classes
    -------
    'sbGRB + blue KN' : Long-lived HMNS engine (M_tot < 1.2 * M_TOV)
    'lbGRB + red KN (HMNS)' : Short-lived HMNS + BH engine (1.2*M_TOV <= M_tot < M_thresh)
    'lbGRB + red KN (disk)' : Prompt collapse, massive disk (M_tot >= M_thresh, q >= q_thresh)
    'Faint lbGRB'     : Prompt collapse, small disk (M_thresh, q_no_disk <= q < q_thresh)
    'No GRB / No KN'  : Near-equal-mass prompt collapse, no disk (q < q_no_disk)
    """
    m1, m2 = np.asarray(m1), np.asarray(m2)
    m_heavy = np.maximum(m1, m2)
    m_light = np.minimum(m1, m2)
    M_tot = m_heavy + m_light
    q = m_heavy / m_light

    hmns_split = 1.2 * m_tov

    return {
        'sbGRB + blue KN':       (M_tot < hmns_split),
        'lbGRB + red KN (HMNS)': (M_tot >= hmns_split) & (M_tot < m_thresh),
        'lbGRB + red KN (disk)': (M_tot >= m_thresh) & (q >= q_thresh),
        'Faint lbGRB':           (M_tot >= m_thresh) & (q >= q_no_disk) & (q < q_thresh),
        'No GRB / No KN':        (M_tot >= m_thresh) & (q < q_no_disk),
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
    0: 'No GRB / No KN',
    1: 'Faint lbGRB (BNS)',
    2: 'lbGRB + red KN (HMNS)',
    3: 'sbGRB + blue KN',
    4: 'lbGRB + red KN (BNS disk)',
    5: 'Faint lbGRB (BHNS)',
    6: 'lbGRB + red KN (BHNS disk)',
}


def classify_grid(m1g, m2g, m_tov=M_TOV, m_thresh=M_THRESH,
                  q_thresh=Q_THRESH_BNS, a_bh=0.5, q_no_disk=Q_NO_DISK):
    """Return an integer class map on a (M1, M2) grid.

    2024 Gottlieb hybrid model: all BH-powered jets produce lbGRBs;
    only long-lived HMNS (NS engine) produces sbGRBs.

    Classes (grouped by physical hierarchy; see ``GRID_CLASS_LABELS``):

        BNS region:
        0 = No GRB / No KN        (near-equal-mass prompt collapse, no disk)
        1 = Faint lbGRB + red KN  (prompt collapse, small disk)
        2 = lbGRB + red KN        (short-lived HMNS, BNS)
        3 = sbGRB + blue KN       (long-lived HMNS, BNS)
        4 = lbGRB + red KN        (BNS prompt collapse, massive disk q >= q_thresh)

        BHNS region:
        5 = Faint lbGRB + red KN  (BHNS, 0.01 <= Md < 0.1)
        6 = lbGRB + red KN        (BHNS, massive disk Md >= 0.1)
    """
    cls = np.full_like(m1g, 0, dtype=int)
    m_tot = m1g + m2g
    q = np.where(m2g > 0, m1g / m2g, 999.0)

    # ns_max sets the upper mass edge of the BNS region on the grid.
    # M_TOV + 0.15 ~ 2.20 M_sun is a pragmatic buffer above the maximum
    # gravitational mass, allowing for numerical/EOS uncertainty.
    # If your COMPAS output has NS masses above this, increase accordingly.
    ns_max = m_tov + 0.15

    # BNS region
    is_bns = (m1g <= ns_max) & (m2g <= ns_max) & (m1g >= m2g)
    hmns_split = 1.2 * m_tov

    cls[is_bns & (m_tot < hmns_split)] = 3
    cls[is_bns & (m_tot >= hmns_split) & (m_tot < m_thresh)] = 2
    cls[is_bns & (m_tot >= m_thresh) & (q >= q_thresh)] = 4
    cls[is_bns & (m_tot >= m_thresh) & (q >= q_no_disk) & (q < q_thresh)] = 1
    cls[is_bns & (m_tot >= m_thresh) & (q < q_no_disk)] = 0

    # BHNS region: m2g > 0.8 M_sun floor excludes sub-NS-mass companions
    # that would not form a realistic NS.  Adjust if your COMPAS sample
    # includes ultra-low-mass or ECSN NS below this floor.
    is_bhns = (m1g > ns_max) & (m2g <= ns_max) & (m2g > 0.8)
    if np.any(is_bhns):
        md = foucart_disk_mass(m1g[is_bhns], m2g[is_bhns], a_BH=a_bh)
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
