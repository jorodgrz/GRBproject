"""
Shared physics functions for GRB classification from compact binary mergers.

Foucart et al. (2018) remnant mass formula, Kruger & Foucart (2020)
dynamical ejecta fits, NS equation-of-state helpers, and
Gottlieb et al. (2023, 2024) classification thresholds.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gottlieb et al. (2023) classification thresholds
# ---------------------------------------------------------------------------
M_CRIT_BNS = 2.8       # BNS prompt-collapse total mass threshold [Msun]
Q_THRESH_BNS = 1.2     # BNS mass ratio threshold (q = M_max / M_min >= 1)
MDISK_SHORT = 0.01     # BHNS Short cbGRB disk mass threshold [Msun]
MDISK_LONG = 0.1       # BHNS Long cbGRB disk mass threshold [Msun]

# ---------------------------------------------------------------------------
# Gottlieb et al. (2024) additions
# ---------------------------------------------------------------------------
M_TOV = 2.2             # Maximum non-rotating NS mass [Msun]
                         # Raaijmakers et al. (2021, arXiv:2105.06981) combined
                         # NICER + GW + KN analysis: M_TOV = 2.23 +0.14/-0.23 (PP)
                         # and 2.11 +0.29/-0.16 (CS).  2.2 is a central estimate.
M_THRESH = M_CRIT_BNS   # Alias: prompt-collapse threshold [Msun]
                         # M_THRESH / M_TOV ~ 1.27 here; literature range is
                         # ~1.3-1.7 (Bauswein+ 2013, 2020; Koppel+ 2019).
Q_NO_DISK = 1.05        # Below this q, near-equal-mass prompt collapse suppresses disk
# NOTE: The HMNS short/long-lived split used in the notebook is at
# 1.2 * M_TOV ~ 2.64 M_sun (total gravitational mass).  Gottlieb (2024)
# discusses this near ~2.7 M_sun (close to M_THRESH).  The 1.2 * M_TOV
# choice captures the concept that remnants significantly above M_TOV but
# below M_THRESH collapse on viscous timescales rather than surviving long
# enough to power an extended GRB engine.

# ---------------------------------------------------------------------------
# Legacy remnant-to-disk fraction (deprecated; kept for back-compat)
# ---------------------------------------------------------------------------
# The proper decomposition is now M_disk = M_rem - M_dyn, using the
# Kruger & Foucart (2020) dynamical ejecta fit.  F_DISK is retained only
# as a fallback when the legacy f_disk keyword is explicitly passed.
F_DISK = 0.4

# ---------------------------------------------------------------------------
# COMPAS simulation constants
# ---------------------------------------------------------------------------
MEAN_MASS_EVOLVED = 77708655  # M_sun formed per simulation; Kroupa IMF, 5-150 Msun primaries

# ---------------------------------------------------------------------------
# EOS reference models
# ---------------------------------------------------------------------------
# M_crit : BNS prompt-collapse total-mass threshold [Msun]
# R_1p4  : NS radius at 1.4 Msun [km]  (Read et al. 2008, Table III)
# M_TOV  : maximum non-rotating NS mass [Msun]  (Read et al. 2008)
EOS_MODELS = {
    'APR4':  {'M_crit': 2.46, 'R_1p4': 11.1, 'M_TOV': 2.20},
    'SFHo':  {'M_crit': 2.60, 'R_1p4': 11.9, 'M_TOV': 2.06},
    'LS220': {'M_crit': 2.72, 'R_1p4': 12.7, 'M_TOV': 2.04},
    'DD2':   {'M_crit': 3.35, 'R_1p4': 13.2, 'M_TOV': 2.42},
}


# ---------------------------------------------------------------------------
# ISCO
# ---------------------------------------------------------------------------
def r_isco(a_BH):
    """ISCO radius in units of G*M_BH/c^2 (Bardeen et al. 1972).

    Uses np.where so that a_BH = 0 maps to the prograde ISCO (sign = +1)
    without silently mis-handling small negative (retrograde) spins.
    """
    Z1 = 1 + (1 - a_BH**2)**(1/3) * ((1 + a_BH)**(1/3) + (1 - a_BH)**(1/3))
    Z2 = np.sqrt(3*a_BH**2 + Z1**2)
    sign = np.where(a_BH >= 0, 1.0, -1.0)
    return 3 + Z2 - sign * np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))


# ---------------------------------------------------------------------------
# Neutron star helpers
# ---------------------------------------------------------------------------
def ns_baryon_mass(M_NS):
    """Approximate NS baryon mass from gravitational mass [M_sun].

    M^b ~ M_g + 0.080 * M_g^2 (Gao et al. 2020; Lattimer & Prakash 2001).
    """
    return M_NS + 0.080 * M_NS**2


def ns_radius(M_NS, R_1p4_km=12.0, M_TOV_local=None):
    """Mass-dependent NS radius [km].

    Asymmetric profile calibrated to the qualitative behaviour of
    tabulated EOS (APR4, SFHo, DD2) from Read et al. (2008, Table III):
      - Nearly flat plateau from ~1.0 to ~1.6 Msun
      - Steepening drop approaching M_TOV (~15% below R_1.4)
      - No neutron star above M_TOV (returns NaN)

    R_1p4_km = 12.0 km is consistent with NICER + GW170817 constraints
    (Raaijmakers et al. 2021: R_1.4 = 12.18 +0.56/-0.79 km, CS model).

    For calculations requiring EOS-specific accuracy, pass R_NS_km
    directly to foucart_disk_mass() instead.
    """
    if M_TOV_local is None:
        M_TOV_local = M_TOV
    M_NS = np.asarray(M_NS, dtype=float)
    x = np.clip((M_NS - 1.0) / (M_TOV_local - 1.0), 0.0, 1.0)
    R = R_1p4_km * (1.0 - 0.15 * x**3)
    R = np.where(M_NS > M_TOV_local, np.nan, R)
    R = np.where(M_NS < 0.8, R_1p4_km, R)
    return R


def mcrit_to_r14(mc):
    """Linear interpolation from M_crit [Msun] to R_{1.4} [km].

    Anchored at APR4 (M_crit=2.46, R_{1.4}=11.1 km) and
    DD2 (M_crit=3.35, R_{1.4}=13.2 km) from Read et al. (2008).
    """
    apr4 = EOS_MODELS['APR4']
    dd2 = EOS_MODELS['DD2']
    return apr4['R_1p4'] + ((dd2['R_1p4'] - apr4['R_1p4'])
                            / (dd2['M_crit'] - apr4['M_crit'])
                            * (mc - apr4['M_crit']))


# ---------------------------------------------------------------------------
# NS compactness helper
# ---------------------------------------------------------------------------
def _compactness(M_NS, R_km):
    """Dimensionless NS compactness C = G*M/(R*c^2)."""
    G = 6.674e-11; c = 3e8; Msun = 1.989e30
    return G * np.asarray(M_NS, dtype=float) * Msun / (np.asarray(R_km, dtype=float) * 1e3 * c**2)


# ---------------------------------------------------------------------------
# Foucart et al. (2018) total remnant baryon mass
# ---------------------------------------------------------------------------
def foucart_remnant_mass(M_BH, M_NS, a_BH=0.0, R_NS_km=None, R_1p4_km=12.0):
    """Foucart et al. (2018) Eq. (4) & (6) [arXiv:1807.00011].

    Returns the *total* baryon mass outside the BH after merger (disk +
    tidal tail + dynamical ejecta), **not** the disk mass alone.
    """
    R_km = R_NS_km if R_NS_km is not None else ns_radius(M_NS, R_1p4_km=R_1p4_km)
    C_NS = _compactness(M_NS, R_km)
    eta = M_NS * M_BH / (M_NS + M_BH)**2
    R_hat = r_isco(a_BH)

    alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761
    bracket = alpha * (1 - 2*C_NS) / eta**(1/3) - beta * R_hat * C_NS / eta + gamma

    M_b = ns_baryon_mass(M_NS)
    return np.maximum(0.0, bracket)**delta * M_b


# ---------------------------------------------------------------------------
# Kruger & Foucart (2020) dynamical ejecta  [arXiv:2002.07728]
# ---------------------------------------------------------------------------
def bhns_dynamical_ejecta(M_BH, M_NS, a_BH, R_NS_km=None, R_1p4_km=12.0):
    """BHNS dynamical ejecta mass [Msun] -- Kruger & Foucart (2020) Eq. (9).

    Fitted to 45+ NR simulations (Kawaguchi+ 2015, Foucart+ 2019).
    Valid for Q ~ 3-7, chi_BH ~ 0-0.9, C_NS ~ 0.14-0.18.
    """
    R_km = R_NS_km if R_NS_km is not None else ns_radius(M_NS, R_1p4_km=R_1p4_km)
    C_NS = _compactness(M_NS, R_km)
    M_b = ns_baryon_mass(M_NS)
    Q = np.asarray(M_BH, dtype=float) / np.asarray(M_NS, dtype=float)
    R_hat = r_isco(a_BH)

    a1, a2, a4 = 0.007116, 0.001436, -0.02762
    n1, n2 = 0.8636, 1.6840
    bracket = (a1 * Q**n1 * (1 - 2*C_NS) / C_NS
               - a2 * Q**n2 * R_hat
               + a4)
    return np.maximum(0.0, bracket) * M_b


def bns_dynamical_ejecta(M1, M2, R1_km=None, R2_km=None, R_1p4_km=12.0):
    """BNS dynamical ejecta mass [Msun] -- Kruger & Foucart (2020) Eq. (6).

    Fitted to 200 NR simulations (Dietrich & Ujevic 2017, Kiuchi+ 2019).
    Result is in solar masses (the formula output is in units of 1e-3 Msun).
    """
    M1 = np.asarray(M1, dtype=float)
    M2 = np.asarray(M2, dtype=float)
    R1 = R1_km if R1_km is not None else ns_radius(M1, R_1p4_km=R_1p4_km)
    R2 = R2_km if R2_km is not None else ns_radius(M2, R_1p4_km=R_1p4_km)
    C1 = _compactness(M1, R1)
    C2 = _compactness(M2, R2)

    a, b, c, n = -9.3335, 114.17, -337.56, 1.5465
    term1 = (a / C1 + b * (M2 / M1)**n + c * C1) * M1
    term2 = (a / C2 + b * (M1 / M2)**n + c * C2) * M2
    return np.maximum(0.0, term1 + term2) * 1e-3


# ---------------------------------------------------------------------------
# Disk mass = remnant mass - dynamical ejecta
# ---------------------------------------------------------------------------
def foucart_disk_mass(M_BH, M_NS, a_BH=0.0, R_NS_km=None, R_1p4_km=12.0,
                      f_disk=None):
    """BHNS accretion disk mass [Msun].

    Default behaviour (f_disk=None): computes
        M_disk = M_rem(Foucart 2018) - M_dyn(Kruger & Foucart 2020)
    which properly separates the disk from dynamical ejecta.

    Legacy behaviour (f_disk=<float>): returns f_disk * M_rem as before.
    """
    M_rem = foucart_remnant_mass(M_BH, M_NS, a_BH=a_BH,
                                 R_NS_km=R_NS_km, R_1p4_km=R_1p4_km)
    if f_disk is not None:
        return f_disk * M_rem

    M_dyn = bhns_dynamical_ejecta(M_BH, M_NS, a_BH,
                                   R_NS_km=R_NS_km, R_1p4_km=R_1p4_km)
    return np.maximum(0.0, M_rem - M_dyn)
