"""
Shared physics functions for GRB classification from compact binary mergers.

Foucart et al. (2018) disk mass formula, NS equation-of-state helpers,
and Gottlieb et al. (2023, 2024) classification thresholds.
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
M_TOV = 2.05            # Maximum non-rotating NS mass [Msun]
M_THRESH = M_CRIT_BNS   # Alias: prompt-collapse threshold [Msun]
Q_NO_DISK = 1.05        # Below this q, near-equal-mass prompt collapse suppresses disk
# NOTE: The HMNS short/long-lived split used in the notebook is at
# 1.2 * M_TOV ~ 2.46 M_sun (total baryonic remnant mass).  Gottlieb (2024)
# discusses this near ~2.7 M_sun total *gravitational* mass (close to
# M_THRESH).  The 1.2 * M_TOV choice is a convenient proxy; physically it
# captures the concept that remnants significantly above M_TOV but below
# M_THRESH collapse on viscous timescales rather than surviving long enough
# to power an extended GRB engine.

# ---------------------------------------------------------------------------
# Remnant-to-disk fraction
# ---------------------------------------------------------------------------
# F_DISK converts Foucart's total remnant baryon mass M_rem into accretion
# disk mass: M_disk = F_DISK * M_rem.  Foucart (2012, Sec. VI) reports the
# *total* remnant (disk + tidal tail + ejecta); the fraction that settles
# into the disk depends on mass ratio and spin, with NR simulations showing
# ~1/3 to ~2/3.  F_DISK = 0.4 is a midrange choice.  This is a significant
# systematic: shifting to 0.3 or 0.5 moves many systems across the 0.01 and
# 0.1 M_sun classification thresholds.  Use f_disk parameter in
# foucart_disk_mass() for sensitivity sweeps.
F_DISK = 0.4

# ---------------------------------------------------------------------------
# COMPAS simulation constants
# ---------------------------------------------------------------------------
MEAN_MASS_EVOLVED = 77708655  # M_sun formed per simulation; Kroupa IMF, 5-150 Msun primaries

# ---------------------------------------------------------------------------
# EOS reference models  {name: M_crit [Msun]}
# ---------------------------------------------------------------------------
EOS_MODELS = {'APR4': 2.46, 'SFHo': 2.60, 'LS220': 2.72, 'DD2': 3.35}


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


def ns_radius(M_NS, R_1p4_km=12.0):
    """Mass-dependent NS radius [km].

    R(M) = R_{1.4} * max(0.75, 1 - 0.25*(M - 1.4)^2)

    Custom analytic approximation (not from a specific published EOS fit).
    Captures the qualitative trend that heavier NS are more compact, with a
    floor at 0.75*R_{1.4}.  R_{1.4} = 12.0 km is consistent with NICER
    constraints (Miller+ 2021, Riley+ 2021).

    Caveat: this underestimates the R(M) slope compared to actual EOS
    tables.  For APR4, R drops ~16% from 1.4 to 2.0 M_sun; this formula
    gives only ~5%.  The net effect is a slight overestimate of NS radius
    (and therefore disk mass) for heavy NS.  For Foucart disk mass
    calculations requiring EOS-specific accuracy, pass R_NS_km directly.
    """
    return R_1p4_km * np.maximum(0.75, 1.0 - 0.25 * (M_NS - 1.4)**2)


def mcrit_to_r14(mc):
    """Linear interpolation from M_crit [Msun] to R_{1.4} [km].

    Anchored at APR4 (M_crit=2.46, R_{1.4}=11.9 km) and
    DD2 (M_crit=3.35, R_{1.4}=13.2 km).
    """
    return 11.9 + (13.2 - 11.9) / (3.35 - 2.46) * (mc - 2.46)


# ---------------------------------------------------------------------------
# Foucart disk mass
# ---------------------------------------------------------------------------
def foucart_disk_mass(M_BH, M_NS, a_BH=0.0, R_NS_km=None, R_1p4_km=12.0,
                      f_disk=F_DISK):
    """Foucart et al. (2018) Eq. (4) & (6) [arXiv:1807.00011].

    Returns the accretion disk mass M_disk = f_disk * M_rem.
    Uses baryon mass and mass-dependent NS radius by default.
    """
    G = 6.674e-11; c = 3e8; Msun = 1.989e30

    R_km = R_NS_km if R_NS_km is not None else ns_radius(M_NS, R_1p4_km=R_1p4_km)

    C_NS  = G * M_NS * Msun / (R_km * 1e3 * c**2)
    eta   = M_NS * M_BH / (M_NS + M_BH)**2
    R_hat = r_isco(a_BH)

    alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761
    bracket = alpha * (1 - 2*C_NS) / eta**(1/3) - beta * R_hat * C_NS / eta + gamma

    M_b = ns_baryon_mass(M_NS)
    M_rem = np.maximum(0.0, bracket)**delta * M_b
    return f_disk * M_rem
