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

# ---------------------------------------------------------------------------
# Remnant-to-disk fraction
# ---------------------------------------------------------------------------
F_DISK = 0.4            # Midrange estimate (Foucart 2012, Sec. VI; NR range ~1/3 to ~2/3)

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

    Approximates a moderate-stiffness EOS (APR4/SLy family).
    R_{1.4} = 12.0 km is consistent with NICER constraints.
    """
    return R_1p4_km * np.maximum(0.75, 1.0 - 0.25 * (M_NS - 1.4)**2)


def mcrit_to_r14(mc):
    """Linear interpolation from M_crit [Msun] to R_{1.4} [km].

    Based on APR4 (M_crit=2.46, R=11.9) and DD2 (M_crit=3.35, R=13.2).
    """
    return 11.9 + (13.2 - 11.9) / (3.35 - 2.60) * (mc - 2.60)


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
