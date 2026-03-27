"""
Shared physics functions for GRB classification from compact binary mergers.

Foucart et al. (2018) disk mass formula, NS equation-of-state helpers,
and Gottlieb et al. (2023) classification thresholds.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gottlieb et al. (2023) classification thresholds
# ---------------------------------------------------------------------------
M_CRIT_BNS = 2.8       # BNS total mass threshold [Msun]
Q_THRESH_BNS = 1.2     # BNS mass ratio threshold (q = M_max / M_min >= 1)
MDISK_SHORT = 0.01     # BHNS Short cbGRB disk mass threshold [Msun]
MDISK_LONG = 0.1       # BHNS Long cbGRB disk mass threshold [Msun]

# ---------------------------------------------------------------------------
# Remnant-to-disk fraction
# ---------------------------------------------------------------------------
F_DISK = 0.4           # Midrange estimate (Foucart 2012, Sec. VI; NR range ~1/3 to ~2/3)


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
    Coefficient A ~ 0.080 for non-rotating NS.  Accurate to ~2% across
    typical EOS for M_NS in [1.0, 2.5] M_sun.
    """
    return M_NS + 0.080 * M_NS**2


def ns_radius(M_NS, R_1p4_km=12.0):
    """Mass-dependent NS radius [km].

    R(M) = R_{1.4} * max(0.75, 1 - 0.25*(M - 1.4)^2)

    Approximates a moderate-stiffness EOS (APR4/SLy family) within
    M_NS in [1.0, 2.5] M_sun.  R_{1.4} = 12.0 km is consistent with
    NICER constraints (Miller+ 2021, Riley+ 2021).  Clamped at 0.75*R_{1.4}
    to avoid unphysical radii near M_max.
    """
    return R_1p4_km * np.maximum(0.75, 1.0 - 0.25 * (M_NS - 1.4)**2)


# ---------------------------------------------------------------------------
# Foucart disk mass
# ---------------------------------------------------------------------------
def foucart_disk_mass(M_BH, M_NS, a_BH=0.0, R_NS_km=None, R_1p4_km=12.0,
                      f_disk=F_DISK):
    """Foucart et al. (2018) Eq. (4) & (6) [arXiv:1807.00011].

    Returns the accretion disk mass M_disk = f_disk * M_rem, where M_rem is
    the total remnant baryon mass from the fitting formula.

    Implementation details
    ----------------------
    1. Uses baryon mass M^b_NS (not gravitational mass) to scale M_rem,
       correcting a ~10-15% systematic underestimate in disk mass.
    2. Uses mass-dependent NS radius by default, avoiding unphysical
       compactness for M_NS >> 1.4 M_sun.

    Parameters
    ----------
    M_BH : float or array
        Black hole gravitational mass [M_sun].
    M_NS : float or array
        Neutron star gravitational mass [M_sun].
    a_BH : float or array
        Dimensionless BH spin (prograde >= 0).
    R_NS_km : float or None
        If given, use this fixed radius for ALL NS (for EOS sensitivity
        sweeps).  If None (default), use mass-dependent ns_radius().
    R_1p4_km : float
        Fiducial NS radius at 1.4 M_sun [km].  Only used when
        R_NS_km is None.  Default 12.0 km (NICER).
    f_disk : float
        Fraction of remnant baryon mass that forms the accretion disk.
        Default 0.4, a midrange estimate consistent with NR simulations
        showing ~1/3 to ~2/3 of remnant material in a disk (Foucart 2012,
        Sec. VI).  Use 1/3 and 1/2 as brackets for sensitivity analysis.
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
