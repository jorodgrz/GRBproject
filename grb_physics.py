"""
Shared physics functions for GRB classification from compact binary mergers.

Foucart et al. (2018) remnant mass formula, Kruger & Foucart (2020)
dynamical ejecta fits, NS equation-of-state helpers, and
Gottlieb et al. (2023, 2024) classification thresholds.

Cosmology
---------
Throughout this pipeline we adopt the same cosmological parameters used
by COMPAS ``FastCosmicIntegration`` (Planck 2015 / TNG-consistent):
  H0 = 67.74 km/s/Mpc,  Omega_m = 0.3089,  Omega_Lambda = 0.6911
All lookback-time ↔ redshift conversions, comoving volumes, and SFR
integrals use these values.  Mixing with Planck 2018 parameters would
introduce ~2% inconsistencies at high z.

Supernova engine
----------------
The COMPAS simulations (Broekgaarden et al. 2021, Model A and K) use the
Fryer et al. (2012) *rapid* supernova explosion mechanism.  This produces
a narrow NS mass distribution peaked near 1.26–1.28 M_sun with a gap
around 2–5 M_sun.  The delayed mechanism yields broader NS masses and a
less pronounced mass gap, which would shift GRB class fractions
(especially the sbGRB + blue KN boundary at 1.2 × M_TOV).

Jet efficiency caveat
---------------------
All disk-mass-based GRB classifications assume 100% jet launching
efficiency above the disk-mass threshold.  In reality, the jet must
break out of the merger ejecta and the disk must reach the MAD
(magnetically arrested disk) state on a timescale shorter than the
accretion time (Gottlieb 2023).  Predicted GRB fractions and rates
should therefore be interpreted as *upper bounds*.
"""

import warnings
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
                         # M_THRESH / M_TOV ~ 1.27 here.  Bauswein+ 2013, 2020
                         # and Koppel+ 2019 find ratios ~1.3-1.7 across EOSs;
                         # the fiducial 2.8 Msun is from Gottlieb+ 2023 and sits
                         # slightly below the NR-derived range.  See Section 17
                         # for the full EOS sensitivity analysis.
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
_MEAN_MASS_EVOLVED_VALUE = 77708655

# ---------------------------------------------------------------------------
# EOS reference models
# ---------------------------------------------------------------------------
# M_crit : BNS prompt-collapse total-mass threshold [Msun]
#          Bauswein et al. (2013, arXiv:1307.5191) Table I
# R_1p4  : NS radius at 1.4 Msun [km]  (Read et al. 2009, Table III)
# M_TOV  : maximum non-rotating NS mass [Msun]  (Read et al. 2009)
EOS_MODELS = {
    'APR4':  {'M_crit': 2.88, 'R_1p4': 11.1, 'M_TOV': 2.20},
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
    """Mass-dependent NS radius [km] (heuristic model).

    Phenomenological interpolation NOT derived from a specific EOS table.
    The cubic suppression R = R_1p4*(1 - 0.15*x^3) with x = (M - 1.4) /
    (M_TOV - 1.4) is anchored at M = 1.4 Msun so that R(1.4) = R_1p4_km
    exactly, matching the observational anchor (Raaijmakers et al. 2021,
    arXiv:2105.06981: R_1.4 = 12.18 +0.56/-0.79 km, CS model).  Below
    1.4 Msun the radius is held at R_1p4_km (flat plateau), and above
    1.4 Msun the cubic suppression drops R by ~15% at M_TOV.

    Qualitative behaviour:
      - Flat plateau at R_1p4_km for M <= 1.4 Msun
      - Steepening drop from 1.4 Msun to M_TOV (~15% below R_1.4)
      - No neutron star above M_TOV (returns NaN)

    Quantitative comparison at M_NS = 1.4 Msun with default R_1p4_km:
      - This heuristic: 12.0 km (by construction)
      - APR4: 11.1 km  |  SFHo: 11.9 km  |  DD2: 13.2 km
    The ~10% spread in R propagates as ~10% in compactness C_NS, which
    can shift disk masses near the MDISK_SHORT/MDISK_LONG thresholds.

    For EOS-consistent calculations, use ``ns_radius_from_eos()`` or
    pass R_NS_km directly to ``foucart_disk_mass()`` with the R_1p4
    value from ``EOS_MODELS``.
    """
    if M_TOV_local is None:
        M_TOV_local = M_TOV
    M_NS = np.asarray(M_NS, dtype=float)
    x = np.clip((M_NS - 1.4) / (M_TOV_local - 1.4), 0.0, 1.0)
    R = R_1p4_km * (1.0 - 0.15 * x**3)
    R = np.where(M_NS > M_TOV_local, np.nan, R)
    R = np.where(M_NS < 1.4, R_1p4_km, R)
    return R


def ns_radius_from_eos(M_NS, eos_name):
    """EOS-anchored NS radius [km] using ``EOS_MODELS`` parameters.

    Calls ``ns_radius`` with R_1p4 and M_TOV from the named EOS,
    ensuring self-consistent compactness for the Foucart formula.

    Parameters
    ----------
    M_NS : float or array
        Gravitational mass [Msun].
    eos_name : str
        Key in ``EOS_MODELS`` (e.g. 'APR4', 'SFHo', 'DD2').
    """
    eos = EOS_MODELS[eos_name]
    return ns_radius(M_NS, R_1p4_km=eos['R_1p4'], M_TOV_local=eos['M_TOV'])


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
def foucart_remnant_mass(M_BH, M_NS, a_BH=0.0, R_NS_km=None, R_1p4_km=12.0,
                          clip_Q=None, clip_chi=None):
    """Foucart et al. (2018) Eq. (4) & (6) [arXiv:1807.00011].

    Returns the *total* baryon mass outside the BH after merger (disk +
    tidal tail + dynamical ejecta), **not** the disk mass alone.

    Validity ranges (Foucart+ 2018):
      - *Calibration range* (abstract / Eq. 4 fit validity):
            Q in [1, 7],  chi_BH in [-0.5, 0.9],  C_NS in [0.13, 0.182]
        Stay within these bounds for the documented ~15% accuracy.
      - *Data range* (Table II coverage):
            individual NR simulations extend to chi_BH = 0.97, but with
            larger residuals (e.g. Q=3, chi=0.97 underestimates the NR
            disk mass by ~30%).

    A warning is emitted when Q > 7 OR |a_BH| > 0.9 (count and max
    value reported per call).  Use ``clip_Q`` and ``clip_chi`` to zero
    out extrapolated systems instead of returning fit values.

    Note on bulk usage: the warnings are aggregated to a single
    ``warnings.warn`` per call (with a count), so they do NOT flood
    when called once on a large array.  If you call this in a Python
    loop over many systems and want to suppress repeated messages,
    wrap the loop in ``warnings.catch_warnings()`` with
    ``warnings.simplefilter('once')`` (see the standard library
    ``warnings`` documentation).

    Parameters
    ----------
    clip_Q : float, optional
        If given, systems with Q > clip_Q are assigned zero remnant mass
        instead of extrapolating the fit.  Set to 7.0 for conservative
        results within the calibration range.
    clip_chi : float, optional
        If given, systems with |a_BH| > clip_chi are assigned zero
        remnant mass instead of extrapolating the fit.  Set to 0.9 to
        stay strictly within the Foucart+ 2018 calibration range.
    """
    M_BH_a = np.asarray(M_BH, dtype=float)
    M_NS_a = np.asarray(M_NS, dtype=float)
    a_BH_a = np.asarray(a_BH, dtype=float)
    Q = M_BH_a / M_NS_a
    if np.any(Q > 7.0):
        n_extrap = int(np.sum(Q > 7.0))
        warnings.warn(
            f"Foucart (2018) formula applied to {n_extrap} systems with "
            f"Q > 7 (max Q={float(np.max(Q)):.1f}); "
            f"calibrated for Q in [1, 7]",
            stacklevel=2)
    if np.any(np.abs(a_BH_a) > 0.9):
        n_chi = int(np.sum(np.abs(a_BH_a) > 0.9))
        warnings.warn(
            f"Foucart (2018) formula applied to {n_chi} systems with "
            f"|chi_BH| > 0.9 (max |chi|={float(np.max(np.abs(a_BH_a))):.3f}); "
            f"calibrated for chi_BH in [-0.5, 0.9]",
            stacklevel=2)

    R_km = R_NS_km if R_NS_km is not None else ns_radius(M_NS, R_1p4_km=R_1p4_km)
    C_NS = _compactness(M_NS, R_km)
    eta = M_NS_a * M_BH_a / (M_NS_a + M_BH_a)**2
    R_hat = r_isco(a_BH)

    alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761
    bracket = alpha * (1 - 2*C_NS) / eta**(1/3) - beta * R_hat * C_NS / eta + gamma

    M_b = ns_baryon_mass(M_NS)
    result = np.maximum(0.0, bracket)**delta * M_b

    if clip_Q is not None:
        result = np.where(Q > clip_Q, 0.0, result)
    if clip_chi is not None:
        result = np.where(np.abs(a_BH_a) > clip_chi, 0.0, result)

    return result


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


def bns_disk_mass(M1, M2, R1_km=None, R2_km=None, R_1p4_km=12.0,
                  M_TOV_local=None):
    """BNS post-merger accretion disk mass [Msun] -- Kruger & Foucart (2020) Eq. (4).

    Distinct from ``foucart_disk_mass`` (which is BHNS-only).  KF2020
    fit the disk mass of BNS NR simulations to the compactness of the
    *lighter* NS:

        M_disk / M_b^tot = max(5e-4, a * C_1 + c)

    where C_1 is the compactness of the less-massive component and
    M_b^tot is the total baryon mass of the binary.  Best-fit
    coefficients from KF2020 Table I:

        a = -8.1580,  c = 1.2695

    Note: this function is provided for completeness and downstream
    use; the BNS classifiers in ``grb_classify`` use mass-ratio /
    total-mass thresholds (Gottlieb 2023, 2024) rather than disk mass,
    so ``bns_disk_mass`` is not currently wired into them.

    Parameters
    ----------
    M1, M2 : array-like
        Component gravitational masses [Msun] (any ordering).
    R1_km, R2_km : array-like, optional
        Component radii [km].  If None, computed via ``ns_radius``.
    R_1p4_km : float, optional
        Reference R_{1.4} for the heuristic ``ns_radius`` (only used if
        the per-component radii are not supplied).
    M_TOV_local : float, optional
        TOV mass forwarded to ``ns_radius``; defaults to ``M_TOV``.

    Returns
    -------
    M_disk : float or array
        BNS disk mass [Msun].
    """
    M1 = np.asarray(M1, dtype=float)
    M2 = np.asarray(M2, dtype=float)
    m_heavy = np.maximum(M1, M2)
    m_light = np.minimum(M1, M2)

    if R1_km is not None and R2_km is not None:
        R1 = np.asarray(R1_km, dtype=float)
        R2 = np.asarray(R2_km, dtype=float)
        R_heavy = np.where(M1 >= M2, R1, R2)
        R_light = np.where(M1 >= M2, R2, R1)
    else:
        R_heavy = ns_radius(m_heavy, R_1p4_km=R_1p4_km, M_TOV_local=M_TOV_local)
        R_light = ns_radius(m_light, R_1p4_km=R_1p4_km, M_TOV_local=M_TOV_local)

    C_1 = _compactness(m_light, R_light)
    M_b_tot = ns_baryon_mass(m_heavy) + ns_baryon_mass(m_light)

    a, c = -8.1580, 1.2695  # KF2020 Table I (BNS disk fit)
    return M_b_tot * np.maximum(5e-4, a * C_1 + c)


def bns_dynamical_ejecta(M1, M2, R1_km=None, R2_km=None, R_1p4_km=12.0):
    """BNS dynamical ejecta mass [Msun] -- Kruger & Foucart (2020) Eq. (6).

    Fitted to 200 NR simulations (Dietrich & Ujevic 2017, Kiuchi+ 2019).
    Result is in solar masses (the formula output is in units of 1e-3 Msun).

    Sanity check (GW170817-like, M1=1.46, M2=1.27, R_1p4=12.0 km):
        expected M_ej_dyn ~ 0.003-0.010 Msun
    The fit residual quoted by Kruger & Foucart (2020) is sigma ~
    0.004 Msun, so a single-system value within this band is consistent
    with the fit even though the central value is closer to ~0.003-0.006.
    AT2017gfo total ejecta ~ 0.05-0.08 Msun (Rastinejad+ 2025); the
    dynamical component is a subset, with disk wind adding ~0.01-0.05 Msun.
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
                      f_disk=None, clip_Q=None, clip_chi=None):
    """BHNS accretion disk mass [Msun].

    Default behaviour (f_disk=None): computes
        M_disk = M_rem(Foucart 2018) - M_dyn(Kruger & Foucart 2020)
    which properly separates the disk from dynamical ejecta.

    Legacy behaviour (f_disk=<float>): returns f_disk * M_rem as before.

    Parameters
    ----------
    clip_Q : float, optional
        Forwarded to ``foucart_remnant_mass``.  Systems with Q > clip_Q
        get zero remnant (and therefore zero disk) mass.
    clip_chi : float, optional
        Forwarded to ``foucart_remnant_mass``.  Systems with
        |a_BH| > clip_chi get zero remnant (and therefore zero disk)
        mass; set to 0.9 to stay strictly within the Foucart+ 2018
        calibration range.
    """
    M_rem = foucart_remnant_mass(M_BH, M_NS, a_BH=a_BH,
                                 R_NS_km=R_NS_km, R_1p4_km=R_1p4_km,
                                 clip_Q=clip_Q, clip_chi=clip_chi)
    if f_disk is not None:
        return f_disk * M_rem

    M_dyn = bhns_dynamical_ejecta(M_BH, M_NS, a_BH,
                                   R_NS_km=R_NS_km, R_1p4_km=R_1p4_km)
    return np.maximum(0.0, M_rem - M_dyn)


# ---------------------------------------------------------------------------
# BH spin misalignment (Issue 5)
# ---------------------------------------------------------------------------
def effective_aligned_spin(a_BH, theta_tilt):
    """Project BH spin onto the orbital angular momentum axis.

    Kawaguchi et al. (2015) NR simulations show that the Foucart remnant
    mass formula (derived for aligned spins) overestimates disk mass when
    the BH spin is misaligned.  The disk mass drops to near zero for
    misalignment angles > 50–60 deg.

    Parameters
    ----------
    a_BH : float or array
        Dimensionless BH spin magnitude.
    theta_tilt : float or array
        Angle between BH spin and orbital angular momentum [rad].

    Returns
    -------
    a_eff : float or array
        Effective aligned spin component (clipped to >= 0).
    """
    return np.maximum(0.0, np.asarray(a_BH) * np.cos(np.asarray(theta_tilt)))


MISALIGNMENT_SYSTEMATIC_FACTOR = 0.5
"""Population-averaged reduction in BHNS GRB fractions from spin-orbit
misalignment.  Pop-synth suggests ~50% of BHNS systems have
misalignment > 45 deg (Fragos+ 2010, arXiv:1001.1107; Gerosa+ 2018).
Kawaguchi+ 2015 NR simulations show the BHNS disk mass drops to near
zero for misalignment > 50-60 deg, so a population-averaged factor-of-2
suppression of BHNS GRB rates is the canonical way to fold this in.

Two ways to apply this:

1. Population level (preferred when per-system tilts are unavailable):
   multiply the integrated BHNS GRB rate by this factor.  See
   ``grb_rates.apply_bhns_misalignment`` for the canonical helper.

2. Per-system level (when individual tilt angles ``theta_tilt`` are
   sampled): use ``effective_aligned_spin(a_BH, theta_tilt)`` to
   project the BH spin onto the orbital angular momentum axis and
   pass the result as ``a_BH`` to ``foucart_disk_mass``.  In this
   regime the population factor should NOT also be applied (it would
   double-count the suppression)."""


# ---------------------------------------------------------------------------
# Module-level __getattr__ for deprecated attributes
# ---------------------------------------------------------------------------
def __getattr__(name):
    if name == "MEAN_MASS_EVOLVED":
        warnings.warn(
            "MEAN_MASS_EVOLVED is deprecated; use "
            "grb_rates.calibrate_mean_mass_evolved() instead",
            DeprecationWarning, stacklevel=2)
        return _MEAN_MASS_EVOLVED_VALUE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
