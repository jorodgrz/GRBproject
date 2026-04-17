"""
Physical host-galaxy offset predictions for compact binary mergers.

Integrates post-SN systemic velocity through a Hernquist (1990) galaxy
potential to predict projected galactocentric offsets at merger time.
Replaces the ballistic upper-bound (v_sys * t_delay) with a
gravitationally-bound orbit model following the approach of
Bloom et al. (1999) and Fong & Berger (2013).

Typical usage
-------------
>>> from grb_offsets import compute_offsets_population
>>> offsets = compute_offsets_population(v_sys, delay_time, weights)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
G_CGS = 6.674e-8            # cm^3 g^-1 s^-2
MSUN_G = 1.989e33           # g
KPC_CM = 3.0857e21          # cm
MYR_S = 3.1557e13           # s
KM_CM = 1e5                 # cm

# Default host galaxy: typical SGRB host (Fong & Berger 2013)
DEFAULT_M_GAL = 10**10.5 * MSUN_G          # stellar mass [g]
DEFAULT_R_E = 5.0 * KPC_CM                 # effective radius [cm]

# Representative host galaxy types (Fong & Berger 2013).
# sGRB hosts: ~75% star-forming, ~25% elliptical.
HOST_MODELS = {
    'SF_disk': {
        'M_gal': 10**9.8 * MSUN_G,
        'R_e': 3.6 * KPC_CM,
        'weight': 0.50,
    },
    'SF_massive': {
        'M_gal': 10**10.5 * MSUN_G,
        'R_e': 5.0 * KPC_CM,
        'weight': 0.25,
    },
    'Elliptical': {
        'M_gal': 10**11.0 * MSUN_G,
        'R_e': 8.0 * KPC_CM,
        'weight': 0.25,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Hernquist potential
# ═══════════════════════════════════════════════════════════════════════════
def hernquist_scale_radius(R_e):
    """Convert projected half-light radius to Hernquist scale radius.

    For the Hernquist (1990) profile the *projected* half-light radius
    satisfies R_e / a ~ 1.8153 (numerical solution of the Abel-
    transformed surface-brightness integral; Hernquist 1990, Table 1).
    This differs from the 3-D half-mass relation r_half = a*(1+sqrt(2))
    (Eq. 38).  Observational effective radii from Sersic fits (e.g.
    Fong & Berger 2013 HST data) are projected half-light radii, so
    the projected relation is the correct one to use here.
    """
    return R_e / 1.8153


_R_CAP_FACTOR = 1000.0
"""Hard cap on radii expressed as a multiple of the Hernquist scale
radius ``a``.  Used by ``hernquist_birth_radius`` and ``integrate_orbit``
to bound the rare tail of the Hernquist profile (Hernquist 1990).

For the inverse-CDF sample r = a*sqrt(u)/(1 - sqrt(u)), the fraction
clipped at ``f * a`` is (1 - f/(1+f))^2.  Setting f = 1000 gives a
pile-up of (1/1001)^2 ~ 1e-6, which is negligible for the offset CDF
(Fong & Berger 2013 observed range tops out near 75 kpc, well below
1000*a even for compact SF hosts).  The previous 20*a cap clipped
~9.3% of draws into a delta function at 20*a, biasing the offset tail.
"""


def hernquist_birth_radius(a, rng=None, size=1, r_cap_factor=_R_CAP_FACTOR):
    """Draw birth radii from the Hernquist stellar density profile.

    The enclosed mass fraction is M(<r)/M_tot = (r/(r+a))^2, so the
    inverse CDF gives r = a * sqrt(u) / (1 - sqrt(u)) for uniform
    u in [0, 1).  Capped at ``r_cap_factor * a`` (default 1000*a)
    to bound numerical edge cases without significantly distorting
    the distribution.

    Parameters
    ----------
    a : float
        Hernquist scale radius [same units as output].
    rng : np.random.Generator, optional
    size : int
        Number of samples.
    r_cap_factor : float, optional
        Multiplier of ``a`` for the upper cap.  See ``_R_CAP_FACTOR``
        for the rationale; the default yields a pileup fraction
        ~1e-6.

    Returns
    -------
    r_birth : array of shape (size,)
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0, 1, size=size)
    su = np.sqrt(u)
    r = a * su / (1.0 - su)
    return np.minimum(r, r_cap_factor * a)


def hernquist_potential(r, M_gal, a):
    """Hernquist (1990) gravitational potential Phi(r) = -GM/(r+a)."""
    return -G_CGS * M_gal / (r + a)


def hernquist_acceleration(r, M_gal, a):
    """Radial gravitational acceleration (inward, negative) for Hernquist profile."""
    return -G_CGS * M_gal / (r + a)**2


def escape_velocity(r, M_gal, a):
    """Escape velocity at radius r in Hernquist potential [cm/s]."""
    return np.sqrt(2.0 * G_CGS * M_gal / (r + a))


# ═══════════════════════════════════════════════════════════════════════════
# Orbit integration
# ═══════════════════════════════════════════════════════════════════════════
def _orbit_rhs(t, y, M_gal, a):
    """Right-hand side for 2D orbit in Hernquist potential (polar: r, v_r, L)."""
    r, vr = y[0], y[1]
    L = y[2]
    r = max(r, 1e-3 * a)
    a_grav = hernquist_acceleration(r, M_gal, a)
    drdt = vr
    dvrdt = a_grav + L**2 / r**3
    dLdt = 0.0
    return [drdt, dvrdt, dLdt]


def integrate_orbit(v_sys_cm, t_delay_s, M_gal, a, r0=None,
                    theta_launch=None, rng=None):
    """Integrate a single orbit in Hernquist potential.

    Parameters
    ----------
    v_sys_cm : float
        Systemic velocity [cm/s].
    t_delay_s : float
        Delay time (merger minus formation) [s].
    M_gal : float
        Galaxy mass [g].
    a : float
        Hernquist scale radius [cm].
    r0 : float or None
        Launch radius as fraction of scale radius.  If None (default),
        draws from the Hernquist stellar density profile via inverse-CDF
        sampling so that birth sites trace the galaxy light.
    theta_launch : float or None
        Angle between kick velocity and radial direction [rad].
        None (default) draws isotropically from arccos(U(-1,1)),
        following Bloom et al. (1999).
    rng : np.random.Generator, optional
        Random generator for isotropic sampling.

    Returns
    -------
    r_final : float
        Galactocentric distance at merger [cm].
    """
    if r0 is None:
        if rng is None:
            rng = np.random.default_rng()
        r_init = hernquist_birth_radius(a, rng=rng, size=1)[0]
    else:
        r_init = r0 * a

    if v_sys_cm <= 0 or t_delay_s <= 0:
        return r_init

    if theta_launch is None:
        if rng is None:
            rng = np.random.default_rng()
        theta_launch = np.arccos(rng.uniform(-1, 1))

    vr0 = v_sys_cm * np.cos(theta_launch)
    vt0 = v_sys_cm * np.sin(theta_launch)
    L0 = r_init * vt0

    y0 = [r_init, vr0, L0]
    t_span = (0.0, t_delay_s)

    try:
        sol = solve_ivp(_orbit_rhs, t_span, y0, args=(M_gal, a),
                        method='RK45', rtol=1e-8, atol=1e-10,
                        max_step=t_delay_s / 50.0,
                        dense_output=False)
        if sol.success:
            r_final = abs(sol.y[0, -1])
            return min(r_final, _R_CAP_FACTOR * a)
        return r_init
    except Exception:
        return r_init


def compute_offset_single(v_sys_km, t_delay_myr, M_gal=DEFAULT_M_GAL,
                          R_e=DEFAULT_R_E, n_angles=8, rng=None):
    """Compute projected offset for a single binary [kpc].

    Launches the binary from a birthplace near the galaxy center and
    integrates through the Hernquist potential.  The projected offset
    is the median over ``n_angles`` random viewing angles.

    Parameters
    ----------
    v_sys_km : float
        Systemic velocity [km/s].
    t_delay_myr : float
        Delay time [Myr].
    M_gal : float
        Galaxy stellar mass [g].
    R_e : float
        Effective radius [cm].
    n_angles : int
        Number of random projection angles for median.
    rng : np.random.Generator, optional

    Returns
    -------
    offset_kpc : float
        Median projected galactocentric offset [kpc].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    a = hernquist_scale_radius(R_e)
    v_cm = v_sys_km * KM_CM
    t_s = t_delay_myr * MYR_S

    r_3d = integrate_orbit(v_cm, t_s, M_gal, a, rng=rng)

    cos_theta = rng.uniform(-1, 1, size=n_angles)
    projected = r_3d * np.sqrt(1.0 - cos_theta**2)
    offset_kpc = np.median(projected) / KPC_CM
    return offset_kpc


# ═══════════════════════════════════════════════════════════════════════════
# Analytic bound-orbit approximation (fast path)
# ═══════════════════════════════════════════════════════════════════════════
def _hernquist_apocenter(E, L, M_gal, a):
    """Find the apocenter of a bound orbit in a Hernquist potential.

    Solves E = L^2/(2*r^2) - GM/(r+a) for the outer turning point
    where the radial velocity vanishes.
    """
    GM = G_CGS * M_gal

    def eff_potential_minus_E(r):
        return 0.5 * L**2 / r**2 - GM / (r + a) - E

    try:
        return brentq(eff_potential_minus_E, 1e-3 * a, _R_CAP_FACTOR * a)
    except ValueError:
        return _R_CAP_FACTOR * a


def _analytic_offset(v_sys_km, t_delay_myr, M_gal, a, r0_frac=None, rng=None):
    """Fast analytic estimate of 3D galactocentric distance at merger.

    For bound orbits in a Hernquist potential, uses energy and angular
    momentum conservation to find the apocenter via root-finding in the
    true Hernquist effective potential, then estimates the time-averaged
    radius.  Falls back to full integration for unbound orbits and
    short-delay systems.

    The kick direction ``theta_launch`` is drawn isotropically once
    per call (Bloom+ 1999) and is threaded through to ``integrate_orbit``
    so the bound/unbound branching uses the same orbit that is then
    actually integrated.

    Returns r_3d [cm].
    """
    if rng is None:
        rng = np.random.default_rng()

    if r0_frac is None:
        r0 = hernquist_birth_radius(a, rng=rng, size=1)[0]
    else:
        r0 = r0_frac * a
    v_cm = v_sys_km * KM_CM
    t_s = t_delay_myr * MYR_S

    if v_cm <= 0 or t_s <= 0:
        return r0

    theta_launch = np.arccos(rng.uniform(-1, 1))

    E = 0.5 * v_cm**2 + hernquist_potential(r0, M_gal, a)
    v_esc = escape_velocity(r0, M_gal, a)

    if v_cm >= v_esc:
        r0_f = r0 / a
        return integrate_orbit(v_cm, t_s, M_gal, a, r0=r0_f,
                               theta_launch=theta_launch, rng=rng)

    vr = v_cm * np.cos(theta_launch)
    vt = v_cm * np.sin(theta_launch)
    L = r0 * vt

    GM = G_CGS * M_gal

    r_apo = _hernquist_apocenter(E, L, M_gal, a)
    r_mean = (r0 + r_apo) / 2.0

    # Period heuristic.  The exact radial period in a Hernquist
    # potential is T_r = 2 * integral_{r_peri}^{r_apo} dr / sqrt(2*(E - Phi_eff))
    # (Binney & Tremaine 2008, "Galactic Dynamics", Section 3.1).
    # The expression below recovers the Kepler limit for r_mean >> a
    # and is order-of-magnitude correct for r_mean ~ a; for r_mean << a
    # it overestimates the period.  Used here only to branch between
    # the time-averaged-radius shortcut (when t >> P_est) and full
    # numerical integration, so a coarse estimate is sufficient.
    P_est = 2.0 * np.pi * np.sqrt(r_mean**3 / GM) * (1 + a / r_mean)

    if t_s > 5.0 * P_est:
        return r_mean
    else:
        r0_f = r0 / a
        return integrate_orbit(v_cm, t_s, M_gal, a, r0=r0_f,
                               theta_launch=theta_launch, rng=rng)


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized population computation
# ═══════════════════════════════════════════════════════════════════════════
def compute_offsets_population(v_sys_km, t_delay_myr, weights=None,
                               M_gal=DEFAULT_M_GAL, R_e=DEFAULT_R_E,
                               use_analytic=True, max_systems=50000,
                               rng=None):
    """Compute projected offsets for a population of merging binaries.

    Parameters
    ----------
    v_sys_km : array
        Systemic velocities [km/s].
    t_delay_myr : array
        Delay times [Myr].
    weights : array, optional
        STROOPWAFEL weights for subsetting. If more than ``max_systems``
        have finite v_sys, a weight-based subsample is drawn.
    M_gal : float
        Galaxy stellar mass [g].
    R_e : float
        Effective radius [cm].
    use_analytic : bool
        If True, use the fast analytic approximation for bound orbits.
    max_systems : int
        Maximum number of systems to integrate (subsampled if exceeded).
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys:
        'offsets_kpc'  : array of projected offsets [kpc]
        'indices'      : indices into the input arrays
        'weights_sub'  : corresponding weights
    """
    if rng is None:
        rng = np.random.default_rng(42)

    v_sys_km = np.asarray(v_sys_km, dtype=float)
    t_delay_myr = np.asarray(t_delay_myr, dtype=float)

    valid = np.isfinite(v_sys_km) & np.isfinite(t_delay_myr) & (v_sys_km > 0) & (t_delay_myr > 0)
    valid_idx = np.where(valid)[0]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        w_valid = weights[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    if len(valid_idx) > max_systems:
        p = w_valid / w_valid.sum()
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False, p=p)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    a = hernquist_scale_radius(R_e)
    n_proj = 8
    offsets = np.zeros(len(valid_idx))

    for i, idx in enumerate(valid_idx):
        v = v_sys_km[idx]
        t = t_delay_myr[idx]

        if use_analytic:
            r_3d = _analytic_offset(v, t, M_gal, a, rng=rng)
        else:
            r_3d = integrate_orbit(v * KM_CM, t * MYR_S, M_gal, a, rng=rng)

        cos_theta = rng.uniform(-1, 1, size=n_proj)
        projected = r_3d * np.sqrt(1.0 - cos_theta**2)
        offsets[i] = np.median(projected) / KPC_CM

    return {
        'offsets_kpc': offsets,
        'indices': valid_idx,
        'weights_sub': w_valid,
    }


def compute_offsets_mixed_hosts(v_sys_km, t_delay_myr, weights=None,
                                host_models=None, max_systems=50000,
                                rng=None, **kw):
    """Compute offsets for a population using a mixture of host galaxy types.

    A single shared subsample is drawn once and then each host potential
    is evaluated on the *same* set of binaries, avoiding inconsistent
    subsampling across galaxy types.

    Parameters
    ----------
    host_models : dict, optional
        Keys are host names, values are dicts with ``M_gal``, ``R_e``,
        and ``weight`` (mixture fraction, should sum to 1).
        Defaults to ``HOST_MODELS``.
    max_systems : int
        Maximum population size (weight-based subsample if exceeded).
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys:
        'per_host' : {name: compute_offsets_population result}
        'mixed_offsets' : combined offset array
        'mixed_weights' : combined weight array (including host mixture)
    """
    if host_models is None:
        host_models = HOST_MODELS
    if rng is None:
        rng = np.random.default_rng(42)

    v = np.asarray(v_sys_km, dtype=float)
    t = np.asarray(t_delay_myr, dtype=float)
    valid = np.isfinite(v) & np.isfinite(t) & (v > 0) & (t > 0)
    valid_idx = np.where(valid)[0]
    if weights is not None:
        w_valid = np.asarray(weights, dtype=float)[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    if len(valid_idx) > max_systems:
        p = w_valid / w_valid.sum()
        chosen = rng.choice(len(valid_idx), size=max_systems,
                            replace=False, p=p)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    per_host = {}
    all_off, all_w = [], []
    for name, hp in host_models.items():
        res = compute_offsets_population(
            v[valid_idx], t[valid_idx], weights=w_valid,
            M_gal=hp['M_gal'], R_e=hp['R_e'],
            max_systems=len(valid_idx), rng=rng, **kw)
        per_host[name] = res
        all_off.append(res['offsets_kpc'])
        all_w.append(res['weights_sub'] * hp['weight'])

    return {
        'per_host': per_host,
        'mixed_offsets': np.concatenate(all_off),
        'mixed_weights': np.concatenate(all_w),
    }


def assign_host_by_delay(t_delay_myr, t_sf_max=3000.0, rng=None):
    """Assign each binary to a host galaxy type based on delay time.

    Short delay (< *t_sf_max* Myr) systems are placed in star-forming
    hosts (split between SF_disk and SF_massive with 2:1 ratio).
    Long delay systems go to elliptical hosts.  This connects GRB class
    properties (delay time tracks M_tot) to host type, following the
    observation from Fong & Berger (2013) that ~25% of sGRBs are in
    elliptical hosts with older stellar populations.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random generator for the SF_disk / SF_massive split.  If None,
        defaults to ``np.random.default_rng(42)`` for backward
        compatibility.  Pass an external ``rng`` to keep this function
        consistent with a global reproducible stream.

    Returns
    -------
    host_assignment : array of str
        Host model key per system ('SF_disk', 'SF_massive', 'Elliptical').
    """
    t = np.asarray(t_delay_myr)
    out = np.full(len(t), 'Elliptical', dtype='<U12')
    sf_mask = t < t_sf_max
    n_sf = sf_mask.sum()
    if rng is None:
        rng = np.random.default_rng(42)
    disk_frac = rng.random(n_sf) < 0.67
    out[sf_mask] = np.where(disk_frac, 'SF_disk', 'SF_massive')
    return out


def compute_offsets_delay_hosts(v_sys_km, t_delay_myr, weights=None,
                                t_sf_max=3000.0, host_models=None, **kw):
    """Compute offsets with delay-time-dependent host assignment.

    Each system is assigned a host type via ``assign_host_by_delay``,
    then its offset is computed in the corresponding galaxy potential.

    Returns
    -------
    dict with 'offsets_kpc', 'weights_sub', 'host_assignments'.
    """
    if host_models is None:
        host_models = HOST_MODELS
    rng = kw.pop('rng', None)
    if rng is None:
        rng = np.random.default_rng(42)

    v = np.asarray(v_sys_km, dtype=float)
    t = np.asarray(t_delay_myr, dtype=float)
    hosts = assign_host_by_delay(t, t_sf_max=t_sf_max, rng=rng)

    valid = np.isfinite(v) & np.isfinite(t) & (v > 0) & (t > 0)
    valid_idx = np.where(valid)[0]
    if weights is not None:
        w_valid = np.asarray(weights, dtype=float)[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    max_systems = kw.pop('max_systems', 50000)
    if len(valid_idx) > max_systems:
        p = w_valid / w_valid.sum()
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False, p=p)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    offsets = np.zeros(len(valid_idx))
    n_proj = 8
    for i, idx in enumerate(valid_idx):
        hp = host_models[hosts[idx]]
        a_host = hernquist_scale_radius(hp['R_e'])
        r_3d = _analytic_offset(v[idx], t[idx], hp['M_gal'], a_host, rng=rng)
        cos_th = rng.uniform(-1, 1, size=n_proj)
        projected = r_3d * np.sqrt(1.0 - cos_th**2)
        offsets[i] = np.median(projected) / KPC_CM

    return {
        'offsets_kpc': offsets,
        'weights_sub': w_valid,
        'host_assignments': hosts[valid_idx],
    }


# ═══════════════════════════════════════════════════════════════════════════
# CDF helper
# ═══════════════════════════════════════════════════════════════════════════
def weighted_offset_cdf(offsets, weights):
    """Build a weighted CDF from offset and weight arrays.

    Returns
    -------
    sorted_offsets : array
        Sorted offset values.
    cdf : array
        Cumulative probability (0 to 1).
    """
    ok = np.isfinite(offsets) & (offsets > 0) & np.isfinite(weights)
    if ok.sum() < 2:
        return np.array([0.0]), np.array([0.0])

    o = offsets[ok]
    w = weights[ok]
    order = np.argsort(o)
    o_sorted = o[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    cdf /= cdf[-1]
    return o_sorted, cdf


# ═══════════════════════════════════════════════════════════════════════════
# Observed offset data (Fong & Berger 2013, Table 3)
# ═══════════════════════════════════════════════════════════════════════════
OBSERVED_SGRB_OFFSETS_KPC = np.array([
    0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.2, 3.0, 3.8, 4.2,
    4.5, 5.0, 5.4, 7.0, 7.4, 8.0, 10.0, 14.6, 18.0, 29.0,
    39.0, 73.0,
])

OBSERVED_LGRB_MERGER_OFFSETS_KPC = np.array([
    1.2, 2.3, 5.7, 14.7,
])
