"""Regression test for per_system_rate_weights / compute_merger_rate.

Locks in the fix for an indexing bug introduced in commit e6188cb where
``per_system_rate_weights`` raised

    TypeError: NumPy boolean array indexing assignment requires
               a 0 or 1-dimensional input, input has 2 dimensions

on every call.  The bug came from the new ``_interp_formation_rate``
helper: it was passed ``dPdlogZ_binned[:, sys_col[valid]]`` (2-D) and
then row-indexed with ``z_lo`` (1-D), producing a 2-D output that could
not be assigned back into ``out[valid]``.  Fixed by passing the full
2-D array plus paired column indices and using ``[z_lo, cols]`` paired
advanced indexing.

The test also pins the (intended) physical equivalence

    sum_i per_system_rate_weights(z, ...)[i] == compute_merger_rate(...)[j(z)]

which both reproduces the bug fix and guards against future regressions
in the cosmic-rate pipeline.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grb_rates import compute_merger_rate, per_system_rate_weights


def _toy_inputs(n_sys=64, n_z=200, seed=0):
    rng = np.random.default_rng(seed)
    redshifts = np.linspace(0.0, 5.0, n_z)
    times = np.linspace(13700.0, 1100.0, n_z)
    time_first_SF = 1100.0
    n_formed = np.full(n_z, 1.0)
    metallicities = np.logspace(-4, -1.5, 53)
    dPdlogZ = np.ones((n_z, len(metallicities))) / len(metallicities)
    p_draw = 1.0
    COMPAS_Z = rng.choice(metallicities, size=n_sys)
    COMPAS_delay_times = rng.uniform(10.0, 5000.0, size=n_sys)
    COMPAS_weights = np.ones(n_sys)
    return dict(
        redshifts=redshifts, times=times, time_first_SF=time_first_SF,
        n_formed=n_formed, dPdlogZ=dPdlogZ, metallicities=metallicities,
        p_draw=p_draw, COMPAS_Z=COMPAS_Z,
        COMPAS_delay_times=COMPAS_delay_times,
        COMPAS_weights=COMPAS_weights,
    )


def test_per_system_rate_weights_does_not_crash():
    args = _toy_inputs()
    w = per_system_rate_weights(
        0.5, args["redshifts"], args["times"], args["time_first_SF"],
        args["n_formed"], args["dPdlogZ"], args["metallicities"],
        args["p_draw"], args["COMPAS_Z"], args["COMPAS_delay_times"],
        args["COMPAS_weights"])
    assert w.ndim == 1
    assert w.shape == args["COMPAS_weights"].shape
    assert np.all(np.isfinite(w))
    assert w.sum() > 0


def test_per_system_sum_matches_total_rate():
    args = _toy_inputs()
    rate = compute_merger_rate(
        args["redshifts"], args["times"], args["time_first_SF"],
        args["n_formed"], args["dPdlogZ"], args["metallicities"],
        args["p_draw"], args["COMPAS_Z"], args["COMPAS_delay_times"],
        args["COMPAS_weights"], smooth_sigma=0)
    for z_t in (0.0, 0.3, 1.0, 2.0):
        w = per_system_rate_weights(
            z_t, args["redshifts"], args["times"], args["time_first_SF"],
            args["n_formed"], args["dPdlogZ"], args["metallicities"],
            args["p_draw"], args["COMPAS_Z"], args["COMPAS_delay_times"],
            args["COMPAS_weights"])
        j = int(np.argmin(np.abs(args["redshifts"] - z_t)))
        assert np.isclose(rate[j], w.sum(), rtol=1e-10), (
            f"per-system sum != total rate at z={z_t}: "
            f"rate={rate[j]:.6e} sum(w)={w.sum():.6e}")


def test_empty_population_guards():
    args = _toy_inputs(n_sys=0)
    rate = compute_merger_rate(
        args["redshifts"], args["times"], args["time_first_SF"],
        args["n_formed"], args["dPdlogZ"], args["metallicities"],
        args["p_draw"], args["COMPAS_Z"], args["COMPAS_delay_times"],
        args["COMPAS_weights"])
    assert rate.shape == (len(args["redshifts"]),)
    assert np.all(rate == 0.0)
    w = per_system_rate_weights(
        0.5, args["redshifts"], args["times"], args["time_first_SF"],
        args["n_formed"], args["dPdlogZ"], args["metallicities"],
        args["p_draw"], args["COMPAS_Z"], args["COMPAS_delay_times"],
        args["COMPAS_weights"])
    assert w.shape == (0,)


if __name__ == "__main__":
    test_per_system_rate_weights_does_not_crash()
    test_per_system_sum_matches_total_rate()
    test_empty_population_guards()
    print("OK")
