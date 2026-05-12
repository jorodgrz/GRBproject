"""Section 4 of grb_main.ipynb: Cosmic Integration / MSSFR Grid Setup.

Smoke-level invariant on the Neijssel et al. (2019) log-normal MSSFR.
For redshift slices that fall on the figure, the bin-integrated
``dP/d(ln Z)`` over a Z grid that fully covers the log-normal support
sums to ~1 (the wide grid is the relevant assertion; narrowed COMPAS
grids fall short at high z by construction, see ``grb_rates.py``).

Reference: Neijssel et al. (2019), MNRAS 490, 3740, Eq. (2);
Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 2.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_rates import _bin_averaged_dPdlogZ


@pytest.mark.parametrize("z, tol", [(0.0, 2e-2), (0.5, 3e-2), (1.0, 6e-2)])
def test_dPdlogZ_normalises_at_redshift(z, tol):
    """Voronoi-bin probabilities sum to ~1 on a log-uniform Z grid
    (0.01 to 1.0).  Tolerance widens with redshift because the
    log-normal mean drifts toward ``ln Z_min``: at z = 1 the lower
    ~5 percent of the distribution sits below the grid edge."""
    Z_grid = 10.0 ** np.linspace(-2, 0, 53)
    redshifts = np.array([z])
    dPdlogZ_binned, _ = _bin_averaged_dPdlogZ(redshifts, Z_grid, Z_grid=Z_grid)
    integral = dPdlogZ_binned.sum(axis=1)[0]
    assert integral == pytest.approx(1.0, rel=tol, abs=tol), (
        f"dPdlogZ integral at z={z} is {integral:.4f}; expected ~1."
    )
