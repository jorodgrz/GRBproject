"""Canonical project palette and ApJ rcParams.

Fixed colour palette and a canonical
``matplotlib.rcParams`` block for every paper-bound figure.  This
module is the single source of truth: the palette dictionary below
is the project's locked palette, and ``apply_apj_rcparams()``
registers the rcParams the figure-generation code uses across
notebooks.

Tests in ``tests/unit/test_palette_and_rcparams.py`` lock the hex codes
and the rcParams settings against drift.

References
----------
Crameri, Shephard and Heron (2020), Nat. Commun. 11, 5444.
ApJ figure guidelines (single column 3.5 in, double column 7.0 in).
"""

from __future__ import annotations

C_SB_BLUE: str = "#06B6D4"
"""sbGRB plus blue KN (long-lived HMNS engine)."""

C_LB_HMNS: str = "#DC2626"
"""lbGRB plus red KN (short-lived HMNS engine)."""

C_LB_DISK: str = "#DC2626"
"""lbGRB plus red KN (disk; prompt collapse, massive disk)."""

C_FAINT: str = "#F59E0B"
"""Faint lbGRB (prompt collapse, small disk)."""

C_NO_GRB: str = "#334155"
"""No GRB / background."""

C_MODEL_A: str = "#1D4ED8"
"""Model A reference (blue, solid)."""

C_MODEL_F: str = "#243c6e"
"""Model F (alpha_CE=0.5; cividis 0.15, dashed)."""

C_MODEL_G: str = "#7d7c78"
"""Model G (alpha_CE=2.0; cividis 0.50, dash-dot)."""

C_MODEL_J: str = "#d6c35d"
"""Model J (M_NS,max=2.0; cividis 0.85, dotted)."""

C_MODEL_K: str = "#DC2626"
"""Model K reference (red, dash-dot-dot-dot)."""

C_WP15: str = "#6366F1"
"""Wanderman and Piran 2015 R(z) overlay (indigo, dashed)."""

C_OBSERVED: str = "#6B21A8"
"""Observed sample overlay (purple)."""


CLASS_PALETTE: dict[str, str] = {
    "sbGRB + blue KN": C_SB_BLUE,
    "lbGRB + red KN (HMNS)": C_LB_HMNS,
    "lbGRB + red KN (disk)": C_LB_DISK,
    "Faint lbGRB": C_FAINT,
    "No GRB": C_NO_GRB,
}

MODEL_PALETTE: dict[str, str] = {
    "A": C_MODEL_A,
    "F": C_MODEL_F,
    "G": C_MODEL_G,
    "J": C_MODEL_J,
    "K": C_MODEL_K,
}

MODEL_LINESTYLES: dict[str, object] = {
    "A": "-",
    "F": "--",
    "G": "-.",
    "J": ":",
    "K": (0, (3, 1, 1, 1)),
}


APJ_RCPARAMS: dict[str, object] = {
    "font.size": 8,
    "mathtext.fontset": "cm",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "savefig.bbox": "tight",
}


def apply_apj_rcparams(extra: dict[str, object] | None = None) -> None:
    """Register the canonical ApJ rcParams on the active matplotlib runtime.

    Parameters
    ----------
    extra : dict, optional
        Additional rcParams overrides applied on top of ``APJ_RCPARAMS``.
    """
    import matplotlib as mpl

    mpl.rcParams.update(APJ_RCPARAMS)
    if extra:
        mpl.rcParams.update(extra)
