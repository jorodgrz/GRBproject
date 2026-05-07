"""Shared pytest fixtures.

Tests that depend on the COMPAS HDF5 catalogues in ``Data/`` should
mark themselves with ``@pytest.mark.requires_data`` and consume the
``compas_data_available`` fixture so they skip cleanly on machines
where ``Data/`` is empty (e.g. CI runners that do not re-download
the multi-GB Broekgaarden+ 2021 Zenodo archives).
"""

from __future__ import annotations

import os
import sys

import pytest

# Add the repo root to sys.path so the test files can ``import grb_*``
# directly.  The repo is a flat package layout (no setup.py); without
# this the tests can only run from the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _hdf5_present(name: str) -> bool:
    return os.path.exists(os.path.join(_REPO_ROOT, "Data", name))


@pytest.fixture(scope="session")
def repo_root() -> str:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def compas_data_available() -> bool:
    """True if all four COMPAS HDF5 files are present locally."""
    return all(_hdf5_present(name) for name in (
        "COMPASCompactOutput_BNS_A.h5",
        "COMPASCompactOutput_BNS_K.h5",
        "COMPASCompactOutput_BHNS_A.h5",
        "COMPASCompactOutput_BHNS_K.h5",
    ))


@pytest.fixture(scope="session")
def bns_a_path(compas_data_available) -> str:
    if not compas_data_available:
        pytest.skip("COMPASCompactOutput_BNS_A.h5 not present in Data/")
    return os.path.join(_REPO_ROOT, "Data", "COMPASCompactOutput_BNS_A.h5")


@pytest.fixture(scope="session")
def bhns_a_path(compas_data_available) -> str:
    if not compas_data_available:
        pytest.skip("COMPASCompactOutput_BHNS_A.h5 not present in Data/")
    return os.path.join(_REPO_ROOT, "Data", "COMPASCompactOutput_BHNS_A.h5")


@pytest.fixture(scope="session")
def rastinejad_csv_path() -> str:
    path = os.path.join(_REPO_ROOT, "Data", "rastinejad_2024.csv")
    if not os.path.exists(path):
        pytest.skip("Data/rastinejad_2024.csv not present")
    return path
