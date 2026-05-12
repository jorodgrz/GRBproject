"""Shared pytest fixtures.

Tests that depend on the COMPAS HDF5 catalogues in ``Data/`` should
mark themselves with ``@pytest.mark.requires_data``.  Per-file
fixtures (``bns_a_path``, ``bhns_a_path``, or the parametrize-friendly
``compas_file`` indirect fixture) skip cleanly when the requested
file is absent, so a machine with only a partial download (Tier-1 only,
or in-progress staged download) runs the tests whose data is present
and skips the rest.

The Broekgaarden et al. 2021 grid is 20 physics variations x 2
populations = 40 HDF5 files, and the project's downloader
(``tools/download_compas_data.py``) writes them with the project
filename convention ``COMPASCompactOutput_<KIND>_<SUFFIX>.h5``.
The full set of expected filenames lives in
``tools/embed_model_metadata.KNOWN_FILES``; tests that need to iterate
the grid should import that table rather than re-listing names here.
"""

from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ``tools/`` is a flat-layout helper directory (no __init__.py); add it to
# sys.path so tests can ``import embed_model_metadata`` the same way
# ``tools/download_compas_data.py`` does at line 61.
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

_DATA_DIR = os.path.join(_REPO_ROOT, "Data")


def _data_path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)


def _skip_if_absent(name: str) -> str:
    """Return ``Data/<name>`` or call ``pytest.skip`` if the file is missing."""
    path = _data_path(name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present in Data/")
    return path


@pytest.fixture(scope="session")
def repo_root() -> str:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def compas_data_available() -> bool:
    """True if the Tier-1 COMPAS HDF5 files (BNS/BHNS A and K) are present.

    Retained for callers that want a coarse "is Data/ usable" check;
    per-test fixtures below skip individually rather than relying on
    this all-or-nothing boolean.  With 40 files in the Broekgaarden+
    2021 grid, no realistic developer machine has the full set, so
    new tests should consume ``compas_file`` (parametrize-friendly)
    or ``bns_a_path`` / ``bhns_a_path`` (per-file skip).
    """
    return all(
        os.path.exists(_data_path(name))
        for name in (
            "COMPASCompactOutput_BNS_A.h5",
            "COMPASCompactOutput_BNS_K.h5",
            "COMPASCompactOutput_BHNS_A.h5",
            "COMPASCompactOutput_BHNS_K.h5",
        )
    )


@pytest.fixture(scope="session")
def bns_a_path() -> str:
    return _skip_if_absent("COMPASCompactOutput_BNS_A.h5")


@pytest.fixture(scope="session")
def bhns_a_path() -> str:
    return _skip_if_absent("COMPASCompactOutput_BHNS_A.h5")


@pytest.fixture
def compas_file(request) -> str:
    """Parametrize-friendly per-file fixture for the 20-variation grid.

    Pass the HDF5 filename via ``indirect``::

        from tools.embed_model_metadata import KNOWN_FILES

        @pytest.mark.parametrize("compas_file",
                                 sorted(KNOWN_FILES.keys()),
                                 indirect=True)
        def test_audit(compas_file):
            ...

    Each parameterisation skips its own test instance when the requested
    file is absent, so a partial download exercises only the tests whose
    data is present without poisoning the whole audit run.
    """
    return _skip_if_absent(request.param)


@pytest.fixture(scope="session")
def rastinejad_csv_path() -> str:
    path = _data_path("rastinejad_2024.csv")
    if not os.path.exists(path):
        pytest.skip("Data/rastinejad_2024.csv not present")
    return path


_AUTO_FOLDER_MARKERS = {
    "unit": "unit",
    "sections": None,  # populated per-file from filename below
    "integration": "integration",
    "anchors": "anchors",
}

_SECTION_RE = "test_section_"


def pytest_collection_modifyitems(config, items):
    """Auto-apply ``unit`` / ``integration`` / ``anchors`` / ``section_<N>``
    markers based on a test's parent folder and filename.

    Authors do not have to repeat the marker on every test; the layout
    on disk is the source of truth.  Manual marker decorations stack on
    top of the auto-applied ones.
    """
    for item in items:
        rel = os.path.relpath(str(item.fspath), _REPO_ROOT)
        parts = rel.replace(os.sep, "/").split("/")
        if len(parts) < 2 or parts[0] != "tests":
            continue
        folder = parts[1]
        marker_name = _AUTO_FOLDER_MARKERS.get(folder)
        if marker_name is not None:
            item.add_marker(getattr(pytest.mark, marker_name))
        if folder == "sections":
            stem = os.path.splitext(parts[-1])[0]
            if stem.startswith(_SECTION_RE):
                tail = stem[len(_SECTION_RE) :]
                num = tail.split("_", 1)[0]
                if num.isdigit():
                    item.add_marker(getattr(pytest.mark, f"section_{int(num)}"))
