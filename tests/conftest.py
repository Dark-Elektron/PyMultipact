"""Shared test fixtures.

The package reads its sample data (sample_domains/, sample_seys/) relative to
the current working directory, so all tests run from the repository root.
"""
import os
import pathlib

import matplotlib

matplotlib.use('Agg')  # no GUI windows during tests
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: None  # analysis functions must never block

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_repo_root():
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)
