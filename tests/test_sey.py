"""Secondary emission yield table handling."""
import pickle

import numpy as np

from pymultipact.domain import SEY


def test_sey_loads_sample_table():
    sey = SEY('sample_seys/sey')
    assert sey.Emin == 0.0
    assert sey.Emax == 1e12


def test_sey_interpolates_table_nodes_exactly():
    sey = SEY('sample_seys/sey')
    E = np.asarray(sey.data['E'])
    d = np.asarray(sey.data['sey'])
    for Ei, di in zip(E, d):
        assert abs(float(sey.sey(Ei)) - di) < 1e-12


def test_sey_sane_in_sparse_high_energy_gap():
    """The table jumps from 1.9 keV to a 1e12 eV sentinel row. A cubic spline
    used to oscillate to ~1e6 in that gap, poisoning the e20/c0 metric for any
    impact above ~1.9 keV; linear interpolation must stay within the table's
    value range."""
    sey = SEY('sample_seys/sey')
    d = np.asarray(sey.data['sey'])
    for Eq in (2e3, 5e4, 1e6, 1e9, 5e11):
        val = float(sey.sey(Eq))
        assert d.min() - 1e-12 <= val <= d.max() + 1e-12


def test_sey_is_picklable():
    """SEY is passed through multiprocessing arguments by the parallel sweep."""
    sey = SEY('sample_seys/sey')
    clone = pickle.loads(pickle.dumps(sey))
    assert abs(float(clone.sey(123.0)) - float(sey.sey(123.0))) < 1e-12
