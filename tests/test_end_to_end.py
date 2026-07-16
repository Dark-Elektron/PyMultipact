"""End-to-end regression tests (slower: ~1-2 min total).

Builds the TESLA mid-cell with the current defaults (resampled boundary,
3rd-order elements) and checks physics observables against reference values
established during validation against the paper / MultiPac.
"""
import numpy as np
import pytest

from pymultipact.domain import Project, Domain


@pytest.fixture(scope="module")
def domain(tmp_path_factory):
    proj = Project()
    proj.create_project(str(tmp_path_factory.mktemp("tesla_project")))
    d = Domain(proj)
    d.compute_fields()
    return d


def test_fundamental_mode_frequency(domain):
    """TM010 of the TESLA mid-cell: 1300 MHz (paper: 1300.02 MHz)."""
    assert abs(domain.eigen_freq[1] - 1300.1) < 1.0


def test_boundary_was_resampled(domain):
    assert len(domain.boundary) < 400   # dense file has ~1831 points


@pytest.mark.slow
def test_multipacting_band_and_normalisation(domain):
    """One in-band and one above-band field level. In-band CF must be finite
    and the above-band CF ~0; the launchable fraction is ~0.5 by symmetry."""
    domain.analyse_multipacting(mode=1, epks=np.array([43e6, 80e6]),
                                phis=np.linspace(0, 2 * np.pi, 24),
                                xrange=[-0.025, 0.0], step=0.002, proc_count=1)
    cf = np.asarray(domain.cn_c0, dtype=float)
    assert 0.05 < cf[0] < 0.35          # resonant band (ref ~0.17)
    assert cf[1] < 0.02                  # above the band
    frac = domain.launchable_fraction()
    assert 0.4 < frac < 0.6
    # results ordered like epks_v, and persisted metrics consistent
    assert len(domain.particles_objects) == 2
    assert domain.particles_left[0] > domain.particles_left[1]


@pytest.mark.slow
def test_parallel_matches_in_process(domain):
    epks = np.array([43e6, 59e6])
    phis = np.linspace(0, 2 * np.pi, 12)
    domain.analyse_multipacting(mode=1, epks=epks, phis=phis,
                                xrange=[-0.025, 0.0], step=0.002, proc_count=1)
    cf_serial = np.asarray(domain.cn_c0, dtype=float)
    domain.analyse_multipacting(mode=1, epks=epks, phis=phis,
                                xrange=[-0.025, 0.0], step=0.002, proc_count=2)
    cf_par = np.asarray(domain.cn_c0, dtype=float)
    assert np.allclose(cf_serial, cf_par)


@pytest.mark.slow
def test_bright_metrics_are_sane(domain):
    """Ef20 in the band sits on the paper's dome (~30 eV at 43 MV/m) and the
    e20/c0 product metric stays near its 2/n_init baseline (it used to explode
    to ~1e100 through bookkeeping misalignment + SEY spline extrapolation)."""
    domain.analyse_multipacting(mode=1, epks=np.array([43e6]),
                                phis=np.linspace(0, 2 * np.pi, 24),
                                xrange=[-0.025, 0.0], step=0.002, proc_count=1)
    p = domain.particles_objects[0]
    Ef = [be[-1] for be in p.bright_E if len(be)]
    assert 10 < np.mean(Ef) < 60
    secondaries = sum(np.prod(nn) for nn in p.bright_n_secondaries)
    e20_c0 = 2 * (secondaries + 1) / domain.n_init_particles
    assert 0 < e20_c0 < 1.0
