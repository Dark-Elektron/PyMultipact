"""Particles bookkeeping: index mapping, removals, bright-set archiving."""
import numpy as np
import pytest

from pymultipact.particles import Particles


def _wall(n=400):
    """Synthetic dense cavity-like wall (open arc, r > 0)."""
    t = np.linspace(0.0, np.pi, n)
    return np.column_stack([0.05 * np.cos(t), 0.03 + 0.07 * np.sin(t)])


@pytest.fixture
def particles():
    phis = np.linspace(0, 2 * np.pi, 5)
    return Particles([-0.02, 0.02], 2, _wall(), phis, cmap='jet', step=None)


def test_tiling_and_index_mapping(particles):
    p = particles
    assert len(p.x) == p.n_sites * len(p.phis_v)
    # particle i is site i % n_sites at phase i // n_sites
    for i in (0, p.n_sites, p.n_sites + 1, 2 * p.n_sites - 1):
        assert np.allclose(p.x[i], p.sites_init[i % p.n_sites])
        assert np.isclose(p.phi[i, 0], p.phis_v[i // p.n_sites])


def test_bookkeeping_lists_match_particle_count(particles):
    p = particles
    n = len(p.x)
    assert len(p.E) == len(p.n_secondaries) == n
    assert len(p.nhit) == n


def test_remove_keeps_impact_histories_aligned(particles):
    """Removing any particle must shrink E/n_secondaries in sync -- a missed
    pop used to shift every surviving particle's history."""
    p = particles
    for i in range(len(p.x)):
        p.E[i].append(float(i))         # tag each history with its index
        p.n_secondaries[i].append(float(i))
    victim = 3
    survivor_after = p.x[victim + 1].copy()
    p.remove([victim])
    assert len(p.E) == len(p.x)
    # the particle that WAS at victim+1 is now at victim, with ITS history
    assert np.allclose(p.x[victim], survivor_after)
    assert p.E[victim] == [float(victim + 1)]


def test_bright_removal_archives_history_and_identity(particles):
    p = particles
    ind = 2
    p.E[ind].extend([10.0, 20.0])
    p.n_secondaries[ind].extend([1.5, 0.5])
    init_x = p.x_init[ind].copy()
    init_phi = float(p.phi_init[ind, 0])
    n_before = len(p.x)

    p.nhit[ind] = 19                    # next counted hit reaches 20
    p.paths_count = 1                   # keep path bookkeeping trivial
    removed = p.update_hit_count([ind])

    assert removed == [ind]
    assert len(p.x) == n_before - 1
    assert len(p.bright_set) == 1
    assert p.bright_E[-1] == [10.0, 20.0]
    assert p.bright_n_secondaries[-1] == [1.5, 0.5]
    assert np.allclose(p.bright_init_x[-1], init_x)
    assert np.isclose(p.bright_init_phi[-1], init_phi)
    # live lists shrank in sync
    assert len(p.E) == len(p.x) == len(p.n_secondaries)


def test_sites_init_immune_to_removals(particles):
    p = particles
    sites_before = p.sites_init.copy()
    p.remove([0, 1])
    assert np.array_equal(p.sites_init, sites_before)
    assert p.n_sites == len(sites_before)


def test_select_values_with_step():
    vals = np.array([[0.0, 0], [0.001, 0], [0.003, 0], [0.0031, 0], [0.006, 0]])
    picked = Particles._select_values_with_step(vals, 0.002)
    assert np.allclose(picked[:, 0], [0.0, 0.003, 0.006])
