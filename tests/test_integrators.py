"""Integrator helpers: geometry predicates, index adjustment, dummy particles,
loss-model validation."""
import numpy as np
import pytest

from pymultipact.integrators import Integrators, _ParticleDummy
from pymultipact.particles import Particles


def _wall(n=400):
    t = np.linspace(0.0, np.pi, n)
    return np.column_stack([0.05 * np.cos(t), 0.03 + 0.07 * np.sin(t)])


def _integ(loss_model='field'):
    # mesh is unused by the helpers under test
    return Integrators(None, 2 * np.pi * 1.3e9, bounding_rect=[-1, 1, 0, 1],
                       loss_model=loss_model)


def test_cross_dot_norm():
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0]])
    # cross of in-plane velocity with azimuthal B: (u_r*B, -u_z*B)
    assert np.allclose(Integrators.cross(a, b), [[2.0 * 3.0, -1.0 * 3.0]])
    assert np.allclose(Integrators.dot(a, np.array([[5.0, 6.0]])), [[17.0]])
    assert np.allclose(Integrators.norm(np.array([[3.0, 4.0]])), [[5.0]])


def test_segment_intersection_hit_and_miss():
    # wall segments: unit horizontal line split in two
    starts = np.array([[0.0, 0.0], [0.5, 0.0]])
    ends = np.array([[0.5, 0.0], [1.0, 0.0]])
    # particle crossed from above to below inside the second segment
    ok, pt, idx = Integrators.segment_intersection(
        (np.array([0.75, -0.1]), np.array([0.75, 0.1])), (starts, ends))
    assert ok
    assert np.allclose(pt, [0.75, 0.0])
    assert idx == 1
    # path that never reaches the wall
    ok, _, _ = Integrators.segment_intersection(
        (np.array([0.75, 0.2]), np.array([0.75, 0.1])), (starts, ends))
    assert not ok


def test_update_lpi_shifts_indices_after_bright_removal():
    integ = _integ()
    # bright particles 2 and 5 were removed; lost indices above them shift down
    lpi = [1, 3, 6]
    adjusted = integ.update_lpi(lpi, [2, 5])
    assert list(adjusted) == [1, 2, 4]


def test_particle_dummy_matches_particles_state():
    p = Particles([-0.02, 0.02], 2, _wall(), np.linspace(0, 2 * np.pi, 3),
                  cmap='jet', step=None)
    p.save_old()
    d = _ParticleDummy(p)
    d.save_old()
    assert np.array_equal(d.x, p.x)
    assert np.array_equal(d.u, p.u)
    assert np.array_equal(d.phi, p.phi)
    # copies, not views: mutating the dummy must not touch the particles
    d.x += 1.0
    assert not np.allclose(d.x, p.x)
    # identical nearest-surface query
    res_p, idx_p = p.distance(5)
    res_d, idx_d = _ParticleDummy(p).distance(5)
    assert np.allclose(res_p, res_d)
    assert idx_p == idx_d


def test_loss_model_validation():
    for ok in ('field', 'wait', 'always'):
        assert _integ(ok).loss_model == ok
    with pytest.raises(ValueError):
        _integ('bogus')
