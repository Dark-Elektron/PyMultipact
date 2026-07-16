"""Boundary polyline resampling."""
import numpy as np

from pymultipact.domain import Domain


def _dense_ellipse(n=1500):
    t = np.linspace(0, np.pi, n)
    return np.column_stack([0.05 * np.cos(t), 0.02 + 0.08 * np.sin(t)])


def test_resample_reduces_point_count():
    b = _dense_ellipse()
    r = Domain._resample_boundary(b, 200)
    assert len(r) <= 210          # ~target (dedup can only shrink it)
    assert len(r) >= 50


def test_resample_none_is_identity():
    b = _dense_ellipse()
    r = Domain._resample_boundary(b, None)
    assert r is b or np.array_equal(r, b)


def test_resample_short_polyline_untouched():
    b = _dense_ellipse(50)
    r = Domain._resample_boundary(b, 200)
    assert np.array_equal(r, b)


def test_resample_keeps_endpoints_and_extremes():
    b = _dense_ellipse()
    r = Domain._resample_boundary(b, 100)
    assert np.array_equal(r[0], b[0])
    assert np.array_equal(r[-1], b[-1])
    # the extreme-coordinate points define the bounding rectangle and must survive
    for extreme in (b[np.argmin(b[:, 0])], b[np.argmax(b[:, 0])],
                    b[np.argmin(b[:, 1])], b[np.argmax(b[:, 1])]):
        assert any(np.array_equal(extreme, p) for p in r)


def test_resample_points_lie_on_original_profile():
    """Decimation must select original points, never invent new ones."""
    b = _dense_ellipse()
    r = Domain._resample_boundary(b, 120)
    original = {tuple(p) for p in b}
    assert all(tuple(p) in original for p in r)


def test_resample_preserves_order():
    b = _dense_ellipse()
    r = Domain._resample_boundary(b, 120)
    # arc-length ordering along the wall must be preserved
    idx = [np.flatnonzero((b == p).all(axis=1))[0] for p in r]
    assert idx == sorted(idx)
