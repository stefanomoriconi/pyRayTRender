import matplotlib
matplotlib.use('Agg')
import numpy as np

from pyraytrender.utils import dist_p_line, orthog, rotM3D, sect_lin_pln, uvect, isPtInTri


def test_uvect_and_orthog():
    v = uvect(np.array([3.0, 0.0, 4.0]))
    assert np.isclose(np.linalg.norm(v), 1.0)
    o = orthog(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.isclose(np.dot(o, np.array([1.0, 0.0, 0.0])), 0.0)
    assert np.isclose(np.linalg.norm(o), 1.0)


def test_rotm3d_round_trip():
    v = np.array([0.2, -0.3, 0.5])
    R = rotM3D(0.2, -0.1, 0.4)
    restored = R.T @ (R @ v)
    assert np.allclose(restored, v)


def test_dist_p_line():
    pts = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    line_pt = np.array([[0.0, 0.0, 0.0]])
    line_n = np.array([[1.0, 0.0, 0.0]])
    d = dist_p_line(pts, line_pt, line_n)
    assert np.allclose(d, np.array([1.0, 2.0]))


def test_sect_lin_pln_and_point_in_triangle():
    ipt = sect_lin_pln(
        np.array([0.25, 0.25, 1.0]),
        np.array([0.25, 0.25, -1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    assert np.allclose(ipt, np.array([0.25, 0.25, 0.0]))
    tri = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert bool(isPtInTri(ipt, tri)) is True
