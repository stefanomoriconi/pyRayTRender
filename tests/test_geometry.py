import matplotlib
matplotlib.use('Agg')
import numpy as np

from pyraytrender import Geometry


def test_plane_and_triangle_counts():
    plane = Geometry()
    plane.genPlane()
    assert plane.vts.shape == (5, 3)
    assert plane.fcs.shape == (4, 3)
    assert plane.fn.shape == (4, 3)

    tri = Geometry()
    tri.genTriangle()
    assert tri.vts.shape == (3, 3)
    assert tri.fcs.shape == (1, 3)
    assert np.isclose(np.linalg.norm(tri.fn[0]), 1.0)


def test_icosphere_and_subdivide_counts():
    sphere = Geometry()
    sphere.genIcoSphere(nsub=0)
    assert sphere.vts.shape[0] == 12
    assert sphere.fcs.shape[0] == 20

    sphere.subdivide(1)
    assert sphere.vts.shape[0] == 42
    assert sphere.fcs.shape[0] == 80
    assert np.allclose(np.linalg.norm(sphere.fn, axis=1), 1.0)


def test_load_bunny_and_stl_reader():
    bunny = Geometry(label='Bunny')
    bunny.loadBunny()
    assert bunny.vts.shape[0] == 2503
    assert bunny.fcs.shape[0] == 5002
    assert bunny.fn.shape[0] == bunny.fcs.shape[0]

    bunny_stl = Geometry(label='BunnySTL')
    try:
        bunny_stl.readSTL('assets/Bunny.stl')
    except ImportError:
        return
    assert bunny_stl.vts.shape[0] > 0
    assert bunny_stl.fcs.shape[0] > 0
