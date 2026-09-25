import matplotlib
matplotlib.use('Agg')
import numpy as np

from pyraytrender import Camera


def test_camera_motion_preserves_unit_axes():
    cam = Camera(trg=[0, 0, 0])
    cam.lockOnTarget(True)
    cam.lockScreenOrient(True)
    cam.pan3D([0.1, -0.2, 0.05])
    cam.rot3D([0.0, 0.0, 0.1])
    cam.orbit3D([0.1, 0.1, 0.0])
    cam.zoom(0.2)

    assert np.isclose(np.linalg.norm(cam.n1), 1.0)
    assert np.isclose(np.linalg.norm(cam.e1), 1.0)
    assert np.isclose(np.linalg.norm(cam.e2), 1.0)
    assert np.isclose(np.dot(cam.n1, cam.e1), 0.0, atol=1e-6)
    assert cam.imgScreenCrnrs.shape == (4, 3)
