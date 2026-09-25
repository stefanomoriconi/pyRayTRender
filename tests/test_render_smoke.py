import matplotlib
matplotlib.use('Agg')
import numpy as np

from pyraytrender import Camera, Geometry, Light, Renderer


def test_render_smoke_nontrivial_image():
    sphere = Geometry(label='Sphere')
    sphere.genIcoSphere(c3=[0.0, 0.0, -4.0], rad=0.8, nsub=1)

    mirror = Geometry(label='Mirror')
    mirror.genPlane()
    mirror.scale(2.0)
    mirror.tformRigid(-np.pi / 2, 0, np.pi / 8, -0.5, -3.0, -4.0)
    mirror.mat.__init__(ambRGB=[0.2, 0.2, 0.2], dffRGB=[0.02, 0.02, 0.02], spcRGB=[0.85, 0.85, 0.85], shnC=100.0, rflC=0.999)

    floor = Geometry(label='Floor')
    floor.genPlane()
    floor.scale(40.0)
    floor.tformRigid(0, 0, 0, -20.0, -20.0, -5.0)
    floor.mat.__init__(ambRGB=[0.5, 0.5, 0.5], dffRGB=[0.95, 0.95, 0.95], shnC=16.0)

    camera = Camera(w=40, h=30)
    camera.zoom(1.0)
    camera.setTrg3D(sphere.CoM)
    camera.lockOnTarget(True)
    camera.lockScreenOrient(True)

    light = Light(ctr=[2.0, 2.0, 2.0])
    renderer = Renderer(
        cam=camera,
        geo=(sphere, mirror, floor),
        lgt=light,
        shadowsFlag=True,
        reflectionsFlag=True,
        dwnSmpl=1,
    )
    renderer.shoot()
    img = renderer.ImgCanvas

    assert img.shape == (30, 40, 3)
    assert np.issubdtype(img.dtype, np.floating)
    assert np.std(img) > 0.01
    assert not np.allclose(img, img[0, 0])
