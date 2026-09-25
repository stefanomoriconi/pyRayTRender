import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyraytrender import Camera, Geometry, Light, Renderer


def main():
    out_dir = Path(__file__).resolve().parent / 'output'
    out_dir.mkdir(exist_ok=True)

    red = Geometry(label='RedSphere')
    red.genIcoSphere(c3=[0, 0, -4], rad=0.7, nsub=1)
    red.mat.resetMat()

    blue = Geometry(label='BlueSphere')
    blue.genIcoSphere(c3=[-1.8, -1.0, -4.6], rad=0.9, nsub=1)
    blue.mat.__init__(ambRGB=[0.0, 0.0, 0.15], dffRGB=[0.1, 0.2, 0.8], shnC=20.0)

    mirror = Geometry(label='Mirror')
    mirror.genPlane()
    mirror.scale(2.0)
    mirror.tformRigid(-3.141592653589793/2, 0, 3.141592653589793/8, -0.5, -3, -4)
    mirror.mat.__init__(ambRGB=[0.2, 0.2, 0.2], dffRGB=[0.02, 0.02, 0.02], spcRGB=[0.85, 0.85, 0.85], shnC=100.0, rflC=0.999)

    floor = Geometry(label='Floor')
    floor.genPlane()
    floor.scale(100.0)
    floor.tformRigid(0, 0, 0, -50, -50, -5)
    floor.mat.__init__(ambRGB=[0.5, 0.5, 0.5], dffRGB=[0.95, 0.95, 0.95], shnC=16.0)

    sky = Geometry(label='Sky', visibleFlag=False)
    sky.genIcoSphere(c3=[0, 0, -4], rad=15.0, nsub=1)
    sky.flipNormals()
    sky.mat.__init__(ambRGB=[0.9, 0.9, 1.0], dffRGB=[0.9, 0.9, 1.0], shnC=10.0)

    camera = Camera(w=140, h=100)
    camera.zoom(1.0)
    camera.setTrg3D(red.CoM)
    camera.lockOnTarget(True)
    camera.lockScreenOrient(True)
    light = Light(ctr=camera.eye + [1.0, 0.5, 1.5])

    renderer = Renderer(cam=camera, geo=(red, blue, mirror, floor, sky), lgt=light, dwnSmpl=2)
    renderer.shoot(frmFileName=str(out_dir / 'multiple_geometry.png'))


if __name__ == '__main__':
    main()
