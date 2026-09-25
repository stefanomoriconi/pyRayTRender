import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyraytrender import Camera, Geometry, Light, Renderer


def main():
    out_dir = Path(__file__).resolve().parent / 'output'
    out_dir.mkdir(exist_ok=True)

    sphere = Geometry(label='Sphere')
    sphere.genIcoSphere(c3=[0, 0, -4], rad=0.9, nsub=1)
    mirror = Geometry(label='Mirror')
    mirror.genPlane()
    mirror.scale(2.0)
    mirror.tformRigid(-3.141592653589793/2, 0, 0.0, -0.5, -3, -4)
    mirror.mat.__init__(ambRGB=[0.2, 0.2, 0.2], dffRGB=[0.02, 0.02, 0.02], spcRGB=[0.85, 0.85, 0.85], shnC=100.0, rflC=0.999)

    camera = Camera(w=120, h=90)
    camera.zoom(1.0)
    camera.setTrg3D(sphere.CoM)
    light = Light(ctr=[2.0, 2.0, 2.5])
    renderer = Renderer(cam=camera, geo=(sphere, mirror), lgt=light, dwnSmpl=2)

    for idx, angle in enumerate((0.0, 3.141592653589793 / 12)):
        if idx:
            mirror.tformRigid(0, 0, angle, 0, 0, 0)
        renderer.shoot(frmFileName=str(out_dir / f'geometry_motion_{idx:02d}.png'))


if __name__ == '__main__':
    main()
