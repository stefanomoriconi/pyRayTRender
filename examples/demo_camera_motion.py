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
    sphere.genIcoSphere(c3=[0, 0, -4], rad=0.8, nsub=1)
    plane = Geometry(label='Plane')
    plane.genPlane()
    plane.scale(20.0)
    plane.tformRigid(0, 0, 0, -10, -10, -5)

    camera = Camera(w=120, h=90)
    camera.zoom(1.2)
    camera.setTrg3D(sphere.CoM)
    camera.lockOnTarget(True)
    camera.lockScreenOrient(True)
    light = Light(ctr=[2.5, 2.0, 2.0])
    renderer = Renderer(cam=camera, geo=(sphere, plane), lgt=light, dwnSmpl=2)

    for idx, orbit in enumerate(([0.0, 0.0, 0.0], [-0.12, 0.2, 0.0], [-0.12, 0.2, 0.0])):
        if idx:
            camera.orbit3D(orbit)
        renderer.shoot(frmFileName=str(out_dir / f'camera_motion_{idx:02d}.png'))


if __name__ == '__main__':
    main()
