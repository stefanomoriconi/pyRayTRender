import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyraytrender import Camera, Geometry, Light, Renderer


def main():
    out_dir = Path(__file__).resolve().parent / 'output'
    out_dir.mkdir(exist_ok=True)

    geo = Geometry(label='Sphere')
    geo.genIcoSphere(c3=[0, 0, -4], rad=0.9, nsub=1)
    camera = Camera(w=120, h=90)
    camera.zoom(1.2)
    camera.setTrg3D(geo.CoM)

    for idx, light_ctr in enumerate(([2.0, 2.0, 2.5], [-2.5, 2.0, 2.0])):
        renderer = Renderer(cam=camera, geo=geo, lgt=Light(ctr=light_ctr), dwnSmpl=2)
        renderer.shoot(frmFileName=str(out_dir / f'multiple_lights_{idx:02d}.png'))


if __name__ == '__main__':
    main()
