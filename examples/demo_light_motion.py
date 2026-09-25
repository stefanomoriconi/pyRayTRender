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
    light = Light(ctr=[2.0, 2.0, 2.5])
    renderer = Renderer(cam=camera, geo=geo, lgt=light, dwnSmpl=2)

    for idx, ctr in enumerate(([2.0, 2.0, 2.5], [-2.0, 1.5, 2.0])):
        renderer.Light.ctr = ctr
        renderer.shoot(frmFileName=str(out_dir / f'light_motion_{idx:02d}.png'))


if __name__ == '__main__':
    main()
