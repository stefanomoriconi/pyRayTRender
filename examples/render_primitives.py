"""Render a simple scene made only of procedurally-generated primitives
(icospheres + a plane) with shadows and reflections, no external assets."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyraytrender import Camera, Geometry, Light, Renderer


def main():
    out_dir = Path(__file__).resolve().parent / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'render_primitives.png'

    sphere = Geometry(label='Sphere')
    sphere.genIcoSphere(c3=[0, 0, 0.9], rad=0.9, nsub=3)
    sphere.mat.ambRGB = [0.04, 0.04, 0.06]
    sphere.mat.dffRGB = [0.12, 0.18, 0.4]
    sphere.mat.spcRGB = [0.5, 0.5, 0.55]
    sphere.mat.shnC = 32.0
    sphere.mat.rflC = 0.3
    sphere.mat.MaxDepth = 2
    sphere.mat._iniMat()

    sphere2 = Geometry(label='Sphere2')
    sphere2.genIcoSphere(c3=[1.9, 1.1, 0.5], rad=0.5, nsub=2)
    sphere2.mat.ambRGB = [0.07, 0.02, 0.02]
    sphere2.mat.dffRGB = [0.65, 0.1, 0.1]
    sphere2.mat.spcRGB = [0.4, 0.3, 0.3]
    sphere2.mat.shnC = 20.0
    sphere2.mat.rflC = 0.05
    sphere2.mat._iniMat()

    floor = Geometry(label='Floor')
    floor.genPlane(ptA=[-2.5, -2.5, 0], ptB=[2.5, -2.5, 0], ptC=[2.5, 2.5, 0], ptD=[-2.5, 2.5, 0])
    floor.mat.ambRGB = [0.04, 0.04, 0.05]
    floor.mat.dffRGB = [0.5, 0.5, 0.55]
    floor.mat.spcRGB = [0.3, 0.3, 0.3]
    floor.mat.shnC = 16.0
    floor.mat.rflC = 0.05
    floor.mat._iniMat()

    eye = np.array([2.6, -3.0, 1.7])
    camera = Camera(w=200, h=140, eye=eye, trg=[0.6, 0.2, 0.35])
    light = Light(ctr=[1.8, -1.8, 3.0])

    renderer = Renderer(
        cam=camera,
        geo=(sphere, sphere2, floor),
        lgt=light,
        shadowsFlag=True,
        reflectionsFlag=True,
        dwnSmpl=1,
    )
    renderer.shoot(frmFileName=str(out_path))
    print(out_path)


if __name__ == '__main__':
    main()
