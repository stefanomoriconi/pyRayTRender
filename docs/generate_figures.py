"""Generate the demo figures embedded in the project README.

Renders two scenes with the real `pyraytrender` API (no mocked output):
1. The Stanford Bunny on a reflective floor, with shadows + reflections.
2. Two procedurally-generated icospheres on a floor, with shadows + reflections.

Run from the repository root:
    python3 docs/generate_figures.py
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyraytrender import Camera, Geometry, Light, Renderer

OUT_DIR = Path(__file__).resolve().parent / 'images'


def render_bunny():
    bunny = Geometry(label='Bunny')
    bunny.loadBunny()
    bunny.mat.ambRGB = [0.05, 0.1, 0.05]
    bunny.mat.dffRGB = [0.15, 0.65, 0.2]
    bunny.mat.spcRGB = [0.6, 0.8, 0.6]
    bunny.mat.shnC = 32.0
    bunny.mat.rflC = 0.05
    bunny.mat._iniMat()

    zmin = bunny.BBox[0, 2]
    floor = Geometry(label='Floor')
    floor.genPlane(ptA=[-4, -4, 0], ptB=[4, -4, 0], ptC=[4, 4, 0], ptD=[-4, 4, 0])
    floor.tformRigid(0, 0, 0, bunny.CoM[0], bunny.CoM[1], zmin - 0.02)
    floor.mat.ambRGB = [0.05, 0.05, 0.06]
    floor.mat.dffRGB = [0.5, 0.5, 0.55]
    floor.mat.spcRGB = [0.9, 0.9, 0.9]
    floor.mat.shnC = 48.0
    floor.mat.rflC = 0.25
    floor.mat._iniMat()

    eye = bunny.CoM + np.array([1.1, -1.6, 0.8])
    camera = Camera(w=200, h=140, eye=eye, trg=bunny.CoM + np.array([0, 0, 0.15]))
    camera.fLen = 1.4
    camera._getImgScreen()
    light = Light(ctr=bunny.CoM + np.array([2.0, -2.0, 3.0]))

    renderer = Renderer(cam=camera, geo=(bunny, floor), lgt=light,
                         shadowsFlag=True, reflectionsFlag=True, dwnSmpl=1)
    out_path = OUT_DIR / 'bunny_reflections.png'
    renderer.shoot(frmFileName=str(out_path))
    return out_path


def render_primitives():
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
    camera = Camera(w=220, h=150, eye=eye, trg=[0.6, 0.2, 0.35])
    light = Light(ctr=[1.8, -1.8, 3.0])

    renderer = Renderer(cam=camera, geo=(sphere, sphere2, floor), lgt=light,
                         shadowsFlag=True, reflectionsFlag=True, dwnSmpl=1)
    out_path = OUT_DIR / 'primitives_scene.png'
    renderer.shoot(frmFileName=str(out_path))
    return out_path


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p1 = render_bunny()
    p2 = render_primitives()
    print('Wrote', p1, 'and', p2, 'in {:.1f}s'.format(time.time() - t0))
