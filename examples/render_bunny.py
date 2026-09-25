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
    out_path = out_dir / 'render_bunny.png'

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

    renderer = Renderer(
        cam=camera,
        geo=(bunny, floor),
        lgt=light,
        shadowsFlag=True,
        reflectionsFlag=True,
        dwnSmpl=1,
    )
    renderer.shoot(frmFileName=str(out_path))
    print(out_path)


if __name__ == '__main__':
    main()
