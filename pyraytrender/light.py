import time

import matplotlib.pyplot as plt
import numpy as np

from .utils import rotM3D

class Light():
    # Light Source as 3D POINT -- Advanced: Make a Geometry-like Light source
    def __init__(self, ctr=[5.0, 5.0, 5.0], ambRGB=[1.0, 1.0, 1.0], dffRGB=[1.0, 1.0, 1.0], spcRGB=[1.0, 1.0, 1.0], lightPow=10.0):
        self.ctr = np.array(ctr)
        self.ambRGB = np.array(ambRGB) # Light Ambient Colour RGB
        self.dffRGB = np.array(dffRGB) # Light Diffuse Colour RGB
        self.spcRGB = np.array(spcRGB) # Light Specular Colour RGB
        self.lgtPOW = lightPow # Light Power (Strength, Intensity)
        
    def _showLight3D(self, fig=None, supTitle=None, frmFileName=None):# OK
        # Enable interactive mode
        plt.ion()
        
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.001)
            ax = fig.axes[0]
        
        # Displaying Geometry as Triangualr Mesh (Patch)
        ax.scatter(self.ctr[0], self.ctr[1], self.ctr[2],
                   marker='*', c='y', s=100)
        
        ax.set_aspect('equal')
        ax.axes.set_xlabel('X-axis')
        ax.axes.set_ylabel('Y-axis')
        ax.axes.set_zlabel('Z-axis')
        
        if supTitle is not None:
            fig.suptitle(supTitle)
        
        # Re-drawing the figure
        fig.canvas.draw()
        
        if frmFileName is not None:
            fig.savefig(frmFileName, bbox_inches='tight')
        
        return fig        
    
    def tformRigid(self, a_euler=0.0, b_euler=0.0, c_euler=0.0, x_offset=0.0, y_offset = 0.0, z_offset=0.0): # OK
        # E.g for Bunny:
        #   a_euler = np.pi/2,
        #   b_euler = 0,
        #   c_euler = np.pi
        
        # A 4x4 transformation matrix will be generated from the input parameters
        M = self._getRigidMatrix(a_euler, b_euler, c_euler, x_offset, y_offset, z_offset)
        
        # If centred: Center to Origin -> Transform -> ReOffset
        vts = self.ctr
            
        # Transformation
        vts = np.vstack((vts.reshape([3, 1]),1.0))
        vts = np.matmul(M, vts).transpose()
        vts = vts[:, 0:3]
                        
        self.ctr = vts.squeeze()

    
    def _getRigidMatrix(self, a_euler=0.0, b_euler=0.0, c_euler=0.0, x_offset=0.0, y_offset = 0.0, z_offset=0.0): #OK
        
        M = np.eye(4)
        M[0:3, 0:3] = rotM3D(a_euler, b_euler, c_euler)
        M[0,3] = x_offset
        M[1,3] = y_offset
        M[2,3] = z_offset
        
        return M
