import inspect
import time

import matplotlib.pyplot as plt
import numpy as np

from .utils import orthog, projct, rotM3D, rotM3Da2b, rotVec, uvect

class Camera:
    def __init__(self, w=300, h=200, eye=[0, 0, 1], n1=[0, 0, -1], e1=[1, 0, 0], fLen=1, trg=[0, 0, 0]):
        # Initialising fields
        self.width = int(w)
        self.height = int(h)
        
        self.eye = np.array(eye) # Camera pin-hole EYE position
        if trg is None:
            self.trg = None
        else:
            self.trg = np.array(trg) # Camera TARGET - could be the CoM of an object
        self.pln = [] # Camera PLANE: image screen centre
        
        # Definig the camera aspect ratio
        self.aspectRatio = float(self.width)/float(self.height)
        
        self.fLen = fLen
        self.fLenRNG = [0.1, 10]
        
        
        self.imgScreenCrnrs = None # top-left, top-right, bottom-right, bottom-left
        
        self.isLockOnTarget = False
        self.trgLock = None
        self.isLockScreenOrient = False
        self.n1Lock = None
        self.e1Lock = None
        self.e2Lock = None
        
        self.gimbalDmp = 1.0/3
        
        if trg is None: # EXPLICIT TARGET has PRIORITY over input n1
            self.n1 = uvect(np.array(n1))
        else:
            self.n1 = uvect(self.trg - self.eye)
        self.e1 = orthog(uvect(np.array(e1)), self.n1)
        self.e2 = uvect(np.cross(self.e1, self.n1))
        
        
        # Initialising
        self._getImgScreen()
        
    def _getImgScreen(self):
        # Determining the 3D screen range based on camera normal and e1 versor (left direction)
        # NB: this is useful if the camera has an orientation angle around its normal vector.
        # Note: left side is given by e1 (right = -e1)
        # Note: e2 is determined as cross product between (e1, nrm)
        # Note: bottom side is given by e2 (top = -e2)
        if self.trg is None:
            self.trg = self.eye + (self.fLen * self.n1)
        else:
            self.n1 = uvect(self.trg - self.eye)        
        # self.trg = self.eye + (self.fLen * self.n1) # OLD
        self.e1 = orthog(self.e1, self.n1)
        self.e2 = uvect(np.cross(self.e1, self.n1))
        self.pln = self.eye + (self.fLen * self.n1)
            
        lft =   self.e1
        rgt = - self.e1
        top = - self.e2/self.aspectRatio
        btm =   self.e2/self.aspectRatio
        
        # Corners
        self.imgScreenCrnrs = np.array([self.pln + top + lft, 
                                        self.pln + top + rgt,
                                        self.pln + btm + rgt,
                                        self.pln + btm + lft])
        
    def pan3D(self, dspl_v3):
        
        dspl_v3 =  np.array(dspl_v3)
        # Translates the Camera EYE (position)
        self.eye = self.eye + dspl_v3
        
        if self.isLockOnTarget:
            self.n1 = uvect(self.trgLock - self.eye)
            self.e1 = orthog(self.e1, self.n1)
            self.e2 = uvect(np.cross(self.e1, self.n1))
            if self.isLockScreenOrient:
                self.e1 = orthog(self.e1Lock, self.n1)
                self.e2 = uvect(np.cross(self.e1, self.n1))
                
        else:
            self.trg = self.trg + dspl_v3
        
        # Update Image Screem
        self._getImgScreen()
        
    def rot3D(self, thetas_v3, forceFlag=False): # OK
        # WRN: BUG IN ROTATION it is ABSOLUTE - NOT RELATIVE TO THE CAMERA AXES (n1, e1, e2)!
        
        if not forceFlag:
            if self.isLockOnTarget:
                # Allow only rotations around n1!
                print('[wrn] {} - LockOnTarget: Enabled!'.format(inspect.stack()[0][3]))
                thetas_v3[0] = 0
                thetas_v3[1] = 0
                
            if self.isLockScreenOrient:
                print('[wrn] {} - LockScreenOrient: Enabled!'.format(inspect.stack()[0][3]))
                return None
        
        thetas_v3 = np.array(thetas_v3)
        # Rotate of thetas_v3 radiants around (e1, e2, n1)
        R = rotM3D(thetas_v3[0], thetas_v3[1], thetas_v3[2])
        #rot_n1 = np.matmul(R, self.n1.transpose()).transpose()
        #rot_e1 = np.matmul(R, self.e1.transpose()).transpose()
        #rot_e2 = np.matmul(R, self.e2.transpose()).transpose()
        Rref = rotM3Da2b(self.e1, self.e2, [1,0,0], [0,1,0])
        rot_n1 = rotVec(self.n1, Rref, R)
        rot_e1 = rotVec(self.e1, Rref, R)
        rot_e2 = rotVec(self.e2, Rref, R)
        
        trgT = self.trg - self.eye
        #trgR = np.matmul(R, trgT.transpose()).transpose()
        trgR = rotVec(trgT, Rref, R)
        trgO = trgR + self.eye
        
        # Update Camera Versors
        self.trg = trgO
        self.n1 = rot_n1
        self.e1 = rot_e1
        self.e2 = rot_e2
        
        # Update Image Screem
        self._getImgScreen()
        
    def orbit3D(self, thetas_v3, localFlag=True): # OK
        # WRN: BUG IN ROTATION it is ABSOLUTE - NOT RELATIVE TO THE CAMERA AXES (n1, e1, e2)!    
        
        # Move the Camera eye so that the Image Screen Center is kept FIXED 
        # (same position in 3D), but the associated versors are rotated.
        # >> pivoting Camera Eye over the Image Screen Center 
        thetas_v3 = np.array(thetas_v3)
        # Rotate of thetas_v3 radiants around (e1, e2, n1)
        R = rotM3D(thetas_v3[0], thetas_v3[1], thetas_v3[2])
        #rot_n1 = np.matmul(R, self.n1.transpose()).transpose()
        #rot_e1 = np.matmul(R, self.e1.transpose()).transpose()
        #rot_e2 = np.matmul(R, self.e2.transpose()).transpose()
        Rref = rotM3Da2b(self.e1, self.e2, [1,0,0], [0,1,0])
        if localFlag:
            rot_n1 = rotVec(self.n1, Rref, R)
            rot_e1 = rotVec(self.e1, Rref, R)
            rot_e2 = rotVec(self.e2, Rref, R)
        else:
            rot_n1 = rotVec(self.n1, R=R)
            rot_e1 = rotVec(self.e1, R=R)
            rot_e2 = rotVec(self.e2, R=R)
            
        eyeT = self.eye - self.trg
        eyeR =  np.matmul(R, eyeT.transpose()).transpose()
        eyeO = eyeR + self.trg
        
        # Correcting for the rotation around n1 to enable GIMBAL effect
        eyeDn = orthog(uvect(eyeO - self.eye), rot_n1)
        # print(projct(eyeDn, rot_e2))
        if projct(eyeDn, rot_e2) > 0:
            thZ = -np.arccos(projct(eyeDn, rot_e2))
        else:
            thZ = -np.arccos(projct(-eyeDn, rot_e2))

        # Update Camera Versors
        self.n1 = rot_n1
        self.e1 = rot_e1
        self.e2 = rot_e2
        # Update Camera Eye
        self.eye = eyeO
        # Update Image Screem
        self._getImgScreen()
        
        if self.isLockScreenOrient:
            self.rot3D([0,0,thZ*self.gimbalDmp], forceFlag=1)
        
    def zoom(self, d_fLen):
        # Increment/Decrement the Camera focal Length (fLen) by input delta (d_fLen)
        # NOTE [min,max] admissible focal Lenght values are stored in self.fLenRNG
        # Control will saturate to range extremes, if out of bound
        
        self.fLen = self.fLen + d_fLen
        if self.fLen < self.fLenRNG[0]:
            self.fLen = self.fLenRNG[0]
            print('{}: Max wide angle reached'.format(inspect.stack()[0][3]))
        if self.fLen > self.fLenRNG[1]:
            self.fLen = self.fLenRNG[1]
            print('{}: Max zoom (magnification) reached'.format(inspect.stack()[0][3]))
        
        # Update Image Screem
        self._getImgScreen()
        
    def setTrg3D(self, trg_v3):
        
        if not self.isLockOnTarget:
            self.trg = np.array(trg_v3) #
            # self.n1 = uvect(np.array(trg_v3) - self.eye)
        else:
            print('[wrn] {} - LockOnTarget: Enabled!'.format(inspect.stack()[0][3]))
            
        # Update Image Screem
        self._getImgScreen()
        
    def lockOnTarget(self, LoT_Flag=False):
            self.isLockOnTarget = LoT_Flag
            if self.isLockOnTarget:
                self.trgLock = self.trg
            else:
                self.trgLock = None
        
    def lockScreenOrient(self, LSO_Flag=False):
        self.isLockScreenOrient = LSO_Flag
        if self.isLockScreenOrient:
            self.e1Lock = self.e1
            self.e2Lock = self.e2
        else:
            self.e1Lock = None
            self.e2Lock = None
            
    def lockAll(self, Lock_Flag=False):
        self.lockOnTarget(Lock_Flag)
        self.lockScreenOrient(Lock_Flag)
        
    def _showCam3D(self, Scnfig=None, showRays=False, supTitle=None, frmFileName=None):
        
        hs = [] # handles of plotted data
        
        # Enable interactive mode
        plt.ion()
        
        if Scnfig is None:
            Scnfig = plt.figure()
            ax = Scnfig.add_subplot(projection='3d')
        else:
            # to flush the GUI events
            Scnfig.canvas.flush_events()
            time.sleep(0.001)
            ax = Scnfig.axes[0]
        
        # Displaying Camera Eye
        h0 = ax.scatter(self.eye[0], self.eye[1], self.eye[2],
                   marker='o', s=30)
        hs.append(h0)
        
        # Displaying the Camera Focal Length (distance to image plane center)
        h1 = ax.plot([self.eye[0], self.pln[0]],
                [self.eye[1], self.pln[1]], 
                [self.eye[2], self.pln[2]],color='k')
        hs.append(h1)
        
        # Displaying Camera Normal
        h2 = ax.quiver(self.eye[0], self.eye[1], self.eye[2],
                  self.n1[0],  self.n1[1],  self.n1[2],
                  length=self.fLen/2, normalize=True, color='b')
        hs.append(h2)
        
        # Displaying e1 and e2 onimage plane center
        h3 = ax.quiver(self.pln[0], self.pln[1], self.pln[2],
                  self.e1[0], self.e1[1], self.e1[2],
                  length=self.fLen/2, normalize=True, color='r')
        hs.append(h3)
        h4 = ax.quiver(self.pln[0], self.pln[1], self.pln[2],
                  self.e2[0], self.e2[1], self.e2[2],
                  length=self.fLen/2, normalize=True, color='g')
        hs.append(h4)
        
        # Displaying the Image Plane
        for i in range(0, 4):     
            htmp = ax.plot([self.imgScreenCrnrs[i-1][0], self.imgScreenCrnrs[i][0]],
                    [self.imgScreenCrnrs[i-1][1], self.imgScreenCrnrs[i][1]], 
                    [self.imgScreenCrnrs[i-1][2], self.imgScreenCrnrs[i][2]],color='k') # lines
            hs.append(htmp)
        
        if showRays:
            nthRay = 49 # Downsampling the rays
            sC, sN = self._getRays(nthRay)
            
            h5 = ax.quiver(sC[:,0], sC[:,1], sC[:,2],
                      sN[:,0], sN[:,1], sN[:,2],
                      length=self.fLen/2,
                      normalize=True, color='k')
            hs.append(h5)
        
        ax.set_aspect('equal')
        ax.axes.set_xlabel('X-axis')
        ax.axes.set_ylabel('Y-axis')
        ax.axes.set_zlabel('Z-axis')
        
        if supTitle is not None:
            Scnfig.suptitle(supTitle)
        
        # Re-drawing the figure
        Scnfig.canvas.draw()
        
        if frmFileName is not None:
            Scnfig.savefig(frmFileName, bbox_inches='tight')
        
        return Scnfig, hs
   
    def _getRays(self, nthRay=49):
        
        hSmpls = np.linspace(-1.0, 1.0, num=self.height)
        wSmpls = np.linspace(1.0, -1.0, num=self.width)
        sC = []
        sN = []
        
        for i, h in enumerate(hSmpls):
            if i % nthRay == 0:
                for j, w in enumerate(wSmpls):
                    if j % nthRay == 0:
                        sel_sC = self.pln + h*self.e2/self.aspectRatio + w*self.e1
                        sel_sN = uvect(sel_sC - self.eye)
                        sC.append(sel_sC)
                        sN.append(sel_sN)
        
        sC = np.array(sC).astype(np.float32)
        sN = np.array(sN).astype(np.float32)
        return sC, sN        
    
    def _removeALLcamHandles(self, camHandles):
        for ii in range(0, len(camHandles)):
            if isinstance(camHandles[ii], list):
                camHandles[ii][0].remove()
            else:
                camHandles[ii].remove()
     
    def demo_rot3D(self, thetas_v3=[np.pi/2, np.pi/3, np.pi/4], t_steps=10, LockTarget_Flag=False, LockOrient_Flag=False, showRays=False):
        eye = np.random.randn(3)
        n1 = uvect(np.random.randn(3))
        e1 = orthog(uvect(np.random.randn(3)), n1)
        trg = None
        self.__init__(eye=eye, n1=n1, e1=e1, trg=trg)
        self.lockScreenOrient(LockOrient_Flag)
        self.lockOnTarget(LockTarget_Flag)
        d_thetas_v3 = np.array(thetas_v3)/t_steps
        
        if showRays:
            Scnfig = self._showCam3DRays()
        else:
            Scnfig = self._showCam3D()[0]
        #Scnfig = self._showCam3D()
        Scnfig.suptitle('Camera Demo: 3D Rotation of the Camera EYE - LockTRG: {:d}, LockORI: {:d}'.format(LockTarget_Flag, LockOrient_Flag), fontsize=10)
        
        for i in range(0, t_steps):
            self.rot3D(d_thetas_v3)
            # self._showCam3D(Scnfig)
            if showRays:
                self._showCam3DRays(Scnfig)
            else:
                self._showCam3D(Scnfig)
        
        return None
    
    def demo_orbit3D(self, thetas_v3=[np.pi/2, np.pi/3, np.pi/4], t_steps=10, LockTarget_Flag=False, LockOrient_Flag=False, showRays=False):
        eye = np.random.randn(3)
        n1 = uvect(np.random.randn(3))
        e1 = orthog(uvect(np.random.randn(3)), n1)
        trg = None
        self.__init__(eye=eye, n1=n1, e1=e1, trg=trg)
        self.lockScreenOrient(LockOrient_Flag)
        self.lockOnTarget(LockTarget_Flag)
        d_thetas_v3 = np.array(thetas_v3)/t_steps
        
        if showRays:
            Scnfig = self._showCam3DRays()
        else:
            Scnfig = self._showCam3D()[0]
        # Scnfig = self._showCam3D()
        Scnfig.suptitle('Camera Demo: 3D Orbit of the Camera around Target - LockTRG: {:d}, LockORI: {:d}'.format(LockTarget_Flag, LockOrient_Flag), fontsize=10)
        
        for i in range(0, t_steps):
            self.orbit3D(d_thetas_v3)
            # self._showCam3D(Scnfig)
            if showRays:
                self._showCam3DRays(Scnfig)
            else:
                self._showCam3D(Scnfig)
        
        return None
    
    def demo_pan3D(self, dsplc_v3=[1, 3, 5], t_steps=10, LockTarget_Flag=False, LockOrient_Flag=False, showRays=False):
        eye = np.random.randn(3)
        n1 = uvect(np.random.randn(3))
        e1 = orthog(uvect(np.random.randn(3)), n1)
        trg = None
        self.__init__(eye=eye, n1=n1, e1=e1, trg=trg)
        self.lockScreenOrient(LockOrient_Flag)
        self.lockOnTarget(LockTarget_Flag)
        d_dsplc_v3 = np.array(dsplc_v3)/t_steps
        
        if showRays:
            Scnfig = self._showCam3DRays()
        else:
            Scnfig = self._showCam3D()[0]
        # Scnfig = self._showCam3D()
        Scnfig.suptitle('Camera Demo: 3D Pan of the Camera EYE - LockTRG: {:d}, LockORI: {:d}'.format(LockTarget_Flag, LockOrient_Flag), fontsize=10)
        
        for i in range(0, t_steps):
            self.pan3D(d_dsplc_v3)
            #self._showCam3D(Scnfig)
            if showRays:
                self._showCam3DRays(Scnfig)
            else:
                self._showCam3D(Scnfig)
        
        return None
    
    def demo_zoom(self, In=True, t_steps=10, showRays=False):
        eye = np.random.randn(3)
        n1 = uvect(np.random.randn(3))
        e1 = orthog(uvect(np.random.randn(3)), n1)
        trg = None
        self.__init__(eye=eye, n1=n1, e1=e1, trg=trg)
        if In: # Zooming In (Magnification)
            d_fLen = 10.0/t_steps
        else: # Zooming Out (Wide Angle)
            d_fLen = -1/t_steps
        
        if showRays:
            Scnfig = self._showCam3DRays()
        else:
            Scnfig = self._showCam3D()[0]
            
        Scnfig.suptitle('Camera Demo: Zoom (in/out) of the Camera - Focal-length Changes', fontsize=10)
        
        for i in range(0, t_steps):
            self.zoom(d_fLen)
            if showRays:
                self._showCam3DRays(Scnfig)
            else:
                self._showCam3D(Scnfig)
        
        return None
