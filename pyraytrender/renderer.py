import inspect
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .camera import Camera
from .geometry import Geometry
from .light import Light
from .utils import dist_p_line, isOppVdir, isPtInTri, sect_lin_pln, uvect

class Renderer():
    def __init__(self, cam=Camera(), geo=Geometry(), lgt=Light(), shadowsFlag=True, reflectionsFlag=True, ARmaskFlag=False, skyRGB=[.9, .9, 1.0], dwnSmpl=1):
        self.Camera = cam       # Camera Object(s) in the Scene
        self.Geometry = geo     # Geometry Object(s) in the Scene
        self.Light = lgt        # Light Object(s) in the Scene
        self.enableShadows=shadowsFlag
        self.enableReflections=reflectionsFlag
        self.skyRGB = np.array(skyRGB)
        self.dwnSmpl = dwnSmpl
        self.ARmaskFlag=ARmaskFlag
        
        # Initialising
        self.ImgCanvas = self._iniImgCanvas() # << Initialising as Black Canvas
        
    def _iniImgCanvas(self):
        # Initialising: Computing the Image Black Canvas
        '''
        imgCanvas = np.round(np.zeros([self.Camera.height,
                                       self.Camera.width,
                                       3])*255).astype(np.uint8)
        '''
        imgCanvas = np.tile(self.skyRGB.reshape([1,self.skyRGB.shape[0]]),
                            [int(np.ceil(float(self.Camera.height)/self.dwnSmpl)) * 
                             int(np.ceil(float(self.Camera.width)/self.dwnSmpl)), 1])
        return imgCanvas
        
    def _raytraceCamView(self, Envfig=None, verboseFlag=1):
        # PlaceHolder for Camera Rendering Engine 
        t0_raytraceCamView = time.time()
        
        imgCanvas = np.tile(self.skyRGB.reshape([1,self.skyRGB.shape[0]]),
                            [int(np.ceil(float(self.Camera.height)/self.dwnSmpl)) * 
                             int(np.ceil(float(self.Camera.width)/self.dwnSmpl)), 1])
        
        # Retrieving the FACES, RAYS and INTERSECTION POINTS - (CAMERA - GEOMETRY)
        # fidx, ipts3D = self._getCamGeoIntersections(dwnSmpl=dwnSmpl) # ORIG (single GEOMETRY)
        t0_CamGeoIntersect = time.time()
        fidx, ipts3D, rayNs, geoNs = self._getCamGeoIntersectionsMULT() # NEW (MULTIPLE GEOMETRIES) -- SEEMS OK
        t1_CamGeoIntersect = time.time()
        
        # Note: rayNs, geoNs are used for computing REFLECTIONS
        '''
        if Envfig is not None:
            ax = Envfig.axes[0]
            ax.scatter(ipts3D[:,0], ipts3D[:,1], ipts3D[:,2])
            ax.set_aspect('equal')
        '''
        # OK - till here - tested
        
        # RETRIEVING the ILLUMINATED INTESECTING POINTS (LIGHT - GEOMETRY) 
        t0_GeoLgtIntersect = time.time()
        if self.enableShadows and not self.ARmaskFlag: # Compute GEOMETRY-based SHADOWS and OCCLUSIONS
            if isinstance(self.Geometry, Geometry):
                lidx = np.ones(ipts3D.shape[0])
            else:
                # lidx = self._getGeoLightIntersections(ipts3D) # <<<!!!  This is correct # ORIG (single GEOMETRY)
                lidx = self._getGeoLightIntersectionsMULT(ipts3D) # <<<!!!  This is correct # NEW (MULTIPLE GEOMETRIES) -- SEEMS OK
        else:
            # but for single geometry in the scene better avoiding this computation
            lidx = np.ones(ipts3D.shape[0])
        t1_GeoLgtIntersect = time.time()
        
        # Checking CONSISTENCY of intermediate OUTPUTS
        assert lidx.shape[0] == fidx.shape[0], "lidx MUST have same shape as fidx! - BUG "
        
        # RETRIEVING the REFLECTIONS as SETS od INTESECTIONS FROM BOUNCING RAYS 
        t0_RayGeoIntersect = time.time()
        if self.enableReflections: # Compute Reflections in the SCENE
            if isinstance(self.Geometry, Geometry): # No reflection in a scene with a single geometry
                ridx = [] 
                rpts3D = []
            else: # Compute reflections among multiple geometries in the scene
                ridx, rpts3D, rlidx = self._getRaysGeoReflectionsMULT(fidx, ipts3D, rayNs, geoNs, Envfig=Envfig)
        else:
            ridx = [] 
            rpts3D = []
            rlidx = []
        t1_RayGeoIntersect = time.time()
        
        # DETERMINING the RENDERING Colour (RGB) of each PIXEL for the CAMERA view in the SCENE
        t0_BlnPhgIllum = time.time()
        if isinstance(self.Geometry, Geometry):
            clum_RGB = self._getBlinnPhongIllumination(fidx, ipts3D, lidx) # REFLECTIONS are not considered
        else:
            clum_RGB = self._getBlinnPhongIlluminationMULT(fidx, ipts3D, lidx, ridx, rpts3D, rlidx)
        t1_BlnPhgIllum = time.time()
        
        # Assigning Rendered Colours for the pixels with ILLUMINATED GEOMETRY
        imgCanvas[fidx[:,1],0] = clum_RGB[:,0]
        imgCanvas[fidx[:,1],1] = clum_RGB[:,1]
        imgCanvas[fidx[:,1],2] = clum_RGB[:,2]
        
        imgCanvas = imgCanvas.reshape(
            [np.ceil(float(self.Camera.height)/self.dwnSmpl).astype(np.int64), 
             np.ceil(float(self.Camera.width)/self.dwnSmpl).astype(np.int64), 3])

        # Camera rays are sampled bottom-to-top along the image-screen height
        # (row 0 = bottom of the screen), so flip vertically to match the
        # conventional top-to-bottom row order expected by imshow/PIL.
        imgCanvas = np.flipud(imgCanvas)

        t1_raytraceCamView = time.time()
                     
        ## Report Computational Time <DEBUG>
        if verboseFlag:
            print('<D> {} - Computational Time Performance Analysis'.format(inspect.stack()[0][3]))
            print('<D> CamGeoIntersect  {:.3f}s'.format(t1_CamGeoIntersect-t0_CamGeoIntersect))
            print('<D> GeoLgtIntersect  {:.3f}s'.format(t1_GeoLgtIntersect-t0_GeoLgtIntersect))
            print('<D> RayGeoIntersect  {:.3f}s'.format(t1_RayGeoIntersect-t0_RayGeoIntersect))
            print('<D> BlnPhgIllum      {:.3f}s'.format(t1_BlnPhgIllum-t0_BlnPhgIllum))
            print('<D> TOT              {:.3f}s'.format(t1_raytraceCamView-t0_raytraceCamView))
        
        self.ImgCanvas = imgCanvas
    
    def _getAllGeometry(self):
        
        if isinstance(self.Geometry, Geometry):
            ggfcs = np.zeros([self.Geometry.fcs.shape[0],1], dtype=np.uint16)
            fcs_ctr = self.Geometry.fcs_ctr
            fcs = self.Geometry.fcs
            vts = self.Geometry.vts
            fns = self.Geometry.fn
            triSize = self.Geometry.trisize
            triSize_fcs = np.multiply(np.ones([self.Geometry.fcs.shape[0],1]), 
                                      1.5 * self.Geometry.trisize).astype(np.float32)
            
        elif isinstance(self.Geometry, (tuple, list)):
            ggfcs = []
            fcs_ctr = []
            fcs = []
            vts = []
            fns = []            
            vts_cumsum = 0
            triSize = 0
            triSize_fcs = []
            
            for gg in range(0, len(self.Geometry)):
                ggfcs.append(gg*np.ones([self.Geometry[gg].fcs.shape[0],1],
                                        dtype=np.uint16))
                fcs_ctr.append(self.Geometry[gg].fcs_ctr)
                fcs.append(self.Geometry[gg].fcs + vts_cumsum)
                vts.append(self.Geometry[gg].vts)
                fns.append(self.Geometry[gg].fn)
                vts_cumsum += self.Geometry[gg].vts.shape[0]
                triSize = np.max([triSize, self.Geometry[gg].trisize])
                triSize_fcs.append(np.multiply(np.ones([self.Geometry[gg].fcs.shape[0],1]),
                                               1.5 * self.Geometry[gg].trisize))
            
            ggfcs = np.vstack(ggfcs).astype(np.uint16)
            fcs_ctr = np.vstack(fcs_ctr)
            fcs = np.vstack(fcs)
            vts = np.vstack(vts)
            fns = np.vstack(fns) 
            triSize_fcs = np.vstack(triSize_fcs)
            
        else:
            print('[wrn] {}: Unknown Geometry Data!'.format(inspect.stack()[0][3]))
            ggfcs = None
            fcs_ctr = None
            fcs = None
            vts = None
            fns = None
            triSize = None
            triSize_fcs = None
        
        if fcs.shape[0] > 65535:
            raise Exception('Too Many Triangular Faces in the Scene - RENDERING: Abort')
        else:
            # Convert Datatype of the Outputs for FAST Computing
            # print('Conversion of data here... WIP')
            fcs_ctr = fcs_ctr.astype(np.float32)
            fcs = fcs.astype(np.uint16)
            vts = vts.astype(np.float32)
            fns = fns.astype(np.float32)
            ggfcs = ggfcs.astype(np.uint16)
            triSize = triSize.astype(np.float32)
            triSize_fcs = triSize_fcs.astype(np.float32)
        
        return fcs_ctr, fcs, vts, fns, ggfcs, triSize, triSize_fcs
    
    def _getInvisibleGeometry(self):
        bkgdGeo = []
        if isinstance(self.Geometry, Geometry):
            if self.Geometry.visibleFlag == False:
                bkgdGeo.append(0)
        elif isinstance(self.Geometry, (tuple, list)):
            for gg in range(0, len(self.Geometry)):
                if self.Geometry[gg].visibleFlag == False:
                    bkgdGeo.append(gg)
        else:
            print('[wrn] {}: Unknown Geometry Data!'.format(inspect.stack()[0][3]))
            
        return bkgdGeo
    '''
    def _getCamGeoIntersections(self):
        #Function implementing the following steps to retrieve:
        # The hitting Camera RAYS
        # The hit Geometry FACES
        # The INTERSECTING POINTS coordinates of the GEOMETRY
        
        # Algorithmic description
        # for each ray (Cam.Eye -> ScreenPixelCoords):
            # check intersection with Geometry (!all geometry!)* -> Strategy to reduce computation:
                # [Reduce Geometryfaces to check]:
                # A) SEL1: check distance(face_centers, ray) and consider ONLY below THRESHOLD (distance point-line)
                # B) SEL2: among those: check opposite face normals (negative projection(sel_face_normals, cam_normal)))
                # C) hit-test(ray, sel_faces) 
                #   -> intersectionPt(ray, face_normal_plane)
                #   -> check isInTriangle(intersectionPt, triangle)
                #   -> consider ONLY INSIDE-TRIANGLE CAMERA-CLOSEST Intersection Point (otherwise occluded)
        
        # Retrieve the CAMERA pixel coordinates in 3D space (sC) and the associated RAY unit vectors (sN)
        sC, sN = self.Camera._getRays(nthRay=self.dwnSmpl)
        
        # A) SEL1 : faces (vertices) within a triangle-distance from the rays
        ddr = dist_p_line(self.Geometry.fcs_ctr, sC, sN) # Radial Distance
        fidx = np.argwhere(ddr < 1.5*self.Geometry.trisize) 
        # Faces that are close to the RAYS (within a certain radius from RAYS)
        # NOTE: idx[:,0] -> indices of valid (vertices) faces
        # NOTE: idx[:,1] -> indices of valid rays
        
        # B) SEL2 : check for opposite sign of triangular faces (vertices) and rays
        opp = np.sum(np.multiply(self.Geometry.fn[fidx[:,0],:],
                                 sN[fidx[:,1],:]), axis=1) < 0
        fidx = fidx[opp,:] # Face NORMALS are OPPOSITE to the RAYS
        
        # INTERSECTION POINTS among RAYS and triangular FACEs' *PLANES*
        ipts3D = sect_lin_pln(np.tile(self.Camera.eye.reshape([1,3]),[fidx.shape[0],1]),
                              sC[fidx[:,1],:],
                              self.Geometry.vts[self.Geometry.fcs[fidx[:,0],0],:],
                              self.Geometry.vts[self.Geometry.fcs[fidx[:,0],1],:],
                              self.Geometry.vts[self.Geometry.fcs[fidx[:,0],2],:])
        
        
        # Check INTERSECTION POINTS are INSIDE respective FACEs (force to belong to the triangle)
        inn = isPtInTri(ipts3D,
                        np.hstack(
                        (self.Geometry.vts[self.Geometry.fcs[fidx[:,0],0],:],
                         self.Geometry.vts[self.Geometry.fcs[fidx[:,0],1],:],
                         self.Geometry.vts[self.Geometry.fcs[fidx[:,0],2],:])))
        
        ipts3D = ipts3D[inn,:] # Intersection points are INSIDE the TRIANGLES
        fidx = fidx[inn,:] # faces AND rays with VALID intersecting points
        
        # Select only CLOSEST INTERSECTION POINTs to the CAMERA PLANE per RAY - Check OCCLUSION by GEOMETRY
        ddi = np.linalg.norm(ipts3D - sC[fidx[:,1],:], axis=1)
        
        # Evaluating for each individual (unique) RAY 
        unqRays, cntRays = np.unique(fidx[:,1], return_counts=1)
        srtidx = np.argsort(fidx[:,1])
        dd = np.split(ddi[srtidx], np.cumsum(cntRays)[:-1])
        ip = np.split(ipts3D[srtidx,:], np.cumsum(cntRays)[:-1])
        ff = np.split(fidx[srtidx,0], np.cumsum(cntRays)[:-1])
        chk = list(map(np.argmin, dd)) # Finding the position corresponding to the MIN DISTANCE
        ffsel = []
        ipts3Dsel = []
        for i in range(0, len(ff)):
            ffsel.append(ff[i][chk[i]]) # APPENDING the FACE corresponding to the MIN DISTANCE
            ipts3Dsel.append(ip[i][chk[i],:])# APPENDING the INTERSECTION POINT with MIN DISTANCE to CAMERA EYE
        ffsel = np.vstack(ffsel)
        ipts3Dsel = np.vstack(ipts3Dsel)
        fidxsel = np.hstack((ffsel, unqRays.reshape([len(unqRays),1])))
        
        return fidxsel, ipts3Dsel
    '''
    
    def _getCamGeoIntersectionsMULT(self):
        #Function implementing the following steps to retrieve:
        # The hitting Camera RAYS
        # The hit Geometry FACES
        # The INTERSECTING POINTS coordinates of the GEOMETRY
        
        # Algorithmic description
        # for each ray (Cam.Eye -> ScreenPixelCoords):
            # check intersection with Geometry (!all geometry!)* -> Strategy to reduce computation:
                # [Reduce Geometryfaces to check]:
                # A) SEL1: check distance(face_centers, ray) and consider ONLY below THRESHOLD (distance point-line)
                # B) SEL2: among those: check opposite face normals (negative projection(sel_face_normals, cam_normal)))
                # C) hit-test(ray, sel_faces) 
                #   -> intersectionPt(ray, face_normal_plane)
                #   -> check isInTriangle(intersectionPt, triangle)
                #   -> consider ONLY INSIDE-TRIANGLE CAMERA-CLOSEST Intersection Point (otherwise occluded)
        
        # Retrieve the CAMERA pixel coordinates in 3D space (sC) and the associated RAY unit vectors (sN)
        #<D> t0_rays = time.time()
        sC, sN = self.Camera._getRays(nthRay=self.dwnSmpl)
        #<D> t1_rays = time.time()
        
        # Get Geometry faces, vertices, and face_centers (ALL GEOMETRY in the SCENE)
        #<D> t0_allGeo = time.time()
        fcs_ctr, fcs, vts, fns, ggfcs, _, triSizeTHR = self._getAllGeometry()
        #<D> print('<D> _getAllGeometry: TOT fcs: {:d}'.format(fcs.shape[0]))
        #<D> print('<D> _getAllGeometry: TOT vts: {:d}'.format(vts.shape[0]))
        #<D> t1_allGeo = time.time()
        
        # A) SEL1 : faces (vertices) within a triangle-distance from the rays
        #<D> t0_dpline = time.time()
        # ddr = dist_p_line(fcs_ctr, sC, sN) # Radial Distance << ORIGINAL
        
        # PERFORMANCE OPTIMISED
        #fidx = np.argwhere(ddr < 1.5*triSize) 
        
                #fidx = np.argwhere(ddr_chk) # << THIS IS OK!!!
        
        #<D> t1_dpline = time.time()
        
        # Faces that are close to the RAYS (within a certain radius from RAYS)
        # NOTE: idx[:,0] -> indices of valid (vertices) faces
        # NOTE: idx[:,1] -> indices of valid rays
        
        # B) SEL2 : check for opposite sign of triangular faces (vertices) and rays
        #<D> t0_oppNs = time.time()
        #opp = np.sum(np.multiply(fns[fidx[:,0],:],
        #                         sN[fidx[:,1],:]), axis=1) < 0 # << ORIGINAL
        
                
        #fidx = fidx[opp,:] # Face NORMALS are OPPOSITE to the RAYS   # << THIS IS OK!!!
        #<D> t1_oppNs = time.time()
        
        # INTERSECTION POINTS among RAYS and triangular FACEs' *PLANES*
        #<D> t0_sectlp = time.time()
        #ipts3D = sect_lin_pln(np.tile(self.Camera.eye.reshape([1,3]),
        #                              [fidx.shape[0],1]),
        #                      sC[fidx[:,1],:],
        #                      vts[fcs[fidx[:,0],0],:],
        #                      vts[fcs[fidx[:,0],1],:],
        #                      vts[fcs[fidx[:,0],2],:]) # << ORIGINAL
        
        # NEW: PERFORMANCE OPTIMISED
        #                              [fidx.shape[0],1]),
        #                        sC[fidx[:,1],:],
        #                        vts[fcs[fidx[:,0],0],:],
        #                        vts[fcs[fidx[:,0],1],:],
        #                        vts[fcs[fidx[:,0],2],:]) # Can be optimised for memory!!! (TILE)
        
        # Check INTERSECTION POINTS are INSIDE respective FACEs (force to belong to the triangle)
        #inn = isPtInTri(ipts3D,
        #                np.hstack(
        #                (vts[fcs[fidx[:,0],0],:],
        #                 vts[fcs[fidx[:,0],1],:],
        #                 vts[fcs[fidx[:,0],2],:]))) << ORIGINAL
        
                #                   vts[fcs[fidx[:,0],0],:],
        #                   vts[fcs[fidx[:,0],1],:],
        #                   vts[fcs[fidx[:,0],2],:])
       
        
        #<D> t0_dplinopp = time.time()
        ddr = dist_p_line(fcs_ctr, sC, sN)
        fidx = np.argwhere(ddr < triSizeTHR)
        opp = isOppVdir(fns[fidx[:,0],:], sN[fidx[:,1],:])
        fidx = fidx[opp,:]
        #<D> t1_dplinopp = time.time()
        
        #<D> t0_sectlp = time.time()
        ipts3D = sect_lin_pln(np.tile(self.Camera.eye.reshape([1,3]),
                                      [fidx.shape[0],1]),
                              sC[fidx[:,1],:],
                              vts[fcs[fidx[:,0],0],:],
                              vts[fcs[fidx[:,0],1],:],
                              vts[fcs[fidx[:,0],2],:])
        inn = isPtInTri(ipts3D,
                        np.hstack((vts[fcs[fidx[:,0],0],:],
                                   vts[fcs[fidx[:,0],1],:],
                                   vts[fcs[fidx[:,0],2],:])))
        #<D> t1_sectlp = time.time()
        
        #<D> t0_rest = time.time()
        ipts3D = ipts3D[inn,:] # Intersection points are INSIDE the TRIANGLES
        fidx = fidx[inn,:] # faces AND rays with VALID intersecting points
        rays3D = sN[fidx[:,1],:] # Camera Rays unit vectors
        gfns3D = fns[fidx[:,0],:] # Geometry face normals unit vectors
        
        # Select only CLOSEST INTERSECTION POINTs to the CAMERA PLANE per RAY - Check OCCLUSION by GEOMETRY
        ddi = np.linalg.norm(ipts3D - sC[fidx[:,1],:], axis=1)
        
        # Evaluating for each individual (unique) RAY 
        unqRays, cntRays = np.unique(fidx[:,1], return_counts=1)
        srtidx = np.argsort(fidx[:,1])
        dd = np.split(ddi[srtidx], np.cumsum(cntRays)[:-1])
        ip = np.split(ipts3D[srtidx,:], np.cumsum(cntRays)[:-1])
        ff = np.split(fidx[srtidx,0], np.cumsum(cntRays)[:-1])
        rr = np.split(rays3D[srtidx,:], np.cumsum(cntRays)[:-1])
        gn = np.split(gfns3D[srtidx,:], np.cumsum(cntRays)[:-1])
        
        chk = list(map(np.argmin, dd)) # Finding the position corresponding to the MIN DISTANCE
        ffsel = []
        ipts3Dsel = []
        rays3Dsel = []
        gfns3Dsel = []
        
        for i in range(0, len(ff)):
            ffsel.append(ff[i][chk[i]]) # APPENDING the FACE corresponding to the MIN DISTANCE
            ipts3Dsel.append(ip[i][chk[i],:])# APPENDING the INTERSECTION POINT with MIN DISTANCE to CAMERA EYE
            rays3Dsel.append(rr[i][chk[i],:])
            gfns3Dsel.append(gn[i][chk[i],:])
            
        ffsel = np.vstack(ffsel)
        ipts3Dsel = np.vstack(ipts3Dsel)
        rays3Dsel = np.vstack(rays3Dsel)
        gfns3Dsel = np.vstack(gfns3Dsel)
        ggsel = ggfcs[ffsel[:,0],:]
        
        fidxsel = np.hstack((ffsel, unqRays.reshape([len(unqRays),1]), ggsel)).astype(np.uint64)
        #<D> t1_rest = time.time()
        
        ## Report Computational Time <DEBUG>
        #<D> print('<D> Computational Time Performance Analysis: _getCamGeoIntersectionsMULT')
        #<D> print('<D> ._getRays        {:.3f}s'.format(t1_rays-t0_rays))
        #<D> print('<D> ._getAllGeom     {:.3f}s'.format(t1_allGeo-t0_allGeo))
        #<D> print('<D> .distPLinOpp     {:.3f}s'.format(t1_dplinopp-t0_dplinopp))
        #<D> print('<D> .oppNorms        {:.3f}s'.format(t1_oppNs-t0_oppNs))
        #<D> print('<D> .sectLinPlnTri   {:.3f}s'.format(t1_sectlp-t0_sectlp))
        #<D> print('<D> .Remaining       {:.3f}s'.format(t1_rest-t0_rest))
        
        return fidxsel, ipts3Dsel, rays3Dsel, gfns3Dsel
    
    '''
    def _getGeoLightIntersections(self, ipts3D):
        # Algorithmic description
        # For each intersecting point:
            # check intersection between the RAY(IPT,LGH) and GEO faces
            # evaluate the intersections in GEO faces triangles
            # *** remove IPT if OTHER INTERSECTIONS EXIST AND are in-triangles AND are CLOSER to the LIGHT compared to IPT
            # i.e. the intersection point IPT is SHADOWED by OTHER GEOMETRY being in the line-of-sight with the LIGHT RAYS 
            # therefore the input INTERSECTION POINT is NOT ILLUMINATED
            
            # *** in prectice: masking those points
            # >> as they may be illuminated in second-step by REFLECTIONS!
            
        # Filter a subset of GEOMETRY Faces to consider in the Raytracing between LIGHT-GEOMETRY
        ddr = dist_p_line(self.Geometry.fcs_ctr,
                          ipts3D,
                          uvect(ipts3D - np.tile(self.Light.ctr.reshape([1,3]),[ipts3D.shape[0],1]))) # Radial Distance
        widx = np.argwhere(ddr < 1.5*self.Geometry.trisize)
        
        # Computing ALL possible intersection points between LIGHT and GEOMETRY planes
        wpts3D = sect_lin_pln(np.tile(self.Light.ctr.reshape([1,3]),[widx.shape[0],1]),
                              ipts3D[widx[:,1],:],
                              self.Geometry.vts[self.Geometry.fcs[widx[:,0],0],:],
                              self.Geometry.vts[self.Geometry.fcs[widx[:,0],1],:],
                              self.Geometry.vts[self.Geometry.fcs[widx[:,0],2],:]) 
        
        # Intersection points between Geometry-Light that belong to the Geometry (inside Triangles)
        inn = isPtInTri(wpts3D, np.hstack((self.Geometry.vts[self.Geometry.fcs[widx[:,0],0],:],
                                           self.Geometry.vts[self.Geometry.fcs[widx[:,0],1],:],
                                           self.Geometry.vts[self.Geometry.fcs[widx[:,0],2],:])))
        # Intersection points that are different (OTHERS) than the given intersection (ipts3D)
        oth = np.linalg.norm(ipts3D[widx[:,1],:] - wpts3D, axis=1) > 1e-9
        
        # Masking (Inside Triangles) and Re-assigning indices
        widx[:,0] = np.array(range(0,widx.shape[0]))
        widx = widx[inn,:]
        wpts3D = wpts3D[inn,:]
        oth = oth[inn]
        
        
        ## DEBUG in case of ERROR 
        ## It can happen that some of the input ipts3D are not listed in the intersections wpts3D
        ## - this may be due to the THRESHOLD applied to ddr:
        ##     i.e. ddr < 1.5*self.Geometry.trisize
        ## since ALL ipts3D are expected to be in wpts3D, please consider increasing the THRESHOLD
        ## NB: increasing the THRESHOLD will SLOW DOWN computational performance!
        ## 
        ## to chech possible entries of ipts3D that are missing in wpts3D, please run:
        ##     
        ##     for ii in range(0, ipts3D.shape[0]):
        ##         if ~np.any(np.linalg.norm(ipts3D[ii,:] - wpts3D, axis=1)<1e-9):
        ##             print('<DBG> _getGeoLightIntersections - Missing ipts3D[{:d}] from wpts3D!'.format(ii))

        
        if np.any(oth):
            # Determining the Unique set of LIGHT-GEOMETRY RAYS
            unqRays, cntRays = np.unique(widx[:,1], return_counts=1)
            srtidx = np.argsort(widx[:,1])
            # Initialising HIDE INDEX Flag -- Indices of GEOMETRY Intersection points that are
            # NOT DIRECTLY ILLUMINATED by the LIGHT (SHADED by some other parts of the GEOMETRY)
            lidx = np.ones(unqRays.shape, dtype=np.bool_) 
            
            # Evaluating Locally (Indices)
            ot = np.split(oth[srtidx], np.cumsum(cntRays)[:-1])
            wp = np.split(wpts3D[srtidx,:], np.cumsum(cntRays)[:-1])
            
            # Select only the points to Evaluate:
            # i.e. GEOMETRY intersection points that MIGHT be SHADED in the line of the LIGHT RAYS:
            # Intersections of the LIGHT-GEOMETRY occur in more than 1 POINT,
            # where 1 of them being ALWAYS the considered INTERSECTION POINT
            otsel = [ot[i] for i in np.argwhere(cntRays>1).squeeze()]
            wpsel = [wp[i] for i in np.argwhere(cntRays>1).squeeze()]
            uRsel = unqRays[cntRays>1]
            ccsel = cntRays[cntRays>1]
            
            ddsel = np.split( np.linalg.norm( np.vstack(wpsel) - self.Light.ctr, axis=1) , np.cumsum(ccsel)[:-1])
            for rr in range(0, len(ddsel)):
                if np.any(ddsel[rr][~otsel[rr]] > ddsel[rr][otsel[rr]]): # IF EXISTS any 
                # OTHER INTERSECTION POINT in the GEOMETRY that belongs to the SAME LIGHT-GEOMETRY RAY 
                # which has SHORTER DISTANCE to the LIGHT compared to the input INTERSECTION POINT, then: 
                    lidx[uRsel[rr]] = 0 # set the input INTERSECTION POINT to SHADED (or hidden to the LIGHT)
        else:
            lidx = np.ones(ipts3D.shape[0])
                
        return lidx
    '''
    
    def _getGeoLightIntersectionsMULT(self, ipts3D):
        # Algorithmic description
        # For each intersecting point:
            # check intersection between the RAY(IPT,LGH) and GEO faces
            # evaluate the intersections in GEO faces triangles
            # *** remove IPT if OTHER INTERSECTIONS EXIST AND are in-triangles AND are CLOSER to the LIGHT compared to IPT
            # i.e. the intersection point IPT is SHADOWED by OTHER GEOMETRY being in the line-of-sight with the LIGHT RAYS 
            # therefore the input INTERSECTION POINT is NOT ILLUMINATED
            
            # *** in prectice: masking those points
            # >> as they may be illuminated in second-step by REFLECTIONS!
            
        # Get Geometry faces, vertices, and face_centers (ALL GEOMETRY in the SCENE)
        fcs_ctr, fcs, vts, _, _, triSize, triSizeTHR = self._getAllGeometry()
            
        # Filter a subset of GEOMETRY Faces to consider in the Raytracing between LIGHT-GEOMETRY
        #ddr = dist_p_line(fcs_ctr,
        #                  ipts3D,
        #                  uvect(ipts3D - np.tile(self.Light.ctr.reshape([1,3]),[ipts3D.shape[0],1]))) # Radial Distance << ORIGINAL
        
        # PERFORMANCE OPTIMISED
        #                      ipts3D,
                #widx = np.argwhere(ddr < 1.5*triSize)
        
        ddr = dist_p_line(fcs_ctr,
                          ipts3D,
                          uvect(ipts3D - np.tile(self.Light.ctr.reshape([1,3]),[ipts3D.shape[0],1])))
        widx = np.argwhere(ddr < triSizeTHR)
        
        # Computing ALL possible intersection points between LIGHT and GEOMETRY planes
        #wpts3D = sect_lin_pln(np.tile(self.Light.ctr.reshape([1,3]),[widx.shape[0],1]),
        #                      ipts3D[widx[:,1],:],
        #                      vts[fcs[widx[:,0],0],:],
        #                      vts[fcs[widx[:,0],1],:],
        #                      vts[fcs[widx[:,0],2],:]) # << ORIGINAL
        
        # NEW: PERFORMANCE OPTIMISED
        #                        ipts3D[widx[:,1],:],
        #                        vts[fcs[widx[:,0],0],:],
        #                        vts[fcs[widx[:,0],1],:],
        #                        vts[fcs[widx[:,0],2],:]) # CAN BE OPTIMISED FOR MEMEORY (TILE)
        
        # Intersection points between Geometry-Light that belong to the Geometry (inside Triangles)
        #inn = isPtInTri(wpts3D, np.hstack((vts[fcs[widx[:,0],0],:],
        #                                   vts[fcs[widx[:,0],1],:],
        #                                   vts[fcs[widx[:,0],2],:]))) # << ORIGINAL
        
                #                   vts[fcs[widx[:,0],0],:],
        #                   vts[fcs[widx[:,0],1],:],
        #                   vts[fcs[widx[:,0],2],:])
        
        wpts3D = sect_lin_pln(np.tile(self.Light.ctr.reshape([1,3]),[widx.shape[0],1]),
                              ipts3D[widx[:,1],:],
                              vts[fcs[widx[:,0],0],:],
                              vts[fcs[widx[:,0],1],:],
                              vts[fcs[widx[:,0],2],:])
        inn = isPtInTri(wpts3D, np.hstack((vts[fcs[widx[:,0],0],:],
                                           vts[fcs[widx[:,0],1],:],
                                           vts[fcs[widx[:,0],2],:])))
        
        # Intersection points that are different (OTHERS) than the given intersection (ipts3D)
        oth = np.linalg.norm(ipts3D[widx[:,1],:] - wpts3D, axis=1) > 1e-9
        
        # Masking (Inside Triangles) and Re-assigning indices
        widx[:,0] = np.array(range(0,widx.shape[0]))
        widx = widx[inn,:]
        wpts3D = wpts3D[inn,:]
        oth = oth[inn]
        
        ''' DEBUG in case of ERROR 
        It can happen that some of the input ipts3D are not listed in the intersections wpts3D
        - this may be due to the THRESHOLD applied to ddr:
            i.e. ddr < 1.5*self.Geometry.trisize
        since ALL ipts3D are expected to be in wpts3D, please consider increasing the THRESHOLD
        NB: increasing the THRESHOLD will SLOW DOWN computational performance!
        
        to chech possible entries of ipts3D that are missing in wpts3D, please run:
            
            for ii in range(0, ipts3D.shape[0]):
                if ~np.any(np.linalg.norm(ipts3D[ii,:] - wpts3D, axis=1)<1e-9):
                    print('<DBG> _getGeoLightIntersections - Missing ipts3D[{:d}] from wpts3D!'.format(ii))
        '''
        
        if np.any(oth):
            # Determining the Unique set of LIGHT-GEOMETRY RAYS
            unqRays, cntRays = np.unique(widx[:,1], return_counts=1)
            srtidx = np.argsort(widx[:,1])
            # Initialising HIDE INDEX Flag -- Indices of GEOMETRY Intersection points that are
            # NOT DIRECTLY ILLUMINATED by the LIGHT (SHADED by some other parts of the GEOMETRY)
            lidx = np.ones(unqRays.shape, dtype=np.bool_) 
            
            # Evaluating Locally (Indices)
            ot = np.split(oth[srtidx], np.cumsum(cntRays)[:-1])
            wp = np.split(wpts3D[srtidx,:], np.cumsum(cntRays)[:-1])
            
            # Select only the points to Evaluate:
            # i.e. GEOMETRY intersection points that MIGHT be SHADED in the line of the LIGHT RAYS:
            # Intersections of the LIGHT-GEOMETRY occur in more than 1 POINT,
            # where 1 of them being ALWAYS the considered INTERSECTION POINT
            otsel = [ot[i] for i in np.argwhere(cntRays>1).squeeze()]
            wpsel = [wp[i] for i in np.argwhere(cntRays>1).squeeze()]
            uRsel = unqRays[cntRays>1]
            ccsel = cntRays[cntRays>1]
            
            ddsel = np.split( np.linalg.norm( np.vstack(wpsel) - self.Light.ctr, axis=1) , np.cumsum(ccsel)[:-1])
            for rr in range(0, len(ddsel)):
                if np.any(ddsel[rr][~otsel[rr]] > ddsel[rr][otsel[rr]]): # IF EXISTS any 
                # OTHER INTERSECTION POINT in the GEOMETRY that belongs to the SAME LIGHT-GEOMETRY RAY 
                # which has SHORTER DISTANCE to the LIGHT compared to the input INTERSECTION POINT, then: 
                    lidx[uRsel[rr]] = 0 # set the input INTERSECTION POINT to SHADED (or hidden to the LIGHT)
        else:
            lidx = np.ones(ipts3D.shape[0])
                
        return lidx
    
    def _getBlinnPhongIllumination(self, fidx, ipts3D, lidx):
        
        clum_RGB = np.zeros(ipts3D.shape) 
        # * lidx[:,np.newaxis].astype(np.double)
        
        dds = np.power(np.linalg.norm(self.Light.ctr - ipts3D, axis=1), 2)
        dls = np.divide(self.Light.lgtPOW, dds).reshape([dds.shape[0],1])
        
        # Ambient
        clum_RGB += np.multiply(self.Geometry.mat.ambRGB,
                                self.Light.ambRGB) 
        # Diffuse
        LN = np.sum(np.multiply(uvect(self.Light.ctr - ipts3D), 
                                self.Geometry.fn[fidx[:,0],:]), 
                    axis=1).reshape([ipts3D.shape[0],1])
        
        clum_RGB += np.multiply(
                        np.multiply(
                            np.multiply(self.Geometry.mat.dffRGB, 
                                        self.Light.dffRGB).reshape(
                                            [1, clum_RGB.shape[1]]), LN), 
                                dls)
        # Specular
        H = uvect(uvect(self.Light.ctr - ipts3D) + 
                  uvect(self.Camera.eye - ipts3D))
        SN = np.sum(np.multiply(self.Geometry.fn[fidx[:,0],:], H),
                    axis=1).reshape([ipts3D.shape[0],1])
        SN[SN<0] = 0.0
        SN = SN**(self.Geometry.mat.shnC/4)
        clum_RGB += np.multiply(
                        np.multiply(
                            np.multiply(self.Geometry.mat.spcRGB, 
                                        self.Light.spcRGB).reshape(
                                        [1, clum_RGB.shape[1]]), SN),
                                dls)
        
        # Mask for the Illuminated Points (the non-illuminated ones go to 0.0)                                        
        clum_RGB = np.multiply(clum_RGB, 
                               (1 + lidx[:,np.newaxis].astype(np.double))/2) # NOT pitch Black but quite dark
        
        # Saturate by Clipping
        clum_RGB = np.clip(clum_RGB, 0, 1)
        
        return clum_RGB
    
    def _getBlinnPhongIlluminationMULT(self, fidx, ipts3D, lidx, ridx, rpts3D, rlidx):
        # General Function to RENDER pixel colour, given:
        # - raytracing light-geometry-camera system [fidx, ipts3D]
        # - self-geometry shadowing occlusions [lidx]
        # - multiple ray reflections as per material depth (bounces) [ridx, rpts3D]
        
        # Get Geometry faces, vertices, and face_centers (ALL GEOMETRY in the SCENE)
        fns = self._getAllGeometry()[3]
        
        bkgdGeoID=self._getInvisibleGeometry()
        
        clum_RGB = np.zeros(ipts3D.shape) 
        
        dds = np.power(np.linalg.norm(self.Light.ctr - ipts3D, axis=1), 2)
        dls = np.divide(self.Light.lgtPOW, dds)
        
        LN = np.sum(np.multiply(uvect(self.Light.ctr - ipts3D), fns[fidx[:,0],:]), axis=1)
        
        H = uvect(uvect(self.Light.ctr - ipts3D) + uvect(self.Camera.eye - ipts3D))
        SN = np.sum(np.multiply(fns[fidx[:,0],:], H), axis=1)
        SN[SN<0] = 0.0
        
        for i in range(0, ipts3D.shape[0]):
            if self.ARmaskFlag:
                if fidx[i,2] in bkgdGeoID:
                    clum_RGB[i,:] = self.skyRGB.copy()
                else:
                    clum_RGB[i,:] = self.Geometry[fidx[i,2]].mat.dffRGB
                    clum_RGB[i,:] += self._getBlinnPhongIlluminationREFL(fidx[i,:], ipts3D[i,:], ridx, rpts3D, rlidx)
                    clum_RGB[i,:] = np.multiply(clum_RGB[i,:], (1 + lidx[i].astype(np.double))/2) # NOT pitch Black but quite dark
            else:
                if fidx[i,2] in bkgdGeoID:
                    clum_RGB[i,:] = self.skyRGB.copy()
                else:
                    # Ambient
                    clum_RGB[i,:] += np.multiply(
                        np.multiply(self.Geometry[fidx[i,2]].mat.ambRGB, self.Light.ambRGB),
                        self.Geometry[fidx[i,2]].mat.alpha)
                    
                    # Diffuse
                    clum_RGB[i,:] += np.multiply(
                        np.multiply(np.multiply(np.multiply(self.Geometry[fidx[i,2]].mat.dffRGB, self.Light.dffRGB), LN[i]), dls[i]),
                        self.Geometry[fidx[i,2]].mat.alpha)
                    
                    # Specular
                    SN[i] = SN[i]**(self.Geometry[fidx[i,2]].mat.shnC/4)
                    clum_RGB[i,:] += np.multiply( 
                        np.multiply(np.multiply(np.multiply(self.Geometry[fidx[i,2]].mat.spcRGB,self.Light.spcRGB), SN[i]), dls[i]),
                        self.Geometry[fidx[i,2]].mat.alpha)
                    
                    # Reflections by Computed Ray Bounces
                    # <DEBUG> Only Reflections
                    # clum_RGB[i,:] = np.zeros(3)
                    clum_RGB[i,:] += np.multiply( 
                        self._getBlinnPhongIlluminationREFL(fidx[i,:], ipts3D[i,:], ridx, rpts3D, rlidx),
                        self.Geometry[fidx[i,2]].mat.alpha)
                    # Mask for the Illuminated Points (the non-illuminated ones go to 0.0) 
                    clum_RGB[i,:] = np.multiply(clum_RGB[i,:], (1 + lidx[i].astype(np.double))/2) # NOT pitch Black but quite dark
                          
        # Saturate by Clipping
        clum_RGB = np.clip(clum_RGB, 0, 1)
        
        return clum_RGB
    
    def _getBlinnPhongIlluminationREFL(self, fidx_ii, ipts3D_ii, ridx, rpts3D, rlidx, debugMode=False):
        crfl_RGB = np.zeros(3).reshape([1,3])
        MaxBounces = self.Geometry[fidx_ii[2]].mat.MaxDepth
        rflC = self.Geometry[fidx_ii[2]].mat.rflC.copy()
        
        bkgdGeoID=self._getInvisibleGeometry()
        
        if rflC < 1e-12: # MULTIPLIER of reflection COEFFICIENT nearly ZERO
            return crfl_RGB.squeeze()
        
         # WIP
        fns = self._getAllGeometry()[3]
        
        # Initialisation of the Updating variable - starting point of the reflection ray tracing (index of the RAY)
        idx = fidx_ii[1]
        
        if len(ridx) > 0:
            for rr in range(0, len(ridx)):
                if rr > MaxBounces: # Max Depth of bounces REACHED!
                    break
                if rflC < 1e-12: # MULTIPLIER of reflection COEFFICIENT nearly ZERO
                    break
                # Initialise Bounce Illumination Colour RGB
                clum_RGB = np.zeros([1,3])
                
                ## Find the HIT Geometry
                idx = np.argwhere(ridx[rr][:,1] == idx)
                
                if debugMode:
                    # <DEBUG>
                    print(' ')
                    print('Source Ray (Pixel): {:d}'.format(fidx_ii[1]))
                    print('Hit Geometry: {:d} aka {:s} in [{:.3f},{:.3f},{:.3f}]'.format(fidx_ii[2], self.Geometry[fidx_ii[2]].label, ipts3D_ii[0], ipts3D_ii[1], ipts3D_ii[2]))
                    
                if len(idx) == 0:
                    if debugMode:
                        # Reflect Sky (no bounced reflections)
                        print('Bounce {:d} NOT found: Sky Colour - stop'.format(rr))
                    crfl_RGB += rflC * self.skyRGB.copy()
                    break
                elif len(idx) > 1:
                    print('<!> More than one reflection (bounce) found! - BUG?!?')
                    break
                else:
                    idx = idx.squeeze()
                    
                    # Updating variables within for loop
                    ipt3D = rpts3D[rr][idx,:] # Intersection point of Reflection target Geometry
                    gidx = ridx[rr][idx,2] # target Geometry Identifier
                    fidx = ridx[rr][idx,0] # target Geometry Face ID
                    if debugMode:
                        print('Bounce {:d} found at pos: {:d}'.format(rr, idx))
                        print('Bounce {:d} found Hitting Geometry: {:d} aka {:s}'.format(rr, gidx, self.Geometry[gidx].label))
                        print('Bounce {:d} found Face Geometry: {:d}'.format(rr, fidx))
                        print('Bounce {:d} found Intersection at [{:.3f},{:.3f},{:.3f}]'.format(rr, ipt3D[0], ipt3D[1], ipt3D[2]))
                        print('Bounce {:d} found Intrsct. Face Normal [{:.3f},{:.3f},{:.3f}]'.format(rr, fns[fidx,0], fns[fidx,1], fns[fidx,2]))
                        
                    if gidx == fidx_ii[2]: # DEBUG (Special case for just 1 bounce)
                        if debugMode:
                            print('Bounce as BUG self-intersection - skip')
                        break
                    
                    if gidx in bkgdGeoID:
                        crfl_RGB += rflC * self.skyRGB.copy()
                        break
                    
                    if self.ARmaskFlag:
                        crfl_RGB += rflC * self.Geometry[gidx].mat.dffRGB
                        break
                    
                    # Computing Illumination
                    dds = np.power(np.linalg.norm(self.Light.ctr - ipt3D), 2)
                    dls = np.divide(self.Light.lgtPOW, dds)
                    
                    LN = np.sum(np.multiply(uvect(self.Light.ctr - ipt3D), fns[fidx,:]))
                    H = uvect(uvect(self.Light.ctr - ipt3D) + uvect(self.Camera.eye - ipt3D))
                    SN = np.sum(np.multiply(fns[fidx,:], H))
                    if SN < 0.0:
                        SN = 0.0
                    
                    # Ambient
                    clum_RGB += np.multiply(self.Geometry[gidx].mat.ambRGB, self.Light.ambRGB)
                    # Diffuse
                    clum_RGB += np.multiply(np.multiply(np.multiply(self.Geometry[gidx].mat.dffRGB, self.Light.dffRGB), LN), dls)
                    # Specular
                    SN = SN**(self.Geometry[gidx].mat.shnC/4)
                    clum_RGB += np.multiply(np.multiply(np.multiply(self.Geometry[gidx].mat.spcRGB,self.Light.spcRGB), SN), dls)
                    
                    # Initial REFLECTIONs ILLUMINATIONs
                    tmp_RGB = rflC * clum_RGB
                    
                    # Modulating Shadows
                    tmp_RGB = np.multiply(tmp_RGB, 
                                          (1 + rlidx[rr][idx].astype(np.double))/2)
                    
                    # Accumulating the REFLECTIONs ILLUMINATIONs
                    crfl_RGB += tmp_RGB
    
                    # Update rflC
                    rflC *= self.Geometry[gidx].mat.rflC.copy()
                    # Update idx
                    idx = ridx[rr][idx,1]
                    
            # print(crfl_RGB)
        
        return crfl_RGB.squeeze()
    
    def _getRaysGeoReflectionsMULT(self, fidx, ipts3D, rayNs, geoNs, Envfig=None): # WIP!!!
        # Function to iteratively compute the ray-geometry reflections up to a max number of bounces
        ridx_lst = []
        rpts3D_lst = []
        rlidx_lst = []
        
        # Determine max number of ray bounces, according to geometry materials in the scene
        # Note: this is an 'expensive' over-complete computation accounting for the GLOBAL max number of bounces (aka MaxDepth)
        # This can be optimised for specific reflections in specific geometries... but whatev for now
        GLOBAL_MaxDepth = np.max([self.Geometry[i].mat.MaxDepth for i in np.unique(fidx[:,2])])
        
        # Initialising the iterative variables
        ridx_dpt = fidx
        rpts3D_dpt = ipts3D
        rrayNs_dpt = rayNs
        rgeoNs_dpt = geoNs
        
        # DEBUG
        # sC = self.Camera._getRays(nthRay=self.dwnSmpl)[0]
        
        for dpt in range(0, GLOBAL_MaxDepth):
            if rpts3D_dpt.shape[0] < 1: # No more reflection intersections found.
                break
            ridx_dpt, rpts3D_dpt, rrayNs_dpt, rgeoNs_dpt, ipts3D_dpt = self._getRaysGeoReflectionsCore(ridx_dpt, rpts3D_dpt, rrayNs_dpt, rgeoNs_dpt)
            
            if self.enableShadows:
                # Retrievening Shadows for reflections
                rlidx_dpt = self._getGeoLightIntersectionsMULT(rpts3D_dpt)
            else:
                rlidx_dpt = np.ones(rpts3D_dpt.shape[0])
            
            # DEBUG
            '''
            hs = self._debugReflectionBounces(Envfig,
                                              sC[ridx_dpt[:,1],:],
                                              ipts3D_dpt,
                                              rpts3D_dpt,
                                              rayNs[ridx_dpt[:,1],:],
                                              geoNs[ridx_dpt[:,1],:],
                                              rrayNs_dpt,
                                              ridx_dpt, selHitIdx=2, quiverFlag=1)
            
            # self._removeALLPlotBounces(hs)
            '''
            
            ridx_lst.append(ridx_dpt)
            rpts3D_lst.append(rpts3D_dpt)
            rlidx_lst.append(rlidx_dpt)
            
            if dpt > 0:
                if ridx_lst[dpt].shape[0] > ridx_lst[dpt-1].shape[0]:
                    print('[wrn] Reflected Rays are INCREASING at Bounce [#{:d}]... BUG!?!'.format(dpt))
            
        # print('Computing Reflections: DONE')    
        return ridx_lst, rpts3D_lst, rlidx_lst
        
    def _getRaysGeoReflectionsCore(self, fidx, ipts3D, rayNs, geoNs):
        # Core Fun
        RrayNs = self._reflectRays(rayNs, geoNs)
        
        # NOTE: very similar to previous core functions, note the difference in Rays (not from camera, but reflected by geometry)
        # sC = ipts3D
        # sN = RrayNs 
        
        # Get Geometry faces, vertices, and face_centers (ALL GEOMETRY in the SCENE)
        fcs_ctr, fcs, vts, fns, ggfcs, _, triSizeTHR = self._getAllGeometry()
        
        # A) SEL1 : faces (vertices) within a triangle-distance from the rays
        # ddr = dist_p_line(fcs_ctr, ipts3D, RrayNs) # Radial Distance << ORIGINAL
        
        # PERFORMANCE OPTIMISED
        #ridx = np.argwhere(ddr < 1.5*triSize) 
        
        # NEW: PERFORMANCE OPTIMISED + EMBEDDED THRESHOLD
        #ridx = np.argwhere(ddr_chk) # <<< OK!!!
        
        # Faces that are close to the RAYS (within a certain radius from RAYS)
        # NOTE: idx[:,0] -> indices of valid (vertices) faces
        # NOTE: idx[:,1] -> indices of valid rays
        
        # B) SEL2 : check for opposite sign of triangular faces (vertices) and rays
        #opp = np.sum(np.multiply(fns[ridx[:,0],:],
        #                         RrayNs[ridx[:,1],:]), axis=1) < 0 << ORIGINAL
        
                
        #ridx = ridx[opp,:] # Face NORMALS are OPPOSITE to the RAYS    # <<< OK!!!
        
        ddr = dist_p_line(fcs_ctr, ipts3D, RrayNs)
        ridx = np.argwhere(ddr < triSizeTHR)
        opp = isOppVdir(fns[ridx[:,0],:], RrayNs[ridx[:,1],:])
        ridx = ridx[opp,:]
        
        
        # INTERSECTION POINTS among RAYS and triangular FACEs' *PLANES*
        #rpts3D = sect_lin_pln(ipts3D[ridx[:,1],:],
        #                      ipts3D[ridx[:,1],:] + RrayNs[ridx[:,1],:],
        #                      vts[fcs[ridx[:,0],0],:],
        #                      vts[fcs[ridx[:,0],1],:],
        #                      vts[fcs[ridx[:,0],2],:]) # << ORIGINAL
        
        # NEW: PERFORMANCE OPTIMISED
        #                        ipts3D[ridx[:,1],:] + RrayNs[ridx[:,1],:],
        #                        vts[fcs[ridx[:,0],0],:],
        #                        vts[fcs[ridx[:,0],1],:],
        #                        vts[fcs[ridx[:,0],2],:])
        
        
        # Check INTERSECTION POINTS are INSIDE respective FACEs (force to belong to the triangle)
        #inn = isPtInTri(rpts3D,
        #                np.hstack(
        #                (vts[fcs[ridx[:,0],0],:],
        #                 vts[fcs[ridx[:,0],1],:],
        #                 vts[fcs[ridx[:,0],2],:]))) # << ORIGINAL
        
                #                   vts[fcs[ridx[:,0],0],:],
        #                   vts[fcs[ridx[:,0],1],:],
        #                   vts[fcs[ridx[:,0],2],:])
        
        rpts3D = sect_lin_pln(ipts3D[ridx[:,1],:],
                              ipts3D[ridx[:,1],:] + RrayNs[ridx[:,1],:],
                              vts[fcs[ridx[:,0],0],:],
                              vts[fcs[ridx[:,0],1],:],
                              vts[fcs[ridx[:,0],2],:])
        inn = isPtInTri(rpts3D,
                        np.hstack((vts[fcs[ridx[:,0],0],:],
                                   vts[fcs[ridx[:,0],1],:],
                                   vts[fcs[ridx[:,0],2],:])))
        
        rpts3D = rpts3D[inn,:] # REFLECTED Intersection points are INSIDE the TRIANGLES
        ridx = ridx[inn,:] # faces AND rays with VALID intersecting points
        
        # Ensure the Reflections Intersections are not BACKWARDS (along the Reflection Ray line, but bounced backward inside the geometry)
        sdi = np.sign(np.sum(np.multiply(rpts3D - ipts3D[ridx[:,1],:], RrayNs[ridx[:,1],:]), axis=1)) > 0
        
        rpts3D = rpts3D[sdi,:]
        ridx = ridx[sdi,:]
        rays3D = RrayNs[ridx[:,1],:] # REFLECTED Rays unit vectors
        gfns3D = fns[ridx[:,0],:] # Geometry face normals unit vectors
        ipts3Din = ipts3D[ridx[:,1],:] # Selected input points
        
        # Select only CLOSEST INTERSECTION POINTs to the CAMERA PLANE per RAY - Check OCCLUSION by GEOMETRY
        ddi = np.linalg.norm(rpts3D - ipts3D[ridx[:,1],:], axis=1)
        
        # Evaluating for each individual (unique) RAY 
        unqRays, cntRays = np.unique(ridx[:,1], return_counts=1)
        srtidx = np.argsort(ridx[:,1])
        dd = np.split(ddi[srtidx], np.cumsum(cntRays)[:-1])
        rp = np.split(rpts3D[srtidx,:], np.cumsum(cntRays)[:-1])
        ff = np.split(ridx[srtidx,0], np.cumsum(cntRays)[:-1])
        rr = np.split(rays3D[srtidx,:], np.cumsum(cntRays)[:-1])
        gn = np.split(gfns3D[srtidx,:], np.cumsum(cntRays)[:-1])
        ip = np.split(ipts3Din[srtidx,:], np.cumsum(cntRays)[:-1])
        
        chk = list(map(np.argmin, dd)) # Finding the position corresponding to the MIN DISTANCE
        ffsel = []
        rpts3Dsel = []
        rays3Dsel = []
        gfns3Dsel = []
        ipts3Dsel = []
        
        for i in range(0, len(ff)):
            ffsel.append(ff[i][chk[i]]) # APPENDING the FACE corresponding to the MIN DISTANCE
            rpts3Dsel.append(rp[i][chk[i],:])# APPENDING the INTERSECTION POINT with MIN DISTANCE to CAMERA EYE
            rays3Dsel.append(rr[i][chk[i],:])
            gfns3Dsel.append(gn[i][chk[i],:])
            ipts3Dsel.append(ip[i][chk[i],:])
            
        ffsel = np.vstack(ffsel)
        rpts3Dsel = np.vstack(rpts3Dsel)
        rays3Dsel = np.vstack(rays3Dsel)
        gfns3Dsel = np.vstack(gfns3Dsel)
        ipts3Dsel = np.vstack(ipts3Dsel)
        ggsel = ggfcs[ffsel[:,0],:]
        
        # Hit Geometry
        hgsel = fidx[unqRays,2]
        
        ridxsel = np.hstack((ffsel, 
                             unqRays.reshape([len(unqRays),1]),
                             ggsel,
                             hgsel.reshape([len(unqRays),1]))).astype(np.uint64)
        
        return ridxsel, rpts3Dsel, rays3Dsel, gfns3Dsel, ipts3Dsel

    
    def _reflectRays(self, iRns, fns): # OK
        # iRns: incident Rays unit vectors Nx3
        # fns:  respective (paired) face normals hit by the rays Nx3
        # rRns: reflected Rays unit vectors Nx3
        
        pj = np.sum(np.multiply(iRns, fns), axis=1).reshape([iRns.shape[0],1])
        rRns = uvect(iRns - 2*pj*fns)
        return rRns
    
    def _debugReflectionBounces(self, Envfig, srcPts, hitPts, rflPts, srcNss, hitNss, rflNss, ridx, selHitIdx=-1, quiverFlag=False, clr=[0,0,0]):
        
        hs = []
        
        if selHitIdx >= 0:
            chk = ridx[:,3] == selHitIdx
            ridx = ridx[chk,:]
            srcPts = srcPts[chk,:]
            hitPts = hitPts[chk,:]
            rflPts = rflPts[chk,:]
            srcNss = srcNss[chk,:]
            hitNss = hitNss[chk,:]
            rflNss = rflNss[chk,:]
            
        for ee in range(0, ridx.shape[0]):
            
            h_tmp = self._showReflectionBounce(Envfig,
                                               srcPts[ee,:],
                                               hitPts[ee,:],
                                               rflPts[ee,:],
                                               srcNss[ee,:],
                                               hitNss[ee,:],
                                               rflNss[ee,:], quiverFlag=quiverFlag)
            '''
            h_tmp = self._showRay(Envfig, 
                                  srcPts[ee,:],
                                  hitPts[ee,:],
                                  clr=clr)
            '''
            hs.append(h_tmp)
            
        return hs
    
    def _showReflectionBounce(self, Envfig, srcPt, hitPt, rflPt, srcNs, hitNs, rflNs, quiverFlag=False):
        
        if Envfig is not None:
            ax = Envfig.axes[0]    
        else: 
            Envfig = plt.gcf()
            ax = Envfig.axes[0]
        
        # PLot HIT Point
        h0 = ax.scatter(hitPt[0], hitPt[1], hitPt[2], color='k', s=3)
        # Plot REFLECTED POint
        h1 = ax.scatter(rflPt[0], rflPt[1], rflPt[2], color='b', s=3)
        
        # Plot INCIDENT RAY
        h2 = ax.plot3D([srcPt[0], hitPt[0]],
                  [srcPt[1], hitPt[1]],
                  [srcPt[2], hitPt[2]],
                  'k', linewidth=0.1)
        
        # Plot REFLECTED RAY
        h3 = ax.plot3D([hitPt[0], rflPt[0]],
                  [hitPt[1], rflPt[1]],
                  [hitPt[2], rflPt[2]],
                  'b', linewidth=0.1)
        
        hs = [h0, h1, h2, h3]
        
        if quiverFlag:
            # Plot INCIDENT RAY Unit-Vector
            ll = 1/3
            #h4 = ax.quiver3D(hitPt[0]-ll*srcNs[0],
            #            hitPt[1]-ll*srcNs[1],
            #            hitPt[2]-ll*srcNs[2],
            #            srcNs[0],
            #            srcNs[1],
            #            srcNs[2],
            #            color='k', length=ll, normalize=True)
            
            # Plot HIT Geometry Normal Unit-Vector
            h5 = ax.quiver3D(hitPt[0],
                        hitPt[1],
                        hitPt[2],
                        hitNs[0],
                        hitNs[1],
                        hitNs[2],
                        color='r', length=ll, normalize=True)
            
            # Plot REFLECTED RAY Unit-Vector
            #h6 = ax.quiver3D(hitPt[0],
            #            hitPt[1],
            #            hitPt[2],
            #            rflNs[0],
            #            rflNs[1],
            #            rflNs[2],
            #            color='r', length=ll, normalize=True)
            
            hs.append(h5)
        
        ax.set_aspect('equal')
        
        return hs
    
    def _removeALLPlotBounces(self, hs):
        for ii in range(0, len(hs)):
            for jj in range(0, len(hs[ii])):
                if isinstance(hs[ii][jj], list):
                    hs[ii][jj][0].remove()
                else:
                    hs[ii][jj].remove()
    
    def _showRay(self, Envfig, srcPt, trgPt, trgNs=None, clr=[0,0,0]):
        if Envfig is not None:
            ax = Envfig.axes[0]    
        else: 
            Envfig = plt.gcf()
            ax = Envfig.axes[0]
        
        # PLot HIT Point
        h0 = ax.scatter(trgPt[0], trgPt[1], trgPt[2], color=clr, s=1)
        
        # Plot INCIDENT RAY
        h1= ax.plot3D([srcPt[0], trgPt[0]],
                      [srcPt[1], trgPt[1]],
                      [srcPt[2], trgPt[2]],
                      color=clr, linewidth=0.1)
        
        hs = [h0, h1]
        
        if trgNs is not None:
            # Plot HIT Geometry Normal Unit-Vector
            h2 = ax.quiver3D(trgPt[0], trgPt[1], trgPt[2],
                             trgNs[0], trgNs[1], trgNs[2],
                             color=clr, length=1/3, normalize=True)
            hs.append(h2)
            
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return hs
        
    def shoot(self, Rndfig=None, Envfig=None, supTitle=None, frmFileName=None, BitmapFlag=1):
        # Invoking rendering of the Camera View: still-frame
        self._raytraceCamView(Envfig=Envfig)
        
        # Displaying the rendered still-frame
        
        if Rndfig is None:
            Rndfig = plt.figure()
            ax = Rndfig.add_subplot()
        else:
            Rndfig.canvas.flush_events()
            time.sleep(0.001)
            ax = Rndfig.axes[0]
        
        ax.imshow(self.ImgCanvas)
        
        # print('[wrn] {}: WIP!'.format(inspect.stack()[0][3]))
        
        plt.xticks([])
        plt.yticks([])
        
        if supTitle is not None:
            Rndfig.suptitle(supTitle)
        
        Rndfig.canvas.draw()
        
        if frmFileName is not None:
            if BitmapFlag:
                im = Image.fromarray(np.round(self.ImgCanvas*255).astype(np.uint8))
                im.save(frmFileName)
            else:
                Rndfig.savefig(frmFileName, bbox_inches='tight')
        
        return Rndfig
    
    def stream(self, Rndfig=None):
        # Method for rendering the camera view as video-stream
        print('[wrn] {}: WIP!'.format(inspect.stack()[0][3]))
        return Rndfig
