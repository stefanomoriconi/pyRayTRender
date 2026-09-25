import inspect
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import stl_reader
except ImportError:
    stl_reader = None

from .material import Material
from .utils import projct, rotM3D, uvect

class Geometry():
    # Geometry defined as any raster triangulated mesh (vertices, faces)
    def __init__(self, vts=None, fcs=None, label='Geometry', visibleFlag=True):
        self.vts = vts # Vertices list (3D Points)
        self.fcs = fcs # Faces list (Triangles)
        self.mat = Material() # Material of the Geometry
        self.label = label
        self.visibleFlag=visibleFlag
        
        self.fcs_ctr = None
        self.fn = None
        self.vn = None
    
        self.CoM = [0, 0, 0]
        self.BBox = [[0, 0, 0],[0, 0, 0]]
        self.size = 0
        self.trisize = 0
        
        # Initialising Geometry Normals
        self._getFeatures()
        
        
    def _getFeatures(self):# OK - optimised for PERFORMANCE 
    # Initialising mesh features: size, Face Centres, Face & Vertex Normals
        if self.fcs is None or self.vts is None:
            self.fcs_ctr = None
            self.fn = None
            self.vn = None
            self.size = 0
            self.trisize = 0
            self.CoM = [0, 0, 0]
            self.BBox = [[0, 0, 0],[0, 0, 0]]
            return

        # Store the Size of the Geometry
        self.size = self._getSize()     
        self.trisize = self._gettrisize()
        self.CoM = np.mean(self.vts, axis=0) # XYZ coords of the CoM
        self.BBox = np.vstack((np.min(self.vts, axis=0), 
                               np.max(self.vts, axis=0))) # Bounding Box [2x3]
        self.fcs_ctr = self._getFaceCenters() # Face Centers Coordinates
        self.fn = self._getFaceNormals()   # Faces Normals of the Geometry
        self.vn = self._getVertexNormals()   # Vertex Normals of the Geometry
        
    def _getSize(self):# OK
        # Estimate Centre
        c3 = np.mean(self.vts, axis=0)
        # Size: average distance of vertices wrt their centre
        size = np.mean(np.linalg.norm(self.vts - c3, axis=1))
        return size
        
    def _gettrisize(self):#WIP
        e0 = np.linalg.norm(self.vts[self.fcs[:,0],:] - self.vts[self.fcs[:,1],:], axis=1)
        e1 = np.linalg.norm(self.vts[self.fcs[:,1],:] - self.vts[self.fcs[:,2],:], axis=1)
        e2 = np.linalg.norm(self.vts[self.fcs[:,2],:] - self.vts[self.fcs[:,0],:], axis=1)
        trisize = np.mean(np.concatenate((e0, e1, e2)))
        return trisize
    
    def _getFaceCenters(self):# OK
        fcs_vts = np.stack((self.vts[self.fcs[:,0],:],
                            self.vts[self.fcs[:,1],:],
                            self.vts[self.fcs[:,2],:]), axis=2)
        fcs_ctr = np.mean(fcs_vts, axis=2)
        return fcs_ctr
    
    def _getFaceNormals(self):# OK
        # Computing Face Normals (ONLY Triangles)
        v1s = self.vts[self.fcs[:,1],:] - self.vts[self.fcs[:,0],:]
        v2s = self.vts[self.fcs[:,2],:] - self.vts[self.fcs[:,1],:]
        fn = uvect(np.cross(v1s, v2s))
        return fn
    
    def _getVertexNormals(self):# OK
        # Computing Vertex Normals (ONLY Triangles)
        idx = np.argsort(self.fcs.flatten()).astype(np.float64)
        idx = np.floor(idx/np.shape(self.fcs)[1]).astype(np.int64)
        _, cts = np.unique(self.fcs, return_counts=1)
        fns_vts = self.fn[idx, :]
        fns_vts = np.split(fns_vts, np.cumsum(cts)[:-1])
        vn = vn = np.array([np.mean(i, axis=0) for i in fns_vts])
        return vn
    
    def flipNormals(self):# OK
        # Filling the sign of Normals (or 3D unit vectors)
        self.fn = -1*self.fn
        self.vn = -1*self.vn
    
    def loadBunny(self, path=None):# OK
        if path is None:
            path = Path(__file__).resolve().parents[1] / 'assets' / 'Bunny.obj'
        self.readOBJ(path)
        self.tformRigid(np.pi/2, 0, np.pi, 5, -3, 10)
        self.scale(10)
        self.tformRigid(0,0,0,-50,29,-103)
    
    def genPlane(self, ptA=[0,0,0], ptB=[1,0,0], ptC=[1,1,0], ptD=[0,1,0]):
        # Generates a polygonal plane given 4 corners. The triangulation is:
        # ABC, CDA, in that order for normals.
        self.vts = np.vstack((ptA, ptB, ptC, ptD))
        ptM = np.mean(self.vts, axis=0)
        self.vts = np.vstack((self.vts, ptM))
        self.fcs = np.array([[0, 1, 4],
                             [1, 2, 4],
                             [2, 3, 4],
                             [3, 0, 4]])
        
        # Initialising Features
        self._getFeatures()
        
    def genTriangle(self, ptA=[0,1,0], ptB=[-np.sqrt(3)/2,-1/2,0], ptC=[np.sqrt(3)/2,-1/2,0]):
        # Generates a polygonal plane given 4 corners. The triangulation is:
        # ABC, CDA, in that order for normals.
        self.vts = np.vstack((ptA, ptB, ptC))
        self.fcs = np.array([[0, 1, 2]])
        
        # Initialising Features
        self._getFeatures()
        
    def genCircle(self, sbdv=3):
        self.genTriangle()

        # Refine with Subdivision
        for ss in range(0, sbdv):
            edg_bdr = self._getBoundaryEdges()
            vts = self.vts.copy()
            fcs = self.fcs.copy()
            for ee in range(0, edg_bdr.shape[0]):
                m_mod = np.mean([np.linalg.norm(vts[edg_bdr[ee, 0],:]),
                                 np.linalg.norm(vts[edg_bdr[ee, 1],:])])
                m_vec = uvect(np.mean([vts[edg_bdr[ee, 0],:],
                                       vts[edg_bdr[ee, 1],:]], axis=0))
                m = np.reshape(m_vec, [1, 3])*m_mod
                vts = np.vstack((vts, m))
                f = np.array([edg_bdr[ee, 1],
                              edg_bdr[ee, 0],
                              vts.shape[0]-1],
                             dtype=np.uint32)
                fcs = np.vstack((fcs, f))
                
            self.vts = vts
            self.fcs = fcs
            self._getFeatures()
            
    
    def genIcoSphere(self, c3=[0.0, 0.0, 0.0], rad=1.0, nsub=0):# OK
        vts, fcs = self._genIcosahedron(nsub=nsub)
        self.vts = rad*vts + np.array(c3) # Broadcasting!
        self.fcs = fcs
        
        # Initialising Features
        self._getFeatures()
        
    def _genIcosahedron(self, nsub=0): # OK
        # Creating a unit regular icosahedron
        t = (1 + np.sqrt(5.0)) / 2
        # Define Vertices
        vts = np.array([[-1, t, 0], # v1
                        [ 1, t, 0], # v2
                        [-1,-t, 0], # v3
                        [ 1,-t, 0], # v4
                        [ 0,-1, t], # v5
                        [ 0, 1, t], # v6
                        [ 0,-1,-t], # v7
                        [ 0, 1,-t], # v8
                        [ t, 0,-1], # v9
                        [ t, 0, 1], # v10
                        [-t, 0,-1], # v11
                        [-t, 0, 1]]) # v12
        
        # Normalising Vertices to unit vectors        
        vts = vts/np.linalg.norm(vts, axis=1, keepdims=1)
        
        # Define Faces (Triangles)
        fcs = np.array([[ 0, 11,  5], # f1
                        [ 0,  5,  1], # f2
                        [ 0,  1,  7], # f3
                        [ 0,  7, 10], # f4
                        [ 0, 10, 11], # f5
                        [ 1,  5,  9], # f6
                        [ 5, 11,  4], # f7
                        [11, 10,  2], # f8
                        [10,  7,  6], # f9
                        [ 7,  1,  8], # f10
                        [ 3,  9,  4], # f11
                        [ 3,  4,  2], # f12
                        [ 3,  2,  6], # f13
                        [ 3,  6,  8], # f14
                        [ 3,  8,  9], # f15
                        [ 4,  9,  5], # f16
                        [ 2,  4, 11], # f17
                        [ 6,  2, 10], # f18
                        [ 8,  6,  7], # f19
                        [ 9,  8,  1]], dtype=np.uint64)# f20
        
        # Subdivision
        nsub = int(nsub)
        if nsub > 0:
            vts, fcs = self._subdTriFaces(vts, fcs, nsub=nsub)
        
        return vts, fcs
    
    def subdivide(self, nsub=1, nrmFlag=True):
        nsub = int(nsub)
        if nsub > 0:
            vts, fcs = self._subdTriFaces(self.vts, self.fcs, 
                                          nsub=nsub, nrmFlag=nrmFlag)
            
        # Assignment
        self.vts = vts
        self.fcs = fcs
        # Initialising Features
        self._getFeatures()
    
    def _subdTriFaces(self, vts, fcs, nsub, nrmFlag=True):# OK
        # Recursively subdivision of triangular faces
        if nsub == 0:
            return vts, fcs
        
        for sbd in range(0, nsub):
            # Initialising subdivided Faces (ONLY Triangles)
            sbd_fcs = np.zeros([fcs.shape[0]*4, 3])
            
            for ff in range(0, fcs.shape[0]): # for each triangular face    
                # Select the i-th Trianglular face
                fc = fcs[ff, :]

                # Calculate the mid points (add new points to v)
                a, vts = self._appendNormMidPoint(fc[0], fc[1], vts, nrmFlag=nrmFlag)
                b, vts = self._appendNormMidPoint(fc[1], fc[2], vts, nrmFlag=nrmFlag)
                c, vts = self._appendNormMidPoint(fc[2], fc[0], vts, nrmFlag=nrmFlag)
                    
                # Generating new subdivision triangles
                nfc = np.array([[fc[0], a, c],
                                [fc[1], b, a],
                                [fc[2], c, b],
                                [    a, b, c]])
                    
                # Replacing Triangle with subdivision
                idx = list(range((4*ff), (4*(ff+1))))
                sbd_fcs[idx, :] = nfc
                
            # Updating Faces with the Subdivided Faces
            fcs = np.array(sbd_fcs, dtype=np.uint64)  
        
        # Removing duplicate vertices
        # NEED for ROBUST UNIQUE? (tolerance) [Potential source of BUG?]
        vts_unq, vts_idx = np.unique(vts.round(decimals=6), axis=0, return_inverse=1) # << Unique with Real values! (BUG?)
        # Re-assigning faces to trimmed vertex list and remove duplicate faces
        fcs_unq = np.zeros(fcs.shape, dtype=np.uint64)
        for vidx in range(0, len(vts_idx)):
            fcs_unq[fcs == vidx] = vts_idx[vidx]
        fcs_unq = np.unique(fcs_unq, axis=0) # << Unique with integers (OK)
        
        return vts_unq, fcs_unq
        
    def _appendNormMidPoint(self, idx0, idx1, vts, nrmFlag=True):# OK
        # Retrieve vertices 
        v0 = vts[idx0, :]
        v1 = vts[idx1, :]
        if nrmFlag:
            # New length-normalised Mid-Point
            v2mod = np.mean([np.linalg.norm(v0), np.linalg.norm(v1)])
            v2vec = uvect(np.mean([v0, v1], axis=0))
            v2 = np.reshape(v2vec, [1, 3])*v2mod
        else:
            v2 = np.mean([v0, v1], axis=0).reshape([1, 3])
        # Concatenate Mid-Point
        vts = np.concatenate((vts, v2), axis=0)
        
        idx = vts.shape[0] - 1
        return idx, vts
        
    def _getTriEdges(self):
        
        edg = []
        fcs = self.fcs
        for ff in range(0, fcs.shape[0]):
            edg.append([fcs[ff,0], fcs[ff,1]])
            edg.append([fcs[ff,1], fcs[ff,2]])
            edg.append([fcs[ff,2], fcs[ff,0]])
        edg = np.vstack(edg).astype(np.uint64)
        _, eidx = np.unique(np.sort(edg, axis=1), axis=0, return_index=1)
        edg = edg[eidx,:]
        
        return edg
    
    def _getBoundaryEdges(self):
        
        edg = self._getTriEdges()
        chk = []
        for ee in range(0, edg.shape[0]):
            cc = np.sum(np.sum(np.hstack((self.fcs == edg[ee,0],
                                          self.fcs == edg[ee,1])),
                               axis=1) == 2,
                        axis=0)
            chk.append(cc)
        chk = np.vstack(chk) == 1
        edg_bdr = edg[chk.squeeze(),:]
        
        return edg_bdr
    
    def _getBoundaryTriangles(self):
        
        edg_bdr = self._getBoundaryEdges()
        fcs_idx = []
        for ee in range(0, edg_bdr.shape[0]):
            chk = np.sum(np.bitwise_or(self.fcs == edg_bdr[ee,0], 
                                       self.fcs == edg_bdr[ee,1]), axis=1) == 2
            fcs_idx.append(np.argwhere(chk).squeeze())
        fcs_idx = np.hstack(fcs_idx)
        
        fcs_idx = np.unique(fcs_idx)
        return fcs_idx
    
    def decimate(self, fcsRate=0.5):
        maxFcs = int(fcsRate * self.fcs.shape[0])
        maxFcs = max(1, min(self.fcs.shape[0], maxFcs))
        fcs_red = self.fcs[:maxFcs].copy()
        used_vtx, inv = np.unique(fcs_red.reshape(-1), return_inverse=True)
        vts_red = self.vts[used_vtx, :]
        fcs_red = inv.reshape(fcs_red.shape).astype(np.uint64)
        # Assignment
        self.vts = vts_red
        self.fcs = fcs_red
        # Initialising Features
        self._getFeatures()
    
    def _showGeo3D(self, fig=None, fnFlag=False, vnFlag=False, fnIdx=None, vnIdx=None, fcsAlpha=0.5, supTitle=None, frmFileName=None):# OK
        # Enable interactive mode
        plt.ion()
        
        hs = [] # handles to Geometries plotted
        
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.001)
            ax = fig.axes[0]
        
        # Displaying Geometry as Triangualr Mesh (Patch)
        h0 = ax.plot_trisurf(self.vts[:, 0],
                        self.vts[:, 1],
                        self.vts[:, 2],
                        triangles = self.fcs,
                        color=self.mat.dffRGB,
                        edgecolor=[[0,0,0,0.25]],
                        linewidth=1.0,
                        alpha=fcsAlpha,
                        shade=False)
        hs.append(h0)
        
        if fnFlag:
            if fnIdx is None:
                h1 = ax.quiver(self.fcs_ctr[:, 0], self.fcs_ctr[:, 1], self.fcs_ctr[:, 2],
                          self.fn[:, 0],  self.fn[:, 1],  self.fn[:, 2],
                          length=self.size/5, normalize=True, color='k')
            else:
                h1 = ax.quiver(self.fcs_ctr[fnIdx, 0], self.fcs_ctr[fnIdx, 1], self.fcs_ctr[fnIdx, 2],
                          self.fn[fnIdx, 0],  self.fn[fnIdx, 1],  self.fn[fnIdx, 2],
                          length=self.size/5, normalize=True, color='k')
            hs.append(h1)
        
        if vnFlag:
            if vnIdx is None:
                h2 = ax.quiver(self.vts[:, 0], self.vts[:, 1], self.vts[:, 2],
                          self.vn[:, 0],  self.vn[:, 1],  self.vn[:, 2],
                          length=self.size/5, normalize=True, color='b')
            else:
                h2 = ax.quiver(self.vts[vnIdx, 0], self.vts[vnIdx, 1], self.vts[vnIdx, 2],
                           self.vn[vnIdx, 0],  self.vn[vnIdx, 1],  self.vn[vnIdx, 2],
                           length=self.size/5, normalize=True, color='b')  
            hs.append(h2)
        
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
        
        return fig, hs
    
    def _removeALLgeoHandles(self, geoHandles):
        for ii in range(0, len(geoHandles)):
            if isinstance(geoHandles[ii], list):
                geoHandles[ii][0].remove()
            else:
                geoHandles[ii].remove()
    
    def readSTL(self, STLFileName=None):# OK
    # STL in BINARY format! (if it's ASCII - refer to OBJ [w/ MeshLab Format])
        if stl_reader is None:
            raise ImportError('stl_reader is required for Geometry.readSTL()')
        if STLFileName is not None:
            try:
                vts, fcs = stl_reader.read(STLFileName)
                self.vts = np.array(vts)
                self.fcs = np.array(fcs, dtype=np.uint64)
            except:
                print('<!> {}: Unable to read data from: {}'.format(
                    inspect.stack()[0][3], str(STLFileName)))
                self.vts = None
                self.fcs = None
        else:
            self.vts = None
            self.fcs = None
        
        # Initialising Features
        self._getFeatures()
        
    def readOBJ(self, OBJFileName=None):# OK
        try:
            vts = []
            fcs = []
            clr = []
            with open(OBJFileName) as file:
                for line in file:
                    if line[0:2] == "v ":
                        vv = list(map(float, line[2:].strip().split()))
                        if len(vv) <= 3:
                            vts.append(vv)
                        else:
                            vts.append(vv[0:3])
                            clr = vv[3:]
                    elif line[0:2] == "f ":
                        ff = list(map(int, line[2:].strip().replace('/',' ').split()))
                        if len(ff) <=3:
                            fcs.append(ff)
                        else:
                            fcs.append(ff[0::2])
                     
            if len(clr) == 3:
                self.mat.dffRGB = np.array(clr)
                self.mat.ambRGB = np.divide(np.array(clr), 5.0)
                     
            self.vts = np.array(vts)
            self.fcs = np.array(fcs, dtype=np.uint64) - 1 # Indices start from 0 (not from 1)
            
            print('>> {}: Succesfully read data from: {}'.format(
                inspect.stack()[0][3], str(OBJFileName)))
            
        except:
            print('<!> {}: Unable to read data from: {}'.format(
                inspect.stack()[0][3], str(OBJFileName)))
            self.vts = None
            self.fcs = None
        
        # Initialising Features
        self._getFeatures()
        
    def writeOBJ(self, OBJFileName=None, matExport=False): #OK
        try:
            hdr = ['# OBJ File Generated with Python\n',
                   '# Vertices: {:d}\n'.format(np.shape(self.vts)[0]),
                   '# Faces: {:d}\n'.format(np.shape(self.fcs)[0])]
            
            vts = self.vts
            fcs = self.fcs + 1 # Faces indices must start form 1 (not from 0!)
            clrRGB = self.mat.dffRGB
            
            if matExport:
                hdr.append('# Vertex Colour: Enabled\n\n')
            else:
                hdr.append('#\n')
            
            with open(OBJFileName, 'w') as file:
                for hh in range(0, len(hdr)):
                    file.write(hdr[hh])
                
                for vv in range(0, vts.shape[0]):
                    if matExport:
                        file.write('v {:.7f} {:.7f} {:.7f} {:.3f} {:.3f} {:.3f}\n'.format(
                            vts[vv,0], vts[vv,1], vts[vv,2], 
                            clrRGB[0], clrRGB[1], clrRGB[2]))
                    else:
                        file.write('v {:.7f} {:.7f} {:.7f}\n'.format(
                            vts[vv,0], vts[vv,1], vts[vv,2]))
                        
                file.write('\n')
                    
                for ff in range(0, self.fcs.shape[0]):
                    file.write('f {:d} {:d} {:d}\n'.format(
                        fcs[ff,0], fcs[ff,1], fcs[ff,2])) 

                file.write('\n# END of FILE\n')                 
                
            file.close()
            
        except:
            print('<!> {}: Unable to write data to: {}'.format(
                inspect.stack()[0][3], str(OBJFileName)))

        return None
        
    def scale(self, scale=1.0):
        scale = np.array(scale)
        self.vts = np.multiply(self.vts, scale) # multiply by scalar (ISOTROPIC SCALING)
        
        # Initialising Features
        self._getFeatures()

    def scaleAniso(self, scale=[1.0, 1.0, 1.0]):
        scale = np.array(scale)
        self.vts[:, 0] = self.vts[:, 0]*scale[0]
        self.vts[:, 1] = self.vts[:, 1]*scale[1]
        self.vts[:, 2] = self.vts[:, 2]*scale[2]
        
        # Initialising Features
        self._getFeatures()
    
    def tformRigid(self, a_euler=0.0, b_euler=0.0, c_euler=0.0, x_offset=0.0, y_offset = 0.0, z_offset=0.0, centredFlag=1): # OK
        # E.g for Bunny:
        #   a_euler = np.pi/2,
        #   b_euler = 0,
        #   c_euler = np.pi
        
        # A 4x4 transformation matrix will be generated from the input parameters
        M = self._getRigidMatrix(a_euler, b_euler, c_euler, x_offset, y_offset, z_offset)
        
        # If centred: Center to Origin -> Transform -> ReOffset
        if centredFlag:
            CoM = np.nanmean(self.vts, axis=0)
            vts = self.vts - CoM
        else:
            vts = self.vts
            
        # Transformation
        vts = np.concatenate((vts.transpose(), np.ones([1, vts.shape[0]])), axis=0)
        vts = np.matmul(M, vts).transpose()
        vts = vts[:, 0:3]
            
        if centredFlag: 
            vts = vts + CoM
            
        self.vts = vts
        # Initialising Features
        self._getFeatures()
    
    def tformRigidM(self, M=np.eye(4), centredFlag=1):
        # De-offset
        if centredFlag:
            CoM = np.nanmean(self.vts, axis=0)
            vts = self.vts - CoM
        else:
            vts = self.vts
            
        # Transformation
        vts = np.concatenate((vts.transpose(), np.ones([1, vts.shape[0]])), axis=0)
        vts = np.matmul(M, vts).transpose()
        vts = vts[:, 0:3]
           
        # Re-offset
        if centredFlag: 
            vts = vts + CoM
            
        self.vts = vts
        # Initialising Features
        self._getFeatures()
        
    def cullFaces(self, v3=None):
        if v3 is None:
            return None
        v3 = uvect(v3)
        fcs_cull = projct(self.fn, v3) <= 0.0
        return fcs_cull  

    def shdFaces(self, sel_fn=None, v3=None):
        if v3 is None:
            return None
        v3 = uvect(v3)
        if sel_fn is None:
            fcs_shd = projct(self.fc, v3)
        else:
            fcs_shd = projct(sel_fn, v3)
            
        return fcs_shd
    
    def _getRigidMatrix(self, a_euler=0.0, b_euler=0.0, c_euler=0.0, x_offset=0.0, y_offset = 0.0, z_offset=0.0): #OK
        
        M = np.eye(4)
        M[0:3, 0:3] = rotM3D(a_euler, b_euler, c_euler)
        M[0,3] = x_offset
        M[1,3] = y_offset
        M[2,3] = z_offset
        
        return M
        
    def stats(self):
        print(' ')
        print(' * Geometry Stats*')
        print(' - Label: {:s}'.format(self.label))
        print(' - Vertices: {:d} x {:d}'.format(self.vts.shape[0], self.vts.shape[1]))
        print(' - Faces: {:d} x {:d}'.format(self.fcs.shape[0], self.fcs.shape[1]))
        print(' - Face Size: {:.3f}'.format(self.trisize))
        print(' - CoM coords XYZ: [{:.3f},{:.3f},{:.3f}]'.format(
            self.CoM[0], self.CoM[1], self.CoM[2]))
        print(' - CoM-Vertices Avg Dist: {:.3f}'.format(self.size))
        print(' - Bounding Box XYZ: [[{:.3f},{:.3f},{:.3f}],[{:.3f},{:.3f},{:.3f}]]'.format(
            self.BBox[0,0], self.BBox[0,1], self.BBox[0,2],
            self.BBox[1,0], self.BBox[1,1], self.BBox[1,2]))
        print(' - Material (Diffuse RGB): [{:.3f},{:.3f},{:.3f}]'.format(
            self.mat.dffRGB[0], self.mat.dffRGB[1], self.mat.dffRGB[2]))
        print(' - Material (Ambient RGB): [{:.3f},{:.3f},{:.3f}]'.format(
            self.mat.ambRGB[0], self.mat.ambRGB[1], self.mat.ambRGB[2]))
        print(' - Material (Specular RGB): [{:.3f},{:.3f},{:.3f}]'.format(
            self.mat.spcRGB[0], self.mat.spcRGB[1], self.mat.spcRGB[2]))
        print(' - Material (Gloss coeff): {:.3f}'.format(self.mat.shnC))
        print(' - Material (Reflective coeff): {:.3f}'.format(self.mat.rflC))
        print(' - Material (Max Depth - Refl. Bounces): {:d}'.format(self.mat.MaxDepth))
        print(' ')
        return None
        
    def _getFidxsFromVidxs(self, vidx):
        vidx = list(vidx)
        fidx = np.empty([1,1], dtype=np.uint64)
        map_idx = np.empty([1,1], dtype=np.uint64)
        for ii in range(0, len(vidx)):
            fidx_tmp = np.argwhere(np.any(self.fcs == vidx[ii], axis=1))
            fidx = np.vstack((fidx, fidx_tmp.reshape([len(fidx_tmp), 1])))
            map_idx = np.vstack((map_idx, ii*np.ones([len(fidx_tmp), 1])))

        return fidx[1:], map_idx[1:]
