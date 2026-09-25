import numpy as np

def uvect(v3):
    # retrieve unit-vector: v3 expected array [3,]
    uv3 = np.array(v3)
    if len(v3.shape) == 1:
        uv3 = uv3/np.linalg.norm(uv3)
    else:
        dd = np.linalg.norm(uv3, axis=1)
        uv3 = np.divide(uv3, dd.reshape(dd.shape[0], 1))
    return uv3
def projct(a3, b3):
    # projection of vector a3 to vector b3: a3 and b3 expected arrays [3,]
    c = np.dot(a3, b3)
    if len(np.shape(a3)) > 1:
        c = np.reshape(c, [np.shape(c)[0], 1])
    return c
def orthog(a3, b3, uvectFlag=True):
    # orthogonalisation of a3 wrt b3: a3 and b3 expected arrays [3,]
    o3 = a3 - projct(a3, b3)*b3
    if uvectFlag:
        o3 = uvect(o3)
    return o3
def rotM3D(a, b, c):
    # Rotation Matrix as (improper) Euler angles:
    # a -> around X, b -> around Y, c -> around Z
    Rz = np.array([[np.cos(c),    -np.sin(c), 0],
                   [np.sin(c),     np.cos(c), 0],
                   [        0,             0, 1]])

    Ry = np.array([[ np.cos(b),  0, np.sin(b)],
                   [         0,  1,         0],
                   [-np.sin(b),  0, np.cos(b)]])

    Rx = np.array([[1,          0,          0],
                   [0,  np.cos(a), -np.sin(a)],
                   [0,  np.sin(a),  np.cos(a)]])

    # Rotation Matrix Mutliplication 
    R = np.matmul(Rz, np.matmul(Ry, Rx))
    return R
def rotM3Da2b(a0, a1, b0, b1):
    # WIP: estimate the rotation matrix mapping two sets of orthonormal bases:
    # a0, a1: source orthonormal base
    # b0, b1: target orthonormal base
    # all input vectors are arrays: 1x3
    # Output rotation matrix R is 3x3 so that:
    #   b0 = a0*R
    #   b1 = a1*R
    a0 = np.array(a0)
    a1 = np.array(a1)
    b0 = np.array(b0)
    b1 = np.array(b1)
    
    # Estimate partial rotation matrix (R1)
    R1 =  _rot3DMap_a2b(a0, b0)
        
    # Rotate a1 with estimated partial rotation matrix R1
    a1r = np.matmul(a1, R1)
    
    # Estimate partial rotation matrix (R2)
    R2 =  _rot3DMap_a2b(a1r, b1)
        
    # Composition of partial rotations (R1, R2)
    R = np.matmul(R1, R2)
    
    return R.transpose()
def _rot3DMap_a2b(a3, b3, tol=0.001):
    
    if np.any(np.abs(a3-b3) > tol):
        if np.any(np.abs(np.abs(a3) - np.abs(b3)) > tol):
            
            v3 = np.cross(b3, a3) # Cross product, taking into account the target FIRST, the source AFTER
            s = np.linalg.norm(v3) # Norm of the resulting vector (sin)
            c = np.dot(a3, b3) # (cos) between vectors
        
            vmat = np.array([[   0  , -v3[2],   v3[1] ],
                             [ v3[2],    0  ,  -v3[0] ],
                             [-v3[1],  v3[0],      0  ]])
        
            R = np.eye(3) + vmat + np.matmul(vmat, vmat)*((1-c)/s**2)
                  
        else: # case a0 = -b0
            R = np.eye(3)
            for i in range(0,len(a3)):
                if a3[i] * b3[i] < 0:
                    R[i, i] = -1
            
    else: # case a0 = b0 
        R = np.eye(3)
    
    return R
def rotVec(v3=np.zeros(3), R_v3C=np.eye(3), R=np.eye(3)):
    # Map (unit) vector to ORIGIN
    v3R = np.matmul(R_v3C, v3.transpose()).transpose()
    # Apply Rotation for Canonical EULER angles
    v3R = np.matmul(R, v3R.transpose()).transpose()
    # Map the rotated (unit) vector back to orifinal reference frame
    v3R = np.matmul(R_v3C.transpose(), v3R.transpose()).transpose()
    
    return v3R 
def dist_p_line(pts3D, pt3D_lin, nn3D_lin):
    # Function to determine the distance of a set of 3D points (pts3D) from
    # a line in 3D, defined as the t-parametric eq. x = pt3D_lin + t*nn3D_lin
    # INPUTS:
    # pts3D expected array Nx3
    # pt3D_lin expected 1x3 OR Mx3
    # nn3D_lin expected 1x3 OR Mx3
    # OUTPUT: 
    # dd_pts3D must be Nx1 OR NxM
    
    # t0 = time.time()
    if len(pt3D_lin.shape) < 2:
        d1 = pt3D_lin - pts3D # d1 Must be Nx3
        d2 = projct(d1, nn3D_lin) * nn3D_lin # d2 must be Nx3
    else:
        ''' SLOWER
        pt3D_lin = pt3D_lin.transpose().reshape([1, 
                                                 pt3D_lin.shape[1], 
                                                 pt3D_lin.shape[0]])
        nn3D_lin = nn3D_lin.transpose().reshape([1, 
                                                 nn3D_lin.shape[1], 
                                                 nn3D_lin.shape[0]])
        
        d1 = pt3D_lin - pts3D.reshape([pts3D.shape[0], pts3D.shape[1],1]) # d1 must be Nx3xM
        # Convoluted
        p1 = np.moveaxis(d1,2,0) @ np.moveaxis(nn3D_lin,[0,1,2],[2,1,0])
        p1 = np.moveaxis(p1,0,2)
        d2 = p1 * nn3D_lin
        ''' # FASTER
        pt3D_lin = np.moveaxis(pt3D_lin,0,1)
        pt3D_lin = pt3D_lin[np.newaxis,:] 
        nn3D_lin = np.moveaxis(nn3D_lin,0,1)
        nn3D_lin = nn3D_lin[np.newaxis,:]

        d1 = pt3D_lin - pts3D[:,:, np.newaxis] # d1 must be Nx3xM
        p1 = np.sum(np.multiply(d1, nn3D_lin), axis=1)
        d2 = np.multiply(p1[:,np.newaxis,:], nn3D_lin)
        
    dd_pts3D = np.linalg.norm(d1-d2, axis=1).squeeze() # dd_pts3D must be Nx1 OR NxM
    
    # t1 = time.time()
    # print('<DBG> dist_p_line T-elaps: {:.3f} s'.format(t1-t0))
    return dd_pts3D
def sect_lin_pln(ptA3D_lin, ptB3D_lin, ptA3D_pln, ptB3D_pln, ptC3D_pln):# OK! - make it N-dim
    # all inputs must have shape = (3,)
    try:
        AB3D_lin = ptB3D_lin - ptA3D_lin
        t_N = np.sum(np.multiply(np.cross(ptB3D_pln - ptA3D_pln, ptC3D_pln - ptA3D_pln), (ptA3D_lin - ptA3D_pln)), axis=1)
        t_D = np.sum(np.multiply(-AB3D_lin, np.cross( ptB3D_pln - ptA3D_pln, ptC3D_pln - ptA3D_pln)), axis=1)
        
        ipt3D = ptA3D_lin + np.multiply(AB3D_lin, np.tile(np.divide(t_N,t_D).reshape([t_N.shape[0],1]),[1,ptA3D_lin.shape[1]]))
    except:
        AB3D_lin = ptB3D_lin - ptA3D_lin
        t_N = np.sum(np.multiply(np.cross(ptB3D_pln - ptA3D_pln, ptC3D_pln - ptA3D_pln), (ptA3D_lin - ptA3D_pln)))
        t_D = np.sum(np.multiply(-AB3D_lin, np.cross( ptB3D_pln - ptA3D_pln, ptC3D_pln - ptA3D_pln)))
        
        ipt3D = ptA3D_lin + np.multiply(AB3D_lin, np.divide(t_N,t_D))
        
    return ipt3D
def isPtInTri(pts3D, tri3D):
    # function to evaluate if a given point in 3D (or a set of those) is inside
    # a triangle (or a set of those)
    # pts3D expected 1x3 OR Nx3 
    #   with [[p0x, p0y, p0z],...]
    # tri3D expected 1x9 OR Nx9
    #   with [[t0x, t0y, t0z, t1x, t1y, t1z, t2x, t2y, t2z],...] 
    try:
        e0 = tri3D[:,6: ] - tri3D[:,0:3]
        e1 = tri3D[:,3:6] - tri3D[:,0:3]
        e2 = pts3D        - tri3D[:,0:3]
    
        d00 = np.sum(np.multiply(e0, e0), axis=1)
        d01 = np.sum(np.multiply(e0, e1), axis=1)
        d02 = np.sum(np.multiply(e0, e2), axis=1)
        d11 = np.sum(np.multiply(e1, e1), axis=1)
        d12 = np.sum(np.multiply(e1, e2), axis=1)
        
        iD = np.divide(1.0, np.multiply(d00, d11) - np.multiply(d01, d01))
        u = np.multiply(np.multiply(d11, d02) - np.multiply(d01, d12), iD)
        v = np.multiply(np.multiply(d00, d12) - np.multiply(d01, d02), iD)
    
    except:
        e0 = tri3D[6: ] - tri3D[0:3]
        e1 = tri3D[3:6] - tri3D[0:3]
        e2 = pts3D      - tri3D[0:3]
    
        d00 = np.sum(np.multiply(e0, e0))
        d01 = np.sum(np.multiply(e0, e1))
        d02 = np.sum(np.multiply(e0, e2))
        d11 = np.sum(np.multiply(e1, e1))
        d12 = np.sum(np.multiply(e1, e2))
        
        iD = np.divide(1.0, np.multiply(d00, d11) - np.multiply(d01, d01))
        u = np.multiply(np.multiply(d11, d02) - np.multiply(d01, d12), iD)
        v = np.multiply(np.multiply(d00, d12) - np.multiply(d01, d02), iD)
        
    inn = np.logical_and(np.logical_and( u >= 0.0, v >= 0.0), (u + v) <= 1.0)
        
    return inn


def isOppVdir(n0s, n1s):
    return np.sum(np.multiply(n0s, n1s), axis=1) < 0
