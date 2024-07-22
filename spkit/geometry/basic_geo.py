'''
Basic Geomatrical Function & Processings
------------------------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023. Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.qhull import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.tri import Triangulation
import copy

class Inter2DPlane(object):
    def __init__(self,xy_loc,res=[128,128]):
        if isinstance(res,int): res = [res,res]
        self.res = res
        self.xy_loc = xy_loc
        self.xi = np.linspace(xy_loc[:,0].min(),xy_loc[:,0].max(), res[0])
        self.yi = np.linspace(xy_loc[:,1].min(),xy_loc[:,1].max(), res[1])
        self.Xi, self.Yi = np.meshgrid(self.xi, self.yi)
        self.tri = Delaunay(xy_loc)
    def get_image(self,values):
        Interpolator = CloughTocher2DInterpolator(self.tri, values)
        args = [self.Xi, self.Yi]
        Zi = Interpolator(*args)
        return Zi
    def get_image2(self,values):
        Interpolator = CloughTocher2DInterpolator(self.xy_loc, values)
        Zi = Interpolator(self.Xi,self.Yi)
        return Zi

def car2spar(X):
    rad = np.linalg.norm(X, axis=1)
    zen = np.arccos(X[:,2] / rad)   #[0, pi],
    azi = np.arctan2(X[:,1], X[:,0])# + np.pi
    return np.c_[rad,zen,azi]

def spar2car(S):
    # r>=0, th: [0, pi], ph: [0, 2pi]
    r,th,ph = S[:,0],S[:,1],S[:,2]
    x = r*np.cos(ph)*np.sin(th)
    y = r*np.sin(ph)*np.sin(th)
    z = r*np.cos(th)
    return np.c_[x,y,z]

def getTriFaces(V, plot=True):
    V1 = V - V.mean(axis=0)
    rad = np.linalg.norm(V1, axis=1)
    zen = np.arccos(V1[:,-1] / rad)
    azi = np.arctan2(V1[:,1], V1[:,0])
    tris = Triangulation(zen, azi)
    if plot:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(V1[:,0], V1[:,1], V1[:,2], triangles=tris.triangles)
        plt.show()
    return tris

def getTriFaces_V2(V, plot=True,npoint_rotate=10,sep=None,verbose=0,zshift=0):
    V1 = V - V.mean(axis=0) + zshift
    rad = np.linalg.norm(V1, axis=1)
    zen = np.arccos(V1[:,-1] / rad)
    azi = np.arctan2(V1[:,1], V1[:,0])
    idx0 = np.argsort(azi)

    n = npoint_rotate
    if n>0:
        #idx1 = np.r_[idx0,idx0[:n]]
        idx1 = np.r_[idx0,idx0[:n]]
    else:
        idx1 = idx0.copy()

    if sep is None:
        #diff  = np.abs(np.diff(azi[idx0]))
        #sep = np.min(diff[diff>0])
        sep = np.pi - np.max(azi)
        if verbose:
            print('min_diff',sep)
    if n>0:
        azi1 = np.r_[azi[idx0],np.max(azi)+sep+(azi[idx0[:n]]-np.min(azi[idx0[:n]]))]
        zen1 = np.r_[zen[idx0],zen[idx0[:n]]]
        rad1 = np.r_[rad[idx0],rad[idx0[:n]]]
    else:
        azi1 = azi[idx0]
        zen1 = zen[idx0]
        rad1 = rad[idx0]
    tris = Triangulation(zen1, azi1)

    triangles = idx1[tris.triangles]
    triangles = np.unique(triangles,axis=0)
    triangles = np.array([np.sort(fi).tolist() for fi in triangles.tolist()])
    triangles = np.unique(triangles,axis=0)
    if plot:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(V[:,0], V[:,1], V[:,2], triangles=triangles)
        plt.show()
    return triangles

def getEdges_idx(tri):
    E = []
    for s in tri.simplices:
        e1 = [s[0],s[1]]
        e2 = [s[1],s[2]]
        e3 = [s[0],s[2]]
        e1.sort()
        e2.sort()
        e3.sort()

        E.append(e1)
        E.append(e2)
        E.append(e3)

    E = np.array(E)
    E = np.unique(E,axis=0)
    return E

def getEdges_idxSet(tri):
    E = []
    for s in tri.simplices:
        e1 = [s[0],s[1]]
        e2 = [s[1],s[2]]
        e3 = [s[0],s[2]]
        e1.sort()
        e2.sort()
        e3.sort()

        E.append(e1)
        E.append(e2)
        E.append(e3)

    E = np.array(E)
    E = np.unique(E,axis=0)
    E = [set(e) for e in E]
    return E

def indx2points_edges(xy,e):
    i,j = e
    return set([tuple(xy[i]),tuple(xy[j])])

def removeTri_edge(F,e):
    e = list(e)
    Fr = np.array([f for f in F if np.sum(f==e[0])+np.sum(f==e[1])<2])
    return Fr

def isEdge_inTri(T,e):
    e = list(e)
    for f in T:
        if (np.sum(f==e[0])+np.sum(f==e[1]))==2:
            return True
    return False

def removeTri_Edges(F,E):
    Fr = []
    for f in F:
        not_include = True
        for e in E:
            e = list(e)
            if (np.sum(f==e[0]) + np.sum(f==e[1]))==2:
                not_include=False
                break
        if not_include:
            Fr.append(f)
    return np.array(Fr)

def selEdges(xy,E,thr,lesthan=True,axis=0):
    if lesthan:
        idx = np.where(xy[:,0]<thr)[0]
    else:
        idx = np.where(xy[:,0]>thr)[0]
    if axis==1:
        if lesthan:
            idx = np.where(xy[:,1]<thr)[0]
        else:
            idx = np.where(xy[:,1]>thr)[0]

    E_sel = []
    for i,j in E:
        if i in idx and j in idx:
            E_sel.append(set([i,j]))

    E_xy = [indx2points_edges(xy,e) for e in E_sel]

    return E_sel, E_xy

def find_E1inE2(E1_xy, E2_xy):
    E1_in_E2_xy = []
    E1_not_E2_xy = []
    E1_in_E2 = []
    E1_not_E2 = []
    for k,e in enumerate(E1_xy):
        if e in E2_xy:
            E1_in_E2_xy.append(e)
            E1_in_E2.append(k)
        else:
            E1_not_E2_xy.append(e)
            E1_not_E2.append(k)

    return (E1_in_E2, E1_not_E2), (E1_in_E2_xy,E1_not_E2_xy)

#========================================
## Optimal Projection  - 3D to 2D
#======================================

def rotation_matrix(theta=np.pi/4):
    Mx = np.array([[1,              0,           0],
                   [0, np.cos(theta),-np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])

    My = np.array([[np.cos(theta),0, np.sin(theta)],
                   [0,            1,           0],
                   [-np.sin(theta),0, np.cos(theta)]])

    Mz = np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta), np.cos(theta),0],
                  [0,              0,           1]])
    return Mx,My,Mz

def get_neibours(V,p,N=None):
    dist = np.sum((V-p)**2,1)
    idx = np.argsort(dist)
    if N is None:
        return idx
    return idx[:N]

def get_optimal_projection(X,p,d=0,plot=True):
    t = np.arange(np.min(X)/5,np.max(X)/5)[:,None]
    #t.shape
    n1  = p[0] - X[1,:]
    n2  = p[0] - X[2,:]
    n3 = np.cross(n1,n2)
    n3 = n3/np.linalg.norm(n3)

    l1 = p + t*n1
    l2 = p + t*n2
    l3 = p + t*n3


    Xp1 = X - np.multiply((X@n3 + d),n3[:,None]).T
    Xp2 = X - np.multiply(((X-p)@n3 + d),n3[:,None]).T
    #Xp1.shape


    if plot:
        XYZ = get_plane(n=n3,d=d,t=(np.min(X)-10,np.max(X)+10))
        #XYZ.shape

        fig = plt.figure(figsize = [5,4])
        ax = fig.add_subplot(projection="3d")
        u =np.max(X)
        ax.plot([0,u],[0,0],[0,0],'--k',lw=0.5)
        ax.plot([0,0],[0,u],[0,0],'--k',lw=0.5)
        ax.plot([0,0],[0,0],[0,u],'--k',lw=0.5)
        u =-np.max(X)
        ax.plot([0,u],[0,0],[0,0],'--k',lw=0.5)
        ax.plot([0,0],[0,u],[0,0],'--k',lw=0.5)
        ax.plot([0,0],[0,0],[0,u],'--k',lw=0.5)

        ax.plot(X[:,0],X[:,1],X[:,2],'.',ms=5)
        ax.plot(p[:,0],p[:,1],p[:,2],'*r',ms=10)

        ax.plot(l1[:,0],l1[:,1],l1[:,2],'-',ms=1)
        ax.plot(l2[:,0],l2[:,1],l2[:,2],'-',ms=1)
        ax.plot(l3[:,0],l3[:,1],l3[:,2],'-',ms=1)

        ax.plot(XYZ[:,0],XYZ[:,1],XYZ[:,2],'.',ms=1)


        ax.plot(Xp1[:,0],Xp1[:,1],Xp1[:,2],'.',ms=1)
        ax.plot(Xp2[:,0],Xp2[:,1],Xp2[:,2],'.',ms=1)

        ax.axis('off')
        #plt.plot(Xpca[:,1],Xpca[:,0],'*')
        ax.set_xlim(np.min(X),np.max(X))
        ax.set_ylim(np.min(X),np.max(X))
        ax.set_zlim(np.min(X),np.max(X))
        plt.tight_layout()
        plt.show()

    return Xp2

def get_plane(n=[1,1,1],d=0,px =[],t=(-10,10)):
    #n1 = np.array([1,1,1])
    x0 = np.arange(t[0],t[1])
    y0 = np.arange(t[0],t[1])
    Xi,Yi = np.meshgrid(x0,y0)
    Xi = Xi.reshape(-1)
    Yi = Yi.reshape(-1)

    # ax + by + cz + d = 0
    #d = 0
    #n1[0]*Xi +n1[1]*Yi + n1[2]*Zi + d = 0
    #Zi = (d - n1[0]*Xi - n1[1]*Yi)/n1[2]
    Zi = -(n[0]*Xi + n[1]*Yi + d)/n[2]
    #Xi.shape, Yi.shape, Zi.shape
    return np.c_[Xi,Yi,Zi]

def get_PCA(X,whiting=True):
    if whiting:
        Xn = (X - X.mean(0))/(X.std(0)+1e-3)
    else:
        Xn = X.copy()
    CV = np.cov(Xn.T)

    eValues, V = np.linalg.eig(CV)
    idx = np.argsort(eValues)[::-1]

    V = V[:,idx]
    eValues = eValues[idx]
    W = V.copy()
    Xpca = Xn.dot(W)
    return Xpca,V,eValues

def opt_project(X,p,plot=True):
    Xp1 = get_optimal_projection(X,p,d=0,plot=plot)
    Xp2,W,ev = get_PCA(Xp1,whiting=False)
    Xp1c = Xp1 -p
    return Xp1,Xp1c,Xp2

def plot_proj(Xp,Xp1,Xp1c,Xp2,p):
    fig = plt.figure(figsize = [8,4])
    ax = fig.add_subplot(121,projection="3d")
    ax.plot(Xp[:,0],Xp[:,1],Xp[:,2],'.',ms=1)
    ax.plot(Xp1[:,0],Xp1[:,1],Xp1[:,2],'.',ms=1)
    ax.plot(p[:,0],p[:,1],p[:,2],'*r',ms=5)

    ax.plot(Xp1c[:,0],Xp1c[:,1],Xp1c[:,2],'.',ms=1)
    ax.plot([0],[0],[0],'*r',ms=5)

    ax.plot(Xp1c[-2:,0],Xp1c[-2:,1],Xp1c[-2:,2],'ok',ms=3)

    ax.plot(Xp2[:,0],Xp2[:,1],Xp2[:,2],'.',ms=1)
    ax.plot(Xp2[0,0],Xp2[0,1],Xp2[0,2],'*r',ms=5)
    ax.plot(Xp2[-2:,0],Xp2[-2:,1],Xp2[-2:,2],'ok',ms=3)

    #ax.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()


    ax = fig.add_subplot(122)

    ax.plot(Xp2[:,0],Xp2[:,1],'o',ms=1)
    ax.plot(Xp2[0,0],Xp2[0,1],'*r',ms=5)
    ax.plot(Xp2[-2:,0],Xp2[-2:,1],'ok',ms=3)
    ax.grid()
    plt.show()


def dir_vectors(V,AdjM,verbose=False):
    r""" Given V vertices as co-ordinates (n-dimensional space) and AdjM as Adjacency matrix,
    Compute D, as directional vectors, in form of
    D = [i,j, v_i,v_j], which indicates as vertex-i connects to vertex-j and
    and cordinates v_i and v_j, for v_j /= v_i

    D has shape of (nc, 2*n+2), for n-dimensional space, with nc number of connections

    Parameters
    ----------
    V: (m,n), m-vertices in n-dimentional space
    AdjM: (m,m), Adjacency matrix, binary and no self connections, means AdjM[i,i]==0

    Return
    D: directional vectors
    """
    assert set(np.unique(AdjM))==set([0,1])
    assert np.diag(AdjM).sum()==0
    assert AdjM.shape[0]==AdjM.shape[1]==V.shape[0]

    D  = []
    for i in range(AdjM.shape[0]):
        v_i = V[i]#.astype(float)
        ci = np.where(AdjM[i])[0]
        if verbose: print(f'From {i}th vertex to: ')
        for j in ci:
            v_j = V[j]#.astype(float)
            di = [i,j]+list(v_i)+list(v_j)
            if verbose: print(f' - {j} \t| {v_i.round(3)}--> {v_j.round(3)}')
            D.append(di)
    D = np.array(D)
    return D

def get_adjacency_matrix_depth(V,F,depth=1,remove_self_con=True,ignore_matrix=False):
    r"""
    Create Adjacency Matrix based on Trianglution Connection Depth

    Returns
    -------
    AdjM   : Binary Adjacency Matrix
    node2C : Dictionary of node to connection list node2C[node_a] = list_of_nodes_connected_to_node_a
    """

    M = V.shape[0]
    TA = TriAng(F=F,V=V)
    E0 = TA.getEdges_idx()
    node2C ={}
    for e in E0:
        e = list(e)
        for ei in e:
            if ei not in node2C.keys():
                node2C[ei] = set()
            node2C[ei].update(e)

    while depth>1:
        node2Ci = copy.deepcopy(node2C)
        for n in node2Ci.keys():
            nn = node2Ci[n]
            nnd = set()
            for ni in nn:
                nnd.update(node2Ci[ni])
            node2C[n].update(nnd)
        depth -=1

    if ignore_matrix: AdjM=None
    if not(ignore_matrix):
        AdjM = np.zeros([M,M])
        for i in range(M):
            if i in node2C:
                c = list(node2C[i])
                AdjM[i,c]=1

    if remove_self_con:
        if not(ignore_matrix): AdjM = AdjM-np.diag(np.diag(AdjM))
        for i in node2C:
            node2C[i] = node2C[i]-set([i])
            if len(node2C[i])==0: del node2C[i]


    return AdjM, node2C

def get_adjacency_matrix_dist(V,dist=5, remove_self_con=True,ignore_matrix=False):
    r"""
    Create Adjacency Matrix based on Euclidean distance

    Parameters
    ----------
    V   :  vertices, (m,n), m-points in n-dimentional space
    dist: float, distance
    Returns
    -------
    AdjM   : Binary Adjacency Matrix
    node2C : Dictionary of node to connection list node2C[node_a] = list_of_nodes_connected_to_node_a
    """

    M = V.shape[0]
    AdjM=None

    if not(ignore_matrix): AdjM = np.zeros([M,M]).astype(int)
    node2C ={}
    for i in range(M):
        clist = list(np.where(np.sqrt(np.sum((V-V[i])**2,1))<dist)[0])
        if remove_self_con: clist = list(set(clist)-set([i]))
        if len(clist):
            node2C[i] = list(clist)
            if not(ignore_matrix): AdjM[i,clist]=1
    return AdjM,node2C

def get_adjacency_matrix_kNN(V,K=5,remove_self_con=True, verbose=False,ignore_matrix=False):
    r"""
    Create Adjacency Index Matrix based on Euclidean distance

    Parameters
    ----------
    V   : Points, (m,n), m-points in n-dimentional space
    K   : int, number of nearest neibhbour
    remove_self_con: bool, if true, self point is excluded from neareast neighbours
    Returns
    -------
    AdjM   : Adjacency Index Matrix, (m, K) shape, index of neareast points, in accending order of distance
    """

    M = V.shape[0]
    AdjM=None
    #AdjM = np.zeros([M,M]).astype(int)
    #node2C ={}
    if not(ignore_matrix): AdjM = np.zeros([M,M]).astype(int)
    node2C ={}
    for i in range(M):
        if verbose: sp.utils.ProgBar(i, M,style=2)
        idx = list(np.argsort(np.sqrt(np.sum((V-V[i])**2,1)))[:K+1])
        idx.sort()
        if remove_self_con:
            if i in idx: idx.remove(i)
        node2C[i] = idx[:K]
        if not(ignore_matrix): AdjM[i,idx]=1
    return AdjM,node2C

def node2C_to_adjacency_matrix(node2C, M=None):
    r"""
    From node2C to Adjacency Matrix

    """
    assert isinstance(node2C, dict)
    for node in node2C:
        assert isinstance(node,(int,float)) and node>-1

    list_nodes = np.array(list(node2C.keys())).astype(int)
    if M is not None:
        assert M>=np.max(list_nodes)
    else:
         M=np.max(list_nodes)

    AdjM = np.zeros([M,M]).astype(int)
    for i in range(M):
        if i in node2C:
            clist = list(node2C[i])
            AdjM[i,clist]=1
    return AdjM

#========================================
## Surface Reconstruction  - 3D to 2D
#======================================

#from scipy.spatial import ConvexHull

class TriAng(object):
    def __init__(self,F,V=None):
        self.V = V
        #self.F = F
        self.F = np.array([np.sort(f).tolist() for f in F])
        self.F0 = self.F.copy() #copy of original F

        #remove any triangle with two same egdes
        self.F = np.array([f.tolist() for f in self.F if len(set(f))==len(f)])

        #np.where((self.F[:,0]-self.F[:,1])==0 | (self.F[:,2]-self.F[:,1])==0 | (self.F[:,0]-self.F[:,2])==0)

    def getEdges_idx(self,update_edges=True):
        E = []
        for s in self.F:
            e1 = [s[0],s[1]]
            e2 = [s[1],s[2]]
            e3 = [s[0],s[2]]
            e1.sort()
            e2.sort()
            e3.sort()

            E.append(e1)
            E.append(e2)
            E.append(e3)

        E = np.array(E)
        E = np.unique(E,axis=0)
        E = [set(e) for e in E]
        if update_edges: self.E = E
        return E

    def get_Tri_with_edge(self,e):
        e = list(e)
        Fi = self.F[np.where(self.F==e[0])[0],:]
        return Fi[np.where(Fi==e[1])[0]]

    def removeTri_with_edge(self,e):
        e = list(e)
        self.F = np.array([f for f in self.F if np.sum(f==e[0])+np.sum(f==e[1])<2])

    def removeTri_with_edges(self,E,update_F=True):
        Fr = []
        for f in self.F:
            f.sort()
            not_include = True
            for e in E:
                e = list(e)
                e.sort()
                if (np.sum(f==e[0]) + np.sum(f==e[1]))==2:
                    not_include=False
                    break
            if not_include:
                Fr.append(f)
        Fr = np.array(Fr)
        if update_F: self.F = Fr
        return Fr

    def getBorder_edges(self,E=None):
        if E is None:
            E = self.getEdges_idx(update_edges=False)

        bordr_idx = np.where(np.array([len(np.where(self.F[np.where(self.F==ex)[0],:]==ey)[0]) for ex,ey in E])==1)[0]
        E_b = [E[k] for k in bordr_idx]

        return E_b

    def get_edges_deg(self,deg=1):
        'get edges with given degree '
        E = self.getEdges_idx(update_edges=False)
        idx = np.where(np.array([len(np.where(self.F[np.where(self.F==ex)[0],:]==ey)[0]) for ex,ey in E])==deg)[0]
        Ed = [E[k] for k in idx]
        return Ed

    def indx2points_edge(self,xy,e):
        i,j = e
        return set([tuple(xy[i]),tuple(xy[j])])

    def selEdges(self,xy=None,E=None,thr=[-np.pi/4,np.pi/4],axis=0,verbose=0):
        if xy is None:
            xy = self.V.copy()
        if E is None:
            E = self.getEdges_idx(update_edges=False)
        idx = np.where((xy[:,axis]<thr[1]) & (xy[:,axis]>thr[0]))[0]

        if verbose:
            print(len(E), len(idx))
        E_sel = []
        for i,j in E:
            if i in idx and j in idx:
                E_sel.append(set([i,j]))
        E_xy = [self.indx2points_edge(xy,e) for e in E_sel]
        return E_sel, E_xy
    def isEdge_inTri(self,e,T):
        e = list(e)
        for f in T:
            if (np.sum(f==e[0])+np.sum(f==e[1]))==2:
                return True
        return False



def unwrape_3d(V, plot=True,npoint_rotate=10,sep=None,verbose=0,zshift=0):
    V1 = V - V.mean(axis=0) + zshift
    rad = np.linalg.norm(V1, axis=1)
    zen = np.arccos(V1[:,-1] / rad)
    azi = np.arctan2(V1[:,1], V1[:,0])

    idx0 = np.argsort(azi)

    n = npoint_rotate
    if n>0:
        idx1 = np.r_[idx0,idx0[:n]]
    else:
        idx1 = idx0.copy()

    if sep is None:
        #diff  = np.abs(np.diff(azi[idx0]))
        #sep = np.min(diff[diff>0])
        sep = np.pi - np.max(azi)
        if verbose:
            print('min_diff',sep)
    if n>0:
        azi1 = np.r_[azi[idx0],np.max(azi)+sep+(azi[idx0[:n]]-np.min(azi[idx0[:n]]))]
        zen1 = np.r_[zen[idx0],zen[idx0[:n]]]
        rad1 = np.r_[rad[idx0],rad[idx0[:n]]]
    else:
        azi1 = azi[idx0]
        zen1 = zen[idx0]
        rad1 = rad[idx0]
    tris = Triangulation(zen1, azi1)
    triangles = tris.triangles
    triangles = np.unique(triangles,axis=0)
    triangles = np.array([np.sort(fi).tolist() for fi in triangles.tolist()])
    triangles = np.unique(triangles,axis=0)
    if plot:
        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d')
        ax.plot_trisurf(azi1, zen1, 0*rad1, triangles=triangles)
        plt.show()
    return np.c_[azi1,zen1,rad1],triangles

def unwrape_surface(V,F=None,margin_x=np.pi/6, margin_y=np.pi/6,n=9,origin=[0,0,0]):
    V1 = V - V.mean(axis=0) + np.array(origin)
    rad = np.linalg.norm(V1, axis=1)
    zen = np.arccos(V1[:,-1] / rad)
    azi = np.arctan2(V1[:,1], V1[:,0])

    azi1  = azi-np.min(azi)
    zen1  = zen-np.min(zen)
    rad1  = rad

    idx1 = np.arange(len(azi1))

    Vx0 = np.c_[azi1        ,zen1]
    Vx1 = np.c_[2*np.pi+azi1,zen1]


    Vx2 = np.c_[2*np.pi-azi1,2*np.pi-zen1]
    Vx21 = np.c_[2*np.pi+2*np.pi-azi1,2*np.pi-zen1]


    #Vx5 = np.c_[azi1,2*np.pi+zen1]
    #Vx6 = np.c_[2*np.pi+azi1,2*np.pi+zen1]

    Vx3 = np.c_[azi1-2*np.pi,zen1]

    Vx23 = np.c_[-azi1,2*np.pi-zen1]


    Vx4 = np.c_[2*np.pi-azi1,-zen1]

    Vx34 = np.c_[-azi1,-zen1]

    Vx14 = np.c_[4*np.pi-azi1,-zen1]


    Vx = [Vx0,Vx1,Vx2,Vx3,Vx4,Vx21,Vx23,Vx34,Vx14] #,Vx34,Vx14

    azi2 = np.vstack(Vx[:n])[:,0]
    zen2 = np.vstack(Vx[:n])[:,1]
    rad2 = np.hstack([rad1 for _ in range(n)])

    idx2 = np.hstack([idx1 for _ in range(n)])
    #print(azi1.shape, zen1.shape,idx1.shape)

    idx = np.where( (azi2<=2*np.pi+margin_x) & (azi2>=-margin_x) & (zen2<=np.pi+margin_y) & (zen2>=-margin_y))[0]

    azi3 = azi2[idx]
    zen3 = zen2[idx]
    rad3 = rad2[idx]
    idx3 = idx2[idx]

    return np.c_[azi3,zen3,rad3],idx3

def unwrape_surface_mirror(V,margin_x=np.pi/6, margin_y=np.pi/6,n=9,origin=[0,0,0]):
    V1 = V - V.mean(axis=0) + np.array(origin)
    rad = np.linalg.norm(V1, axis=1)
    zen = np.arccos(V1[:,-1] / rad)
    azi = np.arctan2(V1[:,1], V1[:,0])

    azi1  = azi-np.min(azi)
    zen1  = zen-np.min(zen)
    rad1  = rad

    idx1 = np.arange(len(azi1))

    #----------------------------
    #   V00      V01      V02
    #   V10      V11      V12
    #   V20      V21      V22
    #----------------------------

    PI2 = 2*np.pi
    PI4 = 4*np.pi

    V10 = np.c_[PI2-azi1-PI2,zen1]
    V11 = np.c_[azi1        ,zen1]
    V12 = np.c_[PI4-azi1,zen1]

    V00 = np.c_[-azi1,PI2-zen1]
    #V01 = np.c_[PI2-(PI2-azi1),PI2-zen1]
    V01 = np.c_[azi1,PI2-zen1]
    V02 = np.c_[PI2+PI2-azi1,PI2-zen1]

    V20 = np.c_[-azi1,-zen1]
    V21 = np.c_[azi1,-zen1]
    V22 = np.c_[PI4-azi1,-zen1]

    Vx = [V00,V01,V02,V10,V11,V12,V20,V21,V22] #,Vx34,Vx14

    azi2 = np.vstack(Vx[:n])[:,0]
    zen2 = np.vstack(Vx[:n])[:,1]

    rad2 = np.hstack([rad1 for _ in range(n)])
    idx2 = np.hstack([idx1 for _ in range(n)])

    idx = np.where( (azi2<=2*np.pi+margin_x) & (azi2>=-margin_x) & (zen2<=np.pi+margin_y) & (zen2>=-margin_y))[0]

    azi3 = azi2[idx]
    zen3 = zen2[idx]
    rad3 = rad2[idx]
    idx3 = idx2[idx]

    return np.c_[azi3,zen3,rad3],idx3

def get_plane(X,lamd=0,n=0):
    r"""

    Get Plane passing through (3D) points X
    Using Regularised Least Square,
    lamd: regulariser

    input
    ------
    X: 3D points (m,3)

    output
    ------
    W: coefficient
    P: grid of (n,3) on plane, if n>0
    """

    XZ = np.c_[np.ones(X.shape[0]),X[:,0],X[:,2]]
    Y = X[:,1]
    #X1.shape, Y1.shape
    #w = np.linalg.pinv(XZ)@Y1
    W = np.linalg.inv(XZ.T@XZ + lamd*np.eye(3))@XZ.T@Y

    xi,yi,zi=0,0,0
    if n>0:
        xi = np.linspace(X[:,0].min(),X[:,0].max(),n)
        zi = np.linspace(X[:,2].min(),X[:,2].max(),n)
        xi,zi = np.meshgrid(xi,zi)
        xi = xi.reshape(-1)
        zi = zi.reshape(-1)
        #xi.shape, yi.shape
        yi = np.c_[np.ones(xi.shape[0]),xi,zi]@W
    return W,np.c_[xi,yi,zi]

def divide_space(X,D,V=None,lamda=0,plot=0,n=30):
    r"""

    Divide space point of X, by plane passing from D

    Essentially, we are dividing points of X by divider D
    using Least Square Plane, than passes through D
    """

    W,_ = get_plane(D,lamd=lamda,n=0)
    Xi =  np.c_[np.ones(X.shape[0]),X[:,0],X[:,2]]@W
    L = (Xi-X[:,1])>=0
    R = (Xi-X[:,1])<0
    if plot:
        if V is not None:
            mn_ = np.r_[X,D,V].min(0)
            mx_ = np.r_[X,D,V].max(0)
        else:
            mn_ = np.r_[X,D].min(0)
            mx_ = np.r_[X,D].max(0)

        xi = np.linspace(mn_[0],mx_[0],n)
        zi = np.linspace(mn_[2],mx_[2],n)
        xi,zi = np.meshgrid(xi,zi)
        xi = xi.reshape(-1)
        zi = zi.reshape(-1)
        yi = np.c_[np.ones(xi.shape[0]),xi,zi]@W

        fig = plt.figure(figsize = [6,5])
        ax = fig.add_subplot(111,projection="3d")
        ax.plot(X[L,0],X[L,1], X[L,2],'.C1',alpha=0.5)
        ax.plot(X[R,0],X[R,1], X[R,2],'.C2',alpha=0.5)

        #ax1.plot(Vv[:,0],Vv[:,1],Vv[:,2],'.',alpha=0.5)
        ax.plot(D[:,0],D[:,1],D[:,2],'.C3',alpha=0.5)
        ax.plot(xi,yi,zi,'.C4',alpha=0.5)
        ax.axis('off')
        if V is not None: ax.plot(V[:,0],V[:,1],V[:,2],'.C5',alpha=0.5)

    return L,R

def get_center(X):
    p1 = np.median(X,0)
    dist = np.sqrt(np.sum(np.abs(X - p1)**2,1))
    idx = np.argmin(dist)
    c = X[idx]
    return c, idx

def area_tri(p1,p2,p3):
    # p1 = absX[0]
    # p2 = absX[1]
    # p3 = absX[2]
    a = np.linalg.norm(p1-p2)
    b = np.linalg.norm(p2-p3)
    c = np.linalg.norm(p1-p3)
    s = (a+b+c)/2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    return A

def surface_reconstruction(V,shift_mean=True,only_outer=False,return_all=False):
    V0 = V.copy()
    if shift_mean:
        V0 = V0 - V0.mean(0)

    if not(only_outer):
        S = car2spar(V0)
        S[:,0] = S[:,0]*0 + np.max(S[:,0])
        S[:,2] = S[:,2] + np.pi
        V0 = spar2car(S)

    cvx = ConvexHull(V0)
    #x, y, z = X.T
    # cvx.simplices contains an (nfacets, 3) array specifying the indices of
    # the vertices for each simplical facet
    tri = Triangulation(V0[:,0], V0[:,1], triangles=cvx.simplices)
    if return_all:
        S = car2spar(V0)
        return tri.triangles, V0, S
    return tri.triangles

def downsampling_space(X,n_point):
    r"""
    Downsampling space with KMedoids
    --------------------------------
    Getting Equidistance points
    """
    from sklearn_extra.cluster import KMedoids
    kmedoids = KMedoids(n_clusters=n_point,method='alternate',init='k-medoids++').fit(X)
    X_new = kmedoids.cluster_centers_
    return X_new

def downsampling_surface_points(V,n_points=100,remove_long_edges=True,k=1.5,verbose=True):
    V1 = downsampling_space(V, n_point=n_points)
    Idx1 = []
    for i in range(len(V1)):
        ix = np.argmin(np.linalg.norm(V - V1[i],axis=1))
        Idx1.append(ix)
    Idx1 = np.array(Idx1)
    if verbose: print(Idx1.shape)

    F1 = surface_reconstruction(V1-V1.mean(0))
    if verbose: print(F1.shape)

    if remove_long_edges:
        EDis = [np.r_[np.linalg.norm(vp[0]-vp[1]),np.linalg.norm(vp[0]-vp[2]),np.linalg.norm(vp[1]-vp[2])] for vp in V1[F1]]
        EDis = np.array(EDis)
        if verbose: print(EDis.shape)

        q1 = np.quantile(EDis.reshape(-1),0.25)
        q3 = np.quantile(EDis.reshape(-1),0.75)

        thr = q3-q1 + k*q3

        idxF = np.where(np.sum(EDis>thr,axis=1)==0)
        F1 = F1[idxF]
        if verbose: print(F1.shape)

    return V1, F1, Idx1

def remove_long_edges_k(V,F,k=1.5,verbose=False):
    EDis = [np.r_[np.linalg.norm(vp[0]-vp[1]),np.linalg.norm(vp[0]-vp[2]),np.linalg.norm(vp[1]-vp[2])] for vp in V[F]]
    EDis = np.array(EDis)
    if verbose: print(EDis.shape)
    q1 = np.quantile(EDis.reshape(-1),0.25)
    q3 = np.quantile(EDis.reshape(-1),0.75)
    thr = q3-q1 + k*q3
    idxF = np.where(np.sum(EDis>thr,axis=1)==0)
    F1 = F[idxF]
    if verbose: print(F.shape)
    return F1


def surface_plot_mayavi(V,F,X, D=None,colormap_value='cmap',value_range=[None,None],show_points=False,N=1,scale_factor=0.35,scale_mode='scalar',scalars_mag=6,
              colormap_arr='jet',arr_range=[0,1], tip_length=0.5,tip_radius=0.2,shaft_radius=0.04,color_mode ='color_by_scalar',fign=1,show=True,f=0.98,f1=0.95):

    try:
        from mayavi import mlab
    except:
        raise ImportError("Install 'mayavi' to use this funtion | try 'pip install mayavi' ")

    mode = 'arrow'
    opacity=1
    line_width=1

    vmin_arr=arr_range[0]
    vmax_arr=arr_range[1]

    fig = mlab.figure(fign, bgcolor=(1, 1, 1), fgcolor=(1, 1, 1))

    mlab.triangular_mesh(f1*V[:,0], f1*V[:,1], f1*V[:,2], F, scalars=V[:,0]*0,opacity=1,colormap='gray',vmin=-10,vmax=1)

    Idx = np.where(np.isnan(X))[0]
    Fi = F.copy()
    Fi = np.array([fx.tolist() for fx in Fi if not((fx[0] in Idx)+(fx[1] in Idx)+(fx[2] in Idx))])

    mlab.triangular_mesh(f*V[:,0], f*V[:,1], f*V[:,2], Fi, scalars=X,opacity=1,colormap=colormap_value,vmin=value_range[0],vmax=value_range[1])

    if show_points:
        try:
            mlab.plot3d(f*V[:,0], f*V[:,1], f*V[:,2],f*V[:,2]*0+scalars_mag,representation='points')
        except:
            print('mlab.plot3d Not working, turn off: show_points')

    if D is not None:
        U = D[:,:3] - D[:,3:6]
        Xi = D[:,3:6]
        scalars = car2spar(U)[::N,2]

        if scalars_mag>0: scalars = scalars*0+scalars_mag

        obj = mlab.quiver3d(Xi[::N,0],Xi[::N,1],Xi[::N,2], U[::N,0], U[::N,1], U[::N,2], line_width=line_width,
                            scale_factor=scale_factor,scale_mode=scale_mode,mode=mode,opacity=opacity,scalars=scalars,
                            colormap=colormap_arr,vmax=vmax_arr,vmin=vmin_arr)

        obj.glyph.glyph_source.glyph_source.tip_length = tip_length
        obj.glyph.glyph_source.glyph_source.tip_radius = tip_radius
        obj.glyph.glyph_source.glyph_source.shaft_radius = shaft_radius
        obj.glyph.color_mode = color_mode
    if show: mlab.show()



# def getTri(az,ze,idx,remove_dup=True,verbose=0,):
#     tri = Delaunay(np.c_[az,ze])
#     F = tri.simplices
#     Ft = idx[F]
#
#     Ft = np.array([np.sort(f).tolist() for f in Ft])
#     if remove_dup:
#         if verbose: print(Ft.shape)
#         Ft = np.unique(Ft,axis=0)
#         if verbose: print(Ft.shape)
#     return F,Ft,tri
#
# def tri_surface_reconstruction(V,plot=True,verbose=0,remove_dup=True):
#     V1 = V - V.mean(axis=0)
#     rad = np.linalg.norm(V1, axis=1)
#     zen = np.arccos(V1[:,-1] / rad)
#     azi = np.arctan2(V1[:,1], V1[:,0])
#
#     azi1  = azi-np.min(azi)
#     zen1  = zen-np.min(zen)
#     rad1  = rad
#     #idx1 = np.argsort(azi)
#
#     if verbose>2:
#         print('min_max_azi',min(azi1),max(azi1))
#         print('min_max_zen',min(zen1),max(zen1))
#
#     idx1 = np.arange(len(azi1))
#
#
#     #azi1 = np.r_[azi1, 2*np.pi+azi1, azi1,         2*np.pi+azi1, azi1,         2*np.pi+azi1]
#     #zen1 = np.r_[zen1, zen1,         2*np.pi-zen1, 2*np.pi-zen1, 2*np.pi+zen1, 2*np.pi+zen1]
#     #rad1 = np.r_[rad1, rad1,         rad1,         rad1,         rad1,         rad1,]
#
#
#     Vx1 = np.c_[azi1        ,zen1]
#     Vx2 = np.c_[2*np.pi+azi1,zen1]
#     Vx3 = np.c_[2*np.pi-azi1,2*np.pi-zen1]
#     Vx4 = np.c_[2*np.pi+2*np.pi-azi1,2*np.pi-zen1]
#
#     Vx5 = np.c_[azi1,2*np.pi+zen1]
#     Vx6 = np.c_[2*np.pi+azi1,2*np.pi+zen1]
#
#     azi1 = np.vstack([Vx1,Vx2,Vx3,Vx4,Vx5,Vx6])[:,0]
#     zen1 = np.vstack([Vx1,Vx2,Vx3,Vx4,Vx5,Vx6])[:,1]
#     rad1 = np.r_[rad1,rad1,rad1,rad1,rad1,rad1]
#
#     idx1 = np.r_[idx1,idx1,idx1,idx1,idx1, idx1]
#     print(azi1.shape, zen1.shape,idx1.shape)
#
#     F1,F1t,tri1 =  getTri(azi1,zen1,idx1,remove_dup=remove_dup,verbose=verbose)
#
#     if plot:
#         fig = plt.figure(figsize = [6,5])
#         ax  = fig.add_subplot(projection='3d')
#         ax.plot(V[:,0], V[:,1], V[:,2],'.')
#         #mesh = Poly3DCollection(V2[F1t], linewidths=0.3, alpha=0.3,color='C0')
#         mesh = Poly3DCollection(V[F1t], linewidths=0.3, alpha=0.3,color='C0')
#         mesh.set_edgecolor('k')
#         ax.add_collection3d(mesh)
#         plt.show()
#         #plt.stop()
#
#
#         fig = plt.figure(figsize = [8,6])
#         ax = fig.add_subplot()
#         ax.plot(azi1,zen1,'.C0')
#         ax.triplot(azi1,zen1, triangles=F1,color='C0',lw=0.5)
#         ax.vlines([0,2*np.pi, 4*np.pi],min(zen1),max(zen1),color='k',ls='--')
#         ax.hlines([0,np.pi,2*np.pi, 4*np.pi],min(azi1),max(azi1),color='k',ls='--')
#         ax.set_xlim(min(azi1),max(azi1))
#         ax.set_ylim(min(zen1),max(zen1))
#         ax.grid()
#         plt.tight_layout()
#         plt.show()
#
#
#     return (azi1,zen1,rad1,idx1,F1,F1t,tri1)
#
# def Filter_Faces(F,azi,zen,idx,thr_x=[np.pi/4,2*np.pi+np.pi/4], thr_y=[0,np.pi+np.pi/4],method=1,plot=True,verbose=0):
#     Vxy = np.array([azi,zen]).T
#     FT = TriAng(F=F,V = Vxy)
#     if verbose: print(F.shape, FT.F.shape)
#
#     E1x,_ = FT.selEdges(thr=thr_x,axis=0,verbose=0)
#     E1y,_ = FT.selEdges(thr=thr_y,axis=1)
#     E1xy = [e for e in E1x if e in E1y]
#     if verbose: print(len(E1x), len(E1y), len(E1xy))
#
#     if plot:
#         fig = plt.figure(figsize = [8,5])
#         ax = fig.add_subplot()
#         ax.plot(azi,zen,'.C0',alpha=0.2)
#         ax.triplot(azi,zen, triangles=F,color='C0',lw=0.2)
#
#         ax.vlines(thr_x,thr_y[0],thr_y[1],color='k',ls='--',lw=2)
#         ax.hlines(thr_y,thr_x[0],thr_x[1],color='k',ls='--',lw=2)
#
#
#         for i,j in E1xy:
#             xi= np.r_[azi[i],azi[j]]
#             yi= np.r_[zen[i],zen[j]]
#             ax.plot(xi,yi,'-',color='C3',lw=1)
#
#         #ax.set_xlim([2*np.pi-np.pi/6, 2*np.pi+np.pi/4 +np.pi/6])
#         #ax.grid()
#         plt.tight_layout()
#         plt.show()
#         #plt.close()
#
#
#     FinalF = []
#     if method==1:
#         if verbose:  print('all edges')
#         # all triangle includes to selected edge
#         for ex,ey in E1xy:
#             Fi = FT.F[np.where(FT.F==ex)[0]]
#             Fi = Fi[np.where(Fi==ey)[0]]
#             FinalF.append(Fi)
#     else:
#         # all triangle connected to selected nodes
#         if verbose:  print('all nodes')
#         allN = np.unique(np.array([list(e) for e in E1xy]).reshape(-1))
#         for node in allN:
#             FinalF.append(FT.F[np.where(FT.F==node)[0],:])
#
#     FinalF = np.vstack(FinalF)
#     FinalF = np.unique(FinalF,axis=0)
#     if verbose: print(FinalF.shape)
#
#     FinalFT = idx[FinalF]
#     if verbose: print(FinalF.shape, FinalFT.shape)
#
#     FinalFT = np.array([np.sort(f).tolist() for f in FinalFT])
#     FinalFT = np.unique(FinalFT,axis=0)
#     if verbose: print(FinalFT.shape)
#
#
#     if plot:
#         fig = plt.figure(figsize = [8,5])
#         ax = fig.add_subplot()
#         ax.plot(azi,zen,'.C0',alpha=0.1)
#         ax.triplot(azi,zen, triangles=F,color='C0',lw=0.1)
#         ax.triplot(azi,zen, triangles=FinalF,color='C0',lw=0.6)
#
#         #ax.vlines([0,2*np.pi, 4*np.pi],min(zen1),max(zen1),color='k',ls='--')
#         #ax.hlines([0,np.pi,2*np.pi, 4*np.pi],min(azi1),max(azi1),color='k',ls='--')
#
#         ax.vlines(thr_x,thr_y[0],thr_y[1],color='k',ls='--',lw=2)
#         ax.hlines(thr_y,thr_x[0],thr_x[1],color='k',ls='--',lw=2)
#
#         ax.set_xlim(min(azi),max(azi))
#         ax.set_ylim(min(zen),max(zen))
#
#
#
#         plt.tight_layout()
#         plt.show()
#     return FinalFT
#
# def SurfaceReco(V,verbose = 0, plots=False,return_FF=False):
#     FF = tri_surface_reconstruction(V, plot=plots,verbose=verbose,remove_dup=True)
#     (azi1,zen1,rad1,idx1,F1,F1t,tri1) = FF
#
#     FT1 = Filter_Faces(F1,azi1,zen1,idx1,thr_x=[np.pi/2,2*np.pi+np.pi/2], thr_y=[np.pi/4,2*np.pi+np.pi/4],method=2
#                        ,plot=plots,verbose=verbose)
#
#
#     FTx = TriAng(F=FT1,V=V)
#     if verbose>2: print(FTx.F.shape, FTx.V.shape, FT1.shape)
#     if verbose: print('Number be edges with degree')
#     for deg in [0,1,2,3,4,5]:
#         Edx = FTx.get_edges_deg(deg=deg)
#         if verbose: print(f' - deg {deg}\t # {len(Edx)}')
#
#     if return_FF:
#         return FTx.F, FTx, FF
#     return FTx.F, FTx
#
# def getRotM_xyz(theta=np.pi/4):
#     Mx = np.array([[1,              0,           0],
#                    [0, np.cos(theta),-np.sin(theta)],
#                   [0, np.sin(theta), np.cos(theta)]])
#
#     My = np.array([[np.cos(theta),0, np.sin(theta)],
#                    [0,            1,           0],
#                    [-np.sin(theta),0, np.cos(theta)]])
#
#     Mz = np.array([[np.cos(theta),-np.sin(theta),0],
#                   [np.sin(theta), np.cos(theta),0],
#                   [0,              0,           1]])
#     return Mx,My,Mz
# def getNeibours(V,p,N=None):
#     dist = np.sum((V-p)**2,1)
#     idx = np.argsort(dist)
#     if N is None:
#         return idx
#     return idx[:N]



# def getSPlane(S,lamd=0,n=0):
#     XZ = np.c_[np.ones(S.shape[0]),S[:,0],S[:,2]]
#     Y = S[:,1]
#     #X1.shape, Y1.shape
#     #w = np.linalg.pinv(XZ)@Y1
#     w = np.linalg.inv(XZ.T@XZ + lamd*np.eye(3))@XZ.T@Y
#
#     xi,yi,zi=0,0,0
#     if n>0:
#         xi = np.linspace(S[:,0].min(),S[:,0].max(),n)
#         zi = np.linspace(S[:,2].min(),S[:,2].max(),n)
#         xi,zi = np.meshgrid(xi,zi)
#         xi = xi.reshape(-1)
#         zi = zi.reshape(-1)
#         #xi.shape, yi.shape
#         yi = np.c_[np.ones(xi.shape[0]),xi,zi]@w
#     return w,np.c_[xi,yi,zi]
#
# def dividAtria(A,S,V=None,lamda=0,plot=0,n=30):
#     w,_ = getSPlane(S,lamd=lamda,n=0)
#     Ai =  np.c_[np.ones(A.shape[0]),A[:,0],A[:,2]]@w
#     la = (Ai-A[:,1])>=0
#     ra = (Ai-A[:,1])<0
#     if plot:
#         if V is not None:
#             mn_ = np.r_[A,S,V].min(0)
#             mx_ = np.r_[A,S,V].max(0)
#         else:
#             mn_ = np.r_[A,S].min(0)
#             mx_ = np.r_[A,S].max(0)
#
#         xi = np.linspace(mn_[0],mx_[0],n)
#         zi = np.linspace(mn_[2],mx_[2],n)
#         xi,zi = np.meshgrid(xi,zi)
#         xi = xi.reshape(-1)
#         zi = zi.reshape(-1)
#         yi = np.c_[np.ones(xi.shape[0]),xi,zi]@w
#
#         fig = plt.figure(figsize = [6,5])
#         ax = fig.add_subplot(111,projection="3d")
#         ax.plot(A[la,0],A[la,1], A[la,2],'.C1',alpha=0.5)
#         ax.plot(A[ra,0],A[ra,1], A[ra,2],'.C2',alpha=0.5)
#
#         #ax1.plot(Vv[:,0],Vv[:,1],Vv[:,2],'.',alpha=0.5)
#         ax.plot(S[:,0],S[:,1],S[:,2],'.C3',alpha=0.5)
#         ax.plot(xi,yi,zi,'.C4',alpha=0.5)
#         ax.axis('off')
#         if V is not None: ax.plot(V[:,0],V[:,1],V[:,2],'.C5',alpha=0.5)
#
#     return la,ra
