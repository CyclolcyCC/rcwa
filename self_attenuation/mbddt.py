#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:48:14 2020

@author: nikola03
"""

import numpy as np
import types
import matplotlib.pyplot as plt
from copy import copy
from itertools import product
from scipy import linalg
import shapely.geometry as sg
from descartes.patch import PolygonPatch

def default_attenuator(*args):
    return 1,0

def U(q,a,b):
    #TODO: make part of module
    if q == 0:
        return b-a
    else:
        return 1j/q*(np.exp(-1j*q*b)- np.exp(-1j*q*a))

def Z(q,t):
    return U(q,-t,0)
    
def calc_conical(K,chi0,nodes):
        kap = ( K[0]**2 + K[1]**2 + K[2]**2 ) * (1+chi0)
        kp_x, kp_y = K[0], K[1]
        qz2 = []
        n_nodes = len(nodes)
        nodes = nodes[n_nodes//4:3*n_nodes//4+1]
        for hx,hy in nodes:
            qz2_ = kap - ( (kp_x+hx)**2+(kp_y+hy)**2 )
            qz2.append( qz2_ )

        return qz2    

def get_toeplitz(chi_hat : dict):
    VL = list(chi_hat.values())[len(chi_hat)//2:]
    HU = list(chi_hat.values())[:len(chi_hat)//2+1]
    HU = np.flipud(HU)
    return linalg.toeplitz(VL,HU)

def angle2wavevect(ai,phi,wl):
    #TODO: make part of module or Symmetry class
    return 2 * np.pi/wl * np.array( [ np.cos(ai)*np.sin(phi) ,np.cos(ai)*np.cos(phi) ,-np.sin(ai) ] )

class Beam():
    
    def __init__(self,ai,phi,wl):
        K = kx,ky,kz = angle2wavevect(ai,phi,wl)
        self.K = K
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.wl = wl
        self.alpha = ai
        self.phi = phi

class Symmetry():
#        #TODO: fix for 2D vs 3D
    def __init__(self, 
                 period_x : float, 
                 period_y : float, 
                 n_modes_x : int, 
                 n_modes_y : int):
        
        self.cell = ( period_x, period_y ) 
        self.n_modes_x = 2*n_modes_x
        self.n_modes_y = 2*n_modes_y
        self.calc_nodes()
        return 
    
    def calc_nodes(self):
        nodes = []
        Mx = self.n_modes_x
        My = self.n_modes_y
        Dx, Dy = self.cell
        for mx,my in product( range(-Mx,Mx+1), range(-My,My+1)):
            hx, hy = 2*np.pi*mx/Dx, 2*np.pi*my/Dy
            nodes.append((hx,hy))
        self.nodes = nodes
    
        

class Symmetry2D( Symmetry ):
    def __init__(self, 
                 period : float, 
                 n_modes : int):
        
        self.cell = ( period, period ) 
        self.n_modes_x = 2*n_modes
        self.n_modes_y = 0
        self.calc_nodes()
        return 

class Symmetry3D( Symmetry ):
    pass

class Material():
    """
    Class to track which material is fluorescent 
    """
    # Nominee for delition?
    # To be used for self attenuation track
    
    colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    
    def __init__(self, tag = "", color = None, isfluorescent = False, alpha = 0.5):
        self._isfluorescent = isfluorescent
        self._tag = tag
        if not color: 
            self.color = next(self.colors)
        else:
            self.color = color
        self.alpha = alpha

    def set_chi( self, chi ):
        if isinstance( chi, types.FunctionType ):
            self.chi = chi
        elif np.isscalar(chi): # TODO: FIX 
            #Make it callable anyway
            self.chi = lambda _ : chi
        else:
            raise TypeError("chi must be function or scalar")
        return self
    
#    def set_attr( self, attr):
#        if isinstance( attr, types.FunctionType ):
#            self.attr = attr
#        elif np.isscalar(attr): # TODO: FIX 
#            #Make it callable anyway
#            self.attr = lambda _ : attr
#        else:
#            raise TypeError("attr must be function or scalar")
    
#TODO sort private from public fields
class Layer():
    """
    Class container of the properties of a single layer
    (including vacuum and substrate)
    """
    def __init__(self, sym: Symmetry, thickness: float, tag = ''):
        self._sym = sym
        self._thickness = thickness
        self._materials = set()
        self._up = None
        self._dwn = None
        self.tag = tag
        self.boxes = {}
        
        # Deocrator for 2D symmetry
        if isinstance(sym, Symmetry2D):
            def add_2D_box(left : float, 
                        right : float,
                        material : Material,
                        **kwags):
                Dx,Dy = self._sym.cell
                bck = -Dy/2
                fwd = +Dy/2
                return self.__add_box(left, right, bck, fwd, material, **kwags)                                  
            self.add_box = add_2D_box
        else:
            self.add_box = self.__add_box
        
        return
    
    #TODO: decorator for 3D symmetry
    #TODO: Check integration order
    def __add_box(self, 
                left : float, 
                right : float,
                bck : float,
                fwd : float,
                material : Material,
                attenuator = default_attenuator,
                sigma = 0,
                ):
        
        self.boxes.setdefault( material, [] ).append( (left, right, bck, fwd, sigma, attenuator) )
        self._materials.add( material )
        return self
    
    def get_image(self, wl : float):
        """
        Get Fourier image { node : Fourier component }
        """
        Dx, Dy  = self._sym.cell
        nodes = self._sym.nodes
        chi_hat = np.zeros(len(nodes), dtype = complex)
        
        #TODO: We here consider 2D symmetry
                       
        for material in self._materials:
            chi_a = material.chi(wl)
            for box in self.boxes.get(material, []):
                ll,rr,bb,ff, sigma, _ = box
                for i,(hx,hy) in enumerate(nodes):
                    #TODO: We here consider 2D symmetry
                    chi_hat[i] +=  chi_a/Dx/Dy  *  U(hx, ll, rr)  *  U(hy, bb, ff) * np.exp(-(hx**2+hy**2) * sigma**2/2 )
        return dict(zip(nodes,chi_hat))
    
    def freq_recons(self, 
                    wl : float, 
                    X, 
                    Y = [0,]):    
        # TODO type set array of floats
        # TODO: Test on 3D structure
        chi_hat = self.get_image(wl)
        chi_r = np.zeros( (len(X), len(Y)), dtype = complex )
        for i, xi in enumerate(X):
            for j, yj in enumerate(Y):
                r = (xi,0)
                for h, chi_h in chi_hat.items():
                    chi_r[i] += chi_h * np.exp(1j*np.dot(h,r))
        
        return np.squeeze(chi_r)

    def calc_dispersion(self, beam : Beam):
        K = beam.K
        wl = beam.wl
        
        chi_hat = self.get_image(wl)
        chi0 = chi_hat[(0,0)]
        nodes = self._sym.nodes
                
        qz2 = calc_conical(K, chi0, nodes)
        A = (2*np.pi/wl)**2 * get_toeplitz(chi_hat)
        np.fill_diagonal(A, qz2)
        
        return linalg.eig(A)

    def get_PQ(self, beam : Beam):
        
        # TODO: optimize 
        t = self._thickness
        k2, E = self.calc_dispersion(beam)
        
        O = np.zeros(E.shape)
        I = np.eye(E.shape[0])
        
        # Refraction matrix
        kgz = np.sqrt(k2, dtype = complex)
        k_hat = np.diag(kgz)
        PL = np.block([[E,O],[O,E]])
        PR = np.block( [[I,I],[k_hat,-k_hat]] )
        P = np.dot(PL,PR)
        
        # Transmission matrix
        qp = np.diag(np.exp(-1j*np.sqrt(k2)*t))
        qm = np.diag(np.exp(1j*np.sqrt(k2)*t))
        Q = np.block([[qp,O],[O,qm]])
    
        return P, Q, E, kgz 

def split(A):
    M2 = A.shape[0]
    M = M2//2
    A11 = A[:M,:M]
    A12 = A[:M,M:]
    A21 = A[M:,:M]
    A22 = A[M:,M:]
    return A11, A12, A21, A22
    
def calc_M(Pa,Pb,Q):
    X = linalg.inv(Pa) @ Pb
    X11, X12, X21, X22 = split(X)
    Qm = split(Q)[3]
    
    X11_ = linalg.inv(X11)
    M11 = Qm @ X11_
    M12 = -M11 @ X12 @ Qm
    M21 = X21 @ X11_
    M22 = (X22 - M21 @ X12) @ Qm
   
    return  np.block([[M11, M12],[M21, M22]])    

def calc_Wb(Wa,M):
    Wa11, Wa12, Wa21, Wa22 = split(Wa)
    M11, M12, M21, M22 = split(M)
    
    I = np.eye(M11.shape[0])
    
    A = M11 @ linalg.inv( I - Wa12 @ M21  )
    B = Wa22 @ linalg.inv( I - M21 @ Wa12 )
    
    Wb11 = A @ Wa11
    Wb12 = M12 + A @ Wa12 @ M22
    Wb21 = Wa21 + B @ M21 @ Wa11
    Wb22 = B @ M22
    
    return np.block([[Wb11, Wb12],[Wb21, Wb22]])

def calc_RT(M,W,R,T0):
    M11, M12, M21, M22 = split(M)
    W11, W12, W21, W22 = split(W)
    I = np.eye(M11.shape[0])
    Rn = linalg.inv(I - M21 @ W12)  @  (M22 @ R + M21 @ W11 @ T0)
    Tn = W11 @ T0 + W12 @ Rn 
    
    return Rn, Tn

def calc_EHZ( Tg, Rgm, kgz, E, Z, depth):
    TG = []
    RGm = []
    for zi in Z:
        TG.append( Tg * np.exp(-1j*kgz*zi) )
        RGm.append( Rgm * np.exp( +1j*kgz*(zi+depth) ) )
    TG = np.array(TG)
    RGm = np.array(RGm)
    Ehz = (TG+RGm) @ np.transpose(E)    
    return Ehz

class Model():
    """
    Container and iterator for layers
    """
    def __init__(self, sym: Symmetry, attenuation = False):
        self._sym = sym
        self._stack = []
        self._dspace = [0]
        self._coursor = 0 
        self._materials = set()
        self._attenuation = attenuation
        
    def __iter__(self):
        return iter(self._stack)
    
    def add_layer(self, layer: Layer ):
        if self._sym == layer._sym:
            layer = copy(layer)
            layer._up = self._coursor
            self._coursor -= layer._thickness
            layer._dwn = self._coursor
            self._stack.append( layer )
            self._dspace.append( layer._thickness )
            self._materials = self._materials.union( layer._materials )
        else:
            raise Exception("Can not add layer. Wrong symmetry")
        
        return self
    
    def calc_amp(self, beam : Beam):
        """
        main function - calculates amplitudes (see Fig 5 in ref)
        """
        #TODO: Might be a problem with vacuum layer
        #TODO: Create a container of amplitude full vectors
        stack = self._stack
        PQ = []
        for layer in stack:
            PQ.append( layer.get_PQ(beam) )

        # Calc W and M for recursive method            
        MM, WW = [], []
        MM.append( calc_M(PQ[0][0],PQ[1][0],PQ[1][1]) )
        WW.append(MM[-1])
        for i in range(1,len(PQ)-1):
            MM.append( calc_M(PQ[i][0],PQ[i+1][0],PQ[i+1][1]) )
            WW.append( calc_Wb(WW[-1],MM[-1]) )

        # Boundary conditions
        m = MM[0].shape[0]//2 # TODO: Make dim as a member of symmetry class
        T0 = np.zeros(m ,dtype = complex)
        T0[m//2] = 1
        Wn11 = split(WW[-1])[0]
        
        # Recursive method
        RR, TT = [], []
        RR.append( np.zeros(m ,dtype = complex) )
        TT.append( Wn11 @ T0 )
        for i in range(len(WW)-1,0,-1):
            Ri,Ti = calc_RT(MM[i],WW[i-1],RR[-1],T0)
            RR.append( Ri )
            TT.append( Ti )
        W11, W12, W21, W22 = split(WW[-1])

        RR.append( W21 @ T0 )
        TT.append( T0 )
        
        # First element --- surface
        RR, TT = RR[::-1], TT[::-1]
                
        # Full amplitude vectors array
        AA = np.block( [np.array( TT ),np.array( RR )] )
        AA_ = []
        AA_.append(  PQ[0][1] @ AA[0] )
        for i in range(len(PQ)-1):
            AA_.append( linalg.inv( PQ[i+1][0] ) @ PQ[i][0] @ AA[i] ) # TODO: CAREFULL! Might be wrong
        AA_ = np.array(AA_)   
        
        return AA, AA_, PQ
    
    def plot(self, ax):
       
        for material in self._materials:
            for layer in self:
                up = layer._up
                dwn = layer._dwn
                for box in layer.boxes.get(material, []):
                    ll,rr,_,_,_,_ = box
                    extent = [( ll, dwn ),
                              ( rr, dwn ),
                              ( rr, up  ),
                              ( ll, up  ),]
                    poly = sg.Polygon( extent )
                    patch = PolygonPatch(poly,
                                         color = material.color,
                                         linewidth = 0,
                                         alpha = material.alpha,
                                         )
                    ax.plot(*poly.exterior.xy, 
                            linewidth = 0.5,
                            color = "black",
                            zorder=10)            
                    ax.add_patch(patch)
      
    def calc_nf_components(self, z, beam : Beam):
        """
        Near field calvculator
        (implemented only for 2D structure)
        """
        AA, AA_, PQ = self.calc_amp(beam)
        m = AA_.shape[1]
        nf = np.empty( (len(z), m//2), dtype = complex )
        nf[:] = np.nan
        
        bounds = [(l._up,l._dwn) for l in self]
        depth = [l._thickness for l in self]                
        z_up = [l._up for l in self]
        
        for i,(up,dwn) in enumerate(bounds):
            idx = np.logical_and( z <= up, z >= dwn )
            Tg = AA_[i,:m//2]
            Rgm = AA[i,m//2:]
            P,Q,E,kgz = PQ[i]
            nf[idx] = calc_EHZ( Tg, Rgm, kgz, E, z[idx] - z_up[i], depth[i])
        return nf
    
    def calc_de(self, beam: Beam):
        TR = self.calc_amp(beam)[0][0]
        m = TR.size//2
        R = np.abs(TR[m:])**2
        return R

    def calc_nf(self, z, pad : int, beam: Beam ):
        nf = self.calc_nf_components(z,beam)
        padding = 2*pad+1
        f = np.zeros( (nf.shape[0], padding), dtype = complex)
        m,n = padding, nf.shape[1]
        f[:, m//2-n//2 : -m//2+n//2+1 ] = nf

        return np.fft.fftshift(np.fft.fft(f),1)
    
    def calc_yield(self, beam : Beam):
        #TODO: skip if no fluorescent materials
        
        AA, AA_, PQ = self.calc_amp(beam)
        M = AA_.shape[1]
        
        Dx, Dy = self._sym.cell
        Mx = self._sym.n_modes_x 
        My = self._sym.n_modes_y
        H = self._sym.nodes
        H = H[len(H)//4:3*len(H)//4+1]
        
        Y = dict()
        
        for material in self._materials:
            if material._isfluorescent:
                Y[material] = 0
                for idx, layer in enumerate(self):
                    t = layer._thickness
                    T = AA[idx,:M//2]
                    R = AA[idx,M//2:]
                    P, Q, E, kgz = PQ[idx]
                    for box in layer.boxes.get(material, []):
                        ll,rr,bb,ff, sigma, attenuator = box
                        c1,c2 = attenuator(beam)
                        PSI = np.zeros([len(H),len(H)], dtype = complex)
                        for h,g in product(range(len(H)),range(len(H))):
    
                            if not Mx == 0:            
                                phase_x = H[h][0] - np.conj(H[g][0]) 
                                psi_x = U(phase_x, ll, rr)
                            else:
                                psi_x = 1
    
                            if not My == 0:
                                phase_y = H[h][1] - np.conj(H[g][1])
                                psi_y = U(phase_y, bb, ff)
                            else: 
                                psi_y = 1
    
                            PSI[h,g] = psi_x * psi_y
    
                        X = np.zeros([len(H),len(H)], dtype = complex)
                        for m,n in product(range(len(H)),range(len(H))):
                            kzm,kzn = kgz[m], np.conj(kgz[n])
                            X[m,n] = T[m] * np.conj(T[n]) * Z( -kzm+kzn + 1j*c2,t)+ \
                                     T[m] * np.conj(R[n]) * Z( -kzm-kzn + 1j*c2,t)+ \
                                     R[m] * np.conj(T[n]) * Z( +kzm+kzn + 1j*c2,t)+ \
                                     R[m] * np.conj(R[n]) * Z( +kzm-kzn + 1j*c2,t)
                            X[m,n] *= c1
                        Y[material] += np.sum( PSI * ( E @ X @ np.conj(E.T) )).real
        return Y
    
if __name__ == "__main__":
    from tqdm import tqdm
    deg = np.pi/180
    pitch = 100 
    cd = 40
    height = 90
    offset = 80
    wl = 2.4
    
#%% Sample definition    
    SiN = Material( tag = "SiN", isfluorescent = True ).set_chi(-0.004+0.001j)
    Si = Material( tag = "Si" ).set_chi(-0.004+0.001j)
    OO = Material( tag = "vacuum", color = "gray", alpha = 0.1).set_chi(0.0)
       
#    sym = Symmetry2D( pitch, 90)
#    vac = Layer(sym,offset).add_box( -pitch/2,pitch/2,-pitch/2,pitch/2, OO )
#    ll1 = Layer(sym,height).add_box( -cd/2,cd/2,-pitch/2,pitch/2, SiN, )
#    sub = Layer(sym,offset).add_box( -pitch/2,pitch/2,-pitch/2,pitch/2, Si )

    sym = Symmetry2D( pitch, 90)
    vac = Layer(sym,offset).add_box( -pitch/2,pitch/2, OO )
    ll1 = Layer(sym,height).add_box( -cd/2,cd/2, SiN, )
    sub = Layer(sym,offset).add_box( -pitch/2,pitch/2, Si )

    sample = Model(sym, attenuation = True)
    sample.add_layer(vac)
    sample.add_layer(ll1)
    sample.add_layer(sub)

#%% Near field
    beam = Beam(2*deg,0.0*deg,wl)   
    z = np.linspace(0,-2*offset-height,500)
    E = sample.calc_nf(z, 1000, beam)
    
#    plt.figure("NF_new")
#    plt.clf()
#    plt.plot(-z,np.abs(nf))  
#%%
#    nfi =nf[10]

#%%    
    plt.figure(666)
    plt.imshow(np.abs(E)**2,
               extent = [-pitch/2, +pitch/2, np.min(z), np.max(z)],
               )
#    
#    
#    
##    m,n = len(f), len(nfi)
##    f[ m//2-n//2 : -m//2+n//2+1  ] = nfi
##    plt.figure("fsadfs")
##    E = np.fft.fft(f)
##    plt.plot( np.abs(E) )
#    
#    
#    
##
###%%
##    AI = np.linspace(0.5,6,300)
##    PHI = 0
##    Y1 = np.zeros( (AI.size,) )
##    Y2 = np.zeros( (AI.size,) )
##    pbar = tqdm(total = Y1.size)
##    for i, ai in enumerate(AI):
##        beam = Beam(ai*deg,PHI*deg,wl)    
##        mu = 5e-3
##        Y1[i] = sample.calc_yield( beam )[SiN]
##        mu = 0
##        Y2[i] = sample.calc_yield( beam )[SiN]
##        pbar.update()
##    pbar.close()    
##    
###%%
##    plt.figure("Fluor")
##    plt.clf()
##    plt.plot(AI,Y1)
##    plt.plot(AI,Y2)
##
###%% Visualization
##    plt.figure("structure")
##    plt.clf()
##    ax = plt.gca()
##    sample.plot(ax)
##    plt.axis("equal")
###
###
###
#####%% Near field
####    k0 = angle2wavevect(4.5*deg,0*deg,wl)   
####    z = np.linspace(0,-2*offset-height,200)
####    nf = sample.calc_nf(z, k0, wl)
####    plt.figure("NF_new")
####    plt.clf()
####    plt.plot(-z,np.abs(nf))
####    
#####%% 
####    AI = np.linspace(0.5,3,50)
####    PHI = np.linspace(-1.5,1.5,50)
####    Y = np.zeros( (AI.size,PHI.size) )
####    pbar = tqdm(total = Y.size)
####    for i, ai in enumerate(AI):
####        for j, phi in enumerate(PHI):
####            k0 = angle2wavevect(ai*deg,phi*deg,wl)    
####            Y[i,j] = sample.calc_yield( k0,wl )[SiN]
####            pbar.update()
####    pbar.close()
####
####    plt.figure("Yield")
####    plt.clf()
####    plt.imshow(Y,
####               origin = "lower",
####               extent = [min(PHI), max(PHI), min(AI), max(AI)],
####               )