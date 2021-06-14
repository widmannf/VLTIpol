import numpy as np
import math
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
from .polfunctions import *



degtorad = np.pi/180

class CalibVLTI(PolFunctions):
    """
    Class to simulate the propagation of polarization through the VLTI 
    Train, in order to calibrate polarization measurements
    
    plot: show the light path [False]
    
    Only important function is:
    M = muellerVLTI(Az, El)
    """
        
    def __init__(self, plot=False):
        self.mfirst = 3
        self.mlast = 18
        self.plot = plot
        self.setupVLTI()


    def setupVLTI(self):
        """
        Sets up the light path and the s & p vectors of each mirror
        """
        
        posfile = resource_filename('VLTIpol', 'Models/mirrorPos.txt')
        uvw = np.genfromtxt(posfile)
        u = uvw[:,0]
        v = uvw[:,1]
        w = uvw[:,2]

        polSin = np.zeros_like(uvw)*np.nan
        polSout = np.zeros_like(uvw)*np.nan
        polPin = np.zeros_like(uvw)*np.nan
        polPout = np.zeros_like(uvw)*np.nan
        normMirror = np.zeros_like(uvw)*np.nan
        incAngle = np.zeros(len(uvw))*np.nan
        
        for m in range(len(uvw)-1):
            lightIn = uvw[m,:]-uvw[m-1,:]
            lightIn = lightIn/np.linalg.norm(lightIn)
            lightOut = uvw[m+1,:]-uvw[m,:]
            lightOut = lightOut/np.linalg.norm(lightOut)
            polSin[m,:], polSout[m,:], polPin[m,:], polPout[m,:], normMirror[m,:], incAngle[m] = self.mirrorReflection(lightIn,lightOut) 

        if self.plot:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(u, v, w, color='k', ls='--', lw=0.5)
            ax.plot(u[self.mfirst-2: self.mlast+1], v[self.mfirst-2: self.mlast+1],
                    w[self.mfirst-2: self.mlast+1], color='b', lw=0.5)
            vecoff = 0.15
            veclen = 0.5
            
            first = True
            for m in range(self.mfirst-1, self.mlast, 1):
                VPin = np.zeros((2,3))
                lightIn = uvw[m,:]-uvw[m-1,:]
                lightOut = uvw[m+1,:]-uvw[m,:]
                VPin[0] = uvw[m,:] - vecoff*lightIn - veclen*polPin[m,:]
                VPin[1] = uvw[m,:] - vecoff*lightIn + veclen*polPin[m,:]
                if first:
                    ax.plot(VPin[:,0], VPin[:,1], VPin[:,2],'r', label='P-Pol') 
                else:
                    ax.plot(VPin[:,0], VPin[:,1], VPin[:,2],'r')
                    
                VPout = np.zeros_like(VPin)
                VPout[0] = uvw[m,:] + vecoff*lightOut - veclen*polPout[m,:] 
                VPout[1] = uvw[m,:] + vecoff*lightOut + veclen*polPout[m,:] 
                ax.plot(VPout[:,0], VPout[:,1], VPout[:,2],'r')
                
                VSin = np.zeros((2,3))
                VSin[0] = uvw[m,:] - vecoff*lightIn - veclen*polSin[m,:]
                VSin[1] = uvw[m,:] - vecoff*lightIn + veclen*polSin[m,:]
                ax.plot(VSin[:,0], VSin[:,1], VSin[:,2],'g') 
                
                VSout = np.zeros_like(VPin)
                VSout[0] = uvw[m,:] + vecoff*lightOut - veclen*polSout[m,:]
                VSout[1] = uvw[m,:] + vecoff*lightOut + veclen*polSout[m,:]
                if first:
                    ax.plot(VSout[:,0], VSout[:,1], VSout[:,2],'g', label='S-Pol') 
                    first=False
                else:
                    ax.plot(VSout[:,0], VSout[:,1], VSout[:,2],'g')
            plt.legend()
            plt.show()
        
        
        # drop dummy or unprocessed mirrors
        polSin = polSin[self.mfirst-1:self.mlast,:]
        polPin = polPin[self.mfirst-1:self.mlast,:]
        polSout = polSout[self.mfirst-1:self.mlast,:]
        polPout = polPout[self.mfirst-1:self.mlast,:]
        incAngle = incAngle[self.mfirst-1:self.mlast]   
        
        self.polSin = polSin
        self.polSout = polSout
        self.polPin = polPin
        self.polPout = polPout
        self.incAngle = incAngle
        self.usedmirror = np.arange(self.mfirst, self.mlast+1)

    
    
    def _getRotation(self, Az, El, returnRot=False):
        """
        Helper function for fitting
        Computes the rotations between the groups of mirrors
        for normal light path (starlight)
        Only works for full train
        """
        
        Az *= degtorad
        El *= degtorad 
        
        # expect rot before M10, M9 and M4 
        exp_M = [3, 8, 9]
        rot_Mgroup = np.zeros(3)
        
        idx = 0
        nmirr = len(self.polSin)
        rot_tot = 0
        for m in range(nmirr-1):
            # ref frame rotation to next mirror
            # for nominal config the out of the current mirror
            Snow = self.polSout[m,:]
            Pnow = self.polPout[m,:]
            # orientation of next mirror
            # for nominal config the in of the next mirror
            Snext = self.polSin[m+1,:]
            Pnext = self.polPin[m+1,:]     
            # Rotation between the two ref frames
            n = np.cross(Pnow, Snow)
            rot = math.atan2(np.dot(np.cross(Snow, Snext), n), 
                                np.dot(Snow, Snext))%math.pi
            if rot > math.pi/2:
                rot -= math.pi

            if (rot == 0.0) and (self.usedmirror[m] not in [3, 8]):
                continue

            if self.usedmirror[m] not in exp_M:
                raise ValueError('Rotation for an unexpected mirror (M%i)' % 
                                 self.usedmirror[m])
            if self.usedmirror[m] == 8:
                # Field rotation due to azimuth position
                telrot = -(Az + 18.984*degtorad)
                rot += telrot
            elif self.usedmirror[m] == 3:
                # Field rotation due to elevation
                telrot = (math.pi/2 - El)
                rot += telrot

            rot = rot%math.pi
            if rot > math.pi/2:
                rot -= math.pi
                
            rot_tot += rot
            rot_tot = rot_tot%math.pi
            rot_Mgroup[exp_M.index(self.usedmirror[m])] = rot
            
        if returnRot:
            rot_tot = (rot_tot/degtorad)%180
            return rot_Mgroup, rot_tot
        else:
            return rot_Mgroup


    def _getRotationRev(self, Az, El, returnRot=False):
        """
        Helper function for fitting
        Computes the rotations between the groups of mirrors
        for reverse light path (metrology)
        Only works for full train
        """
        Az *= degtorad
        El *= degtorad 

        # expect rot before M9, M8 and M3 
        exp_M = [10, 9, 4]
        rot_Mgroup = np.zeros(3)

        idx = 0
        nmirr = len(self.polSin)
        rot_tot = 0
        for m in range(nmirr-1, 0, -1):
            # ref frame rotation to next mirror
            # for reverse config the in of the current mirror
            Snow = self.polSin[m,:]
            Pnow = self.polPin[m,:]
            # orientation of next mirror
            # for reverse config the out of the previous mirror
            Snext = self.polSout[m-1,:]
            Pnext = self.polPout[m-1,:]
            # Rotation between the two ref frames
            n = np.cross(Snow, Pnow)
            rot = math.atan2(np.dot(np.cross(Snow, Snext), n), 
                                np.dot(Snow, Snext))%math.pi
            if rot > math.pi/2:
                rot -= math.pi
                
            if (rot == 0.0) and (self.usedmirror[m] not in [4,9]):
                continue

            if self.usedmirror[m] not in exp_M:
                raise ValueError('Rotation for an unexpected mirror (M%i)' % 
                                 self.usedmirror[m])
            if self.usedmirror[m] == 9:
                # Field rotation due to azimuth position
                telrot = -(Az + 18.984*degtorad)
                rot += telrot
        
            elif self.usedmirror[m] == 4:
                # Field rotation due to elevation
                telrot = ((math.pi/2) - El)
                rot += telrot
                
            rot = rot%math.pi
            if rot > math.pi/2:
                rot -= math.pi
                
            rot_tot += rot
            rot_tot = rot_tot%math.pi
            rot_Mgroup[exp_M.index(self.usedmirror[m])] = rot
            
        if returnRot:
            rot_tot = (rot_tot/degtorad)%180
            return rot_Mgroup, rot_tot
        else:
            return rot_Mgroup
    

    def _groupedMparams(self, X1, d1, X2, d2, X3, d3, X4, d4):
        M3 = self.Mfromparams(X1, d1)
        M4to8 = self.Mfromparams(X2, d2)
        M9 = self.Mfromparams(X3, d3)
        M10to18 = self.Mfromparams(X4, d4)
        return M3, M4to8, M9, M10to18
    
    
    def muellerfromParams(self, Az, El, X1, d1, X2, d2, X3, d3, X4, d4, 
                          rev=False, norm=True, returnRot=False, rotationFree=False):
        """
        Computes the mueller matrix from the grouped approach
        with the parameters of the froups given as arguments
        Can be in both directions with 
        rev=True for metrology light
        """
        Ms = np.array(self._groupedMparams(X1=X1, d1=d1, X2=X2, d2=d2,
                                           X3=X3, d3=d3, X4=X4, d4=d4))
        if rev:
            rot_Mgroup, rot_tot = self._getRotationRev(Az, El, returnRot=True)
            M = Ms[3]
            rotfac = [-1, 1, -1]
            for idx in range(3):
                M = np.matmul(self.rotationMatMueller(rotfac[idx] * rot_Mgroup[idx]),
                              M)
                M = np.matmul(Ms[2-idx], M)
        else:
            rot_Mgroup, rot_tot = self._getRotation(Az, El, returnRot=True)
            M = Ms[0]
            rotfac = [-1, 1, -1]
            for idx in range(3):
                M = np.matmul(self.rotationMatMueller(rotfac[idx] * rot_Mgroup[idx]),
                              M)
                M = np.matmul(Ms[idx+1], M)

        if norm:
            M /= M[0,0]

        if rotationFree:
            if self.verbose:
                print('Rotation back by %.2f deg' % (-rot_tot))
            M = np.matmul(self.rotationMatMueller(-rot_tot*degtorad), M)  

        if returnRot:
            return M, (rot_tot)
        else:
            return M       
    
    
    def muellerVLTI(self, Az, El, fit='Models/GroupedM_fitparams.txt',
                    rev=False, returnRot=False, rotationFree=False):
        """
        Function to calculate the mueller matrix at a given telescope position
        
        Mandatory parameters:
        Az:  Telescope azimuth in degree
        El:  Telescope elevation in degree

        Options:
        rev:          Calculate the mueller matrix in reverse (metrology)
                      direction
        returnRot:    Returns the value of field rotation in degree [False]
        rotationFree: Matrix is given without vield rotation, ie in the 
                      coordinate system of input light [False]
        """
        fitfile = resource_filename('VLTIpol', fit)
        fitval = np.genfromtxt(fitfile)
        return self.muellerfromParams(Az, El, *fitval,
                                      rev=rev, returnRot=returnRot, 
                                      rotationFree=rotationFree)
