import numpy as np
import math



class PolFunctions():
    
    def __init__(self):
        pass


    def mirrorReflection(self, lightIn, lightOut):
        """
        compute the S and P vectors given the rays 
        from previous and to next mirror or reference point
        also compute normal to mirror and incidence angle
        will fail (zero divide) in case of normal reflection

        S-Pol perpendicular to plane of incidence
        plane of incidence is constructed by in and
        outfalling light
        """
        
        polSin = np.cross(lightIn,lightOut)
        if not np.linalg.norm(polSin) == 0.0:
            polSin = polSin / np.linalg.norm(polSin)
        else:
            polSin *= np.nan
        polSout = polSin
        # P-Pol is parrallel to plane of incidence
        polPin = np.cross(polSin,lightIn)
        polPin = polPin / np.linalg.norm(polPin)
        polPout = np.cross(polSout,lightOut)
        polPout = polPout / np.linalg.norm(polPout)
        normMirror = polPin + polPout
        normMirror = normMirror / np.linalg.norm(normMirror)
        Cos2i = - np.dot(lightIn/np.linalg.norm(lightIn), lightOut/np.linalg.norm(lightOut))
        try:
            incAngle = math.acos(Cos2i) / 2
        except ValueError:
            if Cos2i > 1 and Cos2i < (1+1e-5):
                incAngle = math.acos(1) / 2
            elif Cos2i < -1 and Cos2i > (-1-1e-5):
                Cos2i == -1
        
        return polSin, polSout, polPin, polPout, normMirror, incAngle



    def rotationMatMueller(self, theta):
        """
        Mueller Matrix for roation
        transforms from "old" components to "new" components
        when "new" frame is rotated by theta from "old" frame
        """
        if np.abs(theta) > 5*np.pi:
            raise ValueError('Check the angle, seems to be to big')
        
        RR = np.array([[np.cos(2*theta),  np.sin(2*theta)],
                       [-np.sin(2*theta), np.cos(2*theta)]])
        R = np.identity(4)
        R[1:-1,1:-1] = RR
        R[np.where(np.abs(R) < 1e-14)] = 0
        return R
    
    
    def Mfromparams(self, X, d):
        M_new = np.zeros((4,4))
        X2 = X**2
        M_new[0,0] = 1 + X2
        M_new[1,1] = 1 + X2
        M_new[0,1] = 1 - X2
        M_new[1,0] = 1 - X2

        M_new[2,2] = 2*X*np.cos(d)
        M_new[3,3] = 2*X*np.cos(d)
        M_new[2,3] = 2*X*np.sin(d)
        M_new[3,2] =-2*X*np.sin(d)

        M_new /= M_new[0,0]
        return(M_new)
