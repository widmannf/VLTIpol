import numpy as np
import math
import cmath


class PolFunctions():
    def __init__(self):
        pass

    def mirror_reflection(self, lightIn, lightOut):
        """
        compute the S and P vectors given the rays 
        from previous and to next mirror or reference point
        also compute normal to mirror and incidence angle
        will fail (zero divide) in case of normal reflection

        S-Pol perpendicular to plane of incidence
        plane of incidence is constructed by in and
        outfalling light
        """
        polSin = np.cross(lightIn, lightOut)
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

    def fresnel(self, nc, thetai, smallang=False):
        """
        inputs:  real and imag parts of index and incidence angle (radians!)
        outputs: power reflection coefficients r and phase shifts dp

        Reference Goldstein Chapter 25, Equ. (25-24) & (25-43)

        nc : complex index of refraction
        phase shifts are defined with the convention of Collett / Goldstein
        i.e. for perfect conductor ds = pi and dp = 0
        """
        thetar = cmath.asin(np.sin(thetai)/nc)

        if smallang:
            if(np.abs(thetai)>0.001):
                rs = -np.sin(thetai-thetar) / np.sin(thetai+thetar)
                rp = np.tan(thetai-thetar) / np.tan(thetai+thetar)
            else:
                rs = -(nc-1)/(nc+1)
                rp = -rs
        else:
            rs = -np.sin(thetai-thetar) / np.sin(thetai+thetar)
            rp = np.tan(thetai-thetar) / np.tan(thetai+thetar)

        reflectS = np.abs(rs)
        reflectP = np.abs(rp)
        dphaseS = np.angle(rs)
        dphaseP = np.angle(rp)

        return reflectS, reflectP, dphaseS, dphaseP
        
    def rotation_mueller(self, theta):
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
        R[1:-1, 1:-1] = RR
        R[np.where(np.abs(R) < 1e-14)] = 0
        return R

    def rotation(self, theta):
        """
        2-D rotation matrix 
        transforms from "old" components to "new" components
        when "new" frame is rotated by theta from "old" frame
        angle orientation: y is at +pi/2 from x
        """
        if np.abs(theta) > 5*np.pi:
            raise ValueError('Check the angle, seems to be to big')
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        return R

    def polmat_mirror(self, par, full=False, norm=True, jones=False):
        if jones:
            if len(par) != 3:
                raise ValueError('For full Mueller matrix parameter has to '
                                 'contain three values: rs, rp, & d. '
                                 'If e and d ar given use full=True '
                                 'and jones=False')
            rs, rp, d = par
            J = np.array([[rs*np.exp(1j*d), 0],
                          [0, rp]])
            return J

        else:
            if full:
                if len(par) != 3:
                    raise ValueError('For full Mueller matrix parameter has to'
                                     ' contain three values: rs, rp, & d. '
                                     'If e and d ar given use full=True')
                rs, rp, d = par

                M = np.zeros((4, 4))
                M[0, 0] = rs**2 + rp**2
                M[1, 1] = rs**2 + rp**2
                M[1, 0] = rs**2 - rp**2
                M[0, 1] = rs**2 - rp**2
                M[2, 2] = 2*rs*rp*np.cos(d)
                M[3, 3] = 2*rs*rp*np.cos(d)
                M[2, 3] = 2*rs*rp*np.sin(d)
                M[3, 2] = -2*rs*rp*np.sin(d)
                M *= 0.5
                if norm:
                    M /= M[0, 0]

            else:
                if len(par) != 2:
                    raise ValueError('For Mueller matrix parameter has to '
                                     'contain two values: e & d. If '
                                     'rs, rp, and d are given use full=True')
                e, d = par
                M = np.zeros((4, 4))
                M[0, 0] = 1
                M[1, 1] = 1
                M[0, 1] = e
                M[1, 0] = e

                M[2, 2] = np.sqrt(1-e**2)*np.cos(d)
                M[3, 3] = np.sqrt(1-e**2)*np.cos(d)
                M[2, 3] = np.sqrt(1-e**2)*np.sin(d)
                M[3, 2] = -np.sqrt(1-e**2)*np.sin(d)

                M /= M[0, 0]
            return M
