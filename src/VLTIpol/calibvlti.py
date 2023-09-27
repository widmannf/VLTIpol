import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
import logging

from .polfunctions import *
DEGTORAD = np.pi/180

log_level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class CalibVLTI(PolFunctions):
    """
    Class to simulate the propagation of polarization through the VLTI
    Train, in order to calibrate polarization measurements

    plot: show the light path [False]
    loglevel: logging level [INFO]

    The mueller matrix of the VLTI can be created with:
    M = mueller_VLTI(Az, El)
    And the one of GRAVITY with:
    M = mueller_GRAVITY(kmrot, hwprot)
    """

    def __init__(self, plot=False, loglevel='INFO'):
        self.mfirst = 3
        self.mlast = 18
        self.plot = plot

        log_level = log_level_mapping.get(loglevel, logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG) # not sure if needed
        formatter = logging.Formatter('%(levelname)s: %(name)s - %(message)s')
        ch.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(ch)
        self.logger = logger

        self.setup_VLTI()

    def setup_VLTI(self, degradedSilver=True, protSilver=True,
                   ncSilver = 0.76571 + 1j * 13.408,
                   ncAluminium = 2.753 + 1j * 22.282,
                   ncGold = 0.9918 + 1j * 13.808):
        """
        Sets up the light path and the s & p vectors of each mirror
        """
        posfile = resource_filename('VLTIpol', 'Models/mirrorPos.txt')
        uvw = np.genfromtxt(posfile)
        u = uvw[:, 0]
        v = uvw[:, 1]
        w = uvw[:, 2]

        polSin = np.zeros_like(uvw)*np.nan
        polSout = np.zeros_like(uvw)*np.nan
        polPin = np.zeros_like(uvw)*np.nan
        polPout = np.zeros_like(uvw)*np.nan
        normMirror = np.zeros_like(uvw)*np.nan
        incAngle = np.zeros(len(uvw))*np.nan

        for m in range(len(uvw)-1):
            lightIn = uvw[m, :]-uvw[m-1, :]
            lightIn = lightIn/np.linalg.norm(lightIn)
            lightOut = uvw[m+1, :] - uvw[m, :]
            lightOut = lightOut/np.linalg.norm(lightOut)
            (polSin[m, :], polSout[m, :], polPin[m, :], polPout[m, :],
             normMirror[m, :], incAngle[m]) = self.mirror_reflection(lightIn,
                                                                     lightOut) 

        if self.plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(u, v, w, color='k', ls='--', lw=0.5)
            ax.plot(u[self.mfirst-2: self.mlast+1],
                    v[self.mfirst-2: self.mlast+1],
                    w[self.mfirst-2: self.mlast+1],
                    color='b', lw=0.5)
            vecoff = 0.15
            veclen = 0.5

            first = True
            for m in range(self.mfirst-1, self.mlast, 1):
                VPin = np.zeros((2, 3))
                lightIn = uvw[m, :] - uvw[m-1, :]
                lightOut = uvw[m+1, :] - uvw[m, :]
                VPin[0] = uvw[m, :] - vecoff*lightIn - veclen*polPin[m, :]
                VPin[1] = uvw[m, :] - vecoff*lightIn + veclen*polPin[m, :]
                if first:
                    ax.plot(VPin[:, 0], VPin[:, 1], VPin[:, 2],
                            color='r', label='P-Pol')
                else:
                    ax.plot(VPin[:, 0], VPin[:, 1], VPin[:, 2],
                            color='r')

                VPout = np.zeros_like(VPin)
                VPout[0] = uvw[m, :] + vecoff*lightOut - veclen*polPout[m, :]
                VPout[1] = uvw[m, :] + vecoff*lightOut + veclen*polPout[m, :]
                ax.plot(VPout[:, 0], VPout[:, 1], VPout[:, 2],
                        color='r')

                VSin = np.zeros((2, 3))
                VSin[0] = uvw[m, :] - vecoff*lightIn - veclen*polSin[m, :]
                VSin[1] = uvw[m, :] - vecoff*lightIn + veclen*polSin[m, :]
                ax.plot(VSin[:, 0], VSin[:, 1], VSin[:, 2],
                        color='g')

                VSout = np.zeros_like(VPin)
                VSout[0] = uvw[m, :] + vecoff*lightOut - veclen*polSout[m, :]
                VSout[1] = uvw[m, :] + vecoff*lightOut + veclen*polSout[m, :]
                if first:
                    ax.plot(VSout[:, 0], VSout[:, 1], VSout[:, 2],
                            color='g', label='S-Pol')
                    first = False
                else:
                    ax.plot(VSout[:, 0], VSout[:, 1], VSout[:, 2],
                            color='g')
            plt.legend()
            plt.show()

        reflectP = np.zeros(len(uvw)-1) * np.nan
        reflectS = np.zeros(len(uvw)-1) * np.nan
        dPhase = np.zeros(len(uvw)-1) * np.nan

        facDrude = 3
        deltaProtSilverDeg = 155
        if degradedSilver:
            epst = ncSilver**2
            fps = 1/(epst-1)
            fps = np.real(fps) + 1j * facDrude * np.imag(fps)
            eps = (1+fps)/fps
            ncOldsilver = np.sqrt(eps)
        else:
            ncOldsilver = ncSilver

        for m in range(2, len(uvw)-1):
            # Aluminium
            if m in [2]:
                (reflectS[m], reflectP[m],
                 dphaseS, dphaseP) = self.fresnel(ncAluminium, incAngle[m])
                dPhase[m] = (dphaseS-dphaseP) % (2*math.pi)
            # Silver
            elif m in [3, 4, 5, 6, 7, 9, 10, 11, 15]:
                (reflectS[m], reflectP[m],
                 dphaseS, dphaseP) = self.fresnel(ncOldsilver, incAngle[m])
                dPhase[m] = (dphaseS-dphaseP) % (2*math.pi)
                if protSilver:
                    dPhase[m] = (math.pi + (incAngle[m]/(45*DEGTORAD))**(5./2.)
                                * (deltaProtSilverDeg*DEGTORAD-math.pi))
            # dichroic
            elif m in [8]:
                reflectS[m] = np.sqrt(0.92)
                reflectP[m] = np.sqrt(0.82)
                dPhase[m] = 165/180*math.pi

            # catseye
            elif m in [12, 13, 14]:
                reflectS[m] = 0.979
                reflectP[m] = 0.979
                dPhase[m] = math.pi
            
            # Gold
            elif m in [16, 17]:
                (reflectS[m], reflectP[m],
                 dphaseS, dphaseP) = self.fresnel(ncGold, incAngle[m])
                dPhase[m] = (dphaseS-dphaseP) % (2*math.pi)
            
            else:
                self.logger.warning('Mirror %i not defined' % m)

        polSin = polSin[self.mfirst-1:self.mlast, :]
        polPin = polPin[self.mfirst-1:self.mlast, :]
        polSout = polSout[self.mfirst-1:self.mlast, :]
        polPout = polPout[self.mfirst-1:self.mlast, :]
        reflectS = reflectS[self.mfirst-1:self.mlast]
        reflectP = reflectP[self.mfirst-1:self.mlast]
        dPhase = dPhase[self.mfirst-1:self.mlast]
        incAngle = incAngle[self.mfirst-1:self.mlast]
        incAngleDeg = incAngle/DEGTORAD

        self.logger.debug('>>>>>>>>>>>>>>> SetupVltiTrain <<<<<<<<<<<<<<<<<<')
        mirrN = np.arange(len(polSin))
        realM = mirrN+self.mfirst
        self.logger.debug('      refl.Coef.S   refl.Coef.P  IncAngle  dPhase')
        for m in range(len(polSin)):
            if reflectS[m] != 1.0:
                self.logger.debug(f'M{realM[m]:02d}     {reflectS[m]:.4f}        {reflectP[m]:.4f}'
                                 f'        {int(incAngleDeg[m]):02d}      {dPhase[m]/DEGTORAD:03.1f}')

        self.polSin = polSin
        self.polSout = polSout
        self.polPin = polPin
        self.polPout = polPout
        self.reflectS = reflectS
        self.reflectP = reflectP
        self.dPhase = dPhase
        self.incAngle = incAngle
        self.usedmirror = np.arange(self.mfirst, self.mlast+1)

    def polmat(self, Az, El, jones=False, return_rot=False,
               rev=False):
        """
        Computes Mueller matrix for normal propagating
        Input (s,p) axes defined by the plane of incidence on first mirror
        For M3 (s,p) axes defined by the plane of incidence on M3
        Output with (s,p) axes defined by the plane of incidence on
        last mirror, ideally M18
        For M18 s is vertical (+w) in lab space

        Az, El       : telescop position
        return_rot    : returns als angle of full field rotation
        jones        : if true returns jones matrix instead of mueller
        rev          : calculates matrix in opposite direction 
                       (along metrology light path)
        """

        Az *= DEGTORAD
        El *= DEGTORAD
        MN = np.identity(4)
        JN = np.identity(2)
        rot_tot = 0
        midx = 0
        nmirr = len(self.polSin)
        if jones:
            self.logger.info('Getting Jones matrix for theoretical mirror properties')
        else:
            self.logger.info('Getting Mueller matrix for theoretical mirror properties')
        if 8 not in self.usedmirror:
            self.logger.warning('M8 not in path, ignore Azimuth angle')
        if 3 not in self.usedmirror:
            self.logger.warning('M3 not in path, ignore Elevation angle')

        if rev:
            for m in range(nmirr-1, -1, -1):
                mirrM = self.polmat_mirror([self.reflectS[m], self.reflectP[m],
                                            self.dPhase[m]], full=True)
                MN = np.dot(mirrM, MN)
                mirrJ = self.polmat_mirror([self.reflectS[m], self.reflectP[m],
                                            self.dPhase[m]], jones=True)
                JN = np.dot(mirrJ, JN)

                if m == 0:
                    self.logger.debug(f'M{self.usedmirror[m]} reached')
                else:
                    Snow = self.polSin[m, :]
                    Pnow = self.polPin[m, :]
                    Snext = self.polSout[m-1, :]
                    n = np.cross(Snow, Pnow)
                    rot = math.atan2(np.dot(np.cross(Snow, Snext), n),
                                    np.dot(Snow, Snext)) % math.pi

                    if self.usedmirror[m] == 9:
                        telrot = -(Az + 18.984*DEGTORAD)
                        rot += telrot

                    elif self.usedmirror[m] == 4:
                        telrot = ((math.pi/2) - El)
                        rot += telrot

                    rot = rot%math.pi
                    if rot > math.pi/2: rot -= math.pi

                    if rot != 0:
                        self.logger.debug(f'Rotation between M{self.usedmirror[m]} '
                                        f'and M{self.usedmirror[m-1]} is {rot/DEGTORAD:.2f}')

                    if (midx % 2) == 0:  rotfac = -1
                    else:  rotfac = 1
                    frameRot = self.rotation_mueller(rot*rotfac)
                    MN = np.dot(frameRot, MN)
                    frameRotJ = self.rotation(rot*rotfac)
                    JN = np.dot(frameRotJ, JN)

                    rot_tot += rot
                    midx += 1

        else:
            for m in range(nmirr):
                mirrM = self.polmat_mirror([self.reflectS[m], self.reflectP[m],
                                            self.dPhase[m]], full=True)
                MN = np.dot(mirrM, MN)

                mirrJ = self.polmat_mirror([self.reflectS[m], self.reflectP[m],
                                            self.dPhase[m]], jones=True)
                JN = np.dot(mirrJ, JN)

                if m == (nmirr-1):
                    self.logger.debug(f'M{self.usedmirror[m]} reached')
                else:
                    Snow = self.polSout[m, :]
                    Pnow = self.polPout[m, :]
                    Snext = self.polSin[m+1, :]
                    n = np.cross(Pnow, Snow)
                    rot = math.atan2(np.dot(np.cross(Snow, Snext), n),
                                    np.dot(Snow, Snext)) % math.pi
                    if rot > math.pi/2: rot -= math.pi

                    if self.usedmirror[m] == 8:
                        telrot = -(Az + 18.984*DEGTORAD)
                        rot += telrot

                    elif self.usedmirror[m] == 3:
                        # Field rotation due to elevation
                        telrot = (math.pi/2-El)
                        rot += telrot

                    rot = rot % math.pi
                    if rot > math.pi/2: rot -= math.pi
                    if rot != 0:
                        self.logger.debug(f'Rotation between M{self.usedmirror[m]} '
                                        f'and M{self.usedmirror[m+1]} is {rot/DEGTORAD:.2f}')

                    if (midx % 2) == 0: rotfac = -1
                    else: rotfac = 1
                    frameRot = self.rotation_mueller(rot*rotfac)
                    MN = np.dot(frameRot, MN)
                    frameRotJ = self.rotation(rot*rotfac)
                    JN = np.dot(frameRotJ, JN)

                    rot_tot += rot
                    midx += 1

        MN /= MN[0, 0]

        if jones:
            if return_rot:
                return JN, (rot_tot/DEGTORAD) % 180
            else:
                return JN
        else:
            if return_rot:
                return MN, (rot_tot/DEGTORAD) % 180
            else:
                return MN


    def _get_rotation(self, Az, El, rev=False, return_rot=False):
        """
        Helper function for fitting
        Computes the rotations between the groups of mirrors
        for normal light path (starlight)
        """
        Az *= DEGTORAD
        El *= DEGTORAD
        exp_M = [3, 8, 9]
        rot_Mgroup = np.zeros(3)
        nmirr = len(self.polSin)
        rot_tot = 0

        if rev:
            # expect rot before M9, M8 and M3
            exp_M = [10, 9, 4]
            for m in range(nmirr-1, 0, -1):
                Snow = self.polSin[m, :]
                Pnow = self.polPin[m, :]
                Snext = self.polSout[m-1, :]
                Pnext = self.polPout[m-1, :]
                n = np.cross(Snow, Pnow)
                rot = math.atan2(np.dot(np.cross(Snow, Snext), n),
                                np.dot(Snow, Snext)) % math.pi
                if rot > math.pi/2: rot -= math.pi

                if (rot == 0.0) and (self.usedmirror[m] not in [4, 9]):
                    continue

                if self.usedmirror[m] not in exp_M:
                    raise ValueError('Rotation for an unexpected mirror (M%i)' %
                                    self.usedmirror[m])
                if self.usedmirror[m] == 9:
                    telrot = -(Az + 18.984*DEGTORAD)
                    rot += telrot

                elif self.usedmirror[m] == 4:
                    telrot = ((math.pi/2) - El)
                    rot += telrot

                rot = rot % math.pi
                if rot > math.pi/2: rot -= math.pi

                rot_tot += rot
                rot_tot = rot_tot % math.pi
                rot_Mgroup[exp_M.index(self.usedmirror[m])] = rot

        else:
            # expect rot before M10, M9 and M4
            exp_M = [3, 8, 9]
            for m in range(nmirr-1):
                Snow = self.polSout[m, :]
                Pnow = self.polPout[m, :]
                Snext = self.polSin[m+1, :]
                Pnext = self.polPin[m+1, :]
                n = np.cross(Pnow, Snow)
                rot = math.atan2(np.dot(np.cross(Snow, Snext), n),
                                np.dot(Snow, Snext)) % math.pi
                if rot > math.pi/2: rot -= math.pi

                if (rot == 0.0) and (self.usedmirror[m] not in [3, 8]):
                    continue

                if self.usedmirror[m] not in exp_M:
                    raise ValueError('Rotation for an unexpected mirror (M%i)' %
                                    self.usedmirror[m])
                if self.usedmirror[m] == 8:
                    # Field rotation due to azimuth position
                    telrot = -(Az + 18.984*DEGTORAD)
                    rot += telrot
                elif self.usedmirror[m] == 3:
                    # Field rotation due to elevation
                    telrot = (math.pi/2 - El)
                    rot += telrot

                rot = rot % math.pi
                if rot > math.pi/2: rot -= math.pi

                rot_tot += rot
                rot_tot = rot_tot % math.pi
                rot_Mgroup[exp_M.index(self.usedmirror[m])] = rot

        if return_rot:
            return rot_Mgroup, (rot_tot/DEGTORAD) % 180
        else:
            return rot_Mgroup


    def _grouped_polmat_params(self, par, jones=False, full=False):
        """
        Computes the mueller matrix for each mirror group in the VLTIpol
        based on an input list of parameters
        
        par    : diattenuation and phaseshift of each group
        jones  : if true returns jones matrix instead of mueller
        full   : if true takes rs & rp instead of diattenuation
        """
        if jones:
            if len(par) != 12:
                raise ValueError('For Jones matrix parameter has to '
                                 'contain 12 values: (rs, rp, & d) x 4. '
                                 'If e and d are given use full=True '
                                 'and jones=False')
            (rs1, rp1, d1, rs2, rp2, d2, rs3, rp3, d3, rs4, rp4, d4) = par
            J3 = self.polmat_mirror([rs1, rp1, d1], jones=True)
            J4to8 = self.polmat_mirror([rs2, rp2, d2], jones=True)
            J9 = self.polmat_mirror([rs3, rp3, d3], jones=True)
            J10to18 = self.polmat_mirror([rs4, rp4, d4], jones=True)
            return J3, J4to8, J9, J10to18
        else:
            if full:
                if len(par) != 12:
                    raise ValueError('For full Mueller matrix parameter has to'
                                     ' contain 12 values: (rs, rp, & d) x 4. '
                                     'If e and d are given use full=True')
                (rs1, rp1, d1, rs2, rp2, d2, rs3, rp3, d3, rs4, rp4, d4) = par
                M3 = self.polmat_mirror([rs1, rp1, d1], full=True)
                M4to8 = self.polmat_mirror([rs2, rp2, d2], full=True)
                M9 = self.polmat_mirror([rs3, rp3, d3], full=True)
                M10to18 = self.polmat_mirror([rs4, rp4, d4], full=True)
            else:
                if len(par) != 8:
                    raise ValueError('For Mueller matrix parameter has to '
                                     'contain 8 values: (e & d) x 4. If '
                                     'rs, rp, and d are given use full=True')
                (e1, d1, e2, d2, e3, d3, e4, d4) = par
                M3 = self.polmat_mirror([e1, d1])
                M4to8 = self.polmat_mirror([e2, d2])
                M9 = self.polmat_mirror([e3, d3])
                M10to18 = self.polmat_mirror([e4, d4])
            return M3, M4to8, M9, M10to18

    def polmat_from_params(self, Az, El, par, rev=False, jones=False,
                           full=False, return_rot=False):
        """
        Computes the polarization matrix from the grouped approach
        with the parameters of the froups given as arguments

        Az, El : Telescope position
        par    : diattenuation and phaseshift of each group
        rev    : if true calculates in reverse direction (metrology)
        jones  : if true returns jones matrix instead of mueller
        full   : if true takes rs & rp instead of diattenuation
        """
        Ms = self._grouped_polmat_params(par, jones=jones, full=full)
        angs, rot_tot = self._get_rotation(Az, El, rev=rev, return_rot=True)
        rotfac = [-1, 1, -1]
        angs = [ang*rotfac[idx] for idx, ang in enumerate(angs)]

        if jones:
            Rs = [self.rotation(ang) for ang in angs]
        else:
            Rs = [self.rotation_mueller(ang) for ang in angs]
        if rev:
            M = Ms[3]
            for idx in range(3):
                M = np.matmul(Rs[idx], M)
                M = np.matmul(Ms[2-idx], M)
        else:
            M = Ms[0]
            for idx in range(3):
                M = np.matmul(Rs[idx], M)
                M = np.matmul(Ms[idx+1], M)

        if not jones: M /= M[0, 0]

        if return_rot:
            return M, rot_tot
        else:
            return M

    def polmat_VLTI(self, Az, el, jones=False, rev=False):
        """
        Function to calculate the polarization matrix at a given telescope position
        based on the best fit paramters

        Mandatory parameters:
        Az:  Telescope azimuth in degree
        El:  Telescope elevation in degree

        Options:
        rev:          Calculate the mueller matrix in reverse (metrology)
                      direction
        jones:        if true returns jones matrix instead of mueller
        """
        if jones: 
            fitfile = resource_filename('VLTIpol', 'Models/GroupedJ_fitparams.txt')
            fitval = np.genfromtxt(fitfile)
        else:
            fitfile = resource_filename('VLTIpol', 'Models/GroupedM_fitparams.txt')
            fitval = np.genfromtxt(fitfile)
        return self.polmat_from_params(Az, el, fitval, rev=rev, jones=jones)

    def mueller_gravity_rot(self, phiK, phiH, M, fiberrot=15.8, rev=False,
                            norm=True):
        """
        Put the GRAVITY mueller matrix together by adding the rotations
        """
        MK, MHWP, Mrest = M
        phiK *= DEGTORAD
        phiH *= DEGTORAD
        fiberrot *= DEGTORAD

        if rev:
            MG = np.matmul(self.rotation_mueller(-fiberrot), MG)

            MG = np.matmul(self.rotation_mueller(phiH), Mrest)
            MG = np.matmul(MHWP, MG)
            MG = np.matmul(self.rotation_mueller(-phiH), MG)

            MG = np.matmul(self.rotation_mueller(phiK), MG)
            MG = np.matmul(MK, MG)
            MG = np.matmul(self.rotation_mueller(-phiK), MG)

            MG = np.matmul(self.rotation_mueller(-math.pi/2), MG)

        else:
            MG = np.identity(4)
            MG = np.matmul(self.rotation_mueller(math.pi/2), MG)

            MG = np.matmul(self.rotation_mueller(-phiK), MG)
            MG = np.matmul(MK, MG)
            MG = np.matmul(self.rotation_mueller(phiK), MG)

            MG = np.matmul(self.rotation_mueller(-phiH), MG)
            MG = np.matmul(MHWP, MG)
            MG = np.matmul(self.rotation_mueller(phiH), MG)

            MG = np.matmul(Mrest, MG)
            MG = np.matmul(self.rotation_mueller(fiberrot), MG)

        if norm:
            MG /= MG[0, 0]
        return MG

    def mueller_GRAVITY(self, kmrot, hwprot, onaxis=False):
        """
        Function to calculate the mueller matrix of GRAVITY given
        the mirror positions

        Mandatory parameters:
        kmrot:   K-Mirror position in degree
        hwprot:  HWP position in degree

        Options:
        onaxis:  True for on-axis, False for off-axis
        """
        MHWP = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, -1]])
        if onaxis:
            fitMKM_name = resource_filename('polsim',
                                            'Models/GRAVITY_fit_MK_Monaxis.txt')
            fitM_name = resource_filename('polsim',
                                          'Models/GRAVITY_fit_M_onaxis.txt')
        else:
            fitMKM_name = resource_filename('polsim',
                                            'Models/GRAVITY_fit_MKM.txt')
            fitM_name = resource_filename('polsim',
                                          'Models/GRAVITY_fit_M.txt')
        MKM = np.genfromtxt(fitMKM_name)
        Mgra = np.genfromtxt(fitM_name)

        M = [MKM, MHWP, Mgra]
        M = self.mueller_gravity_rot(kmrot, hwprot, M)
        return M


def calib_all(az, el, kmrot, hwprot, pa, return_rot=False,
              onaxis=False, plot=False):
    """
    Gives full mueller matrix of VLTI & GRAVITY
    Mandatory parameters:
    Az:      Telescope azimuth in degree
    El:      Telescope elevation in degree
    kmrot:   K-Mirror position in degree
    hwprot:  HWP position in degree
    pa:      paralactic angle
    """
    vlti = CalibVLTI(plot=plot)

    M_vlti = vlti.polmat_VLTI(az, el)
    M_gra = vlti.mueller_GRAVITY(kmrot, hwprot, onaxis=onaxis)
    M = np.dot(M_gra, M_vlti)
    M = M / M[0, 0]

    if return_rot:
        rotang = pa - 90
        R = vlti.rotation_mueller(((-rotang) % 180) / 180*math.pi)
        return M, R
    else:
        return M

    