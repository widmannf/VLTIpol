# VLTIpol

Small python package to get the instrumental polarization of the VLTI

Mini example to get the polarization in the UT rest position:

```python
import numpy as np
from astropy.io import fits
import VLTIpol as vp
v = vp.CalibVLTI() 

# Values that one gets from the header of the GRAVITY files
header = fits.open('gravity.fits')[0].header
azimuth = header['ESO ISS AZ']
elevation = header['ESO ISS ALT']
paralactic_angle = (header['ESO ISS PARANG START'] + 
                    header['ESO ISS PARANG END'])/2
# HWP and K-Mirror angle have to be calculated from encoder positions:
km_angle = (header['ESO INS DROT1 ENC'] - 8909)/400
hwp_angle = (header['ESO INS DROT1 ENC'] - 10909)/400

# VLTI mueller matrix:
M = v.muellerVLTI(azimuth, elevation) 
iM = np.linalg.inv(M_vlti)

# Additional rotation:
R = pola.rotationMatMueller((pa[idx]+90)/180*np.pi)

# GRAVITY mueller matrix:
M_gra = v.muellerGRAVITY()
        
# K-Mirror mueller matrix:
M_km = v.muellerKM(km_angle)

# HWP mueller matrix:
M_hwp = v.muellerHWP(hwp_angle)

iM_gr = np.linalg.inv(np.dot(np.dot(M_gra, M_hwp), M_km))
        
# Application of the full mueller matrix:
stokes_out = np.dot(iM_gr, np.dot(R, np.dot(iM, stokes_out)))
```
