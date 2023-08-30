# VLTIpol

Small python package to get the instrumental polarization of the VLTI. A paper for documentation is in preparation.

Mini example to get the polarization for a typical GRAVITY file:

```python
import numpy as np
from astropy.io import fits
import VLTIpol as vp

# Values that one gets from the header of the GRAVITY files
header = fits.open('gravity.fits')[0].header
azimuth = header['ESO ISS AZ']
elevation = header['ESO ISS ALT']
paralactic_angle = (header['ESO ISS PARANG START'] + 
                    header['ESO ISS PARANG END'])/2
# HWP and K-Mirror angle have to be calculated from encoder positions:
km_angle = np.mean(([(header[f'ESO INS DROT{i} START'] 
                      + header[f'ESO INS DROT{i} END'])/2 
                     for i in range(1,5)]))
               
hwp_angle = np.mean(([(header[f'ESO INS DROT{i+4} START'] 
                       + header[f'ESO INS DROT{i+4} END'])/2 
                       - header[f'ESO INS HWPOFFSET{i}'])
                       for i in range(1,5)])

# VLTI & GRAVITY mueller matrix:
M = vp.calib_all(azimuth, elevation,
                 km_angle, hwp_angle, paralactic_angle)
iM = np.linalg.inv(M)
        
# Application of the full mueller matrix:
stokes_out = np.dot(iM, stokes_measured)
```
