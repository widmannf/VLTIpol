# VLTIpol

Small python package to get the instrumental polarization of the VLTI.
A paper for documentation is in preparation.

## Installation
run:
```bash
cd VLTIpol
pip install .
```

## Usage
### VLTI polarization

The package can be used to show the polarization properties of the VLTI. A Mueller or Jones matrix can be created for each telescope position:

```python
import VLTIpol as vp

v = vp.CalibVLTI()

az = 10 # deg
el = 80 # deg

# Mueller
M = v.polmat_VLTI(az, el)
print(M)

[[ 1.00000000e+00  2.19203803e-02 -6.16133748e-03  1.38354392e-02]
 [ 2.65928446e-02  8.53388556e-01 -2.34065865e-01  4.65766507e-01]
 [-1.48820507e-03  5.21049075e-01  3.78736731e-01 -7.64433568e-01]
 [ 7.00277534e-04  2.54027106e-03  8.95041752e-01  4.45179083e-01]]


# Jones
J = v.polmat_VLTI(az, el, jones=True)
print(J)

[[-0.11073747+0.69367468j  0.18918713-0.05982619j]
 [-0.03124834+0.18939207j -0.65362468+0.20427131j]]
```

#### Individual telescopes

For the Jones matrices each telescope is available with its individual properties.
To get the matrix for a single telescope one can use:

```python
J = v.polmat_VLTI(az, el, jones=True, telescope=1)
```
 With telescope ranging for 1 - 4 for UT 1 - 4.
 If telescope is not given (or None) or for Mueller matrix, the averaded properties are given.

### VLTI & GRAVITY
Ultimatively the package is created to correct the instrumental effect in GRAVITY observations.
This can be done as follows

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
km_angle = np.mean([(header[f'ESO INS DROT{i} START'] 
                     + header[f'ESO INS DROT{i} END'])/2
                     for i in range(1,5)])
               
hwp_angle = np.mean([(header[f'ESO INS DROT{i+4} START'] 
                      + header[f'ESO INS DROT{i+4} END'])/2
                      for i in range(1,5)])

# VLTI & GRAVITY mueller matrix:
M = vp.calib_all(azimuth, elevation,
                 km_angle, hwp_angle, paralactic_angle)

```
This gives you the full Mueller matrix of the VLTI & GRAVITY. The way how to apply it will then depend on your way on how to process the observing data.