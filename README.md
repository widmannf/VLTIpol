# VLTIpol

Small python package to get the instrumental polarization of the VLTI

Mini example to get the polarization in the UT rest position:

```python
import VLTIpol as vp
v = vp.CalibVLTI() 

azimuth = 0
elevation = 90

M = v.muellerVLTI(azimuth, elevation) 
```
