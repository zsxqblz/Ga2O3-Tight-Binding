#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [3.040000, 3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 
            3.277766, 3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 
            3.327418, 3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 
            3.445864, 3.445864, 3.605804, 3.612253, 4.337574, 4.337574, 
            4.470498, 4.470498, 4.652131, 4.862916, 4.945890, 4.945890]
onsite = [-1, -1, -1, -1]

unit_conv = 3.80998212 # hbar/2mA to eV

m_start = 0.2
m_end = 0.4
m_step = 0.01

r_start = 0
r_end = 1
r_step = .2

m_list = np.arange(m_start, m_end, m_step)
r_list = np.arange(r_start, r_end, r_step)

# for m in m_list:
#     for r in r_list:
m = 0.513
r = 2
print("m=%.2f r-%.2f started\n" %(m, r))
hopping = np.divide(-1, (m*lat**2)) * unit_conv * pow(np.divide(lat, bond_len), r)
my_class = Ga2O3_Ga_s_Class(onsite, hopping, m, r)
print("Fermi Surface plot started\n\n")
for fermi_level in 0.2*np.arange(100) + np.min(my_class.evals[0]):
        print("FE=%.2f started\n" %(fermi_level))
        my_class.plotFermiSurf(fermi_level, "FS_pdf/Ga2O3_%.2fm_%.2fr_%.3fFE.png" %(m, r, fermi_level), real_unit=True)
del my_class
