#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [3.04, 3.04, 3.04, 3.04, 3.10927, 3.10927, 3.2777, 3.2777, 3.30067, 3.30067, 3.30067,
        3.3067, 3.32742, 3.32742, 3.32742, 3.32742, 3.44586, 3.44586, 3.44586, 3.44586, 
        3.6058, 3.61225]

unit_conv = 3.80998212 # hbar/2mA to eV

m_start = 0.2
m_end = 0.8
m_step = 0.01

r_start = 0
r_end = 10
r_step = 0.25

m_list = np.arange(m_start, m_end, m_step)
r_list = np.arange(r_start, r_end, r_step)

outfile = open("scan.csv", "w+")

for m in m_list:
    for r in r_list:
        print("m=%.2f r-%.2f started\n" %(m, r))
        hopping = np.divide(-1, (m*lat**2)) * unit_conv * pow(np.divide(lat, bond_len), r)
        my_class = Ga2O3_Ga_s_Class(hopping, m, r)
        outfile.writelines(my_class.getMassString())
        del my_class
