#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [2.831676, 2.831676, 2.937101, 2.937101, 2.937101, 2.937101, 2.937101, 2.937101, 3.314270, 3.314270, 3.314270, 3.314270, 3.314270, 3.314270, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.884824, 3.884824, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500]
onsite = [-1, -1, -1, -1]

unit_conv = 3.80998212 # hbar/2mA to eV

m_start = 0.2
m_end = 0.8
m_step = 0.1

r_start = 0
r_end = 10
r_step = 0.1

m_list = np.arange(m_start, m_end, m_step)
r_list = np.arange(r_start, r_end, r_step)

outfile = open("Emass_scan.csv", "w+")

for m in m_list:
    for r in r_list:
        print("m=%.2f r-%.2f started\n" %(m, r))
        hopping = m* unit_conv * pow(np.divide(lat, bond_len), r)
        my_class = Ga2O3_Ga_s_Class(onsite, hopping, m, r)
        outfile.writelines(my_class.getMassString())
        my_class.savePlot('Emass_pdf/%.3f_%.3f.pdf' %(m, r))
        del my_class
