#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [2.831676, 2.831676, 2.937101, 2.937101, 2.937101, 2.937101,
    2.937101, 2.937101, 3.314270, 3.314270, 3.314270, 3.314270, 3.314270,
    3.314270, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199,
    3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.884824,
    3.884824, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500,
    4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500]
onsite = [-1, -1, -1, -1]

unit_conv = 3.80998212 # hbar/2mA to eV

t_start = 0.2
t_end = 0.8
t_step = 0.1

r_start = 4
r_end = 5
r_step = 10

t_list = np.arange(t_start, t_end, t_step)
r_list = np.arange(r_start, r_end, r_step)

outfile = open("Emass_scan.csv", "w+")

for t in t_list:
    for r in r_list:
        print("m=%.2f r-%.2f started\n" %(t, r))
        hopping = t * pow(np.divide(lat, bond_len), r)
        my_class = Ga2O3_Ga_s_Class(onsite, hopping, t, r)
        outfile.writelines(my_class.getMassString())
        my_class.savePlot('Emass_pdf/%.3f_%.3f.pdf' %(t, r))
        del my_class
