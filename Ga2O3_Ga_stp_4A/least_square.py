#!/usr/bin/env python

from __future__ import division
import numpy as np
import csv
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [3.040000, 3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 
            3.277766, 3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 
            3.327418, 3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 
            3.445864, 3.445864, 3.605804, 3.612253, 4.337574, 4.337574, 
            4.470498, 4.470498, 4.652131, 4.862916, 4.945890, 4.945890]
onsite = [-1, -1, -1, -1]

unit_conv = 3.80998212 # hbar/2mA to eV

r_start = 0
r_end = 10
r_step = 0.25

r_list = np.arange(r_start, r_end, r_step)
m_list = 0.0029*(r_list**2) -0.0222*r_list + 0.5457


# with open('wannier_band.dat','rb') as file:
#     wannier_band_reader=csv.reader(file)
#     wannier_band_reader.

wannier_band_raw = np.loadtxt('wannier_band.dat', dtype=float)
# shift minimum to 0
wannier_band_raw[:,1] = wannier_band_raw[:,1] - np.min(wannier_band_raw[:,1])
wannier_band = []
# store different bands into different dimensions
wannier_band.append(wannier_band_raw[0:532,1])
wannier_band.append(wannier_band_raw[533:1065,1])
wannier_band.append(wannier_band_raw[1066:1598,1])
wannier_band.append(wannier_band_raw[1599:2131,1])


outfile = open("LS_scan.csv", "w+")

for i in range(len(m_list)):
    print("m=%.3f r=%.3f started\n" %(m_list[i], r_list[i]))
    hopping = np.divide(-1, (m_list[i]*lat**2)) * unit_conv * pow(np.divide(lat, bond_len), r_list[i])
    my_class = Ga2O3_Ga_s_Class(onsite, hopping, m_list[i], r_list[i])
    error = my_class.getLSError(wannier_band, [0, 1, 2, 3])
    outfile.writelines("%.3f\t%.3f\t%f\t%f\n" %(m_list[i], r_list[i], error, my_class.emass_0))
    my_class.savePlot('LS_pdf/%.3fm_%.3fr.pdf' %(m_list[i], r_list[i]))
    my_class.overlapPlot(wannier_band, [0, 1, 2, 3], 'wannier_pdf/%.3fm_%.3fr.pdf' %(m_list[i], r_list[i]))
    del my_class