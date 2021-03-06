#!/usr/bin/env python

from __future__ import division
import numpy as np
import csv
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [3.04, 3.04, 3.04, 3.04, 3.10927, 3.10927, 3.2777, 3.2777, 3.30067, 3.30067, 3.30067,
        3.3067, 3.32742, 3.32742, 3.32742, 3.32742, 3.44586, 3.44586, 3.44586, 3.44586, 
        3.6058, 3.61225]

unit_conv = 3.80998212 # hbar/2mA to eV

m_start = 0.44
m_end = 0.8
m_step = 0.002

m_list = np.arange(m_start, m_end, m_step)
r_list = -135.45*(m_list**2) + 191.93*m_list - 57.386


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
    my_class = Ga2O3_Ga_s_Class(hopping, m_list[i], r_list[i])
    error = my_class.getLSError(wannier_band, [0, 1, 2, 3])
    outfile.writelines("%.3f\t%.3f\t%f\t%f\n" %(m_list[i], r_list[i], error, my_class.emass_0))
    my_class.overlapPlot(wannier_band, [0, 1, 2, 3], 'wannier_pdf/Ga2O3_w_%.3fm_%.3fr.pdf' %(m_list[i], r_list[i]))
    del my_class