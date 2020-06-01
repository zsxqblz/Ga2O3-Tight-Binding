#!/usr/bin/env python

from __future__ import division
import numpy as np
import csv
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [3.040000,
        3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 3.277766,
        3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 3.327418,
        3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 3.445864, 
        3.445864, 3.605804, 3.612253, 4.337574, 4.337574, 4.470498,
        4.470498, 4.652131, 4.716293, 4.716293, 4.721225, 4.721225,
        4.862916, 4.945890, 4.945890]

unit_conv = 3.80998212 # hbar/2mA to eV

##
        # m_start = 0.44
        # m_end = 0.8
        # m_step = 0.002

        # m_list = np.arange(m_start, m_end, m_step)
        # r_list = -135.45*(m_list**2) + 191.93*m_list - 57.386


        # wannier_band_raw = np.loadtxt('wannier_band.dat', dtype=float)
        # # shift minimum to 0
        # wannier_band_raw[:,1] = wannier_band_raw[:,1] - np.min(wannier_band_raw[:,1])
        # wannier_band = []
        # # store different bands into different dimensions
        # wannier_band.append(wannier_band_raw[0:532,1])
        # wannier_band.append(wannier_band_raw[533:1065,1])
        # wannier_band.append(wannier_band_raw[1066:1598,1])
        # wannier_band.append(wannier_band_raw[1599:2131,1])


        # outfile = open("LS_scan.csv", "w+")

        # for i in range(len(m_list)):
        # print("m=%.3f r=%.3f started\n" %(m_list[i], r_list[i]))
        # hopping = np.divide(-1, (m_list[i]*lat**2)) * unit_conv * pow(np.divide(lat, bond_len), r_list[i])
        # my_class = Ga2O3_Ga_s_Class(hopping, m_list[i], r_list[i])
        # error = my_class.getLSError(wannier_band, [0, 1, 2, 3])
        # outfile.writelines("%.3f\t%.3f\t%f\t%f\n" %(m_list[i], r_list[i], error, my_class.emass_0))
        # my_class.overlapPlot(wannier_band, [0, 1, 2, 3], 'wannier_pdf/Ga2O3_w_%.3fm_%.3fr.pdf' %(m_list[i], r_list[i]))
        # del my_class

m = 0.42
r = 2
hopping = np.divide(-1, (m*lat**2)) * unit_conv * pow(np.divide(lat, bond_len), r)

for i in range(4):
        for j in range(4):
                for k in range(4):
                        for l in range(4):
                                onsite = [-1-i, -1-j, -1-k, -1-l]
                                print("%d%d%d%d started\n" %(i, j, k, l))
                                my_class = Ga2O3_Ga_s_Class(onsite, hopping, m, r)
                                my_class.savePlot("pdf/%d%d%d%d" %(i, j, k, l))
