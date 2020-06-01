#!/usr/bin/env python

from __future__ import division
import numpy as np
import csv
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len_no_O = [3.040000, 3.040000, 3.040000, 3.040000, 3.109274, 
                3.327418, 3.327418, 3.605804, 4.337574, 4.337574, 
                4.470498, 4.716293, 4.716293, 4.721225, 4.862916, 
                4.945890, 4.945890, 5.305731, 5.355706, 5.355706, 
                5.420114, 5.436443, 5.509735, 5.557331, 5.557331, 
                5.734942, 5.734942, 5.800000, 5.800000, 5.800000, 
                5.800000, 5.963881]
bond_len_has_O = [3.109274, 3.277766, 3.277766, 3.300672, 3.300672, 
                3.300672, 3.300672, 3.327418, 3.327418, 3.445864, 
                3.445864, 3.445864, 3.612253, 4.470498, 4.652131, 
                4.721225, 5.348829, 5.348829, 5.420114, 5.420114, 
                5.420114, 5.436443, 5.436443, 5.436443, 5.509735, 
                5.509735, 5.963881]
onsite = [-1, -1, -1, -1]

unit_conv = 3.80998212 # hbar/2mA to eV

t_r_data = np.loadtxt('good_Emass_scan.csv', dtype=float)
t1_list = t_r_data[:,0]
r1_list = t_r_data[:,1]
t2_list = t_r_data[:,2]
r2_list = t_r_data[:,3]

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

for i in range(len(t1_list)):
    print("t1=%.3f r1-%.3f t2=%.3f r2-%.3f started\n" %(t1_list[i], r1_list[i], t2_list[i], r2_list[i]))
    hopping_no_O = t1_list[i] * pow(np.divide(lat, bond_len_no_O), r1_list[i])
    hopping_has_O = t2_list[i] * pow(np.divide(lat, bond_len_has_O), r2_list[i])
    hopping = np.concatenate([hopping_no_O, hopping_has_O])
    my_class = Ga2O3_Ga_s_Class(onsite, hopping, t1_list[i], r1_list[i], t2_list[i], r2_list[i])
    error = my_class.getLSError(wannier_band, [0, 1, 2, 3])
    outfile.writelines("%.3f\t%.3f\t%.3f\t%.3f\t%f\t%f\n" %(t1_list[i], r1_list[i], t2_list[i], r2_list[i], error, my_class.emass_0))
    my_class.savePlot('LS_pdf/%.3fm1_%.3fr1_%.3fm2_%.3fr2.pdf' %(t1_list[i], r1_list[i], t2_list[i], r2_list[i]))
    my_class.overlapPlot(wannier_band, [0, 1, 2, 3], 'wannier_pdf/%.3fm1_%.3fr1_%.3fm2_%.3fr2.pdf' %(t1_list[i], r1_list[i], t2_list[i], r2_list[i]))
    del my_class