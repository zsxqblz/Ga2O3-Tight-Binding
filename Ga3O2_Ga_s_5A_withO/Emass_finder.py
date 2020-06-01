#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len_no_O = [3.040000, 3.040000, 3.040000, 3.040000, 3.109274, 
                3.327418, 3.327418, 3.605804, 4.337574, 4.337574, 
                4.470498, 4.716293, 4.716293, 4.721225, 4.862916, 
                4.945890, 4.945890]
bond_len_has_O = [3.109274, 3.277766, 3.277766, 3.300672, 3.300672, 
                3.300672, 3.300672, 3.327418, 3.327418, 3.445864, 
                3.445864, 3.445864, 3.445864, 3.612253, 4.470498, 
                4.652131, 4.721225]
onsite = [-1, -1, -1, -1]

unit_conv = 3.80998212 # hbar/2mA to eV

# m1 r1 are for no oxygen
m1_start = 0.4
m1_end = 0.8
m1_step = 0.02

r1_start = 0
r1_end = 5
r1_step = 0.5

# m2 r2 are for has oxygen
m2_start = 0.4
m2_end = 0.8
m2_step = 0.02

r2_start = 0
r2_end = 5
r2_step = 0.5

m1_list = np.arange(m1_start, m1_end, m1_step)
r1_list = np.arange(r1_start, r1_end, r1_step)
m2_list = np.arange(m2_start, m2_end, m2_step)
r2_list = np.arange(r2_start, r2_end, r2_step)

scan_outfile = open("Emass_scan.csv", "w+")
good_scan_outfile = open("good_Emass_scan.csv", "w+")

target_emass = 0.31
threshold = 0.005

for m1 in m1_list:
    for r1 in r1_list:
        for m2 in m2_list:
            for r2 in r2_list:
                print("m1=%.3f r1-%.3f m2=%.3f r2-%.3f started\n" %(m1, r1, m2, r2))
                hopping_no_O = np.divide(-1, (m1*lat**2)) * unit_conv * pow(np.divide(lat, bond_len_no_O), r1)
                hopping_has_O = np.divide(-1, (m2*lat**2)) * unit_conv * pow(np.divide(lat, bond_len_has_O), r2)
                hopping = np.concatenate([hopping_no_O, hopping_has_O])
                my_class = Ga2O3_Ga_s_Class(onsite, hopping, m1, r1, m2, r2)

                scan_outfile.writelines(my_class.getMassString())
                my_class.savePlot('Emass_pdf/%.3f_%.3f_%.3f_%.3f.pdf' %(m1, r1, m2, r2))

                if(np.abs(my_class.emass_0 - target_emass) < threshold):
                    good_scan_outfile.writelines(my_class.getMassString())
                del my_class
