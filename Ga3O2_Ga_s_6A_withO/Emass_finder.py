#!/usr/bin/env python

from __future__ import division
import numpy as np
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

# # m1 r1 are for no oxygen
# m1_start = -1
# m1_end = 1
# m1_step = 0.1

# # m2 r2 are for has oxygen
# m2_start = -1
# m2_end = 1
# m2_step = 0.1

t1_start = -4
t1_end = 4
t1_step = 0.25

t2_start = -4
t2_end = 4
t2_step = 0.25

r_start = 2
r_end = 6
r_step = 1

# m1_list = np.arange(m1_start, m1_end, m1_step)
# m2_list = np.arange(m2_start, m2_end, m2_step)
t1_list = np.arange(t1_start, t1_end, t1_step)
t2_list = np.arange(t2_start, t2_end, t2_step)
r_list = np.arange(r_start, r_end, r_step)


scan_outfile = open("Emass_scan.csv", "w+")
good_scan_outfile = open("good_Emass_scan.csv", "w+")

target_emass = 0.31
threshold = 0.01

for t1 in t1_list:
    for t2 in t2_list:
        for r in r_list:
            print("t1=%.3f r1-%.3f t2=%.3f r2-%.3f started\n" %(t1, r, t2, r))
            # hopping_no_O = np.divide(-1, (m1*lat**2)) * unit_conv * pow(np.divide(lat, bond_len_no_O), r)
            # hopping_has_O = np.divide(-1, (m2*lat**2)) * unit_conv * pow(np.divide(lat, bond_len_has_O), r)
            hopping_no_O = t1 * pow(np.divide(lat, bond_len_no_O), r)
            hopping_has_O = t2 * pow(np.divide(lat, bond_len_has_O), r)

            hopping = np.concatenate([hopping_no_O, hopping_has_O])
            my_class = Ga2O3_Ga_s_Class(onsite, hopping, t1, r, t2, r)

            scan_outfile.writelines(my_class.getMassString())
            my_class.savePlot('Emass_pdf/%.3f_%.3f_%.3f_%.3f.pdf' %(t1, r, t2, r))

            if(np.abs(my_class.emass_0 - target_emass) < threshold):
                good_scan_outfile.writelines(my_class.getMassString())
            del my_class
