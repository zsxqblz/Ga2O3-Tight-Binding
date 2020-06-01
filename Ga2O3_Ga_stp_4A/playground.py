#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_stp_Class import *

lat = 3.293087727
ss_bond_len = [3.040000, 
    3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 3.277766, 
    3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 3.327418, 
    3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 3.445864, 
    3.445864, 3.605804, 3.612253, 
]
sp_len_cos = [
    [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000], 
    [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000], 
    [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, 1.000000], 
    [3.040000, 1.000000], [3.040000, 1.000000], [3.040000, 1.000000], 
    [3.109274, -0.673094], [3.109274, -0.673094], [3.109274, -0.554942], 
    [3.109274, -0.554942], [3.109274, -0.488860], [3.109274, -0.488860], 
    [3.109274, 0.488860], [3.109274, 0.488860], [3.109274, 0.554942], 
    [3.109274, 0.554942], [3.109274, 0.673094], [3.109274, 0.673094], 
    [3.277766, -0.982253], [3.277766, -0.187560], [3.277766, -0.000000], 
    [3.277766, -0.000000], [3.277766, 0.187560], [3.277766, 0.982253], 
    [3.277766, -0.982253], [3.277766, -0.187560], [3.277766, -0.000000], 
    [3.277766, -0.000000], [3.277766, 0.187560], [3.277766, 0.982253], 
    [3.300672, -0.886903], [3.300672, -0.886903], [3.300672, -0.460512], 
    [3.300672, -0.460512], [3.300672, -0.036498], [3.300672, -0.036498], 
    [3.300672, 0.036498], [3.300672, 0.036498], [3.300672, 0.460512], 
    [3.300672, 0.460512], [3.300672, 0.886903], [3.300672, 0.886903], 
    [3.300672, -0.886903], [3.300672, -0.886903], [3.300672, -0.460512], 
    [3.300672, -0.460512], [3.300672, -0.036498], [3.300672, -0.036498], 
    [3.300672, 0.036498], [3.300672, 0.036498], [3.300672, 0.460512], 
    [3.300672, 0.460512], [3.300672, 0.886903], [3.300672, 0.886903], 
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811], 
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761], 
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811], 
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165], 
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811], 
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761], 
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811], 
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165], 
    [3.445864, -0.785757], [3.445864, -0.785757], [3.445864, -0.785757], 
    [3.445864, -0.785757], [3.445864, -0.441109], [3.445864, -0.441109], 
    [3.445864, -0.441109], [3.445864, -0.441109], [3.445864, -0.433601], 
    [3.445864, -0.433601], [3.445864, -0.433601], [3.445864, -0.433601], 
    [3.445864, 0.433601], [3.445864, 0.433601], [3.445864, 0.433601], 
    [3.445864, 0.433601], [3.445864, 0.441109], [3.445864, 0.441109], 
    [3.445864, 0.441109], [3.445864, 0.441109], [3.445864, 0.785757], 
    [3.445864, 0.785757], [3.445864, 0.785757], [3.445864, 0.785757], 
    [3.605804, -0.921400], [3.605804, -0.388616], [3.605804, -0.000000], 
    [3.605804, -0.000000], [3.605804, 0.388616], [3.605804, 0.921400], 
    [3.612253, -0.768201], [3.612253, -0.640209], [3.612253, -0.000000], 
    [3.612253, -0.000000], [3.612253, 0.640209], [3.612253, 0.768201], 
]
pp_len_cos = [
    [3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000], 
    [3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, -0.000000, 0.000000], [3.040000, -0.000000, 0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], 
    [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], 
    [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], [3.040000, 1.000000, 0.000000], 
    [3.040000, 1.000000, 0.000000], [3.040000, 1.000000, 0.000000], [3.040000, 1.000000, 0.000000], 
    [3.109274, -0.329049, 0.329049], [3.109274, -0.329049, 0.329049], [3.109274, -0.271289, 0.271289], 
    [3.109274, -0.271289, 0.271289], [3.109274, 0.238984, 0.761016], [3.109274, 0.238984, 0.761016], 
    [3.109274, 0.271289, -0.271289], [3.109274, 0.271289, -0.271289], [3.109274, 0.307960, 0.692040], 
    [3.109274, 0.307960, 0.692040], [3.109274, 0.329049, -0.329049], [3.109274, 0.329049, -0.329049], 
    [3.109274, 0.373528, -0.373528], [3.109274, 0.373528, -0.373528], [3.109274, 0.373528, -0.373528], 
    [3.109274, 0.373528, -0.373528], [3.109274, 0.453055, 0.546945], [3.109274, 0.453055, 0.546945], 
    [3.277766, -0.184231, 0.184231], [3.277766, -0.184231, 0.184231], [3.277766, 0.000000, -0.000000], 
    [3.277766, -0.000000, 0.000000], [3.277766, -0.000000, 0.000000], [3.277766, 0.000000, -0.000000], 
    [3.277766, 0.000000, 1.000000], [3.277766, 0.035179, 0.964821], [3.277766, 0.964821, 0.035179], 
    [3.277766, -0.184231, 0.184231], [3.277766, -0.184231, 0.184231], [3.277766, -0.000000, 0.000000], 
    [3.277766, 0.000000, -0.000000], [3.277766, 0.000000, -0.000000], [3.277766, -0.000000, 0.000000], 
    [3.277766, 0.000000, 1.000000], [3.277766, 0.035179, 0.964821], [3.277766, 0.964821, 0.035179], 
    [3.300672, -0.408430, 0.408430], [3.300672, -0.408430, 0.408430], [3.300672, -0.016808, 0.016808], 
    [3.300672, -0.016808, 0.016808], [3.300672, 0.001332, 0.998668], [3.300672, 0.001332, 0.998668], 
    [3.300672, 0.016808, -0.016808], [3.300672, 0.016808, -0.016808], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.212072, 0.787928], [3.300672, 0.212072, 0.787928], [3.300672, 0.408430, -0.408430], 
    [3.300672, 0.408430, -0.408430], [3.300672, 0.786596, 0.213404], [3.300672, 0.786596, 0.213404], 
    [3.300672, -0.408430, 0.408430], [3.300672, -0.408430, 0.408430], [3.300672, -0.016808, 0.016808], 
    [3.300672, -0.016808, 0.016808], [3.300672, 0.001332, 0.998668], [3.300672, 0.001332, 0.998668], 
    [3.300672, 0.016808, -0.016808], [3.300672, 0.016808, -0.016808], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.212072, 0.787928], [3.300672, 0.212072, 0.787928], [3.300672, 0.408430, -0.408430], 
    [3.300672, 0.408430, -0.408430], [3.300672, 0.786596, 0.213404], [3.300672, 0.786596, 0.213404], 
    [3.327418, -0.397501, 0.397501], [3.327418, -0.397501, 0.397501], [3.327418, -0.084401, 0.084401], 
    [3.327418, -0.084401, 0.084401], [3.327418, 0.034137, 0.965863], [3.327418, 0.034137, 0.965863], 
    [3.327418, 0.084401, -0.084401], [3.327418, 0.084401, -0.084401], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.208676, 0.791324], [3.327418, 0.208676, 0.791324], [3.327418, 0.397501, -0.397501], 
    [3.327418, 0.397501, -0.397501], [3.327418, 0.757187, 0.242813], [3.327418, 0.757187, 0.242813], 
    [3.327418, -0.397501, 0.397501], [3.327418, -0.397501, 0.397501], [3.327418, -0.084401, 0.084401], 
    [3.327418, -0.084401, 0.084401], [3.327418, 0.034137, 0.965863], [3.327418, 0.034137, 0.965863], 
    [3.327418, 0.084401, -0.084401], [3.327418, 0.084401, -0.084401], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.208676, 0.791324], [3.327418, 0.208676, 0.791324], [3.327418, 0.397501, -0.397501], 
    [3.327418, 0.397501, -0.397501], [3.327418, 0.757187, 0.242813], [3.327418, 0.757187, 0.242813], 
    [3.445864, -0.346604, 0.346604], [3.445864, -0.346604, 0.346604], [3.445864, -0.346604, 0.346604], 
    [3.445864, -0.346604, 0.346604], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], 
    [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], 
    [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], 
    [3.445864, -0.191265, 0.191265], [3.445864, -0.191265, 0.191265], [3.445864, -0.191265, 0.191265], 
    [3.445864, -0.191265, 0.191265], [3.445864, 0.188010, 0.811990], [3.445864, 0.188010, 0.811990], 
    [3.445864, 0.188010, 0.811990], [3.445864, 0.188010, 0.811990], [3.445864, 0.191265, -0.191265], 
    [3.445864, 0.191265, -0.191265], [3.445864, 0.191265, -0.191265], [3.445864, 0.191265, -0.191265], 
    [3.445864, 0.194577, 0.805423], [3.445864, 0.194577, 0.805423], [3.445864, 0.194577, 0.805423], 
    [3.445864, 0.194577, 0.805423], [3.445864, 0.346604, -0.346604], [3.445864, 0.346604, -0.346604], 
    [3.445864, 0.346604, -0.346604], [3.445864, 0.346604, -0.346604], [3.445864, 0.617413, 0.382587], 
    [3.445864, 0.617413, 0.382587], [3.445864, 0.617413, 0.382587], [3.445864, 0.617413, 0.382587], 
    [3.605804, -0.000000, 0.000000], [3.605804, -0.000000, 0.000000], [3.605804, 0.000000, -0.000000], 
    [3.605804, 0.000000, -0.000000], [3.605804, 0.000000, 1.000000], [3.605804, 0.151022, 0.848978], 
    [3.605804, 0.358071, -0.358071], [3.605804, 0.358071, -0.358071], [3.605804, 0.848978, 0.151022], 
    [3.612253, -0.491809, 0.491809], [3.612253, -0.491809, 0.491809], [3.612253, -0.000000, 0.000000], 
    [3.612253, 0.000000, -0.000000], [3.612253, 0.000000, -0.000000], [3.612253, -0.000000, 0.000000], 
    [3.612253, 0.000000, 1.000000], [3.612253, 0.409868, 0.590132], [3.612253, 0.590132, 0.409868],
]

stp_len_cos = [
    [3.040000, -1.000000], [3.040000, -1.000000], [3.040000, -1.000000], 
    [3.040000, -1.000000], [3.040000, 0.000000], [3.040000, 0.000000], 
    [3.040000, 0.000000], [3.040000, 0.000000], [3.040000, 0.000000], 
    [3.040000, 0.000000], [3.040000, 0.000000], [3.040000, 0.000000], 
    [3.109274, -0.673094], [3.109274, -0.673094], [3.109274, -0.554942], 
    [3.109274, -0.554942], [3.109274, -0.488860], [3.109274, -0.488860], 
    [3.109274, 0.488860], [3.109274, 0.488860], [3.109274, 0.554942], 
    [3.109274, 0.554942], [3.109274, 0.673094], [3.109274, 0.673094], 
    [3.277766, -0.982253], [3.277766, -0.187560], [3.277766, 0.000000], 
    [3.277766, 0.000000], [3.277766, 0.187560], [3.277766, 0.982253], 
    [3.277766, -0.982253], [3.277766, -0.187560], [3.277766, 0.000000], 
    [3.277766, 0.000000], [3.277766, 0.187560], [3.277766, 0.982253], 
    [3.300672, -0.886903], [3.300672, -0.886903], [3.300672, -0.460512], 
    [3.300672, -0.460512], [3.300672, -0.036498], [3.300672, -0.036498], 
    [3.300672, 0.036498], [3.300672, 0.036498], [3.300672, 0.460512], 
    [3.300672, 0.460512], [3.300672, 0.886903], [3.300672, 0.886903], 
    [3.300672, -0.886903], [3.300672, -0.886903], [3.300672, -0.460512], 
    [3.300672, -0.460512], [3.300672, -0.036498], [3.300672, -0.036498], 
    [3.300672, 0.036498], [3.300672, 0.036498], [3.300672, 0.460512], 
    [3.300672, 0.460512], [3.300672, 0.886903], [3.300672, 0.886903], 
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811], 
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761], 
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811], 
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165], 
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811], 
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761], 
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811], 
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165], 
    [3.445864, -0.785757], [3.445864, -0.785757], [3.445864, -0.785757], 
    [3.445864, -0.785757], [3.445864, -0.441109], [3.445864, -0.441109], 
    [3.445864, -0.441109], [3.445864, -0.441109], [3.445864, -0.433601], 
    [3.445864, -0.433601], [3.445864, -0.433601], [3.445864, -0.433601], 
    [3.445864, 0.433601], [3.445864, 0.433601], [3.445864, 0.433601], 
    [3.445864, 0.433601], [3.445864, 0.441109], [3.445864, 0.441109], 
    [3.445864, 0.441109], [3.445864, 0.441109], [3.445864, 0.785757], 
    [3.445864, 0.785757], [3.445864, 0.785757], [3.445864, 0.785757], 
    [3.605804, -0.921400], [3.605804, -0.388616], [3.605804, 0.000000], 
    [3.605804, 0.000000], [3.605804, 0.388616], [3.605804, 0.921400], 
    [3.612253, -0.768201], [3.612253, -0.640209], [3.612253, 0.000000], 
    [3.612253, 0.000000], [3.612253, 0.640209], [3.612253, 0.768201],
]
# convert to numpy array for tuple slicing
ss_bond_len = np.array(ss_bond_len)
sp_len_cos = np.array(sp_len_cos)
pp_len_cos = np.array(pp_len_cos)
stp_len_cos = np.array(stp_len_cos)

# s: -1eV, p: 1eV, st: 3eV. The value is set arbitarily 
# here but follwing the incresing physical trend.
s_onite = 0
p_onsite = 2
st_onsite = 4
onsite = [
    s_onite, s_onite, s_onite, s_onite, 
    p_onsite, p_onsite, p_onsite, p_onsite,
    p_onsite, p_onsite, p_onsite, p_onsite,
    p_onsite, p_onsite, p_onsite, p_onsite,
    st_onsite, st_onsite, st_onsite, st_onsite]
hopping = np.zeros((448,))
unit_conv = 3.80998212 # hbar/2mA to eV

wannier_band_raw = np.loadtxt('wannier_band.dat', dtype=float)
# shift minimum to 0
wannier_band_raw[:,1] = wannier_band_raw[:,1] - np.min(wannier_band_raw[:,1])
wannier_band = []
# store different bands into different dimensions
wannier_band.append(wannier_band_raw[0:532,1])
wannier_band.append(wannier_band_raw[533:1065,1])
wannier_band.append(wannier_band_raw[1066:1598,1])
wannier_band.append(wannier_band_raw[1599:2131,1])


# not too much different
# vss = -0.6
# vsp = 0.4
# vpp_s = 1.15
# vpp_p = 0.2
# vstp = 1.2
# n = 4

# currently the best 0 2 4
vss = -0.65
vsp = 0.3
vpp_s = 0.5
vpp_p = 0.1
vstp = 1.5
n = 4

# better one 0 2 4
# vss = -0.63
# vsp = 0.58
# vpp_s = 1.5
# vpp_p = 0.3
# vstp = 1
# n = 4

# roughly good one 0 5 7
# vss = -0.7
# vsp = 0.7
# vpp_s = 2
# vpp_p = 0.3
# vstp = 1.8
# n = 4

print("vss=%.2f vsp=%.2f vpp_s=%.2f vpp_p=%.2f vstp=%.2f started\n" %(vss, vsp, vpp_s, vpp_p, vstp))
# calculate ss hopping
hopping[0:22] = vss * pow(np.divide(lat, ss_bond_len), n)
# calculate sp hopping
hopping[22:142] = vsp * sp_len_cos[:,1] * pow(np.divide(lat, sp_len_cos[:,0]), n)
# calculate pp hopping
hopping[142:328] = (vpp_s * pp_len_cos[:,1] + vpp_p * pp_len_cos[:,2]) * pow(np.divide(lat, pp_len_cos[:,0]), n)
# calculate stp hopping
hopping[328:448] = vstp * stp_len_cos[:,1] * pow(np.divide(lat, stp_len_cos[:,0]), n)

my_class = Ga2O3_Ga_s_Class(onsite, hopping)
# outfile.writelines(my_class.getMassString())
print(my_class.getMassString())
# my_class.savePlot('%.3f_%.3f_%.3f_%.3f.pdf' %(vss, vsp, vpp_s, vpp_p))
my_class.overlapPlot(wannier_band, [0, 1, 2, 3], 'playground/w_%.3f_%.3f_%.3f_%.3f_%.3f.pdf' %(vss, vsp, vpp_s, vpp_p, vstp))
my_class.savePlot('playground/%.3f_%.3f_%.3f_%.3f_%.3f.pdf' %(vss, vsp, vpp_s, vpp_p, vstp))
my_class.save4BandPlot('playground/4_%.3f_%.3f_%.3f_%.3f_%.3f.pdf' %(vss, vsp, vpp_s, vpp_p, vstp))
# if(np.abs(my_class.emass_0 - target_emass) < threshold):
#     good_scan_outfile.writelines(my_class.getMassString())
del my_class