#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
bond_len = [3.040000, 3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 
            3.277766, 3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 
            3.327418, 3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 
            3.445864, 3.445864, 3.605804, 3.612253, 4.337574, 4.337574, 
            4.470498, 4.470498, 4.652131, 4.862916, 4.945890, 4.945890]
onsite = [-1, -1, -1, -1]

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

m = 0.55
r = 4
print("m=%.2f r-%.2f started\n" %(m, r))
hopping = np.divide(-1, (m*lat**2)) * unit_conv * pow(np.divide(lat, bond_len), r)
my_class = Ga2O3_Ga_s_Class(onsite, hopping, m, r)


fig1, ax1 = plt.subplots()
# specify horizontal axis details
# set range of horizontal axis
ax1.set_xlim(my_class.k_node[0], my_class.k_node[-1])
ax1.set_ylim(0, 12)
# put tickmarks and labels at node positions
ax1.set_xticks(my_class.k_node)
ax1.set_xticklabels(my_class.label)
# add vertical lines at node positions
for n in range(len(my_class.k_node)):
    ax1.axvline(x=my_class.k_node[n], linewidth=0.5, color='k')
# put title
ax1.set_xlabel("Path in k-space")
ax1.set_ylabel("Band energy (eV)")

band_num = range(4)
# plot bands specified by band_num
for i in band_num:
    if(i == 3):
        ax1.plot(my_class.k_dist, my_class.evals[i], 'b-', label='TB')
    else:
        ax1.plot(my_class.k_dist, my_class.evals[i], 'b-')
    if(i == 3):
        ax1.plot(my_class.k_dist, wannier_band[i], 'g--', label='DFT')
    else:
        ax1.plot(my_class.k_dist, wannier_band[i], 'g--')
ax1.legend()

# make an PDF figure of a plot
fig1.tight_layout()
fig1.savefig('200522/s_TB_wannier.png')
plt.close(fig1)

del my_class
