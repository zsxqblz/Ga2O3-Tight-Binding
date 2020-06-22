#!/usr/bin/env python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
    [-6.1149997711,1.5199999809, 0.0000000000], 
    [-1.3736609922, 0.0000000000, 5.6349851545]]

# import E(k) data
evals = np.load('evals.npy')
k_vec = np.load('k_vec.npy')

# total number of points
npoints = len(k_vec)
# 20x volume of the 1BZ is the total state density
# 2pi is already left off in our k definition
# the unit here is cm-3
deltak = 0.02
total_states = (npoints * (deltak*1e10/1e2)**3 
    * np.abs(np.dot(np.cross(lat[0], lat[1]), lat[2])) * 20)

# step of numerical DOS
deltaE = 0.05
# calculate ODS
E = np.arange(0, np.amax(evals), deltaE)
DOS = np.zeros(E.shape)
for band in evals:
    for e in band:
        i = int(e / deltaE)
        DOS[i] = DOS[i] + 1

# use cm-3eV-1 unit
# DOS = DOS / npoints * total_states
# use eV-1 in primitive cell
DOS = DOS / np.sum(DOS) / deltaE * 20
print(np.sum(DOS*np.diff(E, prepend=0)))
print(len(DOS))

fig, ax = plt.subplots()
ax.set_xlim(0, np.max(E))
ax.plot(E, DOS)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("DOS (cm-3*eV-1)")
fig.tight_layout()
fig.savefig('opt_Emass_DOS.png')
plt.close(fig)

dft_DOS_raw = np.genfromtxt('beta_Ga2O3.dos', skip_header=1)
dft_E = dft_DOS_raw[:,0]
dft_DOS = dft_DOS_raw[:,1]
dft_int_DOS = dft_DOS_raw[:,2]


# obtained from wannier_band.dat
mcb = 11.728068
# get the index of the lowest energy of the conduction band
mcb_i = np.argmin(np.abs(dft_E-mcb))
# cut the valence band off
dft_E = dft_E[mcb_i:]
dft_E = dft_E - mcb
dft_DOS = dft_DOS[mcb_i:]
dft_int_DOS = dft_int_DOS[mcb_i:]

print(np.sum(dft_DOS*np.diff(dft_E, prepend=0)))
print(len(dft_DOS))


fig, ax = plt.subplots()
ax.plot(E, DOS)
ax.set_xlim(0, 8)
ax.set_xlabel("Energy (eV)")
# ax.set_ylabel("DOS (cm-3*eV-1)")
ax.set_ylabel("DOS (arb. unit)")
fig.tight_layout()
fig.savefig('opt_Emass_DOS.png')
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(dft_E, dft_DOS)
ax.set_xlim(0, 8)
ax.set_xlabel("Energy (eV)")
# ax.set_ylabel("DOS (cm-3*eV-1)")
ax.set_ylabel("DOS (arb. unit)")
fig.tight_layout()
fig.savefig('dft_DOS.png')
plt.close(fig)

# for now int_DOS is garbage
# fig, ax = plt.subplots()
# ax.set_xlim(np.min(dft_E), np.max(dft_E))
# ax.plot(dft_E, dft_int_DOS)
# ax.set_xlabel("Energy (eV)")
# # ax.set_ylabel("DOS (cm-3*eV-1)")
# ax.set_ylabel("DOS (arb. unit)")
# fig.tight_layout()
# fig.savefig('dft_int_DOS.png')
# plt.close(fig)