#!/usr/bin/env python

# Toy graphene model

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat=[[ 6.1149997711, 1.5199999809, 0.0000000000]
    ,[-6.1149997711, 1.5199999809, 0.0000000000]
    ,[-1.3736609922, 0.0000000000, 5.6349851545]]
# define coordinates of orbitals
#orb=[[0.090417297,-0.090417297, 0.794989099],[0.909582703, 0.090417297, 0.205010901],[0.658542076, 0.341457924, 0.313736601],[0.341457924,-0.341457924, 0.686263399]]
orb=[[0.0000, 0.0000, 0.0000]
    ,[0.8192,-0.8192,-0.5896]
    ,[0.2510,-0.2510,-0.1091]
    ,[0.5682,-0.5682,-0.4805]]
# make three-dimensional tight-binding graphene model
my_model=tb_model(3,3,lat,orb)

# set model parameters
delta =  -1.00
# set on-site energies
my_model.set_onsite([delta,delta,delta,delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(-1.37420,0,0,[1,1,0])
my_model.set_hop(-1.37420,1,1,[1,1,0])
my_model.set_hop(-1.37420,2,2,[1,1,0])
my_model.set_hop(-1.37420,3,3,[1,1,0])
my_model.set_hop(-0.97330,0,1,[-1,0,0])
my_model.set_hop(-0.97678,0,1,[-1,0,-1])
my_model.set_hop(-1.14710,0,2,[0,0,-1])
my_model.set_hop(-1.14710,0,2,[-1,-1,-1])
my_model.set_hop(-1.18210,0,2,[0,-1,-1])
my_model.set_hop(-1.16570,0,3,[0,0,0])
my_model.set_hop(-1.06960,0,3,[0,0,-1])
my_model.set_hop(-1.16570,0,3,[-1,-1,0])
my_model.set_hop(-1.06960,0,3,[-1,-1,-1])
my_model.set_hop(-1.06960,1,2,[1,0,0])
my_model.set_hop(-1.06960,1,2,[0,-1,0])
my_model.set_hop(-1.16570,1,2,[0,-1,-1])
my_model.set_hop(-1.16570,1,2,[1,0,-1])
my_model.set_hop(-1.18210,1,3,[0,0,0])
my_model.set_hop(-1.14710,1,3,[1,0,0])
my_model.set_hop(-1.14710,1,3,[0,-1,0])
my_model.set_hop(-1.31370,2,3,[0,1,0])
my_model.set_hop(-1.31370,2,3,[-1,0,0])

# print tight-binding model
my_model.display()
    
# generate list of k-points following a segmented path in the BZ
# list of nodes (high-symmetry points) that will be connected
path=[[0.5,0.0,0.0],[0.0,0.0,0.0],[0.5,0.0,0.0]]
# labels of the nodes
label=(r'$X $',r'$\Gamma $', r'$X $')
# total number of interpolated k-points along the path
nk=41

# call function k_path to construct the actual path
(k_vec,k_dist,k_node)=my_model.k_path(path,nk)
# inputs:
#   path, nk: see above
#   my_model: the pythtb model
# outputs:
#   k_vec: list of interpolated k-points
#   k_dist: horizontal axis position of each k-point in the list
#   k_node: horizontal axis position of each original node

print('---------------------------------------')
print('starting calculation')
print('---------------------------------------')
print('Calculating bands...')

# obtain eigenvalues to be plotted
evals=my_model.solve_all(k_vec)

# figure for bandstructure

fig, ax = plt.subplots()
# specify horizontal axis details
# set range of horizontal axis
ax.set_xlim(k_node[0],k_node[-1])
# put tickmarks and labels at node positions
ax.set_xticks(k_node)
ax.set_xticklabels(label)
# add vertical lines at node positions
for n in range(len(k_node)):
  ax.axvline(x=k_node[n],linewidth=0.5, color='k')
# put title
ax.set_title("Ga2O3 band structure")
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy (eV)")

# plot first and second band
ax.plot(k_dist,evals[0])
ax.plot(k_dist,evals[1])
ax.plot(k_dist,evals[2])
ax.plot(k_dist,evals[3])

# make an PDF figure of a plot
fig.tight_layout()
fig.savefig("Ga2O3.pdf")

print('Done.\n')
