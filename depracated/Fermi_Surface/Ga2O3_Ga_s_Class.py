#!/usr/bin/env python

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import *  # import TB model class
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Ga2O3_Ga_s_Class:
    def __init__(self, hopping, m, r):
        self.hopping = hopping
        self.m = m
        self.r = r
    
        # define lattice vectors
        lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
            [-6.1149997711,1.5199999809, 0.0000000000], 
            [-1.3736609922, 0.0000000000, 5.6349851545]]
        # define coordinates of orbitals
        orb = [[0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805]]

        # make three-dimensional tight-binding model
        my_model = tb_model(3, 3, lat, orb)

        # set model parameters
        delta = -1.00
        # set on-site energies
        my_model.set_onsite([delta, delta, delta, delta])
        # set hoppings (one for each connected pair of orbitals)
        # (amplitude, i, j, [lattice vector to cell containing j])
        my_model.set_hop(hopping[0],0,0,[1,1,0])
        my_model.set_hop(hopping[1],1,1,[1,1,0])
        my_model.set_hop(hopping[2],2,2,[1,1,0])
        my_model.set_hop(hopping[3],3,3,[1,1,0])
        my_model.set_hop(hopping[4],0,1,[-1,0,0])
        my_model.set_hop(hopping[5],0,1,[-1,0,-1])
        my_model.set_hop(hopping[6],0,2,[0,0,-1])
        my_model.set_hop(hopping[7],0,2,[-1,-1,-1])
        my_model.set_hop(hopping[8],0,2,[0,-1,-1])
        my_model.set_hop(hopping[9],0,3,[0,0,0])
        my_model.set_hop(hopping[10],0,3,[0,0,-1])
        my_model.set_hop(hopping[11],0,3,[-1,-1,0])
        my_model.set_hop(hopping[12],0,3,[-1,-1,-1])
        my_model.set_hop(hopping[13],1,2,[1,0,0])
        my_model.set_hop(hopping[14],1,2,[0,-1,0])
        my_model.set_hop(hopping[15],1,2,[0,-1,-1])
        my_model.set_hop(hopping[16],1,2,[1,0,-1])
        my_model.set_hop(hopping[17],1,3,[0,0,0])
        my_model.set_hop(hopping[18],1,3,[1,0,0])
        my_model.set_hop(hopping[19],1,3,[0,-1,0])
        my_model.set_hop(hopping[20],2,3,[0,1,0])
        my_model.set_hop(hopping[21],2,3,[-1,0,0])


        # print tight-binding model
        # my_model.display()

        # generate list of k-points following a segmented path in the BZ
        # list of nodes (high-symmetry points) that will be connected
        path = [[0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [
            0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        # labels of the nodes
        label = (r'$F $', r'$\Gamma $', r'$T $', r'$\Gamma $', r'$L $')
        # total number of interpolated k-points along the path
        nk = 533

        self.k_vec = []
        # sample inside the FBZ
        for i in np.arange(-0.5, 0.5, 0.02):
            for j in np.arange(-0.5, 0.5, 0.02):
                for k in np.arange(-0.5, 0.5, 0.02):
                    inFBZ = 0
                    if (np.sqrt(i**2 + j**2 + k**2) > np.sqrt(0.5**2 + 0.5**2)):# L[0 0.5 0.5]
                        inFBZ = inFBZ + 1
                    if (np.sqrt(i**2 + j**2 + k**2) > np.sqrt(0.5**2 + 0.5**2)):# Y[0.5 0.5 0]
                        inFBZ = inFBZ + 1
                    if (np.sqrt(i**2 + j**2 + k**2) > np.sqrt(0.5**2 + 0.5**2 + 0.5**2)):# M[0.5 0.5 0.5]
                        inFBZ = inFBZ + 1
                    if(inFBZ == 0):
                        self.k_vec.append([i, j, k])

        # print('---------------------------------------')
        # print('starting calculation')
        # print('---------------------------------------')
        # print('Calculating bands...')

        # obtain eigenvalues to be plotted
        self.evals = my_model.solve_all(self.k_vec)

        # save the raw data
        outdata = []
        outdata.append(self.k_vec)
        outdata.append(self.evals)
        np.save('data/Ga2O3_%.2fm_%.2fr' %(self.m, self.r), outdata)

    def plotFermiSurf(self, fermi_level, real_unit=False):
        # Assume fermi level is at the middle of the bandgap
        threshold = 0.1
        # reciprocal lattice vector as column
        rec_mat = np.array([[0.5137518841, -0.5137518841, 0.0],
                            [2.0668372980, 2.0668372980, 0.0000000000],
                            [0.1252391805, 0.1252391805, 1.1150313860]])

        min_Xarr = []
        min_Yarr = []
        min_Zarr = []
        max_Xarr = []
        max_Yarr = []
        max_Zarr = []

        current_x = 0
        current_y = 0
        min_z = 100
        max_z = -100
        min_indice = -1
        max_indice = -1
        for i in range(np.shape(self.k_vec)[0]):
            if(self.k_vec[i][0] != current_x or self.k_vec[i][1] != current_y):
                if(min_indice > 0 and max_indice > 0):
                    # convert unit to 1/A
                    if(real_unit):
                        min_vec = np.matmul(rec_mat, self.k_vec[min_indice])
                        max_vec = np.matmul(rec_mat, self.k_vec[max_indice])
                    else:
                        min_vec = self.k_vec[min_indice]
                        max_vec = self.k_vec[max_indice]
                    # store the maximum and minimum
                    min_Xarr.append(min_vec[0])
                    min_Yarr.append(min_vec[1])
                    min_Zarr.append(min_vec[2])
                    max_Xarr.append(max_vec[0])
                    max_Yarr.append(max_vec[1])
                    max_Zarr.append(max_vec[2])
                # reset stored value
                current_x = self.k_vec[i][0]
                current_y = self.k_vec[i][1]
                min_z = 100
                max_z = -100
                min_indice = 0
                max_indice = 0
            
            if(np.abs(self.evals[0][i] - fermi_level) < threshold):
                if(self.k_vec[i][2] < min_z):
                    min_z = self.k_vec[i][2]
                    min_indice = i
                if(self.k_vec[i][2] > max_z):
                    max_z = self.k_vec[i][2]
                    max_indice = i
            
        # make into numpy array
        X_arr = np.array(np.concatenate((min_Xarr, max_Xarr[::-1]), axis = 0))
        Y_arr = np.array(np.concatenate((min_Yarr, max_Yarr[::-1]), axis = 0))
        Z_arr = np.array(np.concatenate((min_Zarr, max_Zarr[::-1]), axis = 0))

        

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax = fig.gca(projection='3d')
        if(np.shape(min_Xarr)[0]>3):
            surf_b = ax.plot_trisurf(min_Xarr, min_Yarr, min_Zarr, color = 'blue')
            surf_t = ax.plot_trisurf(max_Xarr, max_Yarr, max_Zarr, color = 'blue')
        ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
        fig.tight_layout()
        fig.savefig("FS_pdf/Ga2O3_%.2fm_%.2fr_%.3fFE.png" %(self.m, self.r, fermi_level), format='png')
        plt.close(fig)
