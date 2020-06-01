#!/usr/bin/env python

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import *  # import TB model class
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

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

        # call function k_path to construct the actual path
        (k_vec, k_dist, k_node) = my_model.k_path(path, nk, report = False)

        # print('---------------------------------------')
        # print('starting calculation')
        # print('---------------------------------------')
        # print('Calculating bands...')

        # obtain eigenvalues to be plotted
        evals = my_model.solve_all(k_vec)

        # figure for bandstructure

        fig, ax = plt.subplots()
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(k_node[0], k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        # add vertical lines at node positions
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k')
        # put title
        ax.set_title("Ga2O3 band structure")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy (eV)")

        # plot first and second band
        ax.plot(k_dist, evals[0])
        ax.plot(k_dist, evals[1])
        ax.plot(k_dist, evals[2])
        ax.plot(k_dist, evals[3])

        # make an PDF figure of a plot
        fig.tight_layout()
        fig.savefig("pdf/Ga2O3_%.2fm_%.2fr.pdf" %(self.m, self.r))
        plt.close(fig)

        # save the raw data
        outdata = []
        outdata.append(k_dist)
        outdata.append(evals)        
        np.save('data/Ga2O3_%.2fm_%.2fr' %(self.m, self.r), outdata)
        ######### calculate the effective mass ##########


        def findBMass(E, k_norm, start, end):
            """
            return the band mass given the energy and kpath
            by calculating the 2nd derivative between start and end, including themselves
            parameters:
            E: the energy band in eV
            k: the kpath in 1/angstrom
            start: the index where the derivative starts
            endL the index where the derivative ends
            return:
            the band mass given in the unit of electron mass
            """
            k_norm = np.multiply(k_norm, 6.28e10)  # 1/Ang to 1/m and add a 2pi
            E = E*1.60218e-19  # eV to J
            hbar = 1.05457e-34  # reduced planck constant
            me = 9.1094e-31  # electron mass
            deriv1 = np.divide(np.gradient(
                E[start:end+1]), np.gradient(k_norm[start:end+1]))
            deriv2 = np.divide(np.gradient(deriv1), np.gradient(k_norm[start:end+1]))
            return hbar**2/np.mean(deriv2)/me

        # find out where are the k-points in k_dist
        Gamma1 = np.where(k_dist == k_node[1])[0][0] # the 1st Gamma point
        Gamma2 = np.where(k_dist == k_node[3])[0][0] # the 2nd Gamma point
        interval = 10 # interval length for calculating the derivative

        # calculate band mass & effective mass for all bands
        self.bmass_0_f = findBMass(evals[0], k_dist, Gamma1-interval, Gamma1)
        self.bmass_0_t = findBMass(evals[0], k_dist, Gamma1, Gamma1+interval)
        self.bmass_0_l = findBMass(evals[0], k_dist, Gamma2, Gamma2+interval)
        self.emass_0 = np.power(self.bmass_0_f*self.bmass_0_t*self.bmass_0_l, 1/3)

        self.bmass_1_f = findBMass(evals[1], k_dist, Gamma1-interval, Gamma1)
        self.bmass_1_t = findBMass(evals[1], k_dist, Gamma1, Gamma1+interval)
        self.bmass_1_l = findBMass(evals[1], k_dist, Gamma2, Gamma2+interval)
        self.emass_1 = np.power(self.bmass_1_f*self.bmass_1_t*self.bmass_1_l, 1/3)

        self.bmass_2_f = findBMass(evals[2], k_dist, Gamma1-interval, Gamma1)
        self.bmass_2_t = findBMass(evals[2], k_dist, Gamma1, Gamma1+interval)
        self.bmass_2_l = findBMass(evals[2], k_dist, Gamma2, Gamma2+interval)
        self.emass_2 = np.power(self.bmass_2_f*self.bmass_2_t*self.bmass_2_l, 1/3)

        self.bmass_3_f = findBMass(evals[3], k_dist, Gamma1-interval, Gamma1)
        self.bmass_3_t = findBMass(evals[3], k_dist, Gamma1, Gamma1+interval)
        self.bmass_3_l = findBMass(evals[3], k_dist, Gamma2, Gamma2+interval)
        self.emass_3 = np.power(self.bmass_3_f*self.bmass_3_t*self.bmass_3_l, 1/3)


    def getMassString(self):
        return ('%.2f\t%.2f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' 
        %(self.m, self.r, 
          self.bmass_0_f, self.bmass_0_t, self.bmass_0_l, self.emass_0,
          self.bmass_1_f, self.bmass_1_t, self.bmass_1_l, self.emass_1,
          self.bmass_2_f, self.bmass_2_t, self.bmass_2_l, self.emass_2,
          self.bmass_3_f, self.bmass_3_t, self.bmass_3_l, self.emass_3))
        