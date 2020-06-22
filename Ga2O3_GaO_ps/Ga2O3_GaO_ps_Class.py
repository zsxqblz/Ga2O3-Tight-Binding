#!/usr/bin/env python

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb_eff import *  # import TB model class
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Ga2O3_Class:

    def __init__(self, onsite, hopping):
        self.onsite = onsite
        self.hopping = hopping
    
        # define lattice vectors
        lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
            [-6.1149997711,1.5199999809, 0.0000000000], 
            [-1.3736609922, 0.0000000000, 5.6349851545]]
        # define coordinates of orbitals
        # the four groups are s, px, py, pz in order
        orb = [
            [0.9096000000, 	-0.9096000000, 	0.2052000000], 
            [1.0904000000, 	-0.0904000000, 	0.7948000000],
            [0.6586000000, 	-0.6586000000, 	0.3143000000], 
            [1.3414000000, 	-0.3414000000, 	0.6857000000],
            [0.8326000000, 	-0.8326000000, 	0.8989000000],
            [1.1674000000, 	-0.1674000000, 	0.1011000000], 
            [0.5043000000, 	-0.5043000000, 	0.7447000000],
            [1.4957000000, 	-0.4957000000, 	0.2553000000],
            [1.1721000000, 	-0.1721000000, 	0.5635000000], 
            [0.8279000000, 	-0.8279000000, 	0.4365000000],
            [0.8326000000, 	-0.8326000000, 	0.8989000000],
            [1.1674000000, 	-0.1674000000, 	0.1011000000], 
            [0.5043000000, 	-0.5043000000, 	0.7447000000],
            [1.4957000000, 	-0.4957000000, 	0.2553000000],
            [1.1721000000, 	-0.1721000000, 	0.5635000000], 
            [0.8279000000, 	-0.8279000000, 	0.4365000000],
            [0.8326000000, 	-0.8326000000, 	0.8989000000],
            [1.1674000000, 	-0.1674000000, 	0.1011000000], 
            [0.5043000000, 	-0.5043000000, 	0.7447000000],
            [1.4957000000, 	-0.4957000000, 	0.2553000000],
            [1.1721000000, 	-0.1721000000, 	0.5635000000], 
            [0.8279000000, 	-0.8279000000, 	0.4365000000]
            ]


        # make three-dimensional tight-binding model
        self.my_model = tb_model(3, 3, lat, orb)

        # set on-site energies
        self.my_model.set_onsite(onsite)
        # set hoppings (one for each connected pair of orbitals)
        # (amplitude, i, j, [lattice vector to cell containing j])
        #region
        # ss
        self.my_model.set_hop(hopping[0],0,0,[-1,-1,0])
        self.my_model.set_hop(hopping[1],1,1,[-1,-1,0])
        self.my_model.set_hop(hopping[2],2,2,[-1,-1,0])
        self.my_model.set_hop(hopping[3],3,3,[-1,-1,0])
        self.my_model.set_hop(hopping[4],2,3,[-1,-1,0])
        self.my_model.set_hop(hopping[5],2,3,[0,0,0])
        self.my_model.set_hop(hopping[6],1,3,[0,0,0])
        self.my_model.set_hop(hopping[7],0,2,[0,0,0])
        self.my_model.set_hop(hopping[8],0,3,[-1,-1,-1])
        self.my_model.set_hop(hopping[9],0,3,[0,0,-1])
        self.my_model.set_hop(hopping[10],1,2,[0,0,1])
        self.my_model.set_hop(hopping[11],1,2,[1,1,1])
        self.my_model.set_hop(hopping[12],0,2,[0,-1,0])
        self.my_model.set_hop(hopping[13],0,2,[1,0,0])
        self.my_model.set_hop(hopping[14],1,3,[-1,0,0])
        self.my_model.set_hop(hopping[15],1,3,[0,1,0])
        self.my_model.set_hop(hopping[16],1,2,[0,0,0])
        self.my_model.set_hop(hopping[17],1,2,[1,1,0])
        self.my_model.set_hop(hopping[18],0,3,[-1,-1,0])
        self.my_model.set_hop(hopping[19],0,3,[0,0,0])
        self.my_model.set_hop(hopping[20],0,1,[0,-1,0])
        self.my_model.set_hop(hopping[21],0,1,[0,-1,-1])
        self.my_model.set_hop(hopping[22],0,1,[-1,-1,-1])
        self.my_model.set_hop(hopping[23],0,1,[0,0,-1])
        self.my_model.set_hop(hopping[24],0,2,[-1,-1,0])
        self.my_model.set_hop(hopping[25],1,3,[-1,-1,0])
        self.my_model.set_hop(hopping[26],0,2,[1,1,0])
        self.my_model.set_hop(hopping[27],1,3,[1,1,0])
        self.my_model.set_hop(hopping[28],2,3,[-1,0,-1])
        self.my_model.set_hop(hopping[29],0,1,[-1,-2,0])
        self.my_model.set_hop(hopping[30],0,1,[1,0,0])
        self.my_model.set_hop(hopping[31],0,1,[-1,-2,-1])
        self.my_model.set_hop(hopping[32],0,1,[1,0,-1])
        self.my_model.set_hop(hopping[33],2,3,[-1,0,0])
        self.my_model.set_hop(hopping[34],2,3,[0,0,-1])
        self.my_model.set_hop(hopping[35],2,3,[-1,-1,-1])
        self.my_model.set_hop(hopping[36],1,17,[0,0,1])
        self.my_model.set_hop(hopping[37],1,5,[0,0,1])
        self.my_model.set_hop(hopping[38],0,10,[0,0,-1])
        self.my_model.set_hop(hopping[39],1,11,[0,0,1])
        self.my_model.set_hop(hopping[40],0,4,[0,0,-1])
        self.my_model.set_hop(hopping[41],0,16,[0,0,-1])
        self.my_model.set_hop(hopping[42],1,12,[1,1,0])
        self.my_model.set_hop(hopping[43],1,18,[0,0,0])
        self.my_model.set_hop(hopping[44],1,18,[1,1,0])
        self.my_model.set_hop(hopping[45],1,6,[0,0,0])
        self.my_model.set_hop(hopping[46],1,6,[1,1,0])
        self.my_model.set_hop(hopping[47],1,12,[0,0,0])
        self.my_model.set_hop(hopping[48],0,13,[0,0,0])
        self.my_model.set_hop(hopping[49],0,7,[-1,-1,0])
        self.my_model.set_hop(hopping[50],0,7,[0,0,0])
        self.my_model.set_hop(hopping[51],0,19,[-1,-1,0])
        self.my_model.set_hop(hopping[52],0,19,[0,0,0])
        self.my_model.set_hop(hopping[53],0,13,[-1,-1,0])
        self.my_model.set_hop(hopping[54],1,8,[0,0,0])
        self.my_model.set_hop(hopping[55],1,14,[0,0,0])
        self.my_model.set_hop(hopping[56],1,20,[0,0,0])
        self.my_model.set_hop(hopping[57],0,21,[0,0,0])
        self.my_model.set_hop(hopping[58],0,15,[0,0,0])
        self.my_model.set_hop(hopping[59],0,9,[0,0,0])
        self.my_model.set_hop(hopping[60],2,13,[-1,0,0])
        self.my_model.set_hop(hopping[61],2,19,[-1,0,0])
        self.my_model.set_hop(hopping[62],2,7,[-1,0,0])
        self.my_model.set_hop(hopping[63],3,6,[1,0,0])
        self.my_model.set_hop(hopping[64],3,18,[1,0,0])
        self.my_model.set_hop(hopping[65],3,12,[1,0,0])
        self.my_model.set_hop(hopping[66],3,10,[1,1,0])
        self.my_model.set_hop(hopping[67],3,16,[0,0,0])
        self.my_model.set_hop(hopping[68],3,16,[1,1,0])
        self.my_model.set_hop(hopping[69],3,4,[1,1,0])
        self.my_model.set_hop(hopping[70],3,4,[0,0,0])
        self.my_model.set_hop(hopping[71],3,10,[0,0,0])
        self.my_model.set_hop(hopping[72],2,11,[0,0,0])
        self.my_model.set_hop(hopping[73],2,5,[-1,-1,0])
        self.my_model.set_hop(hopping[74],2,5,[0,0,0])
        self.my_model.set_hop(hopping[75],2,17,[-1,-1,0])
        self.my_model.set_hop(hopping[76],2,17,[0,0,0])
        self.my_model.set_hop(hopping[77],2,11,[-1,-1,0])
        self.my_model.set_hop(hopping[78],2,9,[0,0,0])
        self.my_model.set_hop(hopping[79],2,21,[0,0,0])
        self.my_model.set_hop(hopping[80],2,15,[0,0,0])
        self.my_model.set_hop(hopping[81],3,14,[0,0,0])
        self.my_model.set_hop(hopping[82],3,20,[0,0,0])
        self.my_model.set_hop(hopping[83],3,8,[0,0,0])
        self.my_model.set_hop(hopping[84],2,20,[0,0,0])
        self.my_model.set_hop(hopping[85],3,9,[0,0,0])
        self.my_model.set_hop(hopping[86],2,8,[0,0,0])
        self.my_model.set_hop(hopping[87],3,21,[0,0,0])
        self.my_model.set_hop(hopping[88],3,15,[1,1,0])
        self.my_model.set_hop(hopping[89],2,20,[-1,-1,0])
        self.my_model.set_hop(hopping[90],3,9,[1,1,0])
        self.my_model.set_hop(hopping[91],2,8,[-1,-1,0])
        self.my_model.set_hop(hopping[92],3,21,[1,1,0])
        self.my_model.set_hop(hopping[93],2,14,[-1,-1,0])
        self.my_model.set_hop(hopping[94],1,4,[0,0,0])
        self.my_model.set_hop(hopping[95],1,4,[1,1,0])
        self.my_model.set_hop(hopping[96],0,11,[0,0,0])
        self.my_model.set_hop(hopping[97],1,10,[1,1,0])
        self.my_model.set_hop(hopping[98],1,16,[0,0,0])
        self.my_model.set_hop(hopping[99],1,16,[1,1,0])
        self.my_model.set_hop(hopping[100],0,17,[-1,-1,0])
        self.my_model.set_hop(hopping[101],0,17,[0,0,0])
        self.my_model.set_hop(hopping[102],0,11,[-1,-1,0])
        self.my_model.set_hop(hopping[103],1,10,[0,0,0])
        self.my_model.set_hop(hopping[104],0,5,[-1,-1,0])
        self.my_model.set_hop(hopping[105],0,5,[0,0,0])
        self.my_model.set_hop(hopping[106],0,5,[0,-1,0])
        self.my_model.set_hop(hopping[107],1,16,[0,1,0])
        self.my_model.set_hop(hopping[108],0,11,[0,-1,0])
        self.my_model.set_hop(hopping[109],1,10,[0,1,0])
        self.my_model.set_hop(hopping[110],0,17,[0,-1,0])
        self.my_model.set_hop(hopping[111],1,4,[0,1,0])
        self.my_model.set_hop(hopping[112],0,8,[0,-1,0])
        self.my_model.set_hop(hopping[113],0,20,[0,-1,0])
        self.my_model.set_hop(hopping[114],0,14,[0,-1,0])
        self.my_model.set_hop(hopping[115],1,15,[0,1,0])
        self.my_model.set_hop(hopping[116],1,21,[0,1,0])
        self.my_model.set_hop(hopping[117],1,9,[0,1,0])
        self.my_model.set_hop(hopping[118],2,12,[0,0,-1])
        self.my_model.set_hop(hopping[119],2,6,[0,0,-1])
        self.my_model.set_hop(hopping[120],2,18,[0,0,-1])
        self.my_model.set_hop(hopping[121],3,19,[0,0,1])
        self.my_model.set_hop(hopping[122],3,7,[0,0,1])
        self.my_model.set_hop(hopping[123],3,13,[0,0,1])
        self.my_model.set_hop(hopping[124],1,13,[0,1,0])
        self.my_model.set_hop(hopping[125],1,7,[-1,0,0])
        self.my_model.set_hop(hopping[126],1,7,[0,1,0])
        self.my_model.set_hop(hopping[127],1,13,[-1,0,0])
        self.my_model.set_hop(hopping[128],1,19,[-1,0,0])
        self.my_model.set_hop(hopping[129],1,19,[0,1,0])
        self.my_model.set_hop(hopping[130],0,18,[0,-1,0])
        self.my_model.set_hop(hopping[131],0,18,[1,0,0])
        self.my_model.set_hop(hopping[132],0,12,[1,0,0])
        self.my_model.set_hop(hopping[133],0,6,[0,-1,0])
        self.my_model.set_hop(hopping[134],0,6,[1,0,0])
        self.my_model.set_hop(hopping[135],0,12,[0,-1,0])
        self.my_model.set_hop(hopping[136],2,12,[0,0,0])
        self.my_model.set_hop(hopping[137],3,13,[0,0,0])
        self.my_model.set_hop(hopping[138],0,6,[0,-1,-1])
        self.my_model.set_hop(hopping[139],0,6,[1,0,-1])
        self.my_model.set_hop(hopping[140],0,12,[1,0,-1])
        self.my_model.set_hop(hopping[141],0,12,[0,-1,-1])
        self.my_model.set_hop(hopping[142],0,18,[0,-1,-1])
        self.my_model.set_hop(hopping[143],0,18,[1,0,-1])
        self.my_model.set_hop(hopping[144],1,19,[-1,0,1])
        self.my_model.set_hop(hopping[145],1,19,[0,1,1])
        self.my_model.set_hop(hopping[146],1,13,[0,1,1])
        self.my_model.set_hop(hopping[147],1,13,[-1,0,1])
        self.my_model.set_hop(hopping[148],1,7,[-1,0,1])
        self.my_model.set_hop(hopping[149],1,7,[0,1,1])
        self.my_model.set_hop(hopping[150],0,10,[1,1,-1])
        self.my_model.set_hop(hopping[151],1,11,[1,1,1])
        self.my_model.set_hop(hopping[152],1,17,[-1,-1,1])
        self.my_model.set_hop(hopping[153],1,17,[1,1,1])
        self.my_model.set_hop(hopping[154],1,5,[-1,-1,1])
        self.my_model.set_hop(hopping[155],1,5,[1,1,1])
        self.my_model.set_hop(hopping[156],0,4,[-1,-1,-1])
        self.my_model.set_hop(hopping[157],0,4,[1,1,-1])
        self.my_model.set_hop(hopping[158],0,16,[-1,-1,-1])
        self.my_model.set_hop(hopping[159],0,16,[1,1,-1])
        self.my_model.set_hop(hopping[160],0,10,[-1,-1,-1])
        self.my_model.set_hop(hopping[161],1,11,[-1,-1,1])
        self.my_model.set_hop(hopping[162],3,5,[0,0,0])
        self.my_model.set_hop(hopping[163],3,17,[0,0,0])
        self.my_model.set_hop(hopping[164],2,16,[0,0,0])
        self.my_model.set_hop(hopping[165],2,4,[0,0,0])
        self.my_model.set_hop(hopping[166],1,14,[1,1,0])
        self.my_model.set_hop(hopping[167],1,8,[1,1,0])
        self.my_model.set_hop(hopping[168],1,20,[1,1,0])
        self.my_model.set_hop(hopping[169],0,15,[1,1,0])
        self.my_model.set_hop(hopping[170],1,8,[-1,-1,0])
        self.my_model.set_hop(hopping[171],0,21,[-1,-1,0])
        self.my_model.set_hop(hopping[172],0,21,[1,1,0])
        self.my_model.set_hop(hopping[173],1,20,[-1,-1,0])
        self.my_model.set_hop(hopping[174],0,9,[-1,-1,0])
        self.my_model.set_hop(hopping[175],0,9,[1,1,0])
        self.my_model.set_hop(hopping[176],0,15,[-1,-1,0])
        self.my_model.set_hop(hopping[177],1,14,[-1,-1,0])
        self.my_model.set_hop(hopping[178],3,17,[0,0,1])
        self.my_model.set_hop(hopping[179],3,11,[0,0,1])
        self.my_model.set_hop(hopping[180],3,5,[0,0,1])
        self.my_model.set_hop(hopping[181],2,4,[0,0,-1])
        self.my_model.set_hop(hopping[182],2,10,[0,0,-1])
        self.my_model.set_hop(hopping[183],2,16,[0,0,-1])
        self.my_model.set_hop(hopping[184],3,12,[2,1,0])
        self.my_model.set_hop(hopping[185],3,6,[0,-1,0])
        self.my_model.set_hop(hopping[186],3,6,[2,1,0])
        self.my_model.set_hop(hopping[187],3,18,[0,-1,0])
        self.my_model.set_hop(hopping[188],3,18,[2,1,0])
        self.my_model.set_hop(hopping[189],2,19,[-2,-1,0])
        self.my_model.set_hop(hopping[190],2,7,[-2,-1,0])
        self.my_model.set_hop(hopping[191],2,13,[-2,-1,0])
        self.my_model.set_hop(hopping[192],3,12,[0,-1,0])
        self.my_model.set_hop(hopping[193],2,13,[0,1,0])
        self.my_model.set_hop(hopping[194],2,19,[0,1,0])
        self.my_model.set_hop(hopping[195],2,7,[0,1,0])
        self.my_model.set_hop(hopping[196],2,15,[1,1,0])
        self.my_model.set_hop(hopping[197],3,14,[1,1,0])
        self.my_model.set_hop(hopping[198],2,9,[-1,-1,0])
        self.my_model.set_hop(hopping[199],2,9,[1,1,0])
        self.my_model.set_hop(hopping[200],2,21,[-1,-1,0])
        self.my_model.set_hop(hopping[201],2,21,[1,1,0])
        self.my_model.set_hop(hopping[202],3,20,[-1,-1,0])
        self.my_model.set_hop(hopping[203],3,20,[1,1,0])
        self.my_model.set_hop(hopping[204],3,8,[-1,-1,0])
        self.my_model.set_hop(hopping[205],3,8,[1,1,0])
        self.my_model.set_hop(hopping[206],2,15,[-1,-1,0])
        self.my_model.set_hop(hopping[207],3,14,[-1,-1,0])
        self.my_model.set_hop(hopping[208],1,9,[1,1,0])
        self.my_model.set_hop(hopping[209],1,15,[1,1,0])
        self.my_model.set_hop(hopping[210],1,15,[0,0,0])
        self.my_model.set_hop(hopping[211],1,21,[1,1,0])
        self.my_model.set_hop(hopping[212],0,20,[-1,-1,0])
        self.my_model.set_hop(hopping[213],0,14,[0,0,0])
        self.my_model.set_hop(hopping[214],0,14,[-1,-1,0])
        self.my_model.set_hop(hopping[215],0,8,[-1,-1,0])
        self.my_model.set_hop(hopping[216],1,20,[0,0,1])
        self.my_model.set_hop(hopping[217],0,9,[0,0,-1])
        self.my_model.set_hop(hopping[218],0,15,[0,0,-1])
        self.my_model.set_hop(hopping[219],1,14,[0,0,1])
        self.my_model.set_hop(hopping[220],1,8,[0,0,1])
        self.my_model.set_hop(hopping[221],0,21,[0,0,-1])
        self.my_model.set_hop(hopping[222],0,10,[0,0,0])
        self.my_model.set_hop(hopping[223],1,11,[0,0,0])
        self.my_model.set_hop(hopping[224],2,10,[0,1,-1])
        self.my_model.set_hop(hopping[225],2,10,[-1,0,-1])
        self.my_model.set_hop(hopping[226],2,16,[-1,0,-1])
        self.my_model.set_hop(hopping[227],2,16,[0,1,-1])
        self.my_model.set_hop(hopping[228],2,4,[-1,0,-1])
        self.my_model.set_hop(hopping[229],2,4,[0,1,-1])
        self.my_model.set_hop(hopping[230],3,5,[0,-1,1])
        self.my_model.set_hop(hopping[231],3,5,[1,0,1])
        self.my_model.set_hop(hopping[232],3,17,[0,-1,1])
        self.my_model.set_hop(hopping[233],3,17,[1,0,1])
        self.my_model.set_hop(hopping[234],3,11,[1,0,1])
        self.my_model.set_hop(hopping[235],3,11,[0,-1,1])
        self.my_model.set_hop(hopping[236],1,21,[0,0,1])
        self.my_model.set_hop(hopping[237],1,21,[1,1,1])
        self.my_model.set_hop(hopping[238],1,9,[0,0,1])
        self.my_model.set_hop(hopping[239],1,9,[1,1,1])
        self.my_model.set_hop(hopping[240],1,15,[1,1,1])
        self.my_model.set_hop(hopping[241],0,14,[0,0,-1])
        self.my_model.set_hop(hopping[242],1,15,[0,0,1])
        self.my_model.set_hop(hopping[243],0,14,[-1,-1,-1])
        self.my_model.set_hop(hopping[244],0,8,[-1,-1,-1])
        self.my_model.set_hop(hopping[245],0,8,[0,0,-1])
        self.my_model.set_hop(hopping[246],0,20,[-1,-1,-1])
        self.my_model.set_hop(hopping[247],0,20,[0,0,-1])
        self.my_model.set_hop(hopping[248],3,12,[1,1,0])
        self.my_model.set_hop(hopping[249],3,18,[1,1,0])
        self.my_model.set_hop(hopping[250],3,6,[1,1,0])
        self.my_model.set_hop(hopping[251],2,7,[-1,-1,0])
        self.my_model.set_hop(hopping[252],2,7,[0,0,0])
        self.my_model.set_hop(hopping[253],2,13,[0,0,0])
        self.my_model.set_hop(hopping[254],3,18,[0,0,0])
        self.my_model.set_hop(hopping[255],2,19,[-1,-1,0])
        self.my_model.set_hop(hopping[256],2,19,[0,0,0])
        self.my_model.set_hop(hopping[257],2,13,[-1,-1,0])
        self.my_model.set_hop(hopping[258],3,12,[0,0,0])
        self.my_model.set_hop(hopping[259],3,6,[0,0,0])
        self.my_model.set_hop(hopping[260],0,5,[-1,-2,0])
        self.my_model.set_hop(hopping[261],0,5,[1,0,0])
        self.my_model.set_hop(hopping[262],0,11,[1,0,0])
        self.my_model.set_hop(hopping[263],1,10,[1,2,0])
        self.my_model.set_hop(hopping[264],1,16,[-1,0,0])
        self.my_model.set_hop(hopping[265],1,16,[1,2,0])
        self.my_model.set_hop(hopping[266],0,17,[-1,-2,0])
        self.my_model.set_hop(hopping[267],0,17,[1,0,0])
        self.my_model.set_hop(hopping[268],0,11,[-1,-2,0])
        self.my_model.set_hop(hopping[269],1,10,[-1,0,0])
        self.my_model.set_hop(hopping[270],1,4,[-1,0,0])
        self.my_model.set_hop(hopping[271],1,4,[1,2,0])
        self.my_model.set_hop(hopping[272],3,8,[0,-1,0])
        self.my_model.set_hop(hopping[273],3,8,[1,0,0])
        self.my_model.set_hop(hopping[274],2,15,[0,1,0])
        self.my_model.set_hop(hopping[275],3,14,[1,0,0])
        self.my_model.set_hop(hopping[276],2,21,[-1,0,0])
        self.my_model.set_hop(hopping[277],2,21,[0,1,0])
        self.my_model.set_hop(hopping[278],3,20,[0,-1,0])
        self.my_model.set_hop(hopping[279],3,20,[1,0,0])
        self.my_model.set_hop(hopping[280],2,15,[-1,0,0])
        self.my_model.set_hop(hopping[281],3,14,[0,-1,0])
        self.my_model.set_hop(hopping[282],2,9,[-1,0,0])
        self.my_model.set_hop(hopping[283],2,9,[0,1,0])
        self.my_model.set_hop(hopping[284],0,14,[1,0,0])
        self.my_model.set_hop(hopping[285],0,8,[1,0,0])
        self.my_model.set_hop(hopping[286],0,20,[1,0,0])
        self.my_model.set_hop(hopping[287],1,15,[1,2,0])
        self.my_model.set_hop(hopping[288],0,8,[-1,-2,0])
        self.my_model.set_hop(hopping[289],0,20,[-1,-2,0])
        self.my_model.set_hop(hopping[290],1,21,[-1,0,0])
        self.my_model.set_hop(hopping[291],1,21,[1,2,0])
        self.my_model.set_hop(hopping[292],1,9,[-1,0,0])
        self.my_model.set_hop(hopping[293],1,9,[1,2,0])
        self.my_model.set_hop(hopping[294],1,15,[-1,0,0])
        self.my_model.set_hop(hopping[295],0,14,[-1,-2,0])
        self.my_model.set_hop(hopping[296],3,19,[-1,-1,1])
        self.my_model.set_hop(hopping[297],3,19,[1,1,1])
        self.my_model.set_hop(hopping[298],2,12,[1,1,-1])
        self.my_model.set_hop(hopping[299],3,13,[1,1,1])
        self.my_model.set_hop(hopping[300],3,7,[-1,-1,1])
        self.my_model.set_hop(hopping[301],3,7,[1,1,1])
        self.my_model.set_hop(hopping[302],2,6,[-1,-1,-1])
        self.my_model.set_hop(hopping[303],2,6,[1,1,-1])
        self.my_model.set_hop(hopping[304],3,13,[-1,-1,1])
        self.my_model.set_hop(hopping[305],2,12,[-1,-1,-1])
        self.my_model.set_hop(hopping[306],2,18,[-1,-1,-1])
        self.my_model.set_hop(hopping[307],2,18,[1,1,-1])
        self.my_model.set_hop(hopping[308],2,12,[1,1,0])
        self.my_model.set_hop(hopping[309],3,13,[1,1,0])
        self.my_model.set_hop(hopping[310],3,7,[-1,-1,0])
        self.my_model.set_hop(hopping[311],3,7,[1,1,0])
        self.my_model.set_hop(hopping[312],2,18,[-1,-1,0])
        self.my_model.set_hop(hopping[313],2,18,[1,1,0])
        self.my_model.set_hop(hopping[314],3,19,[-1,-1,0])
        self.my_model.set_hop(hopping[315],3,19,[1,1,0])
        self.my_model.set_hop(hopping[316],2,6,[-1,-1,0])
        self.my_model.set_hop(hopping[317],2,6,[1,1,0])
        self.my_model.set_hop(hopping[318],3,13,[-1,-1,0])
        self.my_model.set_hop(hopping[319],2,12,[-1,-1,0])
        self.my_model.set_hop(hopping[320],3,13,[-1,0,0])
        self.my_model.set_hop(hopping[321],3,19,[-1,0,0])
        self.my_model.set_hop(hopping[322],3,7,[-1,0,0])
        self.my_model.set_hop(hopping[323],2,6,[0,-1,0])
        self.my_model.set_hop(hopping[324],2,6,[1,0,0])
        self.my_model.set_hop(hopping[325],2,18,[0,-1,0])
        self.my_model.set_hop(hopping[326],2,18,[1,0,0])
        self.my_model.set_hop(hopping[327],3,13,[0,1,0])
        self.my_model.set_hop(hopping[328],2,12,[1,0,0])
        self.my_model.set_hop(hopping[329],2,12,[0,-1,0])
        self.my_model.set_hop(hopping[330],3,19,[0,1,0])
        self.my_model.set_hop(hopping[331],3,7,[0,1,0])
        self.my_model.set_hop(hopping[332],3,21,[0,0,1])
        self.my_model.set_hop(hopping[333],3,21,[1,1,1])
        self.my_model.set_hop(hopping[334],3,15,[1,1,1])
        self.my_model.set_hop(hopping[335],2,14,[0,0,-1])
        self.my_model.set_hop(hopping[336],2,8,[-1,-1,-1])
        self.my_model.set_hop(hopping[337],2,8,[0,0,-1])
        self.my_model.set_hop(hopping[338],3,9,[0,0,1])
        self.my_model.set_hop(hopping[339],3,9,[1,1,1])
        self.my_model.set_hop(hopping[340],3,15,[0,0,1])
        self.my_model.set_hop(hopping[341],2,14,[-1,-1,-1])
        self.my_model.set_hop(hopping[342],2,20,[-1,-1,-1])
        self.my_model.set_hop(hopping[343],2,20,[0,0,-1])
        self.my_model.set_hop(hopping[344],0,13,[1,1,0])
        self.my_model.set_hop(hopping[345],1,12,[2,2,0])
        self.my_model.set_hop(hopping[346],0,7,[-2,-2,0])
        self.my_model.set_hop(hopping[347],0,7,[1,1,0])
        self.my_model.set_hop(hopping[348],0,19,[-2,-2,0])
        self.my_model.set_hop(hopping[349],0,19,[1,1,0])
        self.my_model.set_hop(hopping[350],1,18,[-1,-1,0])
        self.my_model.set_hop(hopping[351],1,18,[2,2,0])
        self.my_model.set_hop(hopping[352],1,6,[-1,-1,0])
        self.my_model.set_hop(hopping[353],1,6,[2,2,0])
        self.my_model.set_hop(hopping[354],0,13,[-2,-2,0])
        self.my_model.set_hop(hopping[355],1,12,[-1,-1,0])
        self.my_model.set_hop(hopping[356],2,16,[-1,-1,0])
        self.my_model.set_hop(hopping[357],2,16,[1,1,0])
        self.my_model.set_hop(hopping[358],3,11,[1,1,0])
        self.my_model.set_hop(hopping[359],2,10,[1,1,0])
        self.my_model.set_hop(hopping[360],2,4,[-1,-1,0])
        self.my_model.set_hop(hopping[361],2,4,[1,1,0])
        self.my_model.set_hop(hopping[362],3,5,[-1,-1,0])
        self.my_model.set_hop(hopping[363],3,5,[1,1,0])
        self.my_model.set_hop(hopping[364],2,10,[-1,-1,0])
        self.my_model.set_hop(hopping[365],3,11,[-1,-1,0])
        self.my_model.set_hop(hopping[366],3,17,[-1,-1,0])
        self.my_model.set_hop(hopping[367],3,17,[1,1,0])
        self.my_model.set_hop(hopping[368],3,11,[1,1,1])
        self.my_model.set_hop(hopping[369],3,17,[-1,-1,1])
        self.my_model.set_hop(hopping[370],3,17,[1,1,1])
        self.my_model.set_hop(hopping[371],3,5,[-1,-1,1])
        self.my_model.set_hop(hopping[372],3,5,[1,1,1])
        self.my_model.set_hop(hopping[373],3,11,[-1,-1,1])
        self.my_model.set_hop(hopping[374],2,10,[1,1,-1])
        self.my_model.set_hop(hopping[375],2,4,[-1,-1,-1])
        self.my_model.set_hop(hopping[376],2,4,[1,1,-1])
        self.my_model.set_hop(hopping[377],2,16,[-1,-1,-1])
        self.my_model.set_hop(hopping[378],2,16,[1,1,-1])
        self.my_model.set_hop(hopping[379],2,10,[-1,-1,-1])
        self.my_model.set_hop(hopping[380],2,11,[1,1,0])
        self.my_model.set_hop(hopping[381],3,10,[2,2,0])
        self.my_model.set_hop(hopping[382],3,16,[-1,-1,0])
        self.my_model.set_hop(hopping[383],3,16,[2,2,0])
        self.my_model.set_hop(hopping[384],2,5,[-2,-2,0])
        self.my_model.set_hop(hopping[385],2,5,[1,1,0])
        self.my_model.set_hop(hopping[386],3,4,[-1,-1,0])
        self.my_model.set_hop(hopping[387],3,4,[2,2,0])
        self.my_model.set_hop(hopping[388],2,17,[-2,-2,0])
        self.my_model.set_hop(hopping[389],2,17,[1,1,0])
        self.my_model.set_hop(hopping[390],2,11,[-2,-2,0])
        self.my_model.set_hop(hopping[391],3,10,[-1,-1,0])
        self.my_model.set_hop(hopping[392],2,14,[1,1,0])
        self.my_model.set_hop(hopping[393],3,15,[2,2,0])
        self.my_model.set_hop(hopping[394],2,20,[-2,-2,0])
        self.my_model.set_hop(hopping[395],2,20,[1,1,0])
        self.my_model.set_hop(hopping[396],3,9,[-1,-1,0])
        self.my_model.set_hop(hopping[397],3,9,[2,2,0])
        self.my_model.set_hop(hopping[398],2,8,[-2,-2,0])
        self.my_model.set_hop(hopping[399],2,8,[1,1,0])
        self.my_model.set_hop(hopping[400],3,21,[-1,-1,0])
        self.my_model.set_hop(hopping[401],3,21,[2,2,0])
        self.my_model.set_hop(hopping[402],2,14,[-2,-2,0])
        self.my_model.set_hop(hopping[403],3,15,[-1,-1,0])
        self.my_model.set_hop(hopping[404],2,17,[-1,-1,1])
        self.my_model.set_hop(hopping[405],2,17,[0,0,1])
        self.my_model.set_hop(hopping[406],2,11,[0,0,1])
        self.my_model.set_hop(hopping[407],3,10,[1,1,-1])
        self.my_model.set_hop(hopping[408],3,4,[1,1,-1])
        self.my_model.set_hop(hopping[409],3,4,[0,0,-1])
        self.my_model.set_hop(hopping[410],2,5,[-1,-1,1])
        self.my_model.set_hop(hopping[411],2,5,[0,0,1])
        self.my_model.set_hop(hopping[412],2,11,[-1,-1,1])
        self.my_model.set_hop(hopping[413],3,10,[0,0,-1])
        self.my_model.set_hop(hopping[414],3,16,[0,0,-1])
        self.my_model.set_hop(hopping[415],3,16,[1,1,-1])
        #endregion

        # print tight-binding model
        # my_model.display()

        # generate list of k-points following a segmented path in the BZ
        # list of nodes (high-symmetry points) that will be connected
        path = [[0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [
            0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        # labels of the nodes
        self.label = (r'$F $', r'$\Gamma $', r'$T $', r'$\Gamma $', r'$L $')
        # total number of interpolated k-points along the path
        nk = 533

        # call function k_path to construct the actual path
        (self.k_vec, self.k_dist, self.k_node) = self.my_model.k_path(path, nk, report = False)

        # print('---------------------------------------')
        # print('starting calculation')
        # print('---------------------------------------')
        # print('Calculating bands...')

        # obtain eigenvalues to be plotted
        self.evals = self.my_model.solve_eff(self.k_vec, roi=4)


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
        Gamma1 = np.where(self.k_dist == self.k_node[1])[0][0] # the 1st Gamma point
        Gamma2 = np.where(self.k_dist == self.k_node[3])[0][0] # the 2nd Gamma point
        interval = 10 # interval length for calculating the derivative

        # calculate band mass & effective mass for all bands
        self.bmass_0_f = findBMass(self.evals[0], self.k_dist, Gamma1-interval, Gamma1)
        self.bmass_0_t = findBMass(self.evals[0], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_0_l = findBMass(self.evals[0], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_0 = np.power(self.bmass_0_f*self.bmass_0_t*self.bmass_0_l, 1/3)

        self.bmass_1_f = findBMass(self.evals[1], self.k_dist, Gamma1-interval, Gamma1)
        self.bmass_1_t = findBMass(self.evals[1], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_1_l = findBMass(self.evals[1], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_1 = np.power(self.bmass_1_f*self.bmass_1_t*self.bmass_1_l, 1/3)

        self.bmass_2_f = findBMass(self.evals[2], self.k_dist, Gamma1-interval, Gamma1)
        self.bmass_2_t = findBMass(self.evals[2], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_2_l = findBMass(self.evals[2], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_2 = np.power(self.bmass_2_f*self.bmass_2_t*self.bmass_2_l, 1/3)

        self.bmass_3_f = findBMass(self.evals[3], self.k_dist, Gamma1-interval, Gamma1)
        self.bmass_3_t = findBMass(self.evals[3], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_3_l = findBMass(self.evals[3], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_3 = np.power(self.bmass_3_f*self.bmass_3_t*self.bmass_3_l, 1/3)


    def getMassString(self):
        """
        return the string containing all band mass and
        effective mass. The order is:
        bmass_0_f, bmass_0_t, bmass_0_l, emass_0,
        bmass_1_f, bmass_1_t, bmass_1_l, emass_1,
        bmass_2_f, bmass_2_t, bmass_2_l, emass_2,
        bmass_3_f, bmass_3_t, bmass_3_l, emass_3
        """
        return ('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' 
        %(
          self.bmass_0_f, self.bmass_0_t, self.bmass_0_l, self.emass_0,
          self.bmass_1_f, self.bmass_1_t, self.bmass_1_l, self.emass_1,
          self.bmass_2_f, self.bmass_2_t, self.bmass_2_l, self.emass_2,
          self.bmass_3_f, self.bmass_3_t, self.bmass_3_l, self.emass_3))
        


    def getLSError(self, fit_band, band_num):
        """
        return the least square error with respect to another band
        this function assume the kapth is the same. The least square error
        is the average of the square error at each point
        parameters:
        fit_band: the band to fit. fit_band has to be a two-dimentiohnal
            array. The first axis is different bands and the second axis
            is the band data. fit_band should have the same shape as evals
            or getLSError will return -1 
        band_num: a list of band index that will be calculated
        return:
        the least square error 
        """
        if(self.evals.shape != np.shape(fit_band)):
            return -1
        error = 0
        for i in band_num:
            error = error + np.sum(np.square(self.evals[i] - fit_band[i]))
        return error/(np.shape(fit_band)[0]*np.shape(fit_band)[1]) # take the average

    def save4BandPlot(self, name):
        fig, ax = plt.subplots()
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(self.k_node[0], self.k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(self.k_node)
        ax.set_xticklabels(self.label)
        # add vertical lines at node positions
        for n in range(len(self.k_node)):
            ax.axvline(x=self.k_node[n], linewidth=0.5, color='k')
        # put title
        ax.set_title("Ga2O3 band structure")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy (eV)")

        # plot first and second band
        ax.plot(self.k_dist, self.evals[0])
        ax.plot(self.k_dist, self.evals[1])
        ax.plot(self.k_dist, self.evals[2])
        ax.plot(self.k_dist, self.evals[3])

        # make an PDF figure of a plot
        fig.tight_layout()
        fig.savefig(name)
        plt.close(fig)

    def savePlot(self, name):
        fig, ax = plt.subplots()
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(self.k_node[0], self.k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(self.k_node)
        ax.set_xticklabels(self.label)
        # add vertical lines at node positions
        for n in range(len(self.k_node)):
            ax.axvline(x=self.k_node[n], linewidth=0.5, color='k')
        # put title
        ax.set_title("Ga2O3 band structure")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy (eV)")

        # plot first and second band
        ax.plot(self.k_dist, self.evals[0])
        ax.plot(self.k_dist, self.evals[1])
        ax.plot(self.k_dist, self.evals[2])
        ax.plot(self.k_dist, self.evals[3])
        ax.plot(self.k_dist, self.evals[4])
        ax.plot(self.k_dist, self.evals[5])
        ax.plot(self.k_dist, self.evals[6])
        ax.plot(self.k_dist, self.evals[7])
        ax.plot(self.k_dist, self.evals[8])
        ax.plot(self.k_dist, self.evals[9])
        ax.plot(self.k_dist, self.evals[10])
        ax.plot(self.k_dist, self.evals[11])
        ax.plot(self.k_dist, self.evals[12])
        ax.plot(self.k_dist, self.evals[13])
        ax.plot(self.k_dist, self.evals[14])
        ax.plot(self.k_dist, self.evals[15])
        ax.plot(self.k_dist, self.evals[16])
        ax.plot(self.k_dist, self.evals[17])
        ax.plot(self.k_dist, self.evals[18])
        ax.plot(self.k_dist, self.evals[19])

        # make an PDF figure of a plot
        fig.tight_layout()
        fig.savefig(name)
        plt.close(fig)

    def overlapPlot(self, fit_band, band_num, name):
        """
        generate a plot with both the calculated TB band and the input band.
        The output will be stored in overlap_pdf/ folder
        parameters:
        fit_band: the input band to plot. fit_band has to be a two-dimentiohnal
            array. The first axis is different bands and the second axis
            is the band data. fit_band should have the same shape as evals
        band_num: a list of band index that will be plotted
        return:
        None 
        """
        # if(self.evals.shape != np.shape(fit_band)):
        #     return -1
        fig, ax = plt.subplots()
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(self.k_node[0], self.k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(self.k_node)
        ax.set_xticklabels(self.label)
        # add vertical lines at node positions
        for n in range(len(self.k_node)):
            ax.axvline(x=self.k_node[n], linewidth=0.5, color='k')
        # put title
        ax.set_title("Ga2O3 band structure")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy (eV)")

        # plot bands specified by band_num
        for i in band_num:
            ax.plot(self.k_dist, self.evals[i])
            ax.plot(self.k_dist, fit_band[i], dashes=[6, 2])
        
        # make an PDF figure of a plot
        fig.tight_layout()
        # fig.savefig("pdf/Ga2O3_%.2fm_%.2fr.pdf" %(self.m, self.r))
        fig.savefig(name)
        plt.close(fig)



    def plotFermiSurf(self, fermi_level, name, real_unit=False):
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
        fig.savefig(name, format='png')
        plt.close(fig)

    def saveData(self, name):
        # save the raw data
        outdata = []
        outdata.append(self.k_dist)
        outdata.append(self.evals)
        np.save(name, outdata)