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

    def __init__(self, onsite, hopping):
        self.onsite = onsite
        self.hopping = hopping
    
        # define lattice vectors
        lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
            [-6.1149997711,1.5199999809, 0.0000000000], 
            [-1.3736609922, 0.0000000000, 5.6349851545]]
        # define coordinates of orbitals
        # the four groups are s, px, py, pz in order
        orb = [[0.9096000000, 	-0.9096000000, 	0.2052000000], 
            [1.0904000000, 	-0.0904000000, 	0.7948000000],
            [0.6586000000, 	-0.6586000000, 	0.3143000000], 
            [1.3414000000, 	-0.3414000000, 	0.6857000000],
            [0.9096000000, 	-0.9096000000, 	0.2052000000], 
            [1.0904000000, 	-0.0904000000, 	0.7948000000],
            [0.6586000000, 	-0.6586000000, 	0.3143000000], 
            [1.3414000000, 	-0.3414000000, 	0.6857000000],
            [0.9096000000, 	-0.9096000000, 	0.2052000000], 
            [1.0904000000, 	-0.0904000000, 	0.7948000000],
            [0.6586000000, 	-0.6586000000, 	0.3143000000], 
            [1.3414000000, 	-0.3414000000, 	0.6857000000],
            [0.9096000000, 	-0.9096000000, 	0.2052000000], 
            [1.0904000000, 	-0.0904000000, 	0.7948000000],
            [0.6586000000, 	-0.6586000000, 	0.3143000000], 
            [1.3414000000, 	-0.3414000000, 	0.6857000000],
            [0.8326000000, 	-0.8326000000, 	0.8989000000],
            [1.1674000000, 	-0.1674000000, 	0.1011000000], 
            [0.5043000000, 	-0.5043000000, 	0.7447000000],
            [1.4957000000, 	-0.4957000000, 	0.2553000000],
            [1.1721000000, 	-0.1721000000, 	0.5635000000], 
            [0.8279000000, 	-0.8279000000, 	0.4365000000]]

        # make three-dimensional tight-binding model
        my_model = tb_model(3, 3, lat, orb)

        # set on-site energies
        my_model.set_onsite(onsite)
        # set hoppings (one for each connected pair of orbitals)
        # (amplitude, i, j, [lattice vector to cell containing j])
        #region
        # ss
        my_model.set_hop(hopping[0],0,0,[-1,-1,0])
        my_model.set_hop(hopping[1],1,1,[-1,-1,0])
        my_model.set_hop(hopping[2],2,2,[-1,-1,0])
        my_model.set_hop(hopping[3],3,3,[-1,-1,0])
        my_model.set_hop(hopping[4],2,3,[-1,-1,0])
        my_model.set_hop(hopping[5],2,3,[0,0,0])
        my_model.set_hop(hopping[6],1,3,[0,0,0])
        my_model.set_hop(hopping[7],0,2,[0,0,0])
        my_model.set_hop(hopping[8],0,3,[-1,-1,-1])
        my_model.set_hop(hopping[9],0,3,[0,0,-1])
        my_model.set_hop(hopping[10],1,2,[0,0,1])
        my_model.set_hop(hopping[11],1,2,[1,1,1])
        my_model.set_hop(hopping[12],0,2,[0,-1,0])
        my_model.set_hop(hopping[13],0,2,[1,0,0])
        my_model.set_hop(hopping[14],1,3,[-1,0,0])
        my_model.set_hop(hopping[15],1,3,[0,1,0])
        my_model.set_hop(hopping[16],1,2,[0,0,0])
        my_model.set_hop(hopping[17],1,2,[1,1,0])
        my_model.set_hop(hopping[18],0,3,[-1,-1,0])
        my_model.set_hop(hopping[19],0,3,[0,0,0])
        my_model.set_hop(hopping[20],0,1,[0,-1,0])
        my_model.set_hop(hopping[21],0,1,[0,-1,-1])
        my_model.set_hop(hopping[22],0,1,[-1,-1,-1])
        my_model.set_hop(hopping[23],0,1,[0,0,-1])
        my_model.set_hop(hopping[24],0,2,[-1,-1,0])
        my_model.set_hop(hopping[25],1,3,[-1,-1,0])
        my_model.set_hop(hopping[26],2,3,[-1,0,-1])
        my_model.set_hop(hopping[27],0,1,[-1,-2,0])
        my_model.set_hop(hopping[28],0,1,[1,0,0])
        my_model.set_hop(hopping[29],0,1,[-1,-2,-1])
        my_model.set_hop(hopping[30],0,1,[1,0,-1])
        my_model.set_hop(hopping[31],2,3,[-1,0,0])
        my_model.set_hop(hopping[32],2,3,[0,0,-1])
        my_model.set_hop(hopping[33],2,3,[-1,-1,-1])
        my_model.set_hop(hopping[34],0,4,[-1,-1,0])
        my_model.set_hop(hopping[35],0,12,[-1,-1,0])
        my_model.set_hop(hopping[36],1,5,[-1,-1,0])
        my_model.set_hop(hopping[37],1,13,[-1,-1,0])
        my_model.set_hop(hopping[38],2,6,[-1,-1,0])
        my_model.set_hop(hopping[39],2,14,[-1,-1,0])
        my_model.set_hop(hopping[40],3,7,[-1,-1,0])
        my_model.set_hop(hopping[41],3,15,[-1,-1,0])
        my_model.set_hop(hopping[42],0,8,[-1,-1,0])
        my_model.set_hop(hopping[43],1,9,[-1,-1,0])
        my_model.set_hop(hopping[44],2,10,[-1,-1,0])
        my_model.set_hop(hopping[45],3,11,[-1,-1,0])
        my_model.set_hop(hopping[46],2,15,[-1,-1,0])
        my_model.set_hop(hopping[47],2,15,[0,0,0])
        my_model.set_hop(hopping[48],2,7,[-1,-1,0])
        my_model.set_hop(hopping[49],2,7,[0,0,0])
        my_model.set_hop(hopping[50],3,10,[1,1,0])
        my_model.set_hop(hopping[51],2,11,[0,0,0])
        my_model.set_hop(hopping[52],3,10,[0,0,0])
        my_model.set_hop(hopping[53],2,11,[-1,-1,0])
        my_model.set_hop(hopping[54],3,6,[0,0,0])
        my_model.set_hop(hopping[55],3,6,[1,1,0])
        my_model.set_hop(hopping[56],3,14,[0,0,0])
        my_model.set_hop(hopping[57],3,14,[1,1,0])
        my_model.set_hop(hopping[58],1,7,[0,0,0])
        my_model.set_hop(hopping[59],3,13,[0,0,0])
        my_model.set_hop(hopping[60],3,9,[0,0,0])
        my_model.set_hop(hopping[61],1,11,[0,0,0])
        my_model.set_hop(hopping[62],1,15,[0,0,0])
        my_model.set_hop(hopping[63],3,5,[0,0,0])
        my_model.set_hop(hopping[64],2,4,[0,0,0])
        my_model.set_hop(hopping[65],0,14,[0,0,0])
        my_model.set_hop(hopping[66],0,10,[0,0,0])
        my_model.set_hop(hopping[67],2,8,[0,0,0])
        my_model.set_hop(hopping[68],2,12,[0,0,0])
        my_model.set_hop(hopping[69],0,6,[0,0,0])
        my_model.set_hop(hopping[70],1,14,[0,0,1])
        my_model.set_hop(hopping[71],3,12,[0,0,1])
        my_model.set_hop(hopping[72],1,14,[1,1,1])
        my_model.set_hop(hopping[73],3,12,[1,1,1])
        my_model.set_hop(hopping[74],0,11,[0,0,-1])
        my_model.set_hop(hopping[75],2,9,[0,0,-1])
        my_model.set_hop(hopping[76],1,10,[1,1,1])
        my_model.set_hop(hopping[77],3,8,[1,1,1])
        my_model.set_hop(hopping[78],3,4,[0,0,1])
        my_model.set_hop(hopping[79],3,4,[1,1,1])
        my_model.set_hop(hopping[80],1,6,[0,0,1])
        my_model.set_hop(hopping[81],1,6,[1,1,1])
        my_model.set_hop(hopping[82],2,5,[-1,-1,-1])
        my_model.set_hop(hopping[83],2,5,[0,0,-1])
        my_model.set_hop(hopping[84],0,7,[-1,-1,-1])
        my_model.set_hop(hopping[85],0,7,[0,0,-1])
        my_model.set_hop(hopping[86],0,11,[-1,-1,-1])
        my_model.set_hop(hopping[87],2,9,[-1,-1,-1])
        my_model.set_hop(hopping[88],1,10,[0,0,1])
        my_model.set_hop(hopping[89],3,8,[0,0,1])
        my_model.set_hop(hopping[90],0,15,[-1,-1,-1])
        my_model.set_hop(hopping[91],2,13,[-1,-1,-1])
        my_model.set_hop(hopping[92],0,15,[0,0,-1])
        my_model.set_hop(hopping[93],2,13,[0,0,-1])
        my_model.set_hop(hopping[94],0,6,[0,-1,0])
        my_model.set_hop(hopping[95],0,6,[1,0,0])
        my_model.set_hop(hopping[96],2,8,[0,1,0])
        my_model.set_hop(hopping[97],0,10,[1,0,0])
        my_model.set_hop(hopping[98],0,14,[0,-1,0])
        my_model.set_hop(hopping[99],0,14,[1,0,0])
        my_model.set_hop(hopping[100],2,12,[-1,0,0])
        my_model.set_hop(hopping[101],2,12,[0,1,0])
        my_model.set_hop(hopping[102],2,8,[-1,0,0])
        my_model.set_hop(hopping[103],0,10,[0,-1,0])
        my_model.set_hop(hopping[104],2,4,[-1,0,0])
        my_model.set_hop(hopping[105],2,4,[0,1,0])
        my_model.set_hop(hopping[106],3,5,[0,-1,0])
        my_model.set_hop(hopping[107],3,5,[1,0,0])
        my_model.set_hop(hopping[108],3,9,[1,0,0])
        my_model.set_hop(hopping[109],1,11,[0,1,0])
        my_model.set_hop(hopping[110],3,13,[0,-1,0])
        my_model.set_hop(hopping[111],3,13,[1,0,0])
        my_model.set_hop(hopping[112],1,15,[-1,0,0])
        my_model.set_hop(hopping[113],1,15,[0,1,0])
        my_model.set_hop(hopping[114],3,9,[0,-1,0])
        my_model.set_hop(hopping[115],1,11,[-1,0,0])
        my_model.set_hop(hopping[116],1,7,[-1,0,0])
        my_model.set_hop(hopping[117],1,7,[0,1,0])
        my_model.set_hop(hopping[118],2,13,[-1,-1,0])
        my_model.set_hop(hopping[119],2,13,[0,0,0])
        my_model.set_hop(hopping[120],2,9,[0,0,0])
        my_model.set_hop(hopping[121],1,10,[1,1,0])
        my_model.set_hop(hopping[122],1,6,[0,0,0])
        my_model.set_hop(hopping[123],1,6,[1,1,0])
        my_model.set_hop(hopping[124],2,5,[-1,-1,0])
        my_model.set_hop(hopping[125],2,5,[0,0,0])
        my_model.set_hop(hopping[126],2,9,[-1,-1,0])
        my_model.set_hop(hopping[127],1,10,[0,0,0])
        my_model.set_hop(hopping[128],1,14,[0,0,0])
        my_model.set_hop(hopping[129],1,14,[1,1,0])
        my_model.set_hop(hopping[130],0,15,[-1,-1,0])
        my_model.set_hop(hopping[131],0,15,[0,0,0])
        my_model.set_hop(hopping[132],0,11,[0,0,0])
        my_model.set_hop(hopping[133],3,8,[1,1,0])
        my_model.set_hop(hopping[134],3,4,[0,0,0])
        my_model.set_hop(hopping[135],3,4,[1,1,0])
        my_model.set_hop(hopping[136],0,7,[-1,-1,0])
        my_model.set_hop(hopping[137],0,7,[0,0,0])
        my_model.set_hop(hopping[138],0,11,[-1,-1,0])
        my_model.set_hop(hopping[139],3,8,[0,0,0])
        my_model.set_hop(hopping[140],3,12,[0,0,0])
        my_model.set_hop(hopping[141],3,12,[1,1,0])
        my_model.set_hop(hopping[142],0,13,[0,-1,0])
        my_model.set_hop(hopping[143],0,5,[0,-1,0])
        my_model.set_hop(hopping[144],0,9,[0,-1,0])
        my_model.set_hop(hopping[145],1,8,[0,1,0])
        my_model.set_hop(hopping[146],1,4,[0,1,0])
        my_model.set_hop(hopping[147],1,12,[0,1,0])
        my_model.set_hop(hopping[148],0,5,[0,-1,-1])
        my_model.set_hop(hopping[149],1,12,[0,1,1])
        my_model.set_hop(hopping[150],0,9,[0,-1,-1])
        my_model.set_hop(hopping[151],1,8,[0,1,1])
        my_model.set_hop(hopping[152],0,13,[0,-1,-1])
        my_model.set_hop(hopping[153],1,4,[0,1,1])
        my_model.set_hop(hopping[154],1,4,[0,0,1])
        my_model.set_hop(hopping[155],1,4,[1,1,1])
        my_model.set_hop(hopping[156],1,12,[0,0,1])
        my_model.set_hop(hopping[157],1,12,[1,1,1])
        my_model.set_hop(hopping[158],0,9,[0,0,-1])
        my_model.set_hop(hopping[159],1,8,[1,1,1])
        my_model.set_hop(hopping[160],0,9,[-1,-1,-1])
        my_model.set_hop(hopping[161],1,8,[0,0,1])
        my_model.set_hop(hopping[162],0,13,[-1,-1,-1])
        my_model.set_hop(hopping[163],0,13,[0,0,-1])
        my_model.set_hop(hopping[164],0,5,[-1,-1,-1])
        my_model.set_hop(hopping[165],0,5,[0,0,-1])
        my_model.set_hop(hopping[166],1,7,[-1,-1,0])
        my_model.set_hop(hopping[167],2,4,[-1,-1,0])
        my_model.set_hop(hopping[168],0,14,[-1,-1,0])
        my_model.set_hop(hopping[169],3,13,[-1,-1,0])
        my_model.set_hop(hopping[170],1,15,[-1,-1,0])
        my_model.set_hop(hopping[171],2,12,[-1,-1,0])
        my_model.set_hop(hopping[172],0,10,[-1,-1,0])
        my_model.set_hop(hopping[173],2,8,[-1,-1,0])
        my_model.set_hop(hopping[174],3,9,[-1,-1,0])
        my_model.set_hop(hopping[175],1,11,[-1,-1,0])
        my_model.set_hop(hopping[176],0,6,[-1,-1,0])
        my_model.set_hop(hopping[177],3,5,[-1,-1,0])
        my_model.set_hop(hopping[178],3,14,[1,0,1])
        my_model.set_hop(hopping[179],3,6,[1,0,1])
        my_model.set_hop(hopping[180],3,10,[1,0,1])
        my_model.set_hop(hopping[181],2,11,[-1,0,-1])
        my_model.set_hop(hopping[182],2,7,[-1,0,-1])
        my_model.set_hop(hopping[183],2,15,[-1,0,-1])
        my_model.set_hop(hopping[184],0,13,[-1,-2,0])
        my_model.set_hop(hopping[185],0,13,[1,0,0])
        my_model.set_hop(hopping[186],0,9,[1,0,0])
        my_model.set_hop(hopping[187],1,8,[1,2,0])
        my_model.set_hop(hopping[188],0,5,[-1,-2,0])
        my_model.set_hop(hopping[189],0,5,[1,0,0])
        my_model.set_hop(hopping[190],1,4,[-1,0,0])
        my_model.set_hop(hopping[191],1,4,[1,2,0])
        my_model.set_hop(hopping[192],0,9,[-1,-2,0])
        my_model.set_hop(hopping[193],1,8,[-1,0,0])
        my_model.set_hop(hopping[194],1,12,[-1,0,0])
        my_model.set_hop(hopping[195],1,12,[1,2,0])
        my_model.set_hop(hopping[196],0,9,[1,0,-1])
        my_model.set_hop(hopping[197],1,8,[1,2,1])
        my_model.set_hop(hopping[198],0,5,[-1,-2,-1])
        my_model.set_hop(hopping[199],0,5,[1,0,-1])
        my_model.set_hop(hopping[200],1,12,[-1,0,1])
        my_model.set_hop(hopping[201],1,12,[1,2,1])
        my_model.set_hop(hopping[202],0,13,[-1,-2,-1])
        my_model.set_hop(hopping[203],0,13,[1,0,-1])
        my_model.set_hop(hopping[204],1,4,[-1,0,1])
        my_model.set_hop(hopping[205],1,4,[1,2,1])
        my_model.set_hop(hopping[206],0,9,[-1,-2,-1])
        my_model.set_hop(hopping[207],1,8,[-1,0,1])
        my_model.set_hop(hopping[208],3,6,[1,0,0])
        my_model.set_hop(hopping[209],2,15,[-1,0,0])
        my_model.set_hop(hopping[210],3,10,[1,0,0])
        my_model.set_hop(hopping[211],2,11,[-1,0,0])
        my_model.set_hop(hopping[212],3,14,[1,0,0])
        my_model.set_hop(hopping[213],2,7,[-1,0,0])
        my_model.set_hop(hopping[214],3,14,[0,0,1])
        my_model.set_hop(hopping[215],2,7,[0,0,-1])
        my_model.set_hop(hopping[216],2,11,[0,0,-1])
        my_model.set_hop(hopping[217],3,10,[0,0,1])
        my_model.set_hop(hopping[218],3,6,[0,0,1])
        my_model.set_hop(hopping[219],2,15,[0,0,-1])
        my_model.set_hop(hopping[220],3,14,[1,1,1])
        my_model.set_hop(hopping[221],2,7,[-1,-1,-1])
        my_model.set_hop(hopping[222],3,10,[1,1,1])
        my_model.set_hop(hopping[223],2,11,[-1,-1,-1])
        my_model.set_hop(hopping[224],3,6,[1,1,1])
        my_model.set_hop(hopping[225],2,15,[-1,-1,-1])
        my_model.set_hop(hopping[226],4,4,[-1,-1,0])
        my_model.set_hop(hopping[227],5,5,[-1,-1,0])
        my_model.set_hop(hopping[228],6,6,[-1,-1,0])
        my_model.set_hop(hopping[229],7,7,[-1,-1,0])
        my_model.set_hop(hopping[230],12,12,[-1,-1,0])
        my_model.set_hop(hopping[231],13,13,[-1,-1,0])
        my_model.set_hop(hopping[232],14,14,[-1,-1,0])
        my_model.set_hop(hopping[233],15,15,[-1,-1,0])
        my_model.set_hop(hopping[234],4,8,[-1,-1,0])
        my_model.set_hop(hopping[235],4,12,[-1,-1,0])
        my_model.set_hop(hopping[236],5,9,[-1,-1,0])
        my_model.set_hop(hopping[237],5,13,[-1,-1,0])
        my_model.set_hop(hopping[238],6,10,[-1,-1,0])
        my_model.set_hop(hopping[239],6,14,[-1,-1,0])
        my_model.set_hop(hopping[240],7,11,[-1,-1,0])
        my_model.set_hop(hopping[241],7,15,[-1,-1,0])
        my_model.set_hop(hopping[242],8,12,[-1,-1,0])
        my_model.set_hop(hopping[243],9,13,[-1,-1,0])
        my_model.set_hop(hopping[244],10,14,[-1,-1,0])
        my_model.set_hop(hopping[245],11,15,[-1,-1,0])
        my_model.set_hop(hopping[246],8,8,[-1,-1,0])
        my_model.set_hop(hopping[247],9,9,[-1,-1,0])
        my_model.set_hop(hopping[248],10,10,[-1,-1,0])
        my_model.set_hop(hopping[249],11,11,[-1,-1,0])
        my_model.set_hop(hopping[250],10,15,[-1,-1,0])
        my_model.set_hop(hopping[251],11,14,[1,1,0])
        my_model.set_hop(hopping[252],6,11,[-1,-1,0])
        my_model.set_hop(hopping[253],7,10,[1,1,0])
        my_model.set_hop(hopping[254],10,11,[0,0,0])
        my_model.set_hop(hopping[255],10,11,[-1,-1,0])
        my_model.set_hop(hopping[256],6,11,[0,0,0])
        my_model.set_hop(hopping[257],7,10,[0,0,0])
        my_model.set_hop(hopping[258],6,7,[0,0,0])
        my_model.set_hop(hopping[259],6,7,[-1,-1,0])
        my_model.set_hop(hopping[260],10,15,[0,0,0])
        my_model.set_hop(hopping[261],11,14,[0,0,0])
        my_model.set_hop(hopping[262],6,15,[0,0,0])
        my_model.set_hop(hopping[263],7,14,[0,0,0])
        my_model.set_hop(hopping[264],6,15,[-1,-1,0])
        my_model.set_hop(hopping[265],7,14,[1,1,0])
        my_model.set_hop(hopping[266],14,15,[-1,-1,0])
        my_model.set_hop(hopping[267],14,15,[0,0,0])
        my_model.set_hop(hopping[268],5,15,[0,0,0])
        my_model.set_hop(hopping[269],7,13,[0,0,0])
        my_model.set_hop(hopping[270],5,11,[0,0,0])
        my_model.set_hop(hopping[271],7,9,[0,0,0])
        my_model.set_hop(hopping[272],9,11,[0,0,0])
        my_model.set_hop(hopping[273],9,15,[0,0,0])
        my_model.set_hop(hopping[274],11,13,[0,0,0])
        my_model.set_hop(hopping[275],13,15,[0,0,0])
        my_model.set_hop(hopping[276],5,7,[0,0,0])
        my_model.set_hop(hopping[277],4,14,[0,0,0])
        my_model.set_hop(hopping[278],6,12,[0,0,0])
        my_model.set_hop(hopping[279],8,10,[0,0,0])
        my_model.set_hop(hopping[280],4,10,[0,0,0])
        my_model.set_hop(hopping[281],6,8,[0,0,0])
        my_model.set_hop(hopping[282],8,14,[0,0,0])
        my_model.set_hop(hopping[283],10,12,[0,0,0])
        my_model.set_hop(hopping[284],12,14,[0,0,0])
        my_model.set_hop(hopping[285],4,6,[0,0,0])
        my_model.set_hop(hopping[286],8,15,[0,0,-1])
        my_model.set_hop(hopping[287],10,13,[0,0,-1])
        my_model.set_hop(hopping[288],9,14,[0,0,1])
        my_model.set_hop(hopping[289],11,12,[0,0,1])
        my_model.set_hop(hopping[290],4,11,[0,0,-1])
        my_model.set_hop(hopping[291],7,8,[0,0,1])
        my_model.set_hop(hopping[292],6,9,[0,0,-1])
        my_model.set_hop(hopping[293],5,10,[0,0,1])
        my_model.set_hop(hopping[294],5,6,[0,0,1])
        my_model.set_hop(hopping[295],5,6,[1,1,1])
        my_model.set_hop(hopping[296],4,7,[-1,-1,-1])
        my_model.set_hop(hopping[297],4,7,[0,0,-1])
        my_model.set_hop(hopping[298],6,9,[-1,-1,-1])
        my_model.set_hop(hopping[299],5,10,[1,1,1])
        my_model.set_hop(hopping[300],4,11,[-1,-1,-1])
        my_model.set_hop(hopping[301],7,8,[1,1,1])
        my_model.set_hop(hopping[302],6,13,[-1,-1,-1])
        my_model.set_hop(hopping[303],6,13,[0,0,-1])
        my_model.set_hop(hopping[304],5,14,[0,0,1])
        my_model.set_hop(hopping[305],5,14,[1,1,1])
        my_model.set_hop(hopping[306],4,15,[-1,-1,-1])
        my_model.set_hop(hopping[307],4,15,[0,0,-1])
        my_model.set_hop(hopping[308],7,12,[0,0,1])
        my_model.set_hop(hopping[309],7,12,[1,1,1])
        my_model.set_hop(hopping[310],8,11,[-1,-1,-1])
        my_model.set_hop(hopping[311],8,11,[0,0,-1])
        my_model.set_hop(hopping[312],9,10,[0,0,1])
        my_model.set_hop(hopping[313],9,10,[1,1,1])
        my_model.set_hop(hopping[314],8,15,[-1,-1,-1])
        my_model.set_hop(hopping[315],10,13,[-1,-1,-1])
        my_model.set_hop(hopping[316],9,14,[1,1,1])
        my_model.set_hop(hopping[317],11,12,[1,1,1])
        my_model.set_hop(hopping[318],12,15,[-1,-1,-1])
        my_model.set_hop(hopping[319],12,15,[0,0,-1])
        my_model.set_hop(hopping[320],13,14,[0,0,1])
        my_model.set_hop(hopping[321],13,14,[1,1,1])
        my_model.set_hop(hopping[322],4,10,[0,-1,0])
        my_model.set_hop(hopping[323],6,8,[0,1,0])
        my_model.set_hop(hopping[324],8,14,[0,-1,0])
        my_model.set_hop(hopping[325],10,12,[0,1,0])
        my_model.set_hop(hopping[326],12,14,[0,-1,0])
        my_model.set_hop(hopping[327],12,14,[1,0,0])
        my_model.set_hop(hopping[328],10,12,[-1,0,0])
        my_model.set_hop(hopping[329],8,14,[1,0,0])
        my_model.set_hop(hopping[330],6,12,[-1,0,0])
        my_model.set_hop(hopping[331],4,14,[0,-1,0])
        my_model.set_hop(hopping[332],6,12,[0,1,0])
        my_model.set_hop(hopping[333],4,14,[1,0,0])
        my_model.set_hop(hopping[334],8,10,[0,-1,0])
        my_model.set_hop(hopping[335],8,10,[1,0,0])
        my_model.set_hop(hopping[336],6,8,[-1,0,0])
        my_model.set_hop(hopping[337],4,10,[1,0,0])
        my_model.set_hop(hopping[338],4,6,[0,-1,0])
        my_model.set_hop(hopping[339],4,6,[1,0,0])
        my_model.set_hop(hopping[340],7,9,[0,-1,0])
        my_model.set_hop(hopping[341],5,11,[0,1,0])
        my_model.set_hop(hopping[342],11,13,[0,-1,0])
        my_model.set_hop(hopping[343],9,15,[0,1,0])
        my_model.set_hop(hopping[344],13,15,[-1,0,0])
        my_model.set_hop(hopping[345],13,15,[0,1,0])
        my_model.set_hop(hopping[346],9,15,[-1,0,0])
        my_model.set_hop(hopping[347],11,13,[1,0,0])
        my_model.set_hop(hopping[348],5,15,[-1,0,0])
        my_model.set_hop(hopping[349],7,13,[1,0,0])
        my_model.set_hop(hopping[350],7,13,[0,-1,0])
        my_model.set_hop(hopping[351],5,15,[0,1,0])
        my_model.set_hop(hopping[352],9,11,[0,1,0])
        my_model.set_hop(hopping[353],9,11,[-1,0,0])
        my_model.set_hop(hopping[354],5,11,[-1,0,0])
        my_model.set_hop(hopping[355],7,9,[1,0,0])
        my_model.set_hop(hopping[356],5,7,[-1,0,0])
        my_model.set_hop(hopping[357],5,7,[0,1,0])
        my_model.set_hop(hopping[358],10,13,[-1,-1,0])
        my_model.set_hop(hopping[359],9,14,[1,1,0])
        my_model.set_hop(hopping[360],6,13,[-1,-1,0])
        my_model.set_hop(hopping[361],5,14,[0,0,0])
        my_model.set_hop(hopping[362],6,13,[0,0,0])
        my_model.set_hop(hopping[363],5,14,[1,1,0])
        my_model.set_hop(hopping[364],5,10,[0,0,0])
        my_model.set_hop(hopping[365],6,9,[0,0,0])
        my_model.set_hop(hopping[366],5,6,[0,0,0])
        my_model.set_hop(hopping[367],5,6,[1,1,0])
        my_model.set_hop(hopping[368],6,9,[-1,-1,0])
        my_model.set_hop(hopping[369],5,10,[1,1,0])
        my_model.set_hop(hopping[370],9,10,[0,0,0])
        my_model.set_hop(hopping[371],9,10,[1,1,0])
        my_model.set_hop(hopping[372],9,14,[0,0,0])
        my_model.set_hop(hopping[373],10,13,[0,0,0])
        my_model.set_hop(hopping[374],13,14,[0,0,0])
        my_model.set_hop(hopping[375],13,14,[1,1,0])
        my_model.set_hop(hopping[376],8,15,[-1,-1,0])
        my_model.set_hop(hopping[377],11,12,[1,1,0])
        my_model.set_hop(hopping[378],4,15,[-1,-1,0])
        my_model.set_hop(hopping[379],4,15,[0,0,0])
        my_model.set_hop(hopping[380],7,12,[0,0,0])
        my_model.set_hop(hopping[381],7,12,[1,1,0])
        my_model.set_hop(hopping[382],4,11,[0,0,0])
        my_model.set_hop(hopping[383],7,8,[0,0,0])
        my_model.set_hop(hopping[384],4,7,[-1,-1,0])
        my_model.set_hop(hopping[385],4,7,[0,0,0])
        my_model.set_hop(hopping[386],4,11,[-1,-1,0])
        my_model.set_hop(hopping[387],7,8,[1,1,0])
        my_model.set_hop(hopping[388],8,11,[-1,-1,0])
        my_model.set_hop(hopping[389],8,11,[0,0,0])
        my_model.set_hop(hopping[390],8,15,[0,0,0])
        my_model.set_hop(hopping[391],11,12,[0,0,0])
        my_model.set_hop(hopping[392],12,15,[-1,-1,0])
        my_model.set_hop(hopping[393],12,15,[0,0,0])
        my_model.set_hop(hopping[394],8,9,[0,-1,0])
        my_model.set_hop(hopping[395],4,9,[0,-1,0])
        my_model.set_hop(hopping[396],8,13,[0,-1,0])
        my_model.set_hop(hopping[397],5,8,[0,1,0])
        my_model.set_hop(hopping[398],9,12,[0,1,0])
        my_model.set_hop(hopping[399],4,5,[0,-1,0])
        my_model.set_hop(hopping[400],4,13,[0,-1,0])
        my_model.set_hop(hopping[401],5,12,[0,1,0])
        my_model.set_hop(hopping[402],12,13,[0,-1,0])
        my_model.set_hop(hopping[403],4,13,[0,-1,-1])
        my_model.set_hop(hopping[404],5,12,[0,1,1])
        my_model.set_hop(hopping[405],8,9,[0,-1,-1])
        my_model.set_hop(hopping[406],4,9,[0,-1,-1])
        my_model.set_hop(hopping[407],8,13,[0,-1,-1])
        my_model.set_hop(hopping[408],5,8,[0,1,1])
        my_model.set_hop(hopping[409],9,12,[0,1,1])
        my_model.set_hop(hopping[410],12,13,[0,-1,-1])
        my_model.set_hop(hopping[411],4,5,[0,-1,-1])
        my_model.set_hop(hopping[412],4,9,[0,0,-1])
        my_model.set_hop(hopping[413],5,8,[0,0,1])
        my_model.set_hop(hopping[414],8,13,[0,0,-1])
        my_model.set_hop(hopping[415],9,12,[0,0,1])
        my_model.set_hop(hopping[416],8,9,[-1,-1,-1])
        my_model.set_hop(hopping[417],8,9,[0,0,-1])
        my_model.set_hop(hopping[418],8,13,[-1,-1,-1])
        my_model.set_hop(hopping[419],9,12,[1,1,1])
        my_model.set_hop(hopping[420],4,9,[-1,-1,-1])
        my_model.set_hop(hopping[421],5,8,[1,1,1])
        my_model.set_hop(hopping[422],12,13,[-1,-1,-1])
        my_model.set_hop(hopping[423],12,13,[0,0,-1])
        my_model.set_hop(hopping[424],4,13,[-1,-1,-1])
        my_model.set_hop(hopping[425],4,13,[0,0,-1])
        my_model.set_hop(hopping[426],5,12,[0,0,1])
        my_model.set_hop(hopping[427],5,12,[1,1,1])
        my_model.set_hop(hopping[428],4,5,[-1,-1,-1])
        my_model.set_hop(hopping[429],4,5,[0,0,-1])
        my_model.set_hop(hopping[430],5,11,[-1,-1,0])
        my_model.set_hop(hopping[431],6,8,[-1,-1,0])
        my_model.set_hop(hopping[432],4,14,[-1,-1,0])
        my_model.set_hop(hopping[433],6,12,[-1,-1,0])
        my_model.set_hop(hopping[434],5,15,[-1,-1,0])
        my_model.set_hop(hopping[435],7,13,[-1,-1,0])
        my_model.set_hop(hopping[436],8,14,[-1,-1,0])
        my_model.set_hop(hopping[437],11,13,[-1,-1,0])
        my_model.set_hop(hopping[438],13,15,[-1,-1,0])
        my_model.set_hop(hopping[439],12,14,[-1,-1,0])
        my_model.set_hop(hopping[440],9,15,[-1,-1,0])
        my_model.set_hop(hopping[441],10,12,[-1,-1,0])
        my_model.set_hop(hopping[442],8,10,[-1,-1,0])
        my_model.set_hop(hopping[443],9,11,[-1,-1,0])
        my_model.set_hop(hopping[444],4,10,[-1,-1,0])
        my_model.set_hop(hopping[445],7,9,[-1,-1,0])
        my_model.set_hop(hopping[446],4,6,[-1,-1,0])
        my_model.set_hop(hopping[447],5,7,[-1,-1,0])
        my_model.set_hop(hopping[448],10,11,[-1,0,-1])
        my_model.set_hop(hopping[449],6,11,[-1,0,-1])
        my_model.set_hop(hopping[450],7,10,[1,0,1])
        my_model.set_hop(hopping[451],10,15,[-1,0,-1])
        my_model.set_hop(hopping[452],11,14,[1,0,1])
        my_model.set_hop(hopping[453],6,7,[-1,0,-1])
        my_model.set_hop(hopping[454],6,15,[-1,0,-1])
        my_model.set_hop(hopping[455],7,14,[1,0,1])
        my_model.set_hop(hopping[456],14,15,[-1,0,-1])
        my_model.set_hop(hopping[457],8,13,[-1,-2,0])
        my_model.set_hop(hopping[458],9,12,[1,2,0])
        my_model.set_hop(hopping[459],4,9,[-1,-2,0])
        my_model.set_hop(hopping[460],5,8,[1,2,0])
        my_model.set_hop(hopping[461],4,5,[-1,-2,0])
        my_model.set_hop(hopping[462],4,5,[1,0,0])
        my_model.set_hop(hopping[463],5,8,[-1,0,0])
        my_model.set_hop(hopping[464],4,9,[1,0,0])
        my_model.set_hop(hopping[465],4,13,[-1,-2,0])
        my_model.set_hop(hopping[466],5,12,[-1,0,0])
        my_model.set_hop(hopping[467],4,13,[1,0,0])
        my_model.set_hop(hopping[468],5,12,[1,2,0])
        my_model.set_hop(hopping[469],8,9,[-1,-2,0])
        my_model.set_hop(hopping[470],8,9,[1,0,0])
        my_model.set_hop(hopping[471],9,12,[-1,0,0])
        my_model.set_hop(hopping[472],8,13,[1,0,0])
        my_model.set_hop(hopping[473],12,13,[-1,-2,0])
        my_model.set_hop(hopping[474],12,13,[1,0,0])
        my_model.set_hop(hopping[475],4,9,[-1,-2,-1])
        my_model.set_hop(hopping[476],5,8,[1,2,1])
        my_model.set_hop(hopping[477],9,12,[-1,0,1])
        my_model.set_hop(hopping[478],8,13,[1,0,-1])
        my_model.set_hop(hopping[479],4,13,[-1,-2,-1])
        my_model.set_hop(hopping[480],5,12,[-1,0,1])
        my_model.set_hop(hopping[481],4,13,[1,0,-1])
        my_model.set_hop(hopping[482],5,12,[1,2,1])
        my_model.set_hop(hopping[483],12,13,[-1,-2,-1])
        my_model.set_hop(hopping[484],12,13,[1,0,-1])
        my_model.set_hop(hopping[485],8,13,[-1,-2,-1])
        my_model.set_hop(hopping[486],9,12,[1,2,1])
        my_model.set_hop(hopping[487],4,5,[-1,-2,-1])
        my_model.set_hop(hopping[488],4,5,[1,0,-1])
        my_model.set_hop(hopping[489],5,8,[-1,0,1])
        my_model.set_hop(hopping[490],4,9,[1,0,-1])
        my_model.set_hop(hopping[491],8,9,[-1,-2,-1])
        my_model.set_hop(hopping[492],8,9,[1,0,-1])
        my_model.set_hop(hopping[493],6,15,[-1,0,0])
        my_model.set_hop(hopping[494],7,14,[1,0,0])
        my_model.set_hop(hopping[495],10,15,[-1,0,0])
        my_model.set_hop(hopping[496],11,14,[1,0,0])
        my_model.set_hop(hopping[497],10,11,[-1,0,0])
        my_model.set_hop(hopping[498],6,11,[-1,0,0])
        my_model.set_hop(hopping[499],7,10,[1,0,0])
        my_model.set_hop(hopping[500],14,15,[-1,0,0])
        my_model.set_hop(hopping[501],6,7,[-1,0,0])
        my_model.set_hop(hopping[502],6,15,[0,0,-1])
        my_model.set_hop(hopping[503],7,14,[0,0,1])
        my_model.set_hop(hopping[504],10,15,[0,0,-1])
        my_model.set_hop(hopping[505],11,14,[0,0,1])
        my_model.set_hop(hopping[506],10,11,[0,0,-1])
        my_model.set_hop(hopping[507],6,11,[0,0,-1])
        my_model.set_hop(hopping[508],7,10,[0,0,1])
        my_model.set_hop(hopping[509],6,7,[0,0,-1])
        my_model.set_hop(hopping[510],14,15,[0,0,-1])
        my_model.set_hop(hopping[511],6,15,[-1,-1,-1])
        my_model.set_hop(hopping[512],7,14,[1,1,1])
        my_model.set_hop(hopping[513],6,11,[-1,-1,-1])
        my_model.set_hop(hopping[514],7,10,[1,1,1])
        my_model.set_hop(hopping[515],10,11,[-1,-1,-1])
        my_model.set_hop(hopping[516],10,15,[-1,-1,-1])
        my_model.set_hop(hopping[517],11,14,[1,1,1])
        my_model.set_hop(hopping[518],6,7,[-1,-1,-1])
        my_model.set_hop(hopping[519],14,15,[-1,-1,-1])
        my_model.set_hop(hopping[520],12,16,[0,0,-1])
        my_model.set_hop(hopping[521],4,16,[0,0,-1])
        my_model.set_hop(hopping[522],9,17,[0,0,1])
        my_model.set_hop(hopping[523],8,16,[0,0,-1])
        my_model.set_hop(hopping[524],5,17,[0,0,1])
        my_model.set_hop(hopping[525],13,17,[0,0,1])
        my_model.set_hop(hopping[526],9,18,[0,0,0])
        my_model.set_hop(hopping[527],5,18,[0,0,0])
        my_model.set_hop(hopping[528],13,18,[0,0,0])
        my_model.set_hop(hopping[529],8,19,[-1,-1,0])
        my_model.set_hop(hopping[530],12,19,[-1,-1,0])
        my_model.set_hop(hopping[531],12,19,[0,0,0])
        my_model.set_hop(hopping[532],4,19,[0,0,0])
        my_model.set_hop(hopping[533],4,19,[-1,-1,0])
        my_model.set_hop(hopping[534],8,19,[0,0,0])
        my_model.set_hop(hopping[535],13,20,[0,0,0])
        my_model.set_hop(hopping[536],9,20,[0,0,0])
        my_model.set_hop(hopping[537],5,20,[0,0,0])
        my_model.set_hop(hopping[538],4,21,[0,0,0])
        my_model.set_hop(hopping[539],8,21,[0,0,0])
        my_model.set_hop(hopping[540],12,21,[0,0,0])
        my_model.set_hop(hopping[541],6,19,[-1,0,0])
        my_model.set_hop(hopping[542],14,19,[-1,0,0])
        my_model.set_hop(hopping[543],10,19,[-1,0,0])
        my_model.set_hop(hopping[544],11,18,[1,0,0])
        my_model.set_hop(hopping[545],15,18,[1,0,0])
        my_model.set_hop(hopping[546],7,18,[1,0,0])
        my_model.set_hop(hopping[547],11,16,[0,0,0])
        my_model.set_hop(hopping[548],7,16,[0,0,0])
        my_model.set_hop(hopping[549],15,16,[0,0,0])
        my_model.set_hop(hopping[550],10,17,[-1,-1,0])
        my_model.set_hop(hopping[551],14,17,[-1,-1,0])
        my_model.set_hop(hopping[552],14,17,[0,0,0])
        my_model.set_hop(hopping[553],6,17,[-1,-1,0])
        my_model.set_hop(hopping[554],6,17,[0,0,0])
        my_model.set_hop(hopping[555],10,17,[0,0,0])
        my_model.set_hop(hopping[556],7,20,[0,0,0])
        my_model.set_hop(hopping[557],15,20,[0,0,0])
        my_model.set_hop(hopping[558],11,20,[0,0,0])
        my_model.set_hop(hopping[559],10,21,[0,0,0])
        my_model.set_hop(hopping[560],14,21,[0,0,0])
        my_model.set_hop(hopping[561],6,21,[0,0,0])
        my_model.set_hop(hopping[562],11,21,[0,0,0])
        my_model.set_hop(hopping[563],15,21,[0,0,0])
        my_model.set_hop(hopping[564],6,20,[0,0,0])
        my_model.set_hop(hopping[565],7,21,[0,0,0])
        my_model.set_hop(hopping[566],14,20,[0,0,0])
        my_model.set_hop(hopping[567],10,20,[0,0,0])
        my_model.set_hop(hopping[568],10,20,[-1,-1,0])
        my_model.set_hop(hopping[569],6,20,[-1,-1,0])
        my_model.set_hop(hopping[570],14,20,[-1,-1,0])
        my_model.set_hop(hopping[571],4,17,[-1,-1,0])
        my_model.set_hop(hopping[572],4,17,[0,0,0])
        my_model.set_hop(hopping[573],8,17,[-1,-1,0])
        my_model.set_hop(hopping[574],9,16,[0,0,0])
        my_model.set_hop(hopping[575],12,17,[-1,-1,0])
        my_model.set_hop(hopping[576],12,17,[0,0,0])
        my_model.set_hop(hopping[577],13,16,[0,0,0])
        my_model.set_hop(hopping[578],13,16,[1,1,0])
        my_model.set_hop(hopping[579],8,17,[0,0,0])
        my_model.set_hop(hopping[580],9,16,[1,1,0])
        my_model.set_hop(hopping[581],5,16,[0,0,0])
        my_model.set_hop(hopping[582],5,16,[1,1,0])
        my_model.set_hop(hopping[583],5,16,[0,1,0])
        my_model.set_hop(hopping[584],12,17,[0,-1,0])
        my_model.set_hop(hopping[585],8,17,[0,-1,0])
        my_model.set_hop(hopping[586],9,16,[0,1,0])
        my_model.set_hop(hopping[587],13,16,[0,1,0])
        my_model.set_hop(hopping[588],4,17,[0,-1,0])
        my_model.set_hop(hopping[589],8,20,[0,-1,0])
        my_model.set_hop(hopping[590],12,20,[0,-1,0])
        my_model.set_hop(hopping[591],4,20,[0,-1,0])
        my_model.set_hop(hopping[592],5,21,[0,1,0])
        my_model.set_hop(hopping[593],13,21,[0,1,0])
        my_model.set_hop(hopping[594],9,21,[0,1,0])
        my_model.set_hop(hopping[595],14,18,[0,0,-1])
        my_model.set_hop(hopping[596],6,18,[0,0,-1])
        my_model.set_hop(hopping[597],10,18,[0,0,-1])
        my_model.set_hop(hopping[598],11,19,[0,0,1])
        my_model.set_hop(hopping[599],7,19,[0,0,1])
        my_model.set_hop(hopping[600],15,19,[0,0,1])
        my_model.set_hop(hopping[601],13,19,[-1,0,0])
        my_model.set_hop(hopping[602],13,19,[0,1,0])
        my_model.set_hop(hopping[603],9,19,[-1,0,0])
        my_model.set_hop(hopping[604],5,19,[-1,0,0])
        my_model.set_hop(hopping[605],5,19,[0,1,0])
        my_model.set_hop(hopping[606],9,19,[0,1,0])
        my_model.set_hop(hopping[607],8,18,[0,-1,0])
        my_model.set_hop(hopping[608],4,18,[0,-1,0])
        my_model.set_hop(hopping[609],4,18,[1,0,0])
        my_model.set_hop(hopping[610],8,18,[1,0,0])
        my_model.set_hop(hopping[611],12,18,[0,-1,0])
        my_model.set_hop(hopping[612],12,18,[1,0,0])
        my_model.set_hop(hopping[613],12,18,[0,-1,-1])
        my_model.set_hop(hopping[614],12,18,[1,0,-1])
        my_model.set_hop(hopping[615],8,18,[0,-1,-1])
        my_model.set_hop(hopping[616],8,18,[1,0,-1])
        my_model.set_hop(hopping[617],4,18,[0,-1,-1])
        my_model.set_hop(hopping[618],4,18,[1,0,-1])
        my_model.set_hop(hopping[619],5,19,[-1,0,1])
        my_model.set_hop(hopping[620],5,19,[0,1,1])
        my_model.set_hop(hopping[621],9,19,[-1,0,1])
        my_model.set_hop(hopping[622],9,19,[0,1,1])
        my_model.set_hop(hopping[623],13,19,[-1,0,1])
        my_model.set_hop(hopping[624],13,19,[0,1,1])
        my_model.set_hop(hopping[625],8,16,[-1,-1,-1])
        my_model.set_hop(hopping[626],9,17,[-1,-1,1])
        my_model.set_hop(hopping[627],12,16,[-1,-1,-1])
        my_model.set_hop(hopping[628],12,16,[1,1,-1])
        my_model.set_hop(hopping[629],4,16,[-1,-1,-1])
        my_model.set_hop(hopping[630],4,16,[1,1,-1])
        my_model.set_hop(hopping[631],5,17,[-1,-1,1])
        my_model.set_hop(hopping[632],5,17,[1,1,1])
        my_model.set_hop(hopping[633],13,17,[-1,-1,1])
        my_model.set_hop(hopping[634],13,17,[1,1,1])
        my_model.set_hop(hopping[635],8,16,[1,1,-1])
        my_model.set_hop(hopping[636],9,17,[1,1,1])
        my_model.set_hop(hopping[637],15,17,[0,0,0])
        my_model.set_hop(hopping[638],7,17,[0,0,0])
        my_model.set_hop(hopping[639],11,17,[0,0,0])
        my_model.set_hop(hopping[640],10,16,[0,0,0])
        my_model.set_hop(hopping[641],6,16,[0,0,0])
        my_model.set_hop(hopping[642],14,16,[0,0,0])
        my_model.set_hop(hopping[643],9,20,[-1,-1,0])
        my_model.set_hop(hopping[644],8,21,[-1,-1,0])
        my_model.set_hop(hopping[645],4,21,[-1,-1,0])
        my_model.set_hop(hopping[646],13,20,[-1,-1,0])
        my_model.set_hop(hopping[647],12,21,[-1,-1,0])
        my_model.set_hop(hopping[648],5,20,[-1,-1,0])
        my_model.set_hop(hopping[649],7,17,[0,0,1])
        my_model.set_hop(hopping[650],11,17,[0,0,1])
        my_model.set_hop(hopping[651],15,17,[0,0,1])
        my_model.set_hop(hopping[652],14,16,[0,0,-1])
        my_model.set_hop(hopping[653],10,16,[0,0,-1])
        my_model.set_hop(hopping[654],6,16,[0,0,-1])
        my_model.set_hop(hopping[655],10,19,[-2,-1,0])
        my_model.set_hop(hopping[656],11,18,[0,-1,0])
        my_model.set_hop(hopping[657],6,19,[-2,-1,0])
        my_model.set_hop(hopping[658],14,19,[-2,-1,0])
        my_model.set_hop(hopping[659],15,18,[0,-1,0])
        my_model.set_hop(hopping[660],15,18,[2,1,0])
        my_model.set_hop(hopping[661],7,18,[0,-1,0])
        my_model.set_hop(hopping[662],7,18,[2,1,0])
        my_model.set_hop(hopping[663],11,18,[2,1,0])
        my_model.set_hop(hopping[664],6,19,[0,1,0])
        my_model.set_hop(hopping[665],14,19,[0,1,0])
        my_model.set_hop(hopping[666],10,19,[0,1,0])
        my_model.set_hop(hopping[667],10,21,[-1,-1,0])
        my_model.set_hop(hopping[668],11,20,[-1,-1,0])
        my_model.set_hop(hopping[669],7,20,[-1,-1,0])
        my_model.set_hop(hopping[670],15,20,[-1,-1,0])
        my_model.set_hop(hopping[671],14,21,[-1,-1,0])
        my_model.set_hop(hopping[672],6,21,[-1,-1,0])
        my_model.set_hop(hopping[673],13,21,[1,1,0])
        my_model.set_hop(hopping[674],9,21,[1,1,0])
        my_model.set_hop(hopping[675],5,21,[1,1,0])
        my_model.set_hop(hopping[676],4,20,[-1,-1,0])
        my_model.set_hop(hopping[677],8,20,[-1,-1,0])
        my_model.set_hop(hopping[678],12,20,[-1,-1,0])
        my_model.set_hop(hopping[679],12,21,[0,0,-1])
        my_model.set_hop(hopping[680],5,20,[0,0,1])
        my_model.set_hop(hopping[681],9,20,[0,0,1])
        my_model.set_hop(hopping[682],8,21,[0,0,-1])
        my_model.set_hop(hopping[683],4,21,[0,0,-1])
        my_model.set_hop(hopping[684],13,20,[0,0,1])
        my_model.set_hop(hopping[685],6,16,[-1,0,-1])
        my_model.set_hop(hopping[686],6,16,[0,1,-1])
        my_model.set_hop(hopping[687],14,16,[-1,0,-1])
        my_model.set_hop(hopping[688],14,16,[0,1,-1])
        my_model.set_hop(hopping[689],10,16,[-1,0,-1])
        my_model.set_hop(hopping[690],10,16,[0,1,-1])
        my_model.set_hop(hopping[691],11,17,[0,-1,1])
        my_model.set_hop(hopping[692],11,17,[1,0,1])
        my_model.set_hop(hopping[693],15,17,[0,-1,1])
        my_model.set_hop(hopping[694],15,17,[1,0,1])
        my_model.set_hop(hopping[695],7,17,[0,-1,1])
        my_model.set_hop(hopping[696],7,17,[1,0,1])
        my_model.set_hop(hopping[697],12,20,[-1,-1,-1])
        my_model.set_hop(hopping[698],12,20,[0,0,-1])
        my_model.set_hop(hopping[699],4,20,[-1,-1,-1])
        my_model.set_hop(hopping[700],4,20,[0,0,-1])
        my_model.set_hop(hopping[701],8,20,[-1,-1,-1])
        my_model.set_hop(hopping[702],9,21,[0,0,1])
        my_model.set_hop(hopping[703],8,20,[0,0,-1])
        my_model.set_hop(hopping[704],9,21,[1,1,1])
        my_model.set_hop(hopping[705],5,21,[0,0,1])
        my_model.set_hop(hopping[706],5,21,[1,1,1])
        my_model.set_hop(hopping[707],13,21,[0,0,1])
        my_model.set_hop(hopping[708],13,21,[1,1,1])
        my_model.set_hop(hopping[709],7,18,[1,1,0])
        my_model.set_hop(hopping[710],15,18,[1,1,0])
        my_model.set_hop(hopping[711],11,18,[1,1,0])
        my_model.set_hop(hopping[712],7,18,[0,0,0])
        my_model.set_hop(hopping[713],11,18,[0,0,0])
        my_model.set_hop(hopping[714],10,19,[-1,-1,0])
        my_model.set_hop(hopping[715],14,19,[-1,-1,0])
        my_model.set_hop(hopping[716],14,19,[0,0,0])
        my_model.set_hop(hopping[717],15,18,[0,0,0])
        my_model.set_hop(hopping[718],10,19,[0,0,0])
        my_model.set_hop(hopping[719],6,19,[-1,-1,0])
        my_model.set_hop(hopping[720],6,19,[0,0,0])
        my_model.set_hop(hopping[721],5,16,[-1,0,0])
        my_model.set_hop(hopping[722],5,16,[1,2,0])
        my_model.set_hop(hopping[723],8,17,[-1,-2,0])
        my_model.set_hop(hopping[724],9,16,[-1,0,0])
        my_model.set_hop(hopping[725],12,17,[-1,-2,0])
        my_model.set_hop(hopping[726],12,17,[1,0,0])
        my_model.set_hop(hopping[727],13,16,[-1,0,0])
        my_model.set_hop(hopping[728],13,16,[1,2,0])
        my_model.set_hop(hopping[729],8,17,[1,0,0])
        my_model.set_hop(hopping[730],9,16,[1,2,0])
        my_model.set_hop(hopping[731],4,17,[-1,-2,0])
        my_model.set_hop(hopping[732],4,17,[1,0,0])
        my_model.set_hop(hopping[733],6,21,[-1,0,0])
        my_model.set_hop(hopping[734],6,21,[0,1,0])
        my_model.set_hop(hopping[735],10,21,[-1,0,0])
        my_model.set_hop(hopping[736],11,20,[0,-1,0])
        my_model.set_hop(hopping[737],15,20,[0,-1,0])
        my_model.set_hop(hopping[738],15,20,[1,0,0])
        my_model.set_hop(hopping[739],14,21,[-1,0,0])
        my_model.set_hop(hopping[740],14,21,[0,1,0])
        my_model.set_hop(hopping[741],10,21,[0,1,0])
        my_model.set_hop(hopping[742],11,20,[1,0,0])
        my_model.set_hop(hopping[743],7,20,[0,-1,0])
        my_model.set_hop(hopping[744],7,20,[1,0,0])
        my_model.set_hop(hopping[745],12,20,[1,0,0])
        my_model.set_hop(hopping[746],4,20,[1,0,0])
        my_model.set_hop(hopping[747],8,20,[1,0,0])
        my_model.set_hop(hopping[748],8,20,[-1,-2,0])
        my_model.set_hop(hopping[749],9,21,[-1,0,0])
        my_model.set_hop(hopping[750],5,21,[-1,0,0])
        my_model.set_hop(hopping[751],5,21,[1,2,0])
        my_model.set_hop(hopping[752],13,21,[-1,0,0])
        my_model.set_hop(hopping[753],13,21,[1,2,0])
        my_model.set_hop(hopping[754],12,20,[-1,-2,0])
        my_model.set_hop(hopping[755],4,20,[-1,-2,0])
        my_model.set_hop(hopping[756],9,21,[1,2,0])
        my_model.set_hop(hopping[757],14,18,[-1,-1,-1])
        my_model.set_hop(hopping[758],14,18,[1,1,-1])
        my_model.set_hop(hopping[759],10,18,[-1,-1,-1])
        my_model.set_hop(hopping[760],11,19,[-1,-1,1])
        my_model.set_hop(hopping[761],6,18,[-1,-1,-1])
        my_model.set_hop(hopping[762],6,18,[1,1,-1])
        my_model.set_hop(hopping[763],7,19,[1,1,1])
        my_model.set_hop(hopping[764],7,19,[-1,-1,1])
        my_model.set_hop(hopping[765],10,18,[1,1,-1])
        my_model.set_hop(hopping[766],11,19,[1,1,1])
        my_model.set_hop(hopping[767],15,19,[-1,-1,1])
        my_model.set_hop(hopping[768],15,19,[1,1,1])
        my_model.set_hop(hopping[769],10,18,[-1,-1,0])
        my_model.set_hop(hopping[770],11,19,[-1,-1,0])
        my_model.set_hop(hopping[771],6,18,[-1,-1,0])
        my_model.set_hop(hopping[772],15,19,[-1,-1,0])
        my_model.set_hop(hopping[773],14,18,[-1,-1,0])
        my_model.set_hop(hopping[774],7,19,[-1,-1,0])
        my_model.set_hop(hopping[775],7,19,[-1,0,0])
        my_model.set_hop(hopping[776],15,19,[-1,0,0])
        my_model.set_hop(hopping[777],11,19,[-1,0,0])
        my_model.set_hop(hopping[778],7,19,[0,1,0])
        my_model.set_hop(hopping[779],15,19,[0,1,0])
        my_model.set_hop(hopping[780],10,18,[0,-1,0])
        my_model.set_hop(hopping[781],10,18,[1,0,0])
        my_model.set_hop(hopping[782],11,19,[0,1,0])
        my_model.set_hop(hopping[783],14,18,[0,-1,0])
        my_model.set_hop(hopping[784],14,18,[1,0,0])
        my_model.set_hop(hopping[785],6,18,[0,-1,0])
        my_model.set_hop(hopping[786],6,18,[1,0,0])
        my_model.set_hop(hopping[787],14,20,[-1,-1,-1])
        my_model.set_hop(hopping[788],14,20,[0,0,-1])
        my_model.set_hop(hopping[789],10,20,[-1,-1,-1])
        my_model.set_hop(hopping[790],11,21,[0,0,1])
        my_model.set_hop(hopping[791],7,21,[1,1,1])
        my_model.set_hop(hopping[792],7,21,[0,0,1])
        my_model.set_hop(hopping[793],6,20,[-1,-1,-1])
        my_model.set_hop(hopping[794],6,20,[0,0,-1])
        my_model.set_hop(hopping[795],10,20,[0,0,-1])
        my_model.set_hop(hopping[796],11,21,[1,1,1])
        my_model.set_hop(hopping[797],15,21,[0,0,1])
        my_model.set_hop(hopping[798],15,21,[1,1,1])
        my_model.set_hop(hopping[799],8,19,[-2,-2,0])
        my_model.set_hop(hopping[800],9,18,[-1,-1,0])
        my_model.set_hop(hopping[801],5,18,[-1,-1,0])
        my_model.set_hop(hopping[802],5,18,[2,2,0])
        my_model.set_hop(hopping[803],13,18,[-1,-1,0])
        my_model.set_hop(hopping[804],13,18,[2,2,0])
        my_model.set_hop(hopping[805],12,19,[-2,-2,0])
        my_model.set_hop(hopping[806],4,19,[-2,-2,0])
        my_model.set_hop(hopping[807],9,18,[2,2,0])
        my_model.set_hop(hopping[808],15,17,[-1,-1,0])
        my_model.set_hop(hopping[809],11,17,[-1,-1,0])
        my_model.set_hop(hopping[810],10,16,[-1,-1,0])
        my_model.set_hop(hopping[811],7,17,[-1,-1,0])
        my_model.set_hop(hopping[812],6,16,[-1,-1,0])
        my_model.set_hop(hopping[813],14,16,[-1,-1,0])
        my_model.set_hop(hopping[814],11,17,[-1,-1,1])
        my_model.set_hop(hopping[815],7,17,[-1,-1,1])
        my_model.set_hop(hopping[816],7,17,[1,1,1])
        my_model.set_hop(hopping[817],15,17,[-1,-1,1])
        my_model.set_hop(hopping[818],15,17,[1,1,1])
        my_model.set_hop(hopping[819],11,17,[1,1,1])
        my_model.set_hop(hopping[820],10,16,[-1,-1,-1])
        my_model.set_hop(hopping[821],14,16,[-1,-1,-1])
        my_model.set_hop(hopping[822],14,16,[1,1,-1])
        my_model.set_hop(hopping[823],6,16,[-1,-1,-1])
        my_model.set_hop(hopping[824],6,16,[1,1,-1])
        my_model.set_hop(hopping[825],10,16,[1,1,-1])
        my_model.set_hop(hopping[826],10,17,[-2,-2,0])
        my_model.set_hop(hopping[827],11,16,[-1,-1,0])
        my_model.set_hop(hopping[828],14,17,[-2,-2,0])
        my_model.set_hop(hopping[829],7,16,[-1,-1,0])
        my_model.set_hop(hopping[830],7,16,[2,2,0])
        my_model.set_hop(hopping[831],6,17,[-2,-2,0])
        my_model.set_hop(hopping[832],15,16,[-1,-1,0])
        my_model.set_hop(hopping[833],15,16,[2,2,0])
        my_model.set_hop(hopping[834],11,16,[2,2,0])
        my_model.set_hop(hopping[835],10,20,[-2,-2,0])
        my_model.set_hop(hopping[836],11,21,[-1,-1,0])
        my_model.set_hop(hopping[837],15,21,[-1,-1,0])
        my_model.set_hop(hopping[838],15,21,[2,2,0])
        my_model.set_hop(hopping[839],6,20,[-2,-2,0])
        my_model.set_hop(hopping[840],7,21,[-1,-1,0])
        my_model.set_hop(hopping[841],7,21,[2,2,0])
        my_model.set_hop(hopping[842],14,20,[-2,-2,0])
        my_model.set_hop(hopping[843],11,21,[2,2,0])
        my_model.set_hop(hopping[844],15,16,[0,0,-1])
        my_model.set_hop(hopping[845],15,16,[1,1,-1])
        my_model.set_hop(hopping[846],10,17,[-1,-1,1])
        my_model.set_hop(hopping[847],11,16,[0,0,-1])
        my_model.set_hop(hopping[848],6,17,[-1,-1,1])
        my_model.set_hop(hopping[849],6,17,[0,0,1])
        my_model.set_hop(hopping[850],7,16,[0,0,-1])
        my_model.set_hop(hopping[851],7,16,[1,1,-1])
        my_model.set_hop(hopping[852],10,17,[0,0,1])
        my_model.set_hop(hopping[853],11,16,[1,1,-1])
        my_model.set_hop(hopping[854],14,17,[-1,-1,1])
        my_model.set_hop(hopping[855],14,17,[0,0,1])
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
        nk = 532

        # call function k_path to construct the actual path
        (self.k_vec, self.k_dist, self.k_node) = my_model.k_path(path, nk, report = False)

        # print('---------------------------------------')
        # print('starting calculation')
        # print('---------------------------------------')
        # print('Calculating bands...')

        # obtain eigenvalues to be plotted
        self.evals = my_model.solve_all(self.k_vec)
        # set the botton of the conduction band to 0
        self.evals = self.evals - np.min(self.evals[0])


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