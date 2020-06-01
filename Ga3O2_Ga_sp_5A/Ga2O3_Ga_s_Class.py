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
    def __init__(self, onsite, hopping, m, r):
        self.onsite = onsite
        self.hopping = hopping
        self.m = m
        self.r = r
    
        # define lattice vectors
        lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
            [-6.1149997711,1.5199999809, 0.0000000000], 
            [-1.3736609922, 0.0000000000, 5.6349851545]]
        # define coordinates of orbitals
        # the four groups are s, px, py, pz in order
        orb = [[0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805], 
            [0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805],
            [0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805],
            [0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805],]

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
        my_model.set_hop(hopping[4],2,3,[-1,0,0])
        my_model.set_hop(hopping[5],2,3,[0,1,0])
        my_model.set_hop(hopping[6],0,2,[0,0,0])
        my_model.set_hop(hopping[7],1,3,[0,0,0])
        my_model.set_hop(hopping[8],1,2,[0,-1,-1])
        my_model.set_hop(hopping[9],1,2,[1,0,-1])
        my_model.set_hop(hopping[10],0,3,[-1,0,1])
        my_model.set_hop(hopping[11],0,3,[0,1,1])
        my_model.set_hop(hopping[12],1,3,[0,-1,0])
        my_model.set_hop(hopping[13],1,3,[1,0,0])
        my_model.set_hop(hopping[14],0,2,[-1,0,0])
        my_model.set_hop(hopping[15],0,2,[0,1,0])
        my_model.set_hop(hopping[16],0,3,[-1,0,0])
        my_model.set_hop(hopping[17],1,2,[0,-1,0])
        my_model.set_hop(hopping[18],0,3,[0,1,0])
        my_model.set_hop(hopping[19],1,2,[1,0,0])
        my_model.set_hop(hopping[20],0,1,[-1,1,0])
        my_model.set_hop(hopping[21],0,1,[-1,1,1])
        my_model.set_hop(hopping[22],0,1,[-1,0,1])
        my_model.set_hop(hopping[23],0,1,[0,1,1])
        my_model.set_hop(hopping[24],0,2,[-1,-1,0])
        my_model.set_hop(hopping[25],1,3,[-1,-1,0])
        my_model.set_hop(hopping[26],2,3,[0,0,1])
        my_model.set_hop(hopping[27],0,1,[-2,0,0])
        my_model.set_hop(hopping[28],0,1,[0,2,0])
        my_model.set_hop(hopping[29],0,1,[-2,0,1])
        my_model.set_hop(hopping[30],0,1,[0,2,1])
        my_model.set_hop(hopping[31],2,3,[0,0,0])
        my_model.set_hop(hopping[32],2,3,[-1,0,1])
        my_model.set_hop(hopping[33],2,3,[0,1,1])
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
        my_model.set_hop(hopping[46],3,14,[0,-1,0])
        my_model.set_hop(hopping[47],3,14,[1,0,0])
        my_model.set_hop(hopping[48],3,6,[0,-1,0])
        my_model.set_hop(hopping[49],3,6,[1,0,0])
        my_model.set_hop(hopping[50],2,11,[0,1,0])
        my_model.set_hop(hopping[51],3,10,[1,0,0])
        my_model.set_hop(hopping[52],2,11,[-1,0,0])
        my_model.set_hop(hopping[53],3,10,[0,-1,0])
        my_model.set_hop(hopping[54],2,7,[-1,0,0])
        my_model.set_hop(hopping[55],2,7,[0,1,0])
        my_model.set_hop(hopping[56],2,15,[-1,0,0])
        my_model.set_hop(hopping[57],2,15,[0,1,0])
        my_model.set_hop(hopping[58],0,6,[0,0,0])
        my_model.set_hop(hopping[59],2,12,[0,0,0])
        my_model.set_hop(hopping[60],0,10,[0,0,0])
        my_model.set_hop(hopping[61],2,8,[0,0,0])
        my_model.set_hop(hopping[62],0,14,[0,0,0])
        my_model.set_hop(hopping[63],2,4,[0,0,0])
        my_model.set_hop(hopping[64],3,5,[0,0,0])
        my_model.set_hop(hopping[65],1,15,[0,0,0])
        my_model.set_hop(hopping[66],1,11,[0,0,0])
        my_model.set_hop(hopping[67],3,9,[0,0,0])
        my_model.set_hop(hopping[68],3,13,[0,0,0])
        my_model.set_hop(hopping[69],1,7,[0,0,0])
        my_model.set_hop(hopping[70],2,13,[-1,0,1])
        my_model.set_hop(hopping[71],2,13,[0,1,1])
        my_model.set_hop(hopping[72],2,9,[0,1,1])
        my_model.set_hop(hopping[73],1,10,[1,0,-1])
        my_model.set_hop(hopping[74],2,5,[-1,0,1])
        my_model.set_hop(hopping[75],2,5,[0,1,1])
        my_model.set_hop(hopping[76],1,6,[0,-1,-1])
        my_model.set_hop(hopping[77],1,6,[1,0,-1])
        my_model.set_hop(hopping[78],2,9,[-1,0,1])
        my_model.set_hop(hopping[79],1,10,[0,-1,-1])
        my_model.set_hop(hopping[80],1,14,[0,-1,-1])
        my_model.set_hop(hopping[81],1,14,[1,0,-1])
        my_model.set_hop(hopping[82],0,15,[-1,0,1])
        my_model.set_hop(hopping[83],0,15,[0,1,1])
        my_model.set_hop(hopping[84],0,11,[0,1,1])
        my_model.set_hop(hopping[85],3,8,[1,0,-1])
        my_model.set_hop(hopping[86],0,7,[-1,0,1])
        my_model.set_hop(hopping[87],0,7,[0,1,1])
        my_model.set_hop(hopping[88],3,4,[0,-1,-1])
        my_model.set_hop(hopping[89],3,4,[1,0,-1])
        my_model.set_hop(hopping[90],0,11,[-1,0,1])
        my_model.set_hop(hopping[91],3,8,[0,-1,-1])
        my_model.set_hop(hopping[92],3,12,[0,-1,-1])
        my_model.set_hop(hopping[93],3,12,[1,0,-1])
        my_model.set_hop(hopping[94],1,7,[0,-1,0])
        my_model.set_hop(hopping[95],1,7,[1,0,0])
        my_model.set_hop(hopping[96],3,9,[0,1,0])
        my_model.set_hop(hopping[97],1,11,[1,0,0])
        my_model.set_hop(hopping[98],1,15,[0,-1,0])
        my_model.set_hop(hopping[99],1,15,[1,0,0])
        my_model.set_hop(hopping[100],3,13,[-1,0,0])
        my_model.set_hop(hopping[101],3,13,[0,1,0])
        my_model.set_hop(hopping[102],3,9,[-1,0,0])
        my_model.set_hop(hopping[103],1,11,[0,-1,0])
        my_model.set_hop(hopping[104],3,5,[-1,0,0])
        my_model.set_hop(hopping[105],3,5,[0,1,0])
        my_model.set_hop(hopping[106],2,4,[0,-1,0])
        my_model.set_hop(hopping[107],2,4,[1,0,0])
        my_model.set_hop(hopping[108],0,10,[0,1,0])
        my_model.set_hop(hopping[109],2,8,[1,0,0])
        my_model.set_hop(hopping[110],2,12,[0,-1,0])
        my_model.set_hop(hopping[111],2,12,[1,0,0])
        my_model.set_hop(hopping[112],0,14,[-1,0,0])
        my_model.set_hop(hopping[113],0,14,[0,1,0])
        my_model.set_hop(hopping[114],0,10,[-1,0,0])
        my_model.set_hop(hopping[115],2,8,[0,-1,0])
        my_model.set_hop(hopping[116],0,6,[-1,0,0])
        my_model.set_hop(hopping[117],0,6,[0,1,0])
        my_model.set_hop(hopping[118],1,14,[0,-1,0])
        my_model.set_hop(hopping[119],1,14,[1,0,0])
        my_model.set_hop(hopping[120],3,12,[0,-1,0])
        my_model.set_hop(hopping[121],3,12,[1,0,0])
        my_model.set_hop(hopping[122],0,11,[0,1,0])
        my_model.set_hop(hopping[123],2,9,[0,1,0])
        my_model.set_hop(hopping[124],1,10,[1,0,0])
        my_model.set_hop(hopping[125],3,8,[1,0,0])
        my_model.set_hop(hopping[126],0,7,[-1,0,0])
        my_model.set_hop(hopping[127],2,5,[-1,0,0])
        my_model.set_hop(hopping[128],0,7,[0,1,0])
        my_model.set_hop(hopping[129],2,5,[0,1,0])
        my_model.set_hop(hopping[130],1,6,[0,-1,0])
        my_model.set_hop(hopping[131],3,4,[0,-1,0])
        my_model.set_hop(hopping[132],1,6,[1,0,0])
        my_model.set_hop(hopping[133],3,4,[1,0,0])
        my_model.set_hop(hopping[134],0,11,[-1,0,0])
        my_model.set_hop(hopping[135],2,9,[-1,0,0])
        my_model.set_hop(hopping[136],1,10,[0,-1,0])
        my_model.set_hop(hopping[137],3,8,[0,-1,0])
        my_model.set_hop(hopping[138],0,15,[-1,0,0])
        my_model.set_hop(hopping[139],0,15,[0,1,0])
        my_model.set_hop(hopping[140],2,13,[-1,0,0])
        my_model.set_hop(hopping[141],2,13,[0,1,0])
        my_model.set_hop(hopping[142],1,12,[1,-1,0])
        my_model.set_hop(hopping[143],1,4,[1,-1,0])
        my_model.set_hop(hopping[144],0,9,[-1,1,0])
        my_model.set_hop(hopping[145],1,8,[1,-1,0])
        my_model.set_hop(hopping[146],0,5,[-1,1,0])
        my_model.set_hop(hopping[147],0,13,[-1,1,0])
        my_model.set_hop(hopping[148],1,4,[1,-1,-1])
        my_model.set_hop(hopping[149],0,13,[-1,1,1])
        my_model.set_hop(hopping[150],0,9,[-1,1,1])
        my_model.set_hop(hopping[151],1,8,[1,-1,-1])
        my_model.set_hop(hopping[152],1,12,[1,-1,-1])
        my_model.set_hop(hopping[153],0,5,[-1,1,1])
        my_model.set_hop(hopping[154],0,5,[-1,0,1])
        my_model.set_hop(hopping[155],0,5,[0,1,1])
        my_model.set_hop(hopping[156],0,13,[-1,0,1])
        my_model.set_hop(hopping[157],0,13,[0,1,1])
        my_model.set_hop(hopping[158],0,9,[0,1,1])
        my_model.set_hop(hopping[159],1,8,[1,0,-1])
        my_model.set_hop(hopping[160],0,9,[-1,0,1])
        my_model.set_hop(hopping[161],1,8,[0,-1,-1])
        my_model.set_hop(hopping[162],1,12,[0,-1,-1])
        my_model.set_hop(hopping[163],1,12,[1,0,-1])
        my_model.set_hop(hopping[164],1,4,[0,-1,-1])
        my_model.set_hop(hopping[165],1,4,[1,0,-1])
        my_model.set_hop(hopping[166],0,6,[-1,-1,0])
        my_model.set_hop(hopping[167],3,5,[-1,-1,0])
        my_model.set_hop(hopping[168],1,15,[-1,-1,0])
        my_model.set_hop(hopping[169],2,12,[-1,-1,0])
        my_model.set_hop(hopping[170],0,14,[-1,-1,0])
        my_model.set_hop(hopping[171],3,13,[-1,-1,0])
        my_model.set_hop(hopping[172],0,10,[-1,-1,0])
        my_model.set_hop(hopping[173],1,11,[-1,-1,0])
        my_model.set_hop(hopping[174],2,8,[-1,-1,0])
        my_model.set_hop(hopping[175],3,9,[-1,-1,0])
        my_model.set_hop(hopping[176],1,7,[-1,-1,0])
        my_model.set_hop(hopping[177],2,4,[-1,-1,0])
        my_model.set_hop(hopping[178],2,15,[0,0,1])
        my_model.set_hop(hopping[179],2,7,[0,0,1])
        my_model.set_hop(hopping[180],3,10,[0,0,-1])
        my_model.set_hop(hopping[181],2,11,[0,0,1])
        my_model.set_hop(hopping[182],3,6,[0,0,-1])
        my_model.set_hop(hopping[183],3,14,[0,0,-1])
        my_model.set_hop(hopping[184],1,12,[0,-2,0])
        my_model.set_hop(hopping[185],1,12,[2,0,0])
        my_model.set_hop(hopping[186],0,9,[0,2,0])
        my_model.set_hop(hopping[187],1,8,[2,0,0])
        my_model.set_hop(hopping[188],1,4,[0,-2,0])
        my_model.set_hop(hopping[189],1,4,[2,0,0])
        my_model.set_hop(hopping[190],0,5,[-2,0,0])
        my_model.set_hop(hopping[191],0,5,[0,2,0])
        my_model.set_hop(hopping[192],0,9,[-2,0,0])
        my_model.set_hop(hopping[193],1,8,[0,-2,0])
        my_model.set_hop(hopping[194],0,13,[-2,0,0])
        my_model.set_hop(hopping[195],0,13,[0,2,0])
        my_model.set_hop(hopping[196],0,9,[0,2,1])
        my_model.set_hop(hopping[197],1,8,[2,0,-1])
        my_model.set_hop(hopping[198],1,4,[0,-2,-1])
        my_model.set_hop(hopping[199],1,4,[2,0,-1])
        my_model.set_hop(hopping[200],0,13,[-2,0,1])
        my_model.set_hop(hopping[201],0,13,[0,2,1])
        my_model.set_hop(hopping[202],1,12,[0,-2,-1])
        my_model.set_hop(hopping[203],1,12,[2,0,-1])
        my_model.set_hop(hopping[204],0,5,[-2,0,1])
        my_model.set_hop(hopping[205],0,5,[0,2,1])
        my_model.set_hop(hopping[206],0,9,[-2,0,1])
        my_model.set_hop(hopping[207],1,8,[0,-2,-1])
        my_model.set_hop(hopping[208],2,7,[0,0,0])
        my_model.set_hop(hopping[209],3,14,[0,0,0])
        my_model.set_hop(hopping[210],2,11,[0,0,0])
        my_model.set_hop(hopping[211],3,10,[0,0,0])
        my_model.set_hop(hopping[212],2,15,[0,0,0])
        my_model.set_hop(hopping[213],3,6,[0,0,0])
        my_model.set_hop(hopping[214],2,15,[-1,0,1])
        my_model.set_hop(hopping[215],2,15,[0,1,1])
        my_model.set_hop(hopping[216],3,6,[0,-1,-1])
        my_model.set_hop(hopping[217],3,6,[1,0,-1])
        my_model.set_hop(hopping[218],2,11,[0,1,1])
        my_model.set_hop(hopping[219],3,10,[1,0,-1])
        my_model.set_hop(hopping[220],2,11,[-1,0,1])
        my_model.set_hop(hopping[221],3,10,[0,-1,-1])
        my_model.set_hop(hopping[222],2,7,[-1,0,1])
        my_model.set_hop(hopping[223],2,7,[0,1,1])
        my_model.set_hop(hopping[224],3,14,[0,-1,-1])
        my_model.set_hop(hopping[225],3,14,[1,0,-1])
        my_model.set_hop(hopping[226],4,8,[-1,-1,0])
        my_model.set_hop(hopping[227],4,12,[-1,-1,0])
        my_model.set_hop(hopping[228],5,9,[-1,-1,0])
        my_model.set_hop(hopping[229],5,13,[-1,-1,0])
        my_model.set_hop(hopping[230],6,10,[-1,-1,0])
        my_model.set_hop(hopping[231],6,14,[-1,-1,0])
        my_model.set_hop(hopping[232],7,11,[-1,-1,0])
        my_model.set_hop(hopping[233],7,15,[-1,-1,0])
        my_model.set_hop(hopping[234],8,12,[-1,-1,0])
        my_model.set_hop(hopping[235],9,13,[-1,-1,0])
        my_model.set_hop(hopping[236],10,14,[-1,-1,0])
        my_model.set_hop(hopping[237],11,15,[-1,-1,0])
        my_model.set_hop(hopping[238],4,4,[-1,-1,0])
        my_model.set_hop(hopping[239],5,5,[-1,-1,0])
        my_model.set_hop(hopping[240],6,6,[-1,-1,0])
        my_model.set_hop(hopping[241],7,7,[-1,-1,0])
        my_model.set_hop(hopping[242],12,12,[-1,-1,0])
        my_model.set_hop(hopping[243],13,13,[-1,-1,0])
        my_model.set_hop(hopping[244],14,14,[-1,-1,0])
        my_model.set_hop(hopping[245],15,15,[-1,-1,0])
        my_model.set_hop(hopping[246],8,8,[-1,-1,0])
        my_model.set_hop(hopping[247],9,9,[-1,-1,0])
        my_model.set_hop(hopping[248],10,10,[-1,-1,0])
        my_model.set_hop(hopping[249],11,11,[-1,-1,0])
        my_model.set_hop(hopping[250],11,14,[0,-1,0])
        my_model.set_hop(hopping[251],10,15,[0,1,0])
        my_model.set_hop(hopping[252],7,10,[0,-1,0])
        my_model.set_hop(hopping[253],6,11,[0,1,0])
        my_model.set_hop(hopping[254],10,11,[-1,0,0])
        my_model.set_hop(hopping[255],10,11,[0,1,0])
        my_model.set_hop(hopping[256],6,11,[-1,0,0])
        my_model.set_hop(hopping[257],7,10,[1,0,0])
        my_model.set_hop(hopping[258],6,7,[-1,0,0])
        my_model.set_hop(hopping[259],6,7,[0,1,0])
        my_model.set_hop(hopping[260],10,15,[-1,0,0])
        my_model.set_hop(hopping[261],11,14,[1,0,0])
        my_model.set_hop(hopping[262],6,15,[-1,0,0])
        my_model.set_hop(hopping[263],7,14,[0,-1,0])
        my_model.set_hop(hopping[264],6,15,[0,1,0])
        my_model.set_hop(hopping[265],7,14,[1,0,0])
        my_model.set_hop(hopping[266],14,15,[-1,0,0])
        my_model.set_hop(hopping[267],14,15,[0,1,0])
        my_model.set_hop(hopping[268],4,14,[0,0,0])
        my_model.set_hop(hopping[269],6,12,[0,0,0])
        my_model.set_hop(hopping[270],4,10,[0,0,0])
        my_model.set_hop(hopping[271],6,8,[0,0,0])
        my_model.set_hop(hopping[272],8,14,[0,0,0])
        my_model.set_hop(hopping[273],10,12,[0,0,0])
        my_model.set_hop(hopping[274],8,10,[0,0,0])
        my_model.set_hop(hopping[275],12,14,[0,0,0])
        my_model.set_hop(hopping[276],4,6,[0,0,0])
        my_model.set_hop(hopping[277],5,15,[0,0,0])
        my_model.set_hop(hopping[278],7,13,[0,0,0])
        my_model.set_hop(hopping[279],5,11,[0,0,0])
        my_model.set_hop(hopping[280],7,9,[0,0,0])
        my_model.set_hop(hopping[281],9,15,[0,0,0])
        my_model.set_hop(hopping[282],11,13,[0,0,0])
        my_model.set_hop(hopping[283],9,11,[0,0,0])
        my_model.set_hop(hopping[284],13,15,[0,0,0])
        my_model.set_hop(hopping[285],5,7,[0,0,0])
        my_model.set_hop(hopping[286],10,13,[-1,0,1])
        my_model.set_hop(hopping[287],9,14,[1,0,-1])
        my_model.set_hop(hopping[288],6,9,[-1,0,1])
        my_model.set_hop(hopping[289],5,10,[1,0,-1])
        my_model.set_hop(hopping[290],5,6,[0,-1,-1])
        my_model.set_hop(hopping[291],5,6,[1,0,-1])
        my_model.set_hop(hopping[292],5,10,[0,-1,-1])
        my_model.set_hop(hopping[293],6,9,[0,1,1])
        my_model.set_hop(hopping[294],6,13,[-1,0,1])
        my_model.set_hop(hopping[295],5,14,[0,-1,-1])
        my_model.set_hop(hopping[296],6,13,[0,1,1])
        my_model.set_hop(hopping[297],5,14,[1,0,-1])
        my_model.set_hop(hopping[298],9,10,[0,-1,-1])
        my_model.set_hop(hopping[299],9,10,[1,0,-1])
        my_model.set_hop(hopping[300],9,14,[0,-1,-1])
        my_model.set_hop(hopping[301],10,13,[0,1,1])
        my_model.set_hop(hopping[302],13,14,[0,-1,-1])
        my_model.set_hop(hopping[303],13,14,[1,0,-1])
        my_model.set_hop(hopping[304],8,15,[-1,0,1])
        my_model.set_hop(hopping[305],11,12,[1,0,-1])
        my_model.set_hop(hopping[306],4,11,[-1,0,1])
        my_model.set_hop(hopping[307],7,8,[1,0,-1])
        my_model.set_hop(hopping[308],4,7,[-1,0,1])
        my_model.set_hop(hopping[309],4,7,[0,1,1])
        my_model.set_hop(hopping[310],7,8,[0,-1,-1])
        my_model.set_hop(hopping[311],4,11,[0,1,1])
        my_model.set_hop(hopping[312],4,15,[-1,0,1])
        my_model.set_hop(hopping[313],7,12,[0,-1,-1])
        my_model.set_hop(hopping[314],4,15,[0,1,1])
        my_model.set_hop(hopping[315],7,12,[1,0,-1])
        my_model.set_hop(hopping[316],8,11,[-1,0,1])
        my_model.set_hop(hopping[317],8,11,[0,1,1])
        my_model.set_hop(hopping[318],11,12,[0,-1,-1])
        my_model.set_hop(hopping[319],8,15,[0,1,1])
        my_model.set_hop(hopping[320],12,15,[-1,0,1])
        my_model.set_hop(hopping[321],12,15,[0,1,1])
        my_model.set_hop(hopping[322],5,11,[0,-1,0])
        my_model.set_hop(hopping[323],7,9,[0,1,0])
        my_model.set_hop(hopping[324],9,15,[0,-1,0])
        my_model.set_hop(hopping[325],11,13,[0,1,0])
        my_model.set_hop(hopping[326],13,15,[0,-1,0])
        my_model.set_hop(hopping[327],13,15,[1,0,0])
        my_model.set_hop(hopping[328],11,13,[-1,0,0])
        my_model.set_hop(hopping[329],9,15,[1,0,0])
        my_model.set_hop(hopping[330],7,13,[-1,0,0])
        my_model.set_hop(hopping[331],5,15,[0,-1,0])
        my_model.set_hop(hopping[332],7,13,[0,1,0])
        my_model.set_hop(hopping[333],5,15,[1,0,0])
        my_model.set_hop(hopping[334],9,11,[0,-1,0])
        my_model.set_hop(hopping[335],9,11,[1,0,0])
        my_model.set_hop(hopping[336],7,9,[-1,0,0])
        my_model.set_hop(hopping[337],5,11,[1,0,0])
        my_model.set_hop(hopping[338],5,7,[0,-1,0])
        my_model.set_hop(hopping[339],5,7,[1,0,0])
        my_model.set_hop(hopping[340],6,8,[0,-1,0])
        my_model.set_hop(hopping[341],4,10,[0,1,0])
        my_model.set_hop(hopping[342],10,12,[0,-1,0])
        my_model.set_hop(hopping[343],8,14,[0,1,0])
        my_model.set_hop(hopping[344],12,14,[-1,0,0])
        my_model.set_hop(hopping[345],12,14,[0,1,0])
        my_model.set_hop(hopping[346],8,14,[-1,0,0])
        my_model.set_hop(hopping[347],10,12,[1,0,0])
        my_model.set_hop(hopping[348],4,14,[-1,0,0])
        my_model.set_hop(hopping[349],6,12,[0,-1,0])
        my_model.set_hop(hopping[350],4,14,[0,1,0])
        my_model.set_hop(hopping[351],6,12,[1,0,0])
        my_model.set_hop(hopping[352],8,10,[-1,0,0])
        my_model.set_hop(hopping[353],8,10,[0,1,0])
        my_model.set_hop(hopping[354],4,10,[-1,0,0])
        my_model.set_hop(hopping[355],6,8,[1,0,0])
        my_model.set_hop(hopping[356],4,6,[-1,0,0])
        my_model.set_hop(hopping[357],4,6,[0,1,0])
        my_model.set_hop(hopping[358],9,14,[0,-1,0])
        my_model.set_hop(hopping[359],11,12,[0,-1,0])
        my_model.set_hop(hopping[360],8,15,[0,1,0])
        my_model.set_hop(hopping[361],10,13,[0,1,0])
        my_model.set_hop(hopping[362],4,15,[-1,0,0])
        my_model.set_hop(hopping[363],6,13,[-1,0,0])
        my_model.set_hop(hopping[364],5,14,[0,-1,0])
        my_model.set_hop(hopping[365],7,12,[0,-1,0])
        my_model.set_hop(hopping[366],4,15,[0,1,0])
        my_model.set_hop(hopping[367],6,13,[0,1,0])
        my_model.set_hop(hopping[368],5,14,[1,0,0])
        my_model.set_hop(hopping[369],7,12,[1,0,0])
        my_model.set_hop(hopping[370],4,11,[-1,0,0])
        my_model.set_hop(hopping[371],6,9,[-1,0,0])
        my_model.set_hop(hopping[372],5,10,[1,0,0])
        my_model.set_hop(hopping[373],7,8,[1,0,0])
        my_model.set_hop(hopping[374],4,7,[-1,0,0])
        my_model.set_hop(hopping[375],5,6,[0,-1,0])
        my_model.set_hop(hopping[376],4,7,[0,1,0])
        my_model.set_hop(hopping[377],5,6,[1,0,0])
        my_model.set_hop(hopping[378],5,10,[0,-1,0])
        my_model.set_hop(hopping[379],7,8,[0,-1,0])
        my_model.set_hop(hopping[380],4,11,[0,1,0])
        my_model.set_hop(hopping[381],6,9,[0,1,0])
        my_model.set_hop(hopping[382],8,11,[-1,0,0])
        my_model.set_hop(hopping[383],9,10,[0,-1,0])
        my_model.set_hop(hopping[384],8,11,[0,1,0])
        my_model.set_hop(hopping[385],9,10,[1,0,0])
        my_model.set_hop(hopping[386],8,15,[-1,0,0])
        my_model.set_hop(hopping[387],10,13,[-1,0,0])
        my_model.set_hop(hopping[388],9,14,[1,0,0])
        my_model.set_hop(hopping[389],11,12,[1,0,0])
        my_model.set_hop(hopping[390],12,15,[-1,0,0])
        my_model.set_hop(hopping[391],12,15,[0,1,0])
        my_model.set_hop(hopping[392],13,14,[0,-1,0])
        my_model.set_hop(hopping[393],13,14,[1,0,0])
        my_model.set_hop(hopping[394],4,9,[-1,1,0])
        my_model.set_hop(hopping[395],8,13,[-1,1,0])
        my_model.set_hop(hopping[396],5,8,[1,-1,0])
        my_model.set_hop(hopping[397],9,12,[1,-1,0])
        my_model.set_hop(hopping[398],8,9,[-1,1,0])
        my_model.set_hop(hopping[399],4,5,[-1,1,0])
        my_model.set_hop(hopping[400],4,13,[-1,1,0])
        my_model.set_hop(hopping[401],5,12,[1,-1,0])
        my_model.set_hop(hopping[402],12,13,[-1,1,0])
        my_model.set_hop(hopping[403],4,13,[-1,1,1])
        my_model.set_hop(hopping[404],5,12,[1,-1,-1])
        my_model.set_hop(hopping[405],4,9,[-1,1,1])
        my_model.set_hop(hopping[406],8,13,[-1,1,1])
        my_model.set_hop(hopping[407],5,8,[1,-1,-1])
        my_model.set_hop(hopping[408],9,12,[1,-1,-1])
        my_model.set_hop(hopping[409],8,9,[-1,1,1])
        my_model.set_hop(hopping[410],12,13,[-1,1,1])
        my_model.set_hop(hopping[411],4,5,[-1,1,1])
        my_model.set_hop(hopping[412],4,9,[-1,0,1])
        my_model.set_hop(hopping[413],5,8,[1,0,-1])
        my_model.set_hop(hopping[414],8,13,[-1,0,1])
        my_model.set_hop(hopping[415],9,12,[1,0,-1])
        my_model.set_hop(hopping[416],8,9,[-1,0,1])
        my_model.set_hop(hopping[417],8,9,[0,1,1])
        my_model.set_hop(hopping[418],9,12,[0,-1,-1])
        my_model.set_hop(hopping[419],8,13,[0,1,1])
        my_model.set_hop(hopping[420],5,8,[0,-1,-1])
        my_model.set_hop(hopping[421],4,9,[0,1,1])
        my_model.set_hop(hopping[422],12,13,[-1,0,1])
        my_model.set_hop(hopping[423],12,13,[0,1,1])
        my_model.set_hop(hopping[424],4,13,[-1,0,1])
        my_model.set_hop(hopping[425],5,12,[0,-1,-1])
        my_model.set_hop(hopping[426],4,13,[0,1,1])
        my_model.set_hop(hopping[427],5,12,[1,0,-1])
        my_model.set_hop(hopping[428],4,5,[-1,0,1])
        my_model.set_hop(hopping[429],4,5,[0,1,1])
        my_model.set_hop(hopping[430],4,10,[-1,-1,0])
        my_model.set_hop(hopping[431],7,9,[-1,-1,0])
        my_model.set_hop(hopping[432],5,15,[-1,-1,0])
        my_model.set_hop(hopping[433],7,13,[-1,-1,0])
        my_model.set_hop(hopping[434],4,14,[-1,-1,0])
        my_model.set_hop(hopping[435],6,12,[-1,-1,0])
        my_model.set_hop(hopping[436],9,15,[-1,-1,0])
        my_model.set_hop(hopping[437],10,12,[-1,-1,0])
        my_model.set_hop(hopping[438],12,14,[-1,-1,0])
        my_model.set_hop(hopping[439],13,15,[-1,-1,0])
        my_model.set_hop(hopping[440],8,14,[-1,-1,0])
        my_model.set_hop(hopping[441],11,13,[-1,-1,0])
        my_model.set_hop(hopping[442],8,10,[-1,-1,0])
        my_model.set_hop(hopping[443],9,11,[-1,-1,0])
        my_model.set_hop(hopping[444],5,11,[-1,-1,0])
        my_model.set_hop(hopping[445],6,8,[-1,-1,0])
        my_model.set_hop(hopping[446],4,6,[-1,-1,0])
        my_model.set_hop(hopping[447],5,7,[-1,-1,0])
        my_model.set_hop(hopping[448],7,10,[0,0,-1])
        my_model.set_hop(hopping[449],11,14,[0,0,-1])
        my_model.set_hop(hopping[450],6,11,[0,0,1])
        my_model.set_hop(hopping[451],10,15,[0,0,1])
        my_model.set_hop(hopping[452],10,11,[0,0,1])
        my_model.set_hop(hopping[453],6,7,[0,0,1])
        my_model.set_hop(hopping[454],7,14,[0,0,-1])
        my_model.set_hop(hopping[455],6,15,[0,0,1])
        my_model.set_hop(hopping[456],14,15,[0,0,1])
        my_model.set_hop(hopping[457],9,12,[0,-2,0])
        my_model.set_hop(hopping[458],8,13,[0,2,0])
        my_model.set_hop(hopping[459],5,8,[0,-2,0])
        my_model.set_hop(hopping[460],4,9,[0,2,0])
        my_model.set_hop(hopping[461],4,5,[-2,0,0])
        my_model.set_hop(hopping[462],4,5,[0,2,0])
        my_model.set_hop(hopping[463],4,9,[-2,0,0])
        my_model.set_hop(hopping[464],5,8,[2,0,0])
        my_model.set_hop(hopping[465],4,13,[-2,0,0])
        my_model.set_hop(hopping[466],5,12,[0,-2,0])
        my_model.set_hop(hopping[467],4,13,[0,2,0])
        my_model.set_hop(hopping[468],5,12,[2,0,0])
        my_model.set_hop(hopping[469],8,9,[-2,0,0])
        my_model.set_hop(hopping[470],8,9,[0,2,0])
        my_model.set_hop(hopping[471],8,13,[-2,0,0])
        my_model.set_hop(hopping[472],9,12,[2,0,0])
        my_model.set_hop(hopping[473],12,13,[-2,0,0])
        my_model.set_hop(hopping[474],12,13,[0,2,0])
        my_model.set_hop(hopping[475],5,8,[0,-2,-1])
        my_model.set_hop(hopping[476],4,9,[0,2,1])
        my_model.set_hop(hopping[477],8,13,[-2,0,1])
        my_model.set_hop(hopping[478],9,12,[2,0,-1])
        my_model.set_hop(hopping[479],4,13,[-2,0,1])
        my_model.set_hop(hopping[480],5,12,[0,-2,-1])
        my_model.set_hop(hopping[481],4,13,[0,2,1])
        my_model.set_hop(hopping[482],5,12,[2,0,-1])
        my_model.set_hop(hopping[483],12,13,[-2,0,1])
        my_model.set_hop(hopping[484],12,13,[0,2,1])
        my_model.set_hop(hopping[485],9,12,[0,-2,-1])
        my_model.set_hop(hopping[486],8,13,[0,2,1])
        my_model.set_hop(hopping[487],4,5,[-2,0,1])
        my_model.set_hop(hopping[488],4,5,[0,2,1])
        my_model.set_hop(hopping[489],4,9,[-2,0,1])
        my_model.set_hop(hopping[490],5,8,[2,0,-1])
        my_model.set_hop(hopping[491],8,9,[-2,0,1])
        my_model.set_hop(hopping[492],8,9,[0,2,1])
        my_model.set_hop(hopping[493],6,15,[0,0,0])
        my_model.set_hop(hopping[494],7,14,[0,0,0])
        my_model.set_hop(hopping[495],6,11,[0,0,0])
        my_model.set_hop(hopping[496],7,10,[0,0,0])
        my_model.set_hop(hopping[497],10,15,[0,0,0])
        my_model.set_hop(hopping[498],11,14,[0,0,0])
        my_model.set_hop(hopping[499],10,11,[0,0,0])
        my_model.set_hop(hopping[500],14,15,[0,0,0])
        my_model.set_hop(hopping[501],6,7,[0,0,0])
        my_model.set_hop(hopping[502],6,15,[-1,0,1])
        my_model.set_hop(hopping[503],7,14,[0,-1,-1])
        my_model.set_hop(hopping[504],6,15,[0,1,1])
        my_model.set_hop(hopping[505],7,14,[1,0,-1])
        my_model.set_hop(hopping[506],10,15,[-1,0,1])
        my_model.set_hop(hopping[507],11,14,[1,0,-1])
        my_model.set_hop(hopping[508],7,10,[0,-1,-1])
        my_model.set_hop(hopping[509],6,11,[0,1,1])
        my_model.set_hop(hopping[510],10,11,[-1,0,1])
        my_model.set_hop(hopping[511],10,11,[0,1,1])
        my_model.set_hop(hopping[512],6,11,[-1,0,1])
        my_model.set_hop(hopping[513],7,10,[1,0,-1])
        my_model.set_hop(hopping[514],11,14,[0,-1,-1])
        my_model.set_hop(hopping[515],10,15,[0,1,1])
        my_model.set_hop(hopping[516],6,7,[-1,0,1])
        my_model.set_hop(hopping[517],6,7,[0,1,1])
        my_model.set_hop(hopping[518],14,15,[-1,0,1])
        my_model.set_hop(hopping[519],14,15,[0,1,1])
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
        return ('%.2f\t%.2f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' 
        %(self.m, self.r, 
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