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
        orb = [[0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805], 
            [0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805],
            [0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805],
            [0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805],
            [0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
            [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805],]

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
        self.my_model.set_hop(hopping[4],2,3,[-1,0,0])
        self.my_model.set_hop(hopping[5],2,3,[0,1,0])
        self.my_model.set_hop(hopping[6],0,2,[0,0,0])
        self.my_model.set_hop(hopping[7],1,3,[0,0,0])
        self.my_model.set_hop(hopping[8],1,2,[0,-1,-1])
        self.my_model.set_hop(hopping[9],1,2,[1,0,-1])
        self.my_model.set_hop(hopping[10],0,3,[-1,0,1])
        self.my_model.set_hop(hopping[11],0,3,[0,1,1])
        self.my_model.set_hop(hopping[12],1,3,[0,-1,0])
        self.my_model.set_hop(hopping[13],1,3,[1,0,0])
        self.my_model.set_hop(hopping[14],0,2,[-1,0,0])
        self.my_model.set_hop(hopping[15],0,2,[0,1,0])
        self.my_model.set_hop(hopping[16],0,3,[-1,0,0])
        self.my_model.set_hop(hopping[17],1,2,[0,-1,0])
        self.my_model.set_hop(hopping[18],0,3,[0,1,0])
        self.my_model.set_hop(hopping[19],1,2,[1,0,0])
        self.my_model.set_hop(hopping[20],0,1,[-1,1,0])
        self.my_model.set_hop(hopping[21],0,1,[-1,1,1])
        self.my_model.set_hop(hopping[22],0,1,[-1,0,1])
        self.my_model.set_hop(hopping[23],0,1,[0,1,1])
        self.my_model.set_hop(hopping[24],0,2,[-1,-1,0])
        self.my_model.set_hop(hopping[25],1,3,[-1,-1,0])
        self.my_model.set_hop(hopping[26],0,2,[1,1,0])
        self.my_model.set_hop(hopping[27],1,3,[1,1,0])
        self.my_model.set_hop(hopping[28],2,3,[0,0,1])
        self.my_model.set_hop(hopping[29],0,1,[-2,0,0])
        self.my_model.set_hop(hopping[30],0,1,[0,2,0])
        self.my_model.set_hop(hopping[31],0,1,[-2,0,1])
        self.my_model.set_hop(hopping[32],0,1,[0,2,1])
        self.my_model.set_hop(hopping[33],2,3,[0,0,0])
        self.my_model.set_hop(hopping[34],2,3,[-1,0,1])
        self.my_model.set_hop(hopping[35],2,3,[0,1,1])
        self.my_model.set_hop(hopping[36],0,8,[1,1,0])
        self.my_model.set_hop(hopping[37],1,9,[1,1,0])
        self.my_model.set_hop(hopping[38],2,10,[1,1,0])
        self.my_model.set_hop(hopping[39],3,11,[1,1,0])
        self.my_model.set_hop(hopping[40],0,4,[-1,-1,0])
        self.my_model.set_hop(hopping[41],0,12,[-1,-1,0])
        self.my_model.set_hop(hopping[42],1,5,[-1,-1,0])
        self.my_model.set_hop(hopping[43],1,13,[-1,-1,0])
        self.my_model.set_hop(hopping[44],2,6,[-1,-1,0])
        self.my_model.set_hop(hopping[45],2,14,[-1,-1,0])
        self.my_model.set_hop(hopping[46],3,7,[-1,-1,0])
        self.my_model.set_hop(hopping[47],3,15,[-1,-1,0])
        self.my_model.set_hop(hopping[48],0,4,[1,1,0])
        self.my_model.set_hop(hopping[49],0,12,[1,1,0])
        self.my_model.set_hop(hopping[50],1,5,[1,1,0])
        self.my_model.set_hop(hopping[51],1,13,[1,1,0])
        self.my_model.set_hop(hopping[52],2,6,[1,1,0])
        self.my_model.set_hop(hopping[53],2,14,[1,1,0])
        self.my_model.set_hop(hopping[54],3,7,[1,1,0])
        self.my_model.set_hop(hopping[55],3,15,[1,1,0])
        self.my_model.set_hop(hopping[56],0,8,[-1,-1,0])
        self.my_model.set_hop(hopping[57],1,9,[-1,-1,0])
        self.my_model.set_hop(hopping[58],2,10,[-1,-1,0])
        self.my_model.set_hop(hopping[59],3,11,[-1,-1,0])
        self.my_model.set_hop(hopping[60],3,14,[0,-1,0])
        self.my_model.set_hop(hopping[61],3,14,[1,0,0])
        self.my_model.set_hop(hopping[62],3,6,[0,-1,0])
        self.my_model.set_hop(hopping[63],3,6,[1,0,0])
        self.my_model.set_hop(hopping[64],2,11,[0,1,0])
        self.my_model.set_hop(hopping[65],3,10,[1,0,0])
        self.my_model.set_hop(hopping[66],2,11,[-1,0,0])
        self.my_model.set_hop(hopping[67],3,10,[0,-1,0])
        self.my_model.set_hop(hopping[68],2,7,[-1,0,0])
        self.my_model.set_hop(hopping[69],2,7,[0,1,0])
        self.my_model.set_hop(hopping[70],2,15,[-1,0,0])
        self.my_model.set_hop(hopping[71],2,15,[0,1,0])
        self.my_model.set_hop(hopping[72],0,6,[0,0,0])
        self.my_model.set_hop(hopping[73],2,12,[0,0,0])
        self.my_model.set_hop(hopping[74],0,10,[0,0,0])
        self.my_model.set_hop(hopping[75],2,8,[0,0,0])
        self.my_model.set_hop(hopping[76],0,14,[0,0,0])
        self.my_model.set_hop(hopping[77],2,4,[0,0,0])
        self.my_model.set_hop(hopping[78],3,5,[0,0,0])
        self.my_model.set_hop(hopping[79],1,15,[0,0,0])
        self.my_model.set_hop(hopping[80],1,11,[0,0,0])
        self.my_model.set_hop(hopping[81],3,9,[0,0,0])
        self.my_model.set_hop(hopping[82],3,13,[0,0,0])
        self.my_model.set_hop(hopping[83],1,7,[0,0,0])
        self.my_model.set_hop(hopping[84],2,13,[-1,0,1])
        self.my_model.set_hop(hopping[85],2,13,[0,1,1])
        self.my_model.set_hop(hopping[86],2,9,[0,1,1])
        self.my_model.set_hop(hopping[87],1,10,[1,0,-1])
        self.my_model.set_hop(hopping[88],2,5,[-1,0,1])
        self.my_model.set_hop(hopping[89],2,5,[0,1,1])
        self.my_model.set_hop(hopping[90],1,6,[0,-1,-1])
        self.my_model.set_hop(hopping[91],1,6,[1,0,-1])
        self.my_model.set_hop(hopping[92],2,9,[-1,0,1])
        self.my_model.set_hop(hopping[93],1,10,[0,-1,-1])
        self.my_model.set_hop(hopping[94],1,14,[0,-1,-1])
        self.my_model.set_hop(hopping[95],1,14,[1,0,-1])
        self.my_model.set_hop(hopping[96],0,15,[-1,0,1])
        self.my_model.set_hop(hopping[97],0,15,[0,1,1])
        self.my_model.set_hop(hopping[98],0,11,[0,1,1])
        self.my_model.set_hop(hopping[99],3,8,[1,0,-1])
        self.my_model.set_hop(hopping[100],0,7,[-1,0,1])
        self.my_model.set_hop(hopping[101],0,7,[0,1,1])
        self.my_model.set_hop(hopping[102],3,4,[0,-1,-1])
        self.my_model.set_hop(hopping[103],3,4,[1,0,-1])
        self.my_model.set_hop(hopping[104],0,11,[-1,0,1])
        self.my_model.set_hop(hopping[105],3,8,[0,-1,-1])
        self.my_model.set_hop(hopping[106],3,12,[0,-1,-1])
        self.my_model.set_hop(hopping[107],3,12,[1,0,-1])
        self.my_model.set_hop(hopping[108],1,7,[0,-1,0])
        self.my_model.set_hop(hopping[109],1,7,[1,0,0])
        self.my_model.set_hop(hopping[110],3,9,[0,1,0])
        self.my_model.set_hop(hopping[111],1,11,[1,0,0])
        self.my_model.set_hop(hopping[112],1,15,[0,-1,0])
        self.my_model.set_hop(hopping[113],1,15,[1,0,0])
        self.my_model.set_hop(hopping[114],3,13,[-1,0,0])
        self.my_model.set_hop(hopping[115],3,13,[0,1,0])
        self.my_model.set_hop(hopping[116],3,9,[-1,0,0])
        self.my_model.set_hop(hopping[117],1,11,[0,-1,0])
        self.my_model.set_hop(hopping[118],3,5,[-1,0,0])
        self.my_model.set_hop(hopping[119],3,5,[0,1,0])
        self.my_model.set_hop(hopping[120],2,4,[0,-1,0])
        self.my_model.set_hop(hopping[121],2,4,[1,0,0])
        self.my_model.set_hop(hopping[122],0,10,[0,1,0])
        self.my_model.set_hop(hopping[123],2,8,[1,0,0])
        self.my_model.set_hop(hopping[124],2,12,[0,-1,0])
        self.my_model.set_hop(hopping[125],2,12,[1,0,0])
        self.my_model.set_hop(hopping[126],0,14,[-1,0,0])
        self.my_model.set_hop(hopping[127],0,14,[0,1,0])
        self.my_model.set_hop(hopping[128],0,10,[-1,0,0])
        self.my_model.set_hop(hopping[129],2,8,[0,-1,0])
        self.my_model.set_hop(hopping[130],0,6,[-1,0,0])
        self.my_model.set_hop(hopping[131],0,6,[0,1,0])
        self.my_model.set_hop(hopping[132],1,14,[0,-1,0])
        self.my_model.set_hop(hopping[133],1,14,[1,0,0])
        self.my_model.set_hop(hopping[134],3,12,[0,-1,0])
        self.my_model.set_hop(hopping[135],3,12,[1,0,0])
        self.my_model.set_hop(hopping[136],0,11,[0,1,0])
        self.my_model.set_hop(hopping[137],2,9,[0,1,0])
        self.my_model.set_hop(hopping[138],1,10,[1,0,0])
        self.my_model.set_hop(hopping[139],3,8,[1,0,0])
        self.my_model.set_hop(hopping[140],0,7,[-1,0,0])
        self.my_model.set_hop(hopping[141],2,5,[-1,0,0])
        self.my_model.set_hop(hopping[142],0,7,[0,1,0])
        self.my_model.set_hop(hopping[143],2,5,[0,1,0])
        self.my_model.set_hop(hopping[144],1,6,[0,-1,0])
        self.my_model.set_hop(hopping[145],3,4,[0,-1,0])
        self.my_model.set_hop(hopping[146],1,6,[1,0,0])
        self.my_model.set_hop(hopping[147],3,4,[1,0,0])
        self.my_model.set_hop(hopping[148],0,11,[-1,0,0])
        self.my_model.set_hop(hopping[149],2,9,[-1,0,0])
        self.my_model.set_hop(hopping[150],1,10,[0,-1,0])
        self.my_model.set_hop(hopping[151],3,8,[0,-1,0])
        self.my_model.set_hop(hopping[152],0,15,[-1,0,0])
        self.my_model.set_hop(hopping[153],0,15,[0,1,0])
        self.my_model.set_hop(hopping[154],2,13,[-1,0,0])
        self.my_model.set_hop(hopping[155],2,13,[0,1,0])
        self.my_model.set_hop(hopping[156],1,12,[1,-1,0])
        self.my_model.set_hop(hopping[157],1,4,[1,-1,0])
        self.my_model.set_hop(hopping[158],0,9,[-1,1,0])
        self.my_model.set_hop(hopping[159],1,8,[1,-1,0])
        self.my_model.set_hop(hopping[160],0,5,[-1,1,0])
        self.my_model.set_hop(hopping[161],0,13,[-1,1,0])
        self.my_model.set_hop(hopping[162],1,4,[1,-1,-1])
        self.my_model.set_hop(hopping[163],0,13,[-1,1,1])
        self.my_model.set_hop(hopping[164],0,9,[-1,1,1])
        self.my_model.set_hop(hopping[165],1,8,[1,-1,-1])
        self.my_model.set_hop(hopping[166],1,12,[1,-1,-1])
        self.my_model.set_hop(hopping[167],0,5,[-1,1,1])
        self.my_model.set_hop(hopping[168],0,5,[-1,0,1])
        self.my_model.set_hop(hopping[169],0,5,[0,1,1])
        self.my_model.set_hop(hopping[170],0,13,[-1,0,1])
        self.my_model.set_hop(hopping[171],0,13,[0,1,1])
        self.my_model.set_hop(hopping[172],0,9,[0,1,1])
        self.my_model.set_hop(hopping[173],1,8,[1,0,-1])
        self.my_model.set_hop(hopping[174],0,9,[-1,0,1])
        self.my_model.set_hop(hopping[175],1,8,[0,-1,-1])
        self.my_model.set_hop(hopping[176],1,12,[0,-1,-1])
        self.my_model.set_hop(hopping[177],1,12,[1,0,-1])
        self.my_model.set_hop(hopping[178],1,4,[0,-1,-1])
        self.my_model.set_hop(hopping[179],1,4,[1,0,-1])
        self.my_model.set_hop(hopping[180],0,6,[-1,-1,0])
        self.my_model.set_hop(hopping[181],3,5,[-1,-1,0])
        self.my_model.set_hop(hopping[182],0,6,[1,1,0])
        self.my_model.set_hop(hopping[183],3,5,[1,1,0])
        self.my_model.set_hop(hopping[184],0,10,[1,1,0])
        self.my_model.set_hop(hopping[185],1,11,[1,1,0])
        self.my_model.set_hop(hopping[186],2,8,[1,1,0])
        self.my_model.set_hop(hopping[187],3,9,[1,1,0])
        self.my_model.set_hop(hopping[188],1,15,[-1,-1,0])
        self.my_model.set_hop(hopping[189],1,15,[1,1,0])
        self.my_model.set_hop(hopping[190],2,12,[-1,-1,0])
        self.my_model.set_hop(hopping[191],2,12,[1,1,0])
        self.my_model.set_hop(hopping[192],0,14,[-1,-1,0])
        self.my_model.set_hop(hopping[193],0,14,[1,1,0])
        self.my_model.set_hop(hopping[194],3,13,[-1,-1,0])
        self.my_model.set_hop(hopping[195],3,13,[1,1,0])
        self.my_model.set_hop(hopping[196],0,10,[-1,-1,0])
        self.my_model.set_hop(hopping[197],1,11,[-1,-1,0])
        self.my_model.set_hop(hopping[198],2,8,[-1,-1,0])
        self.my_model.set_hop(hopping[199],3,9,[-1,-1,0])
        self.my_model.set_hop(hopping[200],1,7,[-1,-1,0])
        self.my_model.set_hop(hopping[201],2,4,[-1,-1,0])
        self.my_model.set_hop(hopping[202],1,7,[1,1,0])
        self.my_model.set_hop(hopping[203],2,4,[1,1,0])
        self.my_model.set_hop(hopping[204],2,15,[0,0,1])
        self.my_model.set_hop(hopping[205],2,7,[0,0,1])
        self.my_model.set_hop(hopping[206],3,10,[0,0,-1])
        self.my_model.set_hop(hopping[207],2,11,[0,0,1])
        self.my_model.set_hop(hopping[208],3,6,[0,0,-1])
        self.my_model.set_hop(hopping[209],3,14,[0,0,-1])
        self.my_model.set_hop(hopping[210],1,12,[0,-2,0])
        self.my_model.set_hop(hopping[211],1,12,[2,0,0])
        self.my_model.set_hop(hopping[212],0,9,[0,2,0])
        self.my_model.set_hop(hopping[213],1,8,[2,0,0])
        self.my_model.set_hop(hopping[214],1,4,[0,-2,0])
        self.my_model.set_hop(hopping[215],1,4,[2,0,0])
        self.my_model.set_hop(hopping[216],0,5,[-2,0,0])
        self.my_model.set_hop(hopping[217],0,5,[0,2,0])
        self.my_model.set_hop(hopping[218],0,9,[-2,0,0])
        self.my_model.set_hop(hopping[219],1,8,[0,-2,0])
        self.my_model.set_hop(hopping[220],0,13,[-2,0,0])
        self.my_model.set_hop(hopping[221],0,13,[0,2,0])
        self.my_model.set_hop(hopping[222],0,9,[0,2,1])
        self.my_model.set_hop(hopping[223],1,8,[2,0,-1])
        self.my_model.set_hop(hopping[224],1,4,[0,-2,-1])
        self.my_model.set_hop(hopping[225],1,4,[2,0,-1])
        self.my_model.set_hop(hopping[226],0,13,[-2,0,1])
        self.my_model.set_hop(hopping[227],0,13,[0,2,1])
        self.my_model.set_hop(hopping[228],1,12,[0,-2,-1])
        self.my_model.set_hop(hopping[229],1,12,[2,0,-1])
        self.my_model.set_hop(hopping[230],0,5,[-2,0,1])
        self.my_model.set_hop(hopping[231],0,5,[0,2,1])
        self.my_model.set_hop(hopping[232],0,9,[-2,0,1])
        self.my_model.set_hop(hopping[233],1,8,[0,-2,-1])
        self.my_model.set_hop(hopping[234],2,7,[0,0,0])
        self.my_model.set_hop(hopping[235],3,14,[0,0,0])
        self.my_model.set_hop(hopping[236],2,11,[0,0,0])
        self.my_model.set_hop(hopping[237],3,10,[0,0,0])
        self.my_model.set_hop(hopping[238],2,15,[0,0,0])
        self.my_model.set_hop(hopping[239],3,6,[0,0,0])
        self.my_model.set_hop(hopping[240],2,15,[-1,0,1])
        self.my_model.set_hop(hopping[241],2,15,[0,1,1])
        self.my_model.set_hop(hopping[242],3,6,[0,-1,-1])
        self.my_model.set_hop(hopping[243],3,6,[1,0,-1])
        self.my_model.set_hop(hopping[244],2,11,[0,1,1])
        self.my_model.set_hop(hopping[245],3,10,[1,0,-1])
        self.my_model.set_hop(hopping[246],2,11,[-1,0,1])
        self.my_model.set_hop(hopping[247],3,10,[0,-1,-1])
        self.my_model.set_hop(hopping[248],2,7,[-1,0,1])
        self.my_model.set_hop(hopping[249],2,7,[0,1,1])
        self.my_model.set_hop(hopping[250],3,14,[0,-1,-1])
        self.my_model.set_hop(hopping[251],3,14,[1,0,-1])
        self.my_model.set_hop(hopping[252],4,4,[-1,-1,0])
        self.my_model.set_hop(hopping[253],5,5,[-1,-1,0])
        self.my_model.set_hop(hopping[254],6,6,[-1,-1,0])
        self.my_model.set_hop(hopping[255],7,7,[-1,-1,0])
        self.my_model.set_hop(hopping[256],12,12,[-1,-1,0])
        self.my_model.set_hop(hopping[257],13,13,[-1,-1,0])
        self.my_model.set_hop(hopping[258],14,14,[-1,-1,0])
        self.my_model.set_hop(hopping[259],15,15,[-1,-1,0])
        self.my_model.set_hop(hopping[260],4,8,[-1,-1,0])
        self.my_model.set_hop(hopping[261],4,12,[-1,-1,0])
        self.my_model.set_hop(hopping[262],5,9,[-1,-1,0])
        self.my_model.set_hop(hopping[263],5,13,[-1,-1,0])
        self.my_model.set_hop(hopping[264],6,10,[-1,-1,0])
        self.my_model.set_hop(hopping[265],6,14,[-1,-1,0])
        self.my_model.set_hop(hopping[266],7,11,[-1,-1,0])
        self.my_model.set_hop(hopping[267],7,15,[-1,-1,0])
        self.my_model.set_hop(hopping[268],8,12,[-1,-1,0])
        self.my_model.set_hop(hopping[269],9,13,[-1,-1,0])
        self.my_model.set_hop(hopping[270],10,14,[-1,-1,0])
        self.my_model.set_hop(hopping[271],11,15,[-1,-1,0])
        self.my_model.set_hop(hopping[272],4,8,[1,1,0])
        self.my_model.set_hop(hopping[273],4,12,[1,1,0])
        self.my_model.set_hop(hopping[274],5,9,[1,1,0])
        self.my_model.set_hop(hopping[275],5,13,[1,1,0])
        self.my_model.set_hop(hopping[276],6,10,[1,1,0])
        self.my_model.set_hop(hopping[277],6,14,[1,1,0])
        self.my_model.set_hop(hopping[278],7,11,[1,1,0])
        self.my_model.set_hop(hopping[279],7,15,[1,1,0])
        self.my_model.set_hop(hopping[280],8,12,[1,1,0])
        self.my_model.set_hop(hopping[281],9,13,[1,1,0])
        self.my_model.set_hop(hopping[282],10,14,[1,1,0])
        self.my_model.set_hop(hopping[283],11,15,[1,1,0])
        self.my_model.set_hop(hopping[284],8,8,[-1,-1,0])
        self.my_model.set_hop(hopping[285],9,9,[-1,-1,0])
        self.my_model.set_hop(hopping[286],10,10,[-1,-1,0])
        self.my_model.set_hop(hopping[287],11,11,[-1,-1,0])
        self.my_model.set_hop(hopping[288],11,14,[0,-1,0])
        self.my_model.set_hop(hopping[289],10,15,[0,1,0])
        self.my_model.set_hop(hopping[290],7,10,[0,-1,0])
        self.my_model.set_hop(hopping[291],6,11,[0,1,0])
        self.my_model.set_hop(hopping[292],10,11,[-1,0,0])
        self.my_model.set_hop(hopping[293],10,11,[0,1,0])
        self.my_model.set_hop(hopping[294],6,11,[-1,0,0])
        self.my_model.set_hop(hopping[295],7,10,[1,0,0])
        self.my_model.set_hop(hopping[296],6,7,[-1,0,0])
        self.my_model.set_hop(hopping[297],6,7,[0,1,0])
        self.my_model.set_hop(hopping[298],10,15,[-1,0,0])
        self.my_model.set_hop(hopping[299],11,14,[1,0,0])
        self.my_model.set_hop(hopping[300],6,15,[-1,0,0])
        self.my_model.set_hop(hopping[301],7,14,[0,-1,0])
        self.my_model.set_hop(hopping[302],6,15,[0,1,0])
        self.my_model.set_hop(hopping[303],7,14,[1,0,0])
        self.my_model.set_hop(hopping[304],14,15,[-1,0,0])
        self.my_model.set_hop(hopping[305],14,15,[0,1,0])
        self.my_model.set_hop(hopping[306],4,14,[0,0,0])
        self.my_model.set_hop(hopping[307],6,12,[0,0,0])
        self.my_model.set_hop(hopping[308],8,10,[0,0,0])
        self.my_model.set_hop(hopping[309],4,10,[0,0,0])
        self.my_model.set_hop(hopping[310],6,8,[0,0,0])
        self.my_model.set_hop(hopping[311],8,14,[0,0,0])
        self.my_model.set_hop(hopping[312],10,12,[0,0,0])
        self.my_model.set_hop(hopping[313],12,14,[0,0,0])
        self.my_model.set_hop(hopping[314],4,6,[0,0,0])
        self.my_model.set_hop(hopping[315],5,15,[0,0,0])
        self.my_model.set_hop(hopping[316],7,13,[0,0,0])
        self.my_model.set_hop(hopping[317],9,11,[0,0,0])
        self.my_model.set_hop(hopping[318],5,11,[0,0,0])
        self.my_model.set_hop(hopping[319],7,9,[0,0,0])
        self.my_model.set_hop(hopping[320],9,15,[0,0,0])
        self.my_model.set_hop(hopping[321],11,13,[0,0,0])
        self.my_model.set_hop(hopping[322],13,15,[0,0,0])
        self.my_model.set_hop(hopping[323],5,7,[0,0,0])
        self.my_model.set_hop(hopping[324],10,13,[-1,0,1])
        self.my_model.set_hop(hopping[325],9,14,[1,0,-1])
        self.my_model.set_hop(hopping[326],6,9,[-1,0,1])
        self.my_model.set_hop(hopping[327],5,10,[1,0,-1])
        self.my_model.set_hop(hopping[328],5,6,[0,-1,-1])
        self.my_model.set_hop(hopping[329],5,6,[1,0,-1])
        self.my_model.set_hop(hopping[330],5,10,[0,-1,-1])
        self.my_model.set_hop(hopping[331],6,9,[0,1,1])
        self.my_model.set_hop(hopping[332],6,13,[-1,0,1])
        self.my_model.set_hop(hopping[333],5,14,[0,-1,-1])
        self.my_model.set_hop(hopping[334],6,13,[0,1,1])
        self.my_model.set_hop(hopping[335],5,14,[1,0,-1])
        self.my_model.set_hop(hopping[336],9,10,[0,-1,-1])
        self.my_model.set_hop(hopping[337],9,10,[1,0,-1])
        self.my_model.set_hop(hopping[338],9,14,[0,-1,-1])
        self.my_model.set_hop(hopping[339],10,13,[0,1,1])
        self.my_model.set_hop(hopping[340],13,14,[0,-1,-1])
        self.my_model.set_hop(hopping[341],13,14,[1,0,-1])
        self.my_model.set_hop(hopping[342],8,15,[-1,0,1])
        self.my_model.set_hop(hopping[343],11,12,[1,0,-1])
        self.my_model.set_hop(hopping[344],4,11,[-1,0,1])
        self.my_model.set_hop(hopping[345],7,8,[1,0,-1])
        self.my_model.set_hop(hopping[346],4,7,[-1,0,1])
        self.my_model.set_hop(hopping[347],4,7,[0,1,1])
        self.my_model.set_hop(hopping[348],7,8,[0,-1,-1])
        self.my_model.set_hop(hopping[349],4,11,[0,1,1])
        self.my_model.set_hop(hopping[350],4,15,[-1,0,1])
        self.my_model.set_hop(hopping[351],7,12,[0,-1,-1])
        self.my_model.set_hop(hopping[352],4,15,[0,1,1])
        self.my_model.set_hop(hopping[353],7,12,[1,0,-1])
        self.my_model.set_hop(hopping[354],8,11,[-1,0,1])
        self.my_model.set_hop(hopping[355],8,11,[0,1,1])
        self.my_model.set_hop(hopping[356],11,12,[0,-1,-1])
        self.my_model.set_hop(hopping[357],8,15,[0,1,1])
        self.my_model.set_hop(hopping[358],12,15,[-1,0,1])
        self.my_model.set_hop(hopping[359],12,15,[0,1,1])
        self.my_model.set_hop(hopping[360],5,11,[0,-1,0])
        self.my_model.set_hop(hopping[361],7,9,[0,1,0])
        self.my_model.set_hop(hopping[362],9,15,[0,-1,0])
        self.my_model.set_hop(hopping[363],11,13,[0,1,0])
        self.my_model.set_hop(hopping[364],13,15,[0,-1,0])
        self.my_model.set_hop(hopping[365],13,15,[1,0,0])
        self.my_model.set_hop(hopping[366],11,13,[-1,0,0])
        self.my_model.set_hop(hopping[367],9,15,[1,0,0])
        self.my_model.set_hop(hopping[368],7,13,[-1,0,0])
        self.my_model.set_hop(hopping[369],5,15,[0,-1,0])
        self.my_model.set_hop(hopping[370],7,13,[0,1,0])
        self.my_model.set_hop(hopping[371],5,15,[1,0,0])
        self.my_model.set_hop(hopping[372],9,11,[0,-1,0])
        self.my_model.set_hop(hopping[373],9,11,[1,0,0])
        self.my_model.set_hop(hopping[374],7,9,[-1,0,0])
        self.my_model.set_hop(hopping[375],5,11,[1,0,0])
        self.my_model.set_hop(hopping[376],5,7,[0,-1,0])
        self.my_model.set_hop(hopping[377],5,7,[1,0,0])
        self.my_model.set_hop(hopping[378],6,8,[0,-1,0])
        self.my_model.set_hop(hopping[379],4,10,[0,1,0])
        self.my_model.set_hop(hopping[380],10,12,[0,-1,0])
        self.my_model.set_hop(hopping[381],8,14,[0,1,0])
        self.my_model.set_hop(hopping[382],12,14,[-1,0,0])
        self.my_model.set_hop(hopping[383],12,14,[0,1,0])
        self.my_model.set_hop(hopping[384],8,14,[-1,0,0])
        self.my_model.set_hop(hopping[385],10,12,[1,0,0])
        self.my_model.set_hop(hopping[386],4,14,[-1,0,0])
        self.my_model.set_hop(hopping[387],6,12,[0,-1,0])
        self.my_model.set_hop(hopping[388],4,14,[0,1,0])
        self.my_model.set_hop(hopping[389],6,12,[1,0,0])
        self.my_model.set_hop(hopping[390],8,10,[-1,0,0])
        self.my_model.set_hop(hopping[391],8,10,[0,1,0])
        self.my_model.set_hop(hopping[392],4,10,[-1,0,0])
        self.my_model.set_hop(hopping[393],6,8,[1,0,0])
        self.my_model.set_hop(hopping[394],4,6,[-1,0,0])
        self.my_model.set_hop(hopping[395],4,6,[0,1,0])
        self.my_model.set_hop(hopping[396],9,14,[0,-1,0])
        self.my_model.set_hop(hopping[397],11,12,[0,-1,0])
        self.my_model.set_hop(hopping[398],8,15,[0,1,0])
        self.my_model.set_hop(hopping[399],10,13,[0,1,0])
        self.my_model.set_hop(hopping[400],4,15,[-1,0,0])
        self.my_model.set_hop(hopping[401],6,13,[-1,0,0])
        self.my_model.set_hop(hopping[402],5,14,[0,-1,0])
        self.my_model.set_hop(hopping[403],7,12,[0,-1,0])
        self.my_model.set_hop(hopping[404],4,15,[0,1,0])
        self.my_model.set_hop(hopping[405],6,13,[0,1,0])
        self.my_model.set_hop(hopping[406],5,14,[1,0,0])
        self.my_model.set_hop(hopping[407],7,12,[1,0,0])
        self.my_model.set_hop(hopping[408],4,11,[-1,0,0])
        self.my_model.set_hop(hopping[409],6,9,[-1,0,0])
        self.my_model.set_hop(hopping[410],5,10,[1,0,0])
        self.my_model.set_hop(hopping[411],7,8,[1,0,0])
        self.my_model.set_hop(hopping[412],4,7,[-1,0,0])
        self.my_model.set_hop(hopping[413],5,6,[0,-1,0])
        self.my_model.set_hop(hopping[414],4,7,[0,1,0])
        self.my_model.set_hop(hopping[415],5,6,[1,0,0])
        self.my_model.set_hop(hopping[416],5,10,[0,-1,0])
        self.my_model.set_hop(hopping[417],7,8,[0,-1,0])
        self.my_model.set_hop(hopping[418],4,11,[0,1,0])
        self.my_model.set_hop(hopping[419],6,9,[0,1,0])
        self.my_model.set_hop(hopping[420],8,11,[-1,0,0])
        self.my_model.set_hop(hopping[421],9,10,[0,-1,0])
        self.my_model.set_hop(hopping[422],8,11,[0,1,0])
        self.my_model.set_hop(hopping[423],9,10,[1,0,0])
        self.my_model.set_hop(hopping[424],8,15,[-1,0,0])
        self.my_model.set_hop(hopping[425],10,13,[-1,0,0])
        self.my_model.set_hop(hopping[426],9,14,[1,0,0])
        self.my_model.set_hop(hopping[427],11,12,[1,0,0])
        self.my_model.set_hop(hopping[428],12,15,[-1,0,0])
        self.my_model.set_hop(hopping[429],12,15,[0,1,0])
        self.my_model.set_hop(hopping[430],13,14,[0,-1,0])
        self.my_model.set_hop(hopping[431],13,14,[1,0,0])
        self.my_model.set_hop(hopping[432],8,9,[-1,1,0])
        self.my_model.set_hop(hopping[433],4,9,[-1,1,0])
        self.my_model.set_hop(hopping[434],8,13,[-1,1,0])
        self.my_model.set_hop(hopping[435],5,8,[1,-1,0])
        self.my_model.set_hop(hopping[436],9,12,[1,-1,0])
        self.my_model.set_hop(hopping[437],4,5,[-1,1,0])
        self.my_model.set_hop(hopping[438],4,13,[-1,1,0])
        self.my_model.set_hop(hopping[439],5,12,[1,-1,0])
        self.my_model.set_hop(hopping[440],12,13,[-1,1,0])
        self.my_model.set_hop(hopping[441],4,13,[-1,1,1])
        self.my_model.set_hop(hopping[442],5,12,[1,-1,-1])
        self.my_model.set_hop(hopping[443],8,9,[-1,1,1])
        self.my_model.set_hop(hopping[444],4,9,[-1,1,1])
        self.my_model.set_hop(hopping[445],8,13,[-1,1,1])
        self.my_model.set_hop(hopping[446],5,8,[1,-1,-1])
        self.my_model.set_hop(hopping[447],9,12,[1,-1,-1])
        self.my_model.set_hop(hopping[448],12,13,[-1,1,1])
        self.my_model.set_hop(hopping[449],4,5,[-1,1,1])
        self.my_model.set_hop(hopping[450],4,9,[-1,0,1])
        self.my_model.set_hop(hopping[451],5,8,[1,0,-1])
        self.my_model.set_hop(hopping[452],8,13,[-1,0,1])
        self.my_model.set_hop(hopping[453],9,12,[1,0,-1])
        self.my_model.set_hop(hopping[454],8,9,[-1,0,1])
        self.my_model.set_hop(hopping[455],8,9,[0,1,1])
        self.my_model.set_hop(hopping[456],9,12,[0,-1,-1])
        self.my_model.set_hop(hopping[457],8,13,[0,1,1])
        self.my_model.set_hop(hopping[458],5,8,[0,-1,-1])
        self.my_model.set_hop(hopping[459],4,9,[0,1,1])
        self.my_model.set_hop(hopping[460],12,13,[-1,0,1])
        self.my_model.set_hop(hopping[461],12,13,[0,1,1])
        self.my_model.set_hop(hopping[462],4,13,[-1,0,1])
        self.my_model.set_hop(hopping[463],5,12,[0,-1,-1])
        self.my_model.set_hop(hopping[464],4,13,[0,1,1])
        self.my_model.set_hop(hopping[465],5,12,[1,0,-1])
        self.my_model.set_hop(hopping[466],4,5,[-1,0,1])
        self.my_model.set_hop(hopping[467],4,5,[0,1,1])
        self.my_model.set_hop(hopping[468],4,10,[-1,-1,0])
        self.my_model.set_hop(hopping[469],7,9,[-1,-1,0])
        self.my_model.set_hop(hopping[470],5,11,[1,1,0])
        self.my_model.set_hop(hopping[471],6,8,[1,1,0])
        self.my_model.set_hop(hopping[472],5,15,[-1,-1,0])
        self.my_model.set_hop(hopping[473],7,13,[-1,-1,0])
        self.my_model.set_hop(hopping[474],5,15,[1,1,0])
        self.my_model.set_hop(hopping[475],7,13,[1,1,0])
        self.my_model.set_hop(hopping[476],4,14,[-1,-1,0])
        self.my_model.set_hop(hopping[477],6,12,[-1,-1,0])
        self.my_model.set_hop(hopping[478],4,14,[1,1,0])
        self.my_model.set_hop(hopping[479],6,12,[1,1,0])
        self.my_model.set_hop(hopping[480],9,15,[-1,-1,0])
        self.my_model.set_hop(hopping[481],11,13,[1,1,0])
        self.my_model.set_hop(hopping[482],10,12,[-1,-1,0])
        self.my_model.set_hop(hopping[483],8,14,[1,1,0])
        self.my_model.set_hop(hopping[484],12,14,[-1,-1,0])
        self.my_model.set_hop(hopping[485],12,14,[1,1,0])
        self.my_model.set_hop(hopping[486],13,15,[-1,-1,0])
        self.my_model.set_hop(hopping[487],13,15,[1,1,0])
        self.my_model.set_hop(hopping[488],8,14,[-1,-1,0])
        self.my_model.set_hop(hopping[489],10,12,[1,1,0])
        self.my_model.set_hop(hopping[490],11,13,[-1,-1,0])
        self.my_model.set_hop(hopping[491],9,15,[1,1,0])
        self.my_model.set_hop(hopping[492],8,10,[-1,-1,0])
        self.my_model.set_hop(hopping[493],9,11,[-1,-1,0])
        self.my_model.set_hop(hopping[494],8,10,[1,1,0])
        self.my_model.set_hop(hopping[495],9,11,[1,1,0])
        self.my_model.set_hop(hopping[496],5,11,[-1,-1,0])
        self.my_model.set_hop(hopping[497],6,8,[-1,-1,0])
        self.my_model.set_hop(hopping[498],4,10,[1,1,0])
        self.my_model.set_hop(hopping[499],7,9,[1,1,0])
        self.my_model.set_hop(hopping[500],4,6,[-1,-1,0])
        self.my_model.set_hop(hopping[501],5,7,[-1,-1,0])
        self.my_model.set_hop(hopping[502],4,6,[1,1,0])
        self.my_model.set_hop(hopping[503],5,7,[1,1,0])
        self.my_model.set_hop(hopping[504],10,11,[0,0,1])
        self.my_model.set_hop(hopping[505],7,10,[0,0,-1])
        self.my_model.set_hop(hopping[506],11,14,[0,0,-1])
        self.my_model.set_hop(hopping[507],6,11,[0,0,1])
        self.my_model.set_hop(hopping[508],10,15,[0,0,1])
        self.my_model.set_hop(hopping[509],6,7,[0,0,1])
        self.my_model.set_hop(hopping[510],7,14,[0,0,-1])
        self.my_model.set_hop(hopping[511],6,15,[0,0,1])
        self.my_model.set_hop(hopping[512],14,15,[0,0,1])
        self.my_model.set_hop(hopping[513],9,12,[0,-2,0])
        self.my_model.set_hop(hopping[514],8,13,[0,2,0])
        self.my_model.set_hop(hopping[515],5,8,[0,-2,0])
        self.my_model.set_hop(hopping[516],4,9,[0,2,0])
        self.my_model.set_hop(hopping[517],4,5,[-2,0,0])
        self.my_model.set_hop(hopping[518],4,5,[0,2,0])
        self.my_model.set_hop(hopping[519],4,9,[-2,0,0])
        self.my_model.set_hop(hopping[520],5,8,[2,0,0])
        self.my_model.set_hop(hopping[521],4,13,[-2,0,0])
        self.my_model.set_hop(hopping[522],5,12,[0,-2,0])
        self.my_model.set_hop(hopping[523],4,13,[0,2,0])
        self.my_model.set_hop(hopping[524],5,12,[2,0,0])
        self.my_model.set_hop(hopping[525],8,9,[-2,0,0])
        self.my_model.set_hop(hopping[526],8,9,[0,2,0])
        self.my_model.set_hop(hopping[527],8,13,[-2,0,0])
        self.my_model.set_hop(hopping[528],9,12,[2,0,0])
        self.my_model.set_hop(hopping[529],12,13,[-2,0,0])
        self.my_model.set_hop(hopping[530],12,13,[0,2,0])
        self.my_model.set_hop(hopping[531],5,8,[0,-2,-1])
        self.my_model.set_hop(hopping[532],4,9,[0,2,1])
        self.my_model.set_hop(hopping[533],8,13,[-2,0,1])
        self.my_model.set_hop(hopping[534],9,12,[2,0,-1])
        self.my_model.set_hop(hopping[535],4,13,[-2,0,1])
        self.my_model.set_hop(hopping[536],5,12,[0,-2,-1])
        self.my_model.set_hop(hopping[537],4,13,[0,2,1])
        self.my_model.set_hop(hopping[538],5,12,[2,0,-1])
        self.my_model.set_hop(hopping[539],12,13,[-2,0,1])
        self.my_model.set_hop(hopping[540],12,13,[0,2,1])
        self.my_model.set_hop(hopping[541],9,12,[0,-2,-1])
        self.my_model.set_hop(hopping[542],8,13,[0,2,1])
        self.my_model.set_hop(hopping[543],4,5,[-2,0,1])
        self.my_model.set_hop(hopping[544],4,5,[0,2,1])
        self.my_model.set_hop(hopping[545],4,9,[-2,0,1])
        self.my_model.set_hop(hopping[546],5,8,[2,0,-1])
        self.my_model.set_hop(hopping[547],8,9,[-2,0,1])
        self.my_model.set_hop(hopping[548],8,9,[0,2,1])
        self.my_model.set_hop(hopping[549],6,15,[0,0,0])
        self.my_model.set_hop(hopping[550],7,14,[0,0,0])
        self.my_model.set_hop(hopping[551],10,11,[0,0,0])
        self.my_model.set_hop(hopping[552],6,11,[0,0,0])
        self.my_model.set_hop(hopping[553],7,10,[0,0,0])
        self.my_model.set_hop(hopping[554],10,15,[0,0,0])
        self.my_model.set_hop(hopping[555],11,14,[0,0,0])
        self.my_model.set_hop(hopping[556],14,15,[0,0,0])
        self.my_model.set_hop(hopping[557],6,7,[0,0,0])
        self.my_model.set_hop(hopping[558],6,15,[-1,0,1])
        self.my_model.set_hop(hopping[559],7,14,[0,-1,-1])
        self.my_model.set_hop(hopping[560],6,15,[0,1,1])
        self.my_model.set_hop(hopping[561],7,14,[1,0,-1])
        self.my_model.set_hop(hopping[562],10,15,[-1,0,1])
        self.my_model.set_hop(hopping[563],11,14,[1,0,-1])
        self.my_model.set_hop(hopping[564],7,10,[0,-1,-1])
        self.my_model.set_hop(hopping[565],6,11,[0,1,1])
        self.my_model.set_hop(hopping[566],10,11,[-1,0,1])
        self.my_model.set_hop(hopping[567],10,11,[0,1,1])
        self.my_model.set_hop(hopping[568],6,11,[-1,0,1])
        self.my_model.set_hop(hopping[569],7,10,[1,0,-1])
        self.my_model.set_hop(hopping[570],11,14,[0,-1,-1])
        self.my_model.set_hop(hopping[571],10,15,[0,1,1])
        self.my_model.set_hop(hopping[572],6,7,[-1,0,1])
        self.my_model.set_hop(hopping[573],6,7,[0,1,1])
        self.my_model.set_hop(hopping[574],14,15,[-1,0,1])
        self.my_model.set_hop(hopping[575],14,15,[0,1,1])
        self.my_model.set_hop(hopping[576],8,16,[-1,-1,0])
        self.my_model.set_hop(hopping[577],9,17,[-1,-1,0])
        self.my_model.set_hop(hopping[578],10,18,[-1,-1,0])
        self.my_model.set_hop(hopping[579],11,19,[-1,-1,0])
        self.my_model.set_hop(hopping[580],4,16,[-1,-1,0])
        self.my_model.set_hop(hopping[581],5,17,[-1,-1,0])
        self.my_model.set_hop(hopping[582],6,18,[-1,-1,0])
        self.my_model.set_hop(hopping[583],7,19,[-1,-1,0])
        self.my_model.set_hop(hopping[584],12,16,[-1,-1,0])
        self.my_model.set_hop(hopping[585],13,17,[-1,-1,0])
        self.my_model.set_hop(hopping[586],14,18,[-1,-1,0])
        self.my_model.set_hop(hopping[587],15,19,[-1,-1,0])
        self.my_model.set_hop(hopping[588],14,19,[-1,0,0])
        self.my_model.set_hop(hopping[589],14,19,[0,1,0])
        self.my_model.set_hop(hopping[590],6,19,[-1,0,0])
        self.my_model.set_hop(hopping[591],6,19,[0,1,0])
        self.my_model.set_hop(hopping[592],10,19,[-1,0,0])
        self.my_model.set_hop(hopping[593],11,18,[0,-1,0])
        self.my_model.set_hop(hopping[594],10,19,[0,1,0])
        self.my_model.set_hop(hopping[595],11,18,[1,0,0])
        self.my_model.set_hop(hopping[596],7,18,[0,-1,0])
        self.my_model.set_hop(hopping[597],7,18,[1,0,0])
        self.my_model.set_hop(hopping[598],15,18,[0,-1,0])
        self.my_model.set_hop(hopping[599],15,18,[1,0,0])
        self.my_model.set_hop(hopping[600],6,16,[0,0,0])
        self.my_model.set_hop(hopping[601],12,18,[0,0,0])
        self.my_model.set_hop(hopping[602],8,18,[0,0,0])
        self.my_model.set_hop(hopping[603],10,16,[0,0,0])
        self.my_model.set_hop(hopping[604],14,16,[0,0,0])
        self.my_model.set_hop(hopping[605],4,18,[0,0,0])
        self.my_model.set_hop(hopping[606],5,19,[0,0,0])
        self.my_model.set_hop(hopping[607],15,17,[0,0,0])
        self.my_model.set_hop(hopping[608],9,19,[0,0,0])
        self.my_model.set_hop(hopping[609],11,17,[0,0,0])
        self.my_model.set_hop(hopping[610],13,19,[0,0,0])
        self.my_model.set_hop(hopping[611],7,17,[0,0,0])
        self.my_model.set_hop(hopping[612],13,18,[0,-1,-1])
        self.my_model.set_hop(hopping[613],13,18,[1,0,-1])
        self.my_model.set_hop(hopping[614],10,17,[-1,0,1])
        self.my_model.set_hop(hopping[615],9,18,[0,-1,-1])
        self.my_model.set_hop(hopping[616],5,18,[0,-1,-1])
        self.my_model.set_hop(hopping[617],5,18,[1,0,-1])
        self.my_model.set_hop(hopping[618],6,17,[-1,0,1])
        self.my_model.set_hop(hopping[619],6,17,[0,1,1])
        self.my_model.set_hop(hopping[620],10,17,[0,1,1])
        self.my_model.set_hop(hopping[621],9,18,[1,0,-1])
        self.my_model.set_hop(hopping[622],14,17,[-1,0,1])
        self.my_model.set_hop(hopping[623],14,17,[0,1,1])
        self.my_model.set_hop(hopping[624],15,16,[0,-1,-1])
        self.my_model.set_hop(hopping[625],15,16,[1,0,-1])
        self.my_model.set_hop(hopping[626],8,19,[-1,0,1])
        self.my_model.set_hop(hopping[627],11,16,[0,-1,-1])
        self.my_model.set_hop(hopping[628],7,16,[0,-1,-1])
        self.my_model.set_hop(hopping[629],7,16,[1,0,-1])
        self.my_model.set_hop(hopping[630],4,19,[-1,0,1])
        self.my_model.set_hop(hopping[631],4,19,[0,1,1])
        self.my_model.set_hop(hopping[632],8,19,[0,1,1])
        self.my_model.set_hop(hopping[633],11,16,[1,0,-1])
        self.my_model.set_hop(hopping[634],12,19,[-1,0,1])
        self.my_model.set_hop(hopping[635],12,19,[0,1,1])
        self.my_model.set_hop(hopping[636],7,17,[-1,0,0])
        self.my_model.set_hop(hopping[637],7,17,[0,1,0])
        self.my_model.set_hop(hopping[638],11,17,[-1,0,0])
        self.my_model.set_hop(hopping[639],9,19,[0,-1,0])
        self.my_model.set_hop(hopping[640],15,17,[-1,0,0])
        self.my_model.set_hop(hopping[641],15,17,[0,1,0])
        self.my_model.set_hop(hopping[642],13,19,[0,-1,0])
        self.my_model.set_hop(hopping[643],13,19,[1,0,0])
        self.my_model.set_hop(hopping[644],11,17,[0,1,0])
        self.my_model.set_hop(hopping[645],9,19,[1,0,0])
        self.my_model.set_hop(hopping[646],5,19,[0,-1,0])
        self.my_model.set_hop(hopping[647],5,19,[1,0,0])
        self.my_model.set_hop(hopping[648],4,18,[-1,0,0])
        self.my_model.set_hop(hopping[649],4,18,[0,1,0])
        self.my_model.set_hop(hopping[650],8,18,[-1,0,0])
        self.my_model.set_hop(hopping[651],10,16,[0,-1,0])
        self.my_model.set_hop(hopping[652],12,18,[-1,0,0])
        self.my_model.set_hop(hopping[653],12,18,[0,1,0])
        self.my_model.set_hop(hopping[654],14,16,[0,-1,0])
        self.my_model.set_hop(hopping[655],14,16,[1,0,0])
        self.my_model.set_hop(hopping[656],8,18,[0,1,0])
        self.my_model.set_hop(hopping[657],10,16,[1,0,0])
        self.my_model.set_hop(hopping[658],6,16,[0,-1,0])
        self.my_model.set_hop(hopping[659],6,16,[1,0,0])
        self.my_model.set_hop(hopping[660],14,17,[-1,0,0])
        self.my_model.set_hop(hopping[661],14,17,[0,1,0])
        self.my_model.set_hop(hopping[662],12,19,[-1,0,0])
        self.my_model.set_hop(hopping[663],12,19,[0,1,0])
        self.my_model.set_hop(hopping[664],8,19,[-1,0,0])
        self.my_model.set_hop(hopping[665],10,17,[-1,0,0])
        self.my_model.set_hop(hopping[666],9,18,[0,-1,0])
        self.my_model.set_hop(hopping[667],11,16,[0,-1,0])
        self.my_model.set_hop(hopping[668],5,18,[0,-1,0])
        self.my_model.set_hop(hopping[669],7,16,[0,-1,0])
        self.my_model.set_hop(hopping[670],5,18,[1,0,0])
        self.my_model.set_hop(hopping[671],7,16,[1,0,0])
        self.my_model.set_hop(hopping[672],4,19,[-1,0,0])
        self.my_model.set_hop(hopping[673],6,17,[-1,0,0])
        self.my_model.set_hop(hopping[674],4,19,[0,1,0])
        self.my_model.set_hop(hopping[675],6,17,[0,1,0])
        self.my_model.set_hop(hopping[676],8,19,[0,1,0])
        self.my_model.set_hop(hopping[677],10,17,[0,1,0])
        self.my_model.set_hop(hopping[678],9,18,[1,0,0])
        self.my_model.set_hop(hopping[679],11,16,[1,0,0])
        self.my_model.set_hop(hopping[680],15,16,[0,-1,0])
        self.my_model.set_hop(hopping[681],15,16,[1,0,0])
        self.my_model.set_hop(hopping[682],13,18,[0,-1,0])
        self.my_model.set_hop(hopping[683],13,18,[1,0,0])
        self.my_model.set_hop(hopping[684],12,17,[-1,1,0])
        self.my_model.set_hop(hopping[685],4,17,[-1,1,0])
        self.my_model.set_hop(hopping[686],8,17,[-1,1,0])
        self.my_model.set_hop(hopping[687],9,16,[1,-1,0])
        self.my_model.set_hop(hopping[688],5,16,[1,-1,0])
        self.my_model.set_hop(hopping[689],13,16,[1,-1,0])
        self.my_model.set_hop(hopping[690],4,17,[-1,1,1])
        self.my_model.set_hop(hopping[691],13,16,[1,-1,-1])
        self.my_model.set_hop(hopping[692],8,17,[-1,1,1])
        self.my_model.set_hop(hopping[693],9,16,[1,-1,-1])
        self.my_model.set_hop(hopping[694],12,17,[-1,1,1])
        self.my_model.set_hop(hopping[695],5,16,[1,-1,-1])
        self.my_model.set_hop(hopping[696],5,16,[0,-1,-1])
        self.my_model.set_hop(hopping[697],5,16,[1,0,-1])
        self.my_model.set_hop(hopping[698],13,16,[0,-1,-1])
        self.my_model.set_hop(hopping[699],13,16,[1,0,-1])
        self.my_model.set_hop(hopping[700],8,17,[-1,0,1])
        self.my_model.set_hop(hopping[701],9,16,[0,-1,-1])
        self.my_model.set_hop(hopping[702],8,17,[0,1,1])
        self.my_model.set_hop(hopping[703],9,16,[1,0,-1])
        self.my_model.set_hop(hopping[704],12,17,[-1,0,1])
        self.my_model.set_hop(hopping[705],12,17,[0,1,1])
        self.my_model.set_hop(hopping[706],4,17,[-1,0,1])
        self.my_model.set_hop(hopping[707],4,17,[0,1,1])
        self.my_model.set_hop(hopping[708],5,19,[-1,-1,0])
        self.my_model.set_hop(hopping[709],6,16,[-1,-1,0])
        self.my_model.set_hop(hopping[710],8,18,[-1,-1,0])
        self.my_model.set_hop(hopping[711],9,19,[-1,-1,0])
        self.my_model.set_hop(hopping[712],10,16,[-1,-1,0])
        self.my_model.set_hop(hopping[713],11,17,[-1,-1,0])
        self.my_model.set_hop(hopping[714],15,17,[-1,-1,0])
        self.my_model.set_hop(hopping[715],12,18,[-1,-1,0])
        self.my_model.set_hop(hopping[716],14,16,[-1,-1,0])
        self.my_model.set_hop(hopping[717],13,19,[-1,-1,0])
        self.my_model.set_hop(hopping[718],4,18,[-1,-1,0])
        self.my_model.set_hop(hopping[719],7,17,[-1,-1,0])
        self.my_model.set_hop(hopping[720],15,18,[0,0,-1])
        self.my_model.set_hop(hopping[721],7,18,[0,0,-1])
        self.my_model.set_hop(hopping[722],11,18,[0,0,-1])
        self.my_model.set_hop(hopping[723],10,19,[0,0,1])
        self.my_model.set_hop(hopping[724],6,19,[0,0,1])
        self.my_model.set_hop(hopping[725],14,19,[0,0,1])
        self.my_model.set_hop(hopping[726],12,17,[-2,0,0])
        self.my_model.set_hop(hopping[727],12,17,[0,2,0])
        self.my_model.set_hop(hopping[728],8,17,[-2,0,0])
        self.my_model.set_hop(hopping[729],9,16,[0,-2,0])
        self.my_model.set_hop(hopping[730],4,17,[-2,0,0])
        self.my_model.set_hop(hopping[731],4,17,[0,2,0])
        self.my_model.set_hop(hopping[732],5,16,[0,-2,0])
        self.my_model.set_hop(hopping[733],5,16,[2,0,0])
        self.my_model.set_hop(hopping[734],8,17,[0,2,0])
        self.my_model.set_hop(hopping[735],9,16,[2,0,0])
        self.my_model.set_hop(hopping[736],13,16,[0,-2,0])
        self.my_model.set_hop(hopping[737],13,16,[2,0,0])
        self.my_model.set_hop(hopping[738],8,17,[-2,0,1])
        self.my_model.set_hop(hopping[739],9,16,[0,-2,-1])
        self.my_model.set_hop(hopping[740],4,17,[-2,0,1])
        self.my_model.set_hop(hopping[741],4,17,[0,2,1])
        self.my_model.set_hop(hopping[742],13,16,[0,-2,-1])
        self.my_model.set_hop(hopping[743],13,16,[2,0,-1])
        self.my_model.set_hop(hopping[744],12,17,[-2,0,1])
        self.my_model.set_hop(hopping[745],12,17,[0,2,1])
        self.my_model.set_hop(hopping[746],5,16,[0,-2,-1])
        self.my_model.set_hop(hopping[747],5,16,[2,0,-1])
        self.my_model.set_hop(hopping[748],8,17,[0,2,1])
        self.my_model.set_hop(hopping[749],9,16,[2,0,-1])
        self.my_model.set_hop(hopping[750],7,18,[0,0,0])
        self.my_model.set_hop(hopping[751],14,19,[0,0,0])
        self.my_model.set_hop(hopping[752],10,19,[0,0,0])
        self.my_model.set_hop(hopping[753],11,18,[0,0,0])
        self.my_model.set_hop(hopping[754],15,18,[0,0,0])
        self.my_model.set_hop(hopping[755],6,19,[0,0,0])
        self.my_model.set_hop(hopping[756],15,18,[0,-1,-1])
        self.my_model.set_hop(hopping[757],15,18,[1,0,-1])
        self.my_model.set_hop(hopping[758],6,19,[-1,0,1])
        self.my_model.set_hop(hopping[759],6,19,[0,1,1])
        self.my_model.set_hop(hopping[760],10,19,[-1,0,1])
        self.my_model.set_hop(hopping[761],11,18,[0,-1,-1])
        self.my_model.set_hop(hopping[762],10,19,[0,1,1])
        self.my_model.set_hop(hopping[763],11,18,[1,0,-1])
        self.my_model.set_hop(hopping[764],7,18,[0,-1,-1])
        self.my_model.set_hop(hopping[765],7,18,[1,0,-1])
        self.my_model.set_hop(hopping[766],14,19,[-1,0,1])
        self.my_model.set_hop(hopping[767],14,19,[0,1,1])

        768 






        3.040000, 
        3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 3.277766, 
        3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 3.327418, 
        3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 3.445864, 
        3.445864, 3.605804, 3.612253, 4.337574, 4.337574, 4.470498, 
        4.470498, 4.470498, 4.470498, 4.652131, 4.716293, 4.716293, 
        4.721225, 4.721225, 4.862916, 4.945890, 4.945890, 
        36 

        [3.040000, -1.000000], 
        [3.040000, -1.000000], [3.040000, -1.000000], [3.040000, -1.000000], 
        [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000], 
        [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000], 
        [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000], 
        [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000], 
        [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000], 
        [3.040000, -0.000000], [3.040000, 1.000000], [3.040000, 1.000000], 
        [3.040000, 1.000000], [3.040000, 1.000000], [3.109274, -0.673094], 
        [3.109274, -0.673094], [3.109274, -0.554942], [3.109274, -0.554942], 
        [3.109274, -0.488860], [3.109274, -0.488860], [3.109274, 0.488860], 
        [3.109274, 0.488860], [3.109274, 0.554942], [3.109274, 0.554942], 
        [3.109274, 0.673094], [3.109274, 0.673094], [3.277766, -0.982253], 
        [3.277766, -0.187560], [3.277766, -0.000000], [3.277766, -0.000000], 
        [3.277766, 0.187560], [3.277766, 0.982253], [3.277766, -0.982253], 
        [3.277766, -0.187560], [3.277766, -0.000000], [3.277766, -0.000000], 
        [3.277766, 0.187560], [3.277766, 0.982253], [3.300672, -0.886903], 
        [3.300672, -0.886903], [3.300672, -0.460512], [3.300672, -0.460512], 
        [3.300672, -0.036498], [3.300672, -0.036498], [3.300672, 0.036498], 
        [3.300672, 0.036498], [3.300672, 0.460512], [3.300672, 0.460512], 
        [3.300672, 0.886903], [3.300672, 0.886903], [3.300672, -0.886903], 
        [3.300672, -0.886903], [3.300672, -0.460512], [3.300672, -0.460512], 
        [3.300672, -0.036498], [3.300672, -0.036498], [3.300672, 0.036498], 
        [3.300672, 0.036498], [3.300672, 0.460512], [3.300672, 0.460512], 
        [3.300672, 0.886903], [3.300672, 0.886903], [3.327418, -0.870165], 
        [3.327418, -0.870165], [3.327418, -0.456811], [3.327418, -0.456811], 
        [3.327418, -0.184761], [3.327418, -0.184761], [3.327418, 0.184761], 
        [3.327418, 0.184761], [3.327418, 0.456811], [3.327418, 0.456811], 
        [3.327418, 0.870165], [3.327418, 0.870165], [3.327418, -0.870165], 
        [3.327418, -0.870165], [3.327418, -0.456811], [3.327418, -0.456811], 
        [3.327418, -0.184761], [3.327418, -0.184761], [3.327418, 0.184761], 
        [3.327418, 0.184761], [3.327418, 0.456811], [3.327418, 0.456811], 
        [3.327418, 0.870165], [3.327418, 0.870165], [3.445864, -0.785757], 
        [3.445864, -0.785757], [3.445864, -0.785757], [3.445864, -0.785757], 
        [3.445864, -0.441109], [3.445864, -0.441109], [3.445864, -0.441109], 
        [3.445864, -0.441109], [3.445864, -0.433601], [3.445864, -0.433601], 
        [3.445864, -0.433601], [3.445864, -0.433601], [3.445864, 0.433601], 
        [3.445864, 0.433601], [3.445864, 0.433601], [3.445864, 0.433601], 
        [3.445864, 0.441109], [3.445864, 0.441109], [3.445864, 0.441109], 
        [3.445864, 0.441109], [3.445864, 0.785757], [3.445864, 0.785757], 
        [3.445864, 0.785757], [3.445864, 0.785757], [3.605804, -0.921400], 
        [3.605804, -0.388616], [3.605804, -0.000000], [3.605804, -0.000000], 
        [3.605804, 0.388616], [3.605804, 0.921400], [3.612253, -0.768201], 
        [3.612253, -0.640209], [3.612253, -0.000000], [3.612253, -0.000000], 
        [3.612253, 0.640209], [3.612253, 0.768201], [4.337574, -0.770031], 
        [4.337574, -0.770031], [4.337574, -0.533155], [4.337574, -0.533155], 
        [4.337574, -0.350426], [4.337574, -0.350426], [4.337574, 0.350426], 
        [4.337574, 0.350426], [4.337574, 0.533155], [4.337574, 0.533155], 
        [4.337574, 0.770031], [4.337574, 0.770031], [4.470498, -0.720187], 
        [4.470498, -0.720187], [4.470498, -0.720187], [4.470498, -0.720187], 
        [4.470498, -0.680014], [4.470498, -0.680014], [4.470498, -0.680014], 
        [4.470498, -0.680014], [4.470498, -0.137519], [4.470498, -0.137519], 
        [4.470498, -0.137519], [4.470498, -0.137519], [4.470498, 0.137519], 
        [4.470498, 0.137519], [4.470498, 0.137519], [4.470498, 0.137519], 
        [4.470498, 0.680014], [4.470498, 0.680014], [4.470498, 0.680014], 
        [4.470498, 0.680014], [4.470498, 0.720187], [4.470498, 0.720187], 
        [4.470498, 0.720187], [4.470498, 0.720187], [4.652131, -0.761404], 
        [4.652131, -0.648278], [4.652131, -0.000000], [4.652131, -0.000000], 
        [4.652131, 0.648278], [4.652131, 0.761404], [4.716293, -0.704449], 
        [4.716293, -0.704449], [4.716293, -0.644574], [4.716293, -0.644574], 
        [4.716293, -0.297113], [4.716293, -0.297113], [4.716293, 0.297113], 
        [4.716293, 0.297113], [4.716293, 0.644574], [4.716293, 0.644574], 
        [4.716293, 0.704449], [4.716293, 0.704449], [4.721225, -0.643901], 
        [4.721225, -0.643901], [4.721225, -0.587757], [4.721225, -0.587757], 
        [4.721225, -0.489830], [4.721225, -0.489830], [4.721225, 0.489830], 
        [4.721225, 0.489830], [4.721225, 0.587757], [4.721225, 0.587757], 
        [4.721225, 0.643901], [4.721225, 0.643901], [4.862916, -0.902655], 
        [4.862916, -0.430366], [4.862916, -0.000000], [4.862916, -0.000000], 
        [4.862916, 0.430366], [4.862916, 0.902655], [4.945890, -0.716181], 
        [4.945890, -0.716181], [4.945890, -0.626607], [4.945890, -0.626607], 
        [4.945890, -0.307326], [4.945890, -0.307326], [4.945890, 0.307326], 
        [4.945890, 0.307326], [4.945890, 0.626607], [4.945890, 0.626607], 
        [4.945890, 0.716181], [4.945890, 0.716181], 
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
        self.evals = self.my_model.solve_all(self.k_vec)
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