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
        self.my_model.set_hop(hopping[26],2,3,[0,0,1])
        self.my_model.set_hop(hopping[27],0,1,[-2,0,0])
        self.my_model.set_hop(hopping[28],0,1,[0,2,0])
        self.my_model.set_hop(hopping[29],0,1,[-2,0,1])
        self.my_model.set_hop(hopping[30],0,1,[0,2,1])
        self.my_model.set_hop(hopping[31],2,3,[0,0,0])
        self.my_model.set_hop(hopping[32],2,3,[-1,0,1])
        self.my_model.set_hop(hopping[33],2,3,[0,1,1])
        self.my_model.set_hop(hopping[34],0,4,[-1,-1,0])
        self.my_model.set_hop(hopping[35],0,12,[-1,-1,0])
        self.my_model.set_hop(hopping[36],1,5,[-1,-1,0])
        self.my_model.set_hop(hopping[37],1,13,[-1,-1,0])
        self.my_model.set_hop(hopping[38],2,6,[-1,-1,0])
        self.my_model.set_hop(hopping[39],2,14,[-1,-1,0])
        self.my_model.set_hop(hopping[40],3,7,[-1,-1,0])
        self.my_model.set_hop(hopping[41],3,15,[-1,-1,0])
        self.my_model.set_hop(hopping[42],0,8,[-1,-1,0])
        self.my_model.set_hop(hopping[43],1,9,[-1,-1,0])
        self.my_model.set_hop(hopping[44],2,10,[-1,-1,0])
        self.my_model.set_hop(hopping[45],3,11,[-1,-1,0])
        self.my_model.set_hop(hopping[46],3,14,[0,-1,0])
        self.my_model.set_hop(hopping[47],3,14,[1,0,0])
        self.my_model.set_hop(hopping[48],3,6,[0,-1,0])
        self.my_model.set_hop(hopping[49],3,6,[1,0,0])
        self.my_model.set_hop(hopping[50],2,11,[0,1,0])
        self.my_model.set_hop(hopping[51],3,10,[1,0,0])
        self.my_model.set_hop(hopping[52],2,11,[-1,0,0])
        self.my_model.set_hop(hopping[53],3,10,[0,-1,0])
        self.my_model.set_hop(hopping[54],2,7,[-1,0,0])
        self.my_model.set_hop(hopping[55],2,7,[0,1,0])
        self.my_model.set_hop(hopping[56],2,15,[-1,0,0])
        self.my_model.set_hop(hopping[57],2,15,[0,1,0])
        self.my_model.set_hop(hopping[58],0,6,[0,0,0])
        self.my_model.set_hop(hopping[59],2,12,[0,0,0])
        self.my_model.set_hop(hopping[60],0,10,[0,0,0])
        self.my_model.set_hop(hopping[61],2,8,[0,0,0])
        self.my_model.set_hop(hopping[62],0,14,[0,0,0])
        self.my_model.set_hop(hopping[63],2,4,[0,0,0])
        self.my_model.set_hop(hopping[64],3,5,[0,0,0])
        self.my_model.set_hop(hopping[65],1,15,[0,0,0])
        self.my_model.set_hop(hopping[66],1,11,[0,0,0])
        self.my_model.set_hop(hopping[67],3,9,[0,0,0])
        self.my_model.set_hop(hopping[68],3,13,[0,0,0])
        self.my_model.set_hop(hopping[69],1,7,[0,0,0])
        self.my_model.set_hop(hopping[70],2,13,[-1,0,1])
        self.my_model.set_hop(hopping[71],2,13,[0,1,1])
        self.my_model.set_hop(hopping[72],2,9,[0,1,1])
        self.my_model.set_hop(hopping[73],1,10,[1,0,-1])
        self.my_model.set_hop(hopping[74],2,5,[-1,0,1])
        self.my_model.set_hop(hopping[75],2,5,[0,1,1])
        self.my_model.set_hop(hopping[76],1,6,[0,-1,-1])
        self.my_model.set_hop(hopping[77],1,6,[1,0,-1])
        self.my_model.set_hop(hopping[78],2,9,[-1,0,1])
        self.my_model.set_hop(hopping[79],1,10,[0,-1,-1])
        self.my_model.set_hop(hopping[80],1,14,[0,-1,-1])
        self.my_model.set_hop(hopping[81],1,14,[1,0,-1])
        self.my_model.set_hop(hopping[82],0,15,[-1,0,1])
        self.my_model.set_hop(hopping[83],0,15,[0,1,1])
        self.my_model.set_hop(hopping[84],0,11,[0,1,1])
        self.my_model.set_hop(hopping[85],3,8,[1,0,-1])
        self.my_model.set_hop(hopping[86],0,7,[-1,0,1])
        self.my_model.set_hop(hopping[87],0,7,[0,1,1])
        self.my_model.set_hop(hopping[88],3,4,[0,-1,-1])
        self.my_model.set_hop(hopping[89],3,4,[1,0,-1])
        self.my_model.set_hop(hopping[90],0,11,[-1,0,1])
        self.my_model.set_hop(hopping[91],3,8,[0,-1,-1])
        self.my_model.set_hop(hopping[92],3,12,[0,-1,-1])
        self.my_model.set_hop(hopping[93],3,12,[1,0,-1])
        self.my_model.set_hop(hopping[94],1,7,[0,-1,0])
        self.my_model.set_hop(hopping[95],1,7,[1,0,0])
        self.my_model.set_hop(hopping[96],3,9,[0,1,0])
        self.my_model.set_hop(hopping[97],1,11,[1,0,0])
        self.my_model.set_hop(hopping[98],1,15,[0,-1,0])
        self.my_model.set_hop(hopping[99],1,15,[1,0,0])
        self.my_model.set_hop(hopping[100],3,13,[-1,0,0])
        self.my_model.set_hop(hopping[101],3,13,[0,1,0])
        self.my_model.set_hop(hopping[102],3,9,[-1,0,0])
        self.my_model.set_hop(hopping[103],1,11,[0,-1,0])
        self.my_model.set_hop(hopping[104],3,5,[-1,0,0])
        self.my_model.set_hop(hopping[105],3,5,[0,1,0])
        self.my_model.set_hop(hopping[106],2,4,[0,-1,0])
        self.my_model.set_hop(hopping[107],2,4,[1,0,0])
        self.my_model.set_hop(hopping[108],0,10,[0,1,0])
        self.my_model.set_hop(hopping[109],2,8,[1,0,0])
        self.my_model.set_hop(hopping[110],2,12,[0,-1,0])
        self.my_model.set_hop(hopping[111],2,12,[1,0,0])
        self.my_model.set_hop(hopping[112],0,14,[-1,0,0])
        self.my_model.set_hop(hopping[113],0,14,[0,1,0])
        self.my_model.set_hop(hopping[114],0,10,[-1,0,0])
        self.my_model.set_hop(hopping[115],2,8,[0,-1,0])
        self.my_model.set_hop(hopping[116],0,6,[-1,0,0])
        self.my_model.set_hop(hopping[117],0,6,[0,1,0])
        self.my_model.set_hop(hopping[118],1,14,[0,-1,0])
        self.my_model.set_hop(hopping[119],1,14,[1,0,0])
        self.my_model.set_hop(hopping[120],3,12,[0,-1,0])
        self.my_model.set_hop(hopping[121],3,12,[1,0,0])
        self.my_model.set_hop(hopping[122],0,11,[0,1,0])
        self.my_model.set_hop(hopping[123],2,9,[0,1,0])
        self.my_model.set_hop(hopping[124],1,10,[1,0,0])
        self.my_model.set_hop(hopping[125],3,8,[1,0,0])
        self.my_model.set_hop(hopping[126],0,7,[-1,0,0])
        self.my_model.set_hop(hopping[127],2,5,[-1,0,0])
        self.my_model.set_hop(hopping[128],0,7,[0,1,0])
        self.my_model.set_hop(hopping[129],2,5,[0,1,0])
        self.my_model.set_hop(hopping[130],1,6,[0,-1,0])
        self.my_model.set_hop(hopping[131],3,4,[0,-1,0])
        self.my_model.set_hop(hopping[132],1,6,[1,0,0])
        self.my_model.set_hop(hopping[133],3,4,[1,0,0])
        self.my_model.set_hop(hopping[134],0,11,[-1,0,0])
        self.my_model.set_hop(hopping[135],2,9,[-1,0,0])
        self.my_model.set_hop(hopping[136],1,10,[0,-1,0])
        self.my_model.set_hop(hopping[137],3,8,[0,-1,0])
        self.my_model.set_hop(hopping[138],0,15,[-1,0,0])
        self.my_model.set_hop(hopping[139],0,15,[0,1,0])
        self.my_model.set_hop(hopping[140],2,13,[-1,0,0])
        self.my_model.set_hop(hopping[141],2,13,[0,1,0])
        self.my_model.set_hop(hopping[142],1,12,[1,-1,0])
        self.my_model.set_hop(hopping[143],1,4,[1,-1,0])
        self.my_model.set_hop(hopping[144],0,9,[-1,1,0])
        self.my_model.set_hop(hopping[145],1,8,[1,-1,0])
        self.my_model.set_hop(hopping[146],0,5,[-1,1,0])
        self.my_model.set_hop(hopping[147],0,13,[-1,1,0])
        self.my_model.set_hop(hopping[148],1,4,[1,-1,-1])
        self.my_model.set_hop(hopping[149],0,13,[-1,1,1])
        self.my_model.set_hop(hopping[150],0,9,[-1,1,1])
        self.my_model.set_hop(hopping[151],1,8,[1,-1,-1])
        self.my_model.set_hop(hopping[152],1,12,[1,-1,-1])
        self.my_model.set_hop(hopping[153],0,5,[-1,1,1])
        self.my_model.set_hop(hopping[154],0,5,[-1,0,1])
        self.my_model.set_hop(hopping[155],0,5,[0,1,1])
        self.my_model.set_hop(hopping[156],0,13,[-1,0,1])
        self.my_model.set_hop(hopping[157],0,13,[0,1,1])
        self.my_model.set_hop(hopping[158],0,9,[0,1,1])
        self.my_model.set_hop(hopping[159],1,8,[1,0,-1])
        self.my_model.set_hop(hopping[160],0,9,[-1,0,1])
        self.my_model.set_hop(hopping[161],1,8,[0,-1,-1])
        self.my_model.set_hop(hopping[162],1,12,[0,-1,-1])
        self.my_model.set_hop(hopping[163],1,12,[1,0,-1])
        self.my_model.set_hop(hopping[164],1,4,[0,-1,-1])
        self.my_model.set_hop(hopping[165],1,4,[1,0,-1])
        self.my_model.set_hop(hopping[166],0,6,[-1,-1,0])
        self.my_model.set_hop(hopping[167],3,5,[-1,-1,0])
        self.my_model.set_hop(hopping[168],1,15,[-1,-1,0])
        self.my_model.set_hop(hopping[169],2,12,[-1,-1,0])
        self.my_model.set_hop(hopping[170],0,14,[-1,-1,0])
        self.my_model.set_hop(hopping[171],3,13,[-1,-1,0])
        self.my_model.set_hop(hopping[172],0,10,[-1,-1,0])
        self.my_model.set_hop(hopping[173],1,11,[-1,-1,0])
        self.my_model.set_hop(hopping[174],2,8,[-1,-1,0])
        self.my_model.set_hop(hopping[175],3,9,[-1,-1,0])
        self.my_model.set_hop(hopping[176],1,7,[-1,-1,0])
        self.my_model.set_hop(hopping[177],2,4,[-1,-1,0])
        self.my_model.set_hop(hopping[178],2,15,[0,0,1])
        self.my_model.set_hop(hopping[179],2,7,[0,0,1])
        self.my_model.set_hop(hopping[180],3,10,[0,0,-1])
        self.my_model.set_hop(hopping[181],2,11,[0,0,1])
        self.my_model.set_hop(hopping[182],3,6,[0,0,-1])
        self.my_model.set_hop(hopping[183],3,14,[0,0,-1])
        self.my_model.set_hop(hopping[184],1,12,[0,-2,0])
        self.my_model.set_hop(hopping[185],1,12,[2,0,0])
        self.my_model.set_hop(hopping[186],0,9,[0,2,0])
        self.my_model.set_hop(hopping[187],1,8,[2,0,0])
        self.my_model.set_hop(hopping[188],1,4,[0,-2,0])
        self.my_model.set_hop(hopping[189],1,4,[2,0,0])
        self.my_model.set_hop(hopping[190],0,5,[-2,0,0])
        self.my_model.set_hop(hopping[191],0,5,[0,2,0])
        self.my_model.set_hop(hopping[192],0,9,[-2,0,0])
        self.my_model.set_hop(hopping[193],1,8,[0,-2,0])
        self.my_model.set_hop(hopping[194],0,13,[-2,0,0])
        self.my_model.set_hop(hopping[195],0,13,[0,2,0])
        self.my_model.set_hop(hopping[196],0,9,[0,2,1])
        self.my_model.set_hop(hopping[197],1,8,[2,0,-1])
        self.my_model.set_hop(hopping[198],1,4,[0,-2,-1])
        self.my_model.set_hop(hopping[199],1,4,[2,0,-1])
        self.my_model.set_hop(hopping[200],0,13,[-2,0,1])
        self.my_model.set_hop(hopping[201],0,13,[0,2,1])
        self.my_model.set_hop(hopping[202],1,12,[0,-2,-1])
        self.my_model.set_hop(hopping[203],1,12,[2,0,-1])
        self.my_model.set_hop(hopping[204],0,5,[-2,0,1])
        self.my_model.set_hop(hopping[205],0,5,[0,2,1])
        self.my_model.set_hop(hopping[206],0,9,[-2,0,1])
        self.my_model.set_hop(hopping[207],1,8,[0,-2,-1])
        self.my_model.set_hop(hopping[208],2,7,[0,0,0])
        self.my_model.set_hop(hopping[209],3,14,[0,0,0])
        self.my_model.set_hop(hopping[210],2,11,[0,0,0])
        self.my_model.set_hop(hopping[211],3,10,[0,0,0])
        self.my_model.set_hop(hopping[212],2,15,[0,0,0])
        self.my_model.set_hop(hopping[213],3,6,[0,0,0])
        self.my_model.set_hop(hopping[214],2,15,[-1,0,1])
        self.my_model.set_hop(hopping[215],2,15,[0,1,1])
        self.my_model.set_hop(hopping[216],3,6,[0,-1,-1])
        self.my_model.set_hop(hopping[217],3,6,[1,0,-1])
        self.my_model.set_hop(hopping[218],2,11,[0,1,1])
        self.my_model.set_hop(hopping[219],3,10,[1,0,-1])
        self.my_model.set_hop(hopping[220],2,11,[-1,0,1])
        self.my_model.set_hop(hopping[221],3,10,[0,-1,-1])
        self.my_model.set_hop(hopping[222],2,7,[-1,0,1])
        self.my_model.set_hop(hopping[223],2,7,[0,1,1])
        self.my_model.set_hop(hopping[224],3,14,[0,-1,-1])
        self.my_model.set_hop(hopping[225],3,14,[1,0,-1])
        self.my_model.set_hop(hopping[226],4,4,[-1,-1,0])
        self.my_model.set_hop(hopping[227],5,5,[-1,-1,0])
        self.my_model.set_hop(hopping[228],6,6,[-1,-1,0])
        self.my_model.set_hop(hopping[229],7,7,[-1,-1,0])
        self.my_model.set_hop(hopping[230],12,12,[-1,-1,0])
        self.my_model.set_hop(hopping[231],13,13,[-1,-1,0])
        self.my_model.set_hop(hopping[232],14,14,[-1,-1,0])
        self.my_model.set_hop(hopping[233],15,15,[-1,-1,0])
        self.my_model.set_hop(hopping[234],4,8,[-1,-1,0])
        self.my_model.set_hop(hopping[235],4,12,[-1,-1,0])
        self.my_model.set_hop(hopping[236],5,9,[-1,-1,0])
        self.my_model.set_hop(hopping[237],5,13,[-1,-1,0])
        self.my_model.set_hop(hopping[238],6,10,[-1,-1,0])
        self.my_model.set_hop(hopping[239],6,14,[-1,-1,0])
        self.my_model.set_hop(hopping[240],7,11,[-1,-1,0])
        self.my_model.set_hop(hopping[241],7,15,[-1,-1,0])
        self.my_model.set_hop(hopping[242],8,12,[-1,-1,0])
        self.my_model.set_hop(hopping[243],9,13,[-1,-1,0])
        self.my_model.set_hop(hopping[244],10,14,[-1,-1,0])
        self.my_model.set_hop(hopping[245],11,15,[-1,-1,0])
        self.my_model.set_hop(hopping[246],8,8,[-1,-1,0])
        self.my_model.set_hop(hopping[247],9,9,[-1,-1,0])
        self.my_model.set_hop(hopping[248],10,10,[-1,-1,0])
        self.my_model.set_hop(hopping[249],11,11,[-1,-1,0])
        self.my_model.set_hop(hopping[250],11,14,[0,-1,0])
        self.my_model.set_hop(hopping[251],10,15,[0,1,0])
        self.my_model.set_hop(hopping[252],7,10,[0,-1,0])
        self.my_model.set_hop(hopping[253],6,11,[0,1,0])
        self.my_model.set_hop(hopping[254],10,11,[-1,0,0])
        self.my_model.set_hop(hopping[255],10,11,[0,1,0])
        self.my_model.set_hop(hopping[256],6,11,[-1,0,0])
        self.my_model.set_hop(hopping[257],7,10,[1,0,0])
        self.my_model.set_hop(hopping[258],6,7,[-1,0,0])
        self.my_model.set_hop(hopping[259],6,7,[0,1,0])
        self.my_model.set_hop(hopping[260],10,15,[-1,0,0])
        self.my_model.set_hop(hopping[261],11,14,[1,0,0])
        self.my_model.set_hop(hopping[262],6,15,[-1,0,0])
        self.my_model.set_hop(hopping[263],7,14,[0,-1,0])
        self.my_model.set_hop(hopping[264],6,15,[0,1,0])
        self.my_model.set_hop(hopping[265],7,14,[1,0,0])
        self.my_model.set_hop(hopping[266],14,15,[-1,0,0])
        self.my_model.set_hop(hopping[267],14,15,[0,1,0])
        self.my_model.set_hop(hopping[268],4,14,[0,0,0])
        self.my_model.set_hop(hopping[269],6,12,[0,0,0])
        self.my_model.set_hop(hopping[270],8,10,[0,0,0])
        self.my_model.set_hop(hopping[271],4,10,[0,0,0])
        self.my_model.set_hop(hopping[272],6,8,[0,0,0])
        self.my_model.set_hop(hopping[273],8,14,[0,0,0])
        self.my_model.set_hop(hopping[274],10,12,[0,0,0])
        self.my_model.set_hop(hopping[275],12,14,[0,0,0])
        self.my_model.set_hop(hopping[276],4,6,[0,0,0])
        self.my_model.set_hop(hopping[277],5,15,[0,0,0])
        self.my_model.set_hop(hopping[278],7,13,[0,0,0])
        self.my_model.set_hop(hopping[279],9,11,[0,0,0])
        self.my_model.set_hop(hopping[280],5,11,[0,0,0])
        self.my_model.set_hop(hopping[281],7,9,[0,0,0])
        self.my_model.set_hop(hopping[282],9,15,[0,0,0])
        self.my_model.set_hop(hopping[283],11,13,[0,0,0])
        self.my_model.set_hop(hopping[284],13,15,[0,0,0])
        self.my_model.set_hop(hopping[285],5,7,[0,0,0])
        self.my_model.set_hop(hopping[286],10,13,[-1,0,1])
        self.my_model.set_hop(hopping[287],9,14,[1,0,-1])
        self.my_model.set_hop(hopping[288],6,9,[-1,0,1])
        self.my_model.set_hop(hopping[289],5,10,[1,0,-1])
        self.my_model.set_hop(hopping[290],5,6,[0,-1,-1])
        self.my_model.set_hop(hopping[291],5,6,[1,0,-1])
        self.my_model.set_hop(hopping[292],5,10,[0,-1,-1])
        self.my_model.set_hop(hopping[293],6,9,[0,1,1])
        self.my_model.set_hop(hopping[294],6,13,[-1,0,1])
        self.my_model.set_hop(hopping[295],5,14,[0,-1,-1])
        self.my_model.set_hop(hopping[296],6,13,[0,1,1])
        self.my_model.set_hop(hopping[297],5,14,[1,0,-1])
        self.my_model.set_hop(hopping[298],9,10,[0,-1,-1])
        self.my_model.set_hop(hopping[299],9,10,[1,0,-1])
        self.my_model.set_hop(hopping[300],9,14,[0,-1,-1])
        self.my_model.set_hop(hopping[301],10,13,[0,1,1])
        self.my_model.set_hop(hopping[302],13,14,[0,-1,-1])
        self.my_model.set_hop(hopping[303],13,14,[1,0,-1])
        self.my_model.set_hop(hopping[304],8,15,[-1,0,1])
        self.my_model.set_hop(hopping[305],11,12,[1,0,-1])
        self.my_model.set_hop(hopping[306],4,11,[-1,0,1])
        self.my_model.set_hop(hopping[307],7,8,[1,0,-1])
        self.my_model.set_hop(hopping[308],4,7,[-1,0,1])
        self.my_model.set_hop(hopping[309],4,7,[0,1,1])
        self.my_model.set_hop(hopping[310],7,8,[0,-1,-1])
        self.my_model.set_hop(hopping[311],4,11,[0,1,1])
        self.my_model.set_hop(hopping[312],4,15,[-1,0,1])
        self.my_model.set_hop(hopping[313],7,12,[0,-1,-1])
        self.my_model.set_hop(hopping[314],4,15,[0,1,1])
        self.my_model.set_hop(hopping[315],7,12,[1,0,-1])
        self.my_model.set_hop(hopping[316],8,11,[-1,0,1])
        self.my_model.set_hop(hopping[317],8,11,[0,1,1])
        self.my_model.set_hop(hopping[318],11,12,[0,-1,-1])
        self.my_model.set_hop(hopping[319],8,15,[0,1,1])
        self.my_model.set_hop(hopping[320],12,15,[-1,0,1])
        self.my_model.set_hop(hopping[321],12,15,[0,1,1])
        self.my_model.set_hop(hopping[322],5,11,[0,-1,0])
        self.my_model.set_hop(hopping[323],7,9,[0,1,0])
        self.my_model.set_hop(hopping[324],9,15,[0,-1,0])
        self.my_model.set_hop(hopping[325],11,13,[0,1,0])
        self.my_model.set_hop(hopping[326],13,15,[0,-1,0])
        self.my_model.set_hop(hopping[327],13,15,[1,0,0])
        self.my_model.set_hop(hopping[328],11,13,[-1,0,0])
        self.my_model.set_hop(hopping[329],9,15,[1,0,0])
        self.my_model.set_hop(hopping[330],7,13,[-1,0,0])
        self.my_model.set_hop(hopping[331],5,15,[0,-1,0])
        self.my_model.set_hop(hopping[332],7,13,[0,1,0])
        self.my_model.set_hop(hopping[333],5,15,[1,0,0])
        self.my_model.set_hop(hopping[334],9,11,[0,-1,0])
        self.my_model.set_hop(hopping[335],9,11,[1,0,0])
        self.my_model.set_hop(hopping[336],7,9,[-1,0,0])
        self.my_model.set_hop(hopping[337],5,11,[1,0,0])
        self.my_model.set_hop(hopping[338],5,7,[0,-1,0])
        self.my_model.set_hop(hopping[339],5,7,[1,0,0])
        self.my_model.set_hop(hopping[340],6,8,[0,-1,0])
        self.my_model.set_hop(hopping[341],4,10,[0,1,0])
        self.my_model.set_hop(hopping[342],10,12,[0,-1,0])
        self.my_model.set_hop(hopping[343],8,14,[0,1,0])
        self.my_model.set_hop(hopping[344],12,14,[-1,0,0])
        self.my_model.set_hop(hopping[345],12,14,[0,1,0])
        self.my_model.set_hop(hopping[346],8,14,[-1,0,0])
        self.my_model.set_hop(hopping[347],10,12,[1,0,0])
        self.my_model.set_hop(hopping[348],4,14,[-1,0,0])
        self.my_model.set_hop(hopping[349],6,12,[0,-1,0])
        self.my_model.set_hop(hopping[350],4,14,[0,1,0])
        self.my_model.set_hop(hopping[351],6,12,[1,0,0])
        self.my_model.set_hop(hopping[352],8,10,[-1,0,0])
        self.my_model.set_hop(hopping[353],8,10,[0,1,0])
        self.my_model.set_hop(hopping[354],4,10,[-1,0,0])
        self.my_model.set_hop(hopping[355],6,8,[1,0,0])
        self.my_model.set_hop(hopping[356],4,6,[-1,0,0])
        self.my_model.set_hop(hopping[357],4,6,[0,1,0])
        self.my_model.set_hop(hopping[358],9,14,[0,-1,0])
        self.my_model.set_hop(hopping[359],11,12,[0,-1,0])
        self.my_model.set_hop(hopping[360],8,15,[0,1,0])
        self.my_model.set_hop(hopping[361],10,13,[0,1,0])
        self.my_model.set_hop(hopping[362],4,15,[-1,0,0])
        self.my_model.set_hop(hopping[363],6,13,[-1,0,0])
        self.my_model.set_hop(hopping[364],5,14,[0,-1,0])
        self.my_model.set_hop(hopping[365],7,12,[0,-1,0])
        self.my_model.set_hop(hopping[366],4,15,[0,1,0])
        self.my_model.set_hop(hopping[367],6,13,[0,1,0])
        self.my_model.set_hop(hopping[368],5,14,[1,0,0])
        self.my_model.set_hop(hopping[369],7,12,[1,0,0])
        self.my_model.set_hop(hopping[370],4,11,[-1,0,0])
        self.my_model.set_hop(hopping[371],6,9,[-1,0,0])
        self.my_model.set_hop(hopping[372],5,10,[1,0,0])
        self.my_model.set_hop(hopping[373],7,8,[1,0,0])
        self.my_model.set_hop(hopping[374],4,7,[-1,0,0])
        self.my_model.set_hop(hopping[375],5,6,[0,-1,0])
        self.my_model.set_hop(hopping[376],4,7,[0,1,0])
        self.my_model.set_hop(hopping[377],5,6,[1,0,0])
        self.my_model.set_hop(hopping[378],5,10,[0,-1,0])
        self.my_model.set_hop(hopping[379],7,8,[0,-1,0])
        self.my_model.set_hop(hopping[380],4,11,[0,1,0])
        self.my_model.set_hop(hopping[381],6,9,[0,1,0])
        self.my_model.set_hop(hopping[382],8,11,[-1,0,0])
        self.my_model.set_hop(hopping[383],9,10,[0,-1,0])
        self.my_model.set_hop(hopping[384],8,11,[0,1,0])
        self.my_model.set_hop(hopping[385],9,10,[1,0,0])
        self.my_model.set_hop(hopping[386],8,15,[-1,0,0])
        self.my_model.set_hop(hopping[387],10,13,[-1,0,0])
        self.my_model.set_hop(hopping[388],9,14,[1,0,0])
        self.my_model.set_hop(hopping[389],11,12,[1,0,0])
        self.my_model.set_hop(hopping[390],12,15,[-1,0,0])
        self.my_model.set_hop(hopping[391],12,15,[0,1,0])
        self.my_model.set_hop(hopping[392],13,14,[0,-1,0])
        self.my_model.set_hop(hopping[393],13,14,[1,0,0])
        self.my_model.set_hop(hopping[394],8,9,[-1,1,0])
        self.my_model.set_hop(hopping[395],4,9,[-1,1,0])
        self.my_model.set_hop(hopping[396],8,13,[-1,1,0])
        self.my_model.set_hop(hopping[397],5,8,[1,-1,0])
        self.my_model.set_hop(hopping[398],9,12,[1,-1,0])
        self.my_model.set_hop(hopping[399],4,5,[-1,1,0])
        self.my_model.set_hop(hopping[400],4,13,[-1,1,0])
        self.my_model.set_hop(hopping[401],5,12,[1,-1,0])
        self.my_model.set_hop(hopping[402],12,13,[-1,1,0])
        self.my_model.set_hop(hopping[403],4,13,[-1,1,1])
        self.my_model.set_hop(hopping[404],5,12,[1,-1,-1])
        self.my_model.set_hop(hopping[405],8,9,[-1,1,1])
        self.my_model.set_hop(hopping[406],4,9,[-1,1,1])
        self.my_model.set_hop(hopping[407],8,13,[-1,1,1])
        self.my_model.set_hop(hopping[408],5,8,[1,-1,-1])
        self.my_model.set_hop(hopping[409],9,12,[1,-1,-1])
        self.my_model.set_hop(hopping[410],12,13,[-1,1,1])
        self.my_model.set_hop(hopping[411],4,5,[-1,1,1])
        self.my_model.set_hop(hopping[412],4,9,[-1,0,1])
        self.my_model.set_hop(hopping[413],5,8,[1,0,-1])
        self.my_model.set_hop(hopping[414],8,13,[-1,0,1])
        self.my_model.set_hop(hopping[415],9,12,[1,0,-1])
        self.my_model.set_hop(hopping[416],8,9,[-1,0,1])
        self.my_model.set_hop(hopping[417],8,9,[0,1,1])
        self.my_model.set_hop(hopping[418],9,12,[0,-1,-1])
        self.my_model.set_hop(hopping[419],8,13,[0,1,1])
        self.my_model.set_hop(hopping[420],5,8,[0,-1,-1])
        self.my_model.set_hop(hopping[421],4,9,[0,1,1])
        self.my_model.set_hop(hopping[422],12,13,[-1,0,1])
        self.my_model.set_hop(hopping[423],12,13,[0,1,1])
        self.my_model.set_hop(hopping[424],4,13,[-1,0,1])
        self.my_model.set_hop(hopping[425],5,12,[0,-1,-1])
        self.my_model.set_hop(hopping[426],4,13,[0,1,1])
        self.my_model.set_hop(hopping[427],5,12,[1,0,-1])
        self.my_model.set_hop(hopping[428],4,5,[-1,0,1])
        self.my_model.set_hop(hopping[429],4,5,[0,1,1])
        self.my_model.set_hop(hopping[430],4,10,[-1,-1,0])
        self.my_model.set_hop(hopping[431],7,9,[-1,-1,0])
        self.my_model.set_hop(hopping[432],5,15,[-1,-1,0])
        self.my_model.set_hop(hopping[433],7,13,[-1,-1,0])
        self.my_model.set_hop(hopping[434],4,14,[-1,-1,0])
        self.my_model.set_hop(hopping[435],6,12,[-1,-1,0])
        self.my_model.set_hop(hopping[436],9,15,[-1,-1,0])
        self.my_model.set_hop(hopping[437],10,12,[-1,-1,0])
        self.my_model.set_hop(hopping[438],12,14,[-1,-1,0])
        self.my_model.set_hop(hopping[439],13,15,[-1,-1,0])
        self.my_model.set_hop(hopping[440],8,14,[-1,-1,0])
        self.my_model.set_hop(hopping[441],11,13,[-1,-1,0])
        self.my_model.set_hop(hopping[442],8,10,[-1,-1,0])
        self.my_model.set_hop(hopping[443],9,11,[-1,-1,0])
        self.my_model.set_hop(hopping[444],5,11,[-1,-1,0])
        self.my_model.set_hop(hopping[445],6,8,[-1,-1,0])
        self.my_model.set_hop(hopping[446],4,6,[-1,-1,0])
        self.my_model.set_hop(hopping[447],5,7,[-1,-1,0])
        self.my_model.set_hop(hopping[448],10,11,[0,0,1])
        self.my_model.set_hop(hopping[449],7,10,[0,0,-1])
        self.my_model.set_hop(hopping[450],11,14,[0,0,-1])
        self.my_model.set_hop(hopping[451],6,11,[0,0,1])
        self.my_model.set_hop(hopping[452],10,15,[0,0,1])
        self.my_model.set_hop(hopping[453],6,7,[0,0,1])
        self.my_model.set_hop(hopping[454],7,14,[0,0,-1])
        self.my_model.set_hop(hopping[455],6,15,[0,0,1])
        self.my_model.set_hop(hopping[456],14,15,[0,0,1])
        self.my_model.set_hop(hopping[457],9,12,[0,-2,0])
        self.my_model.set_hop(hopping[458],8,13,[0,2,0])
        self.my_model.set_hop(hopping[459],5,8,[0,-2,0])
        self.my_model.set_hop(hopping[460],4,9,[0,2,0])
        self.my_model.set_hop(hopping[461],4,5,[-2,0,0])
        self.my_model.set_hop(hopping[462],4,5,[0,2,0])
        self.my_model.set_hop(hopping[463],4,9,[-2,0,0])
        self.my_model.set_hop(hopping[464],5,8,[2,0,0])
        self.my_model.set_hop(hopping[465],4,13,[-2,0,0])
        self.my_model.set_hop(hopping[466],5,12,[0,-2,0])
        self.my_model.set_hop(hopping[467],4,13,[0,2,0])
        self.my_model.set_hop(hopping[468],5,12,[2,0,0])
        self.my_model.set_hop(hopping[469],8,9,[-2,0,0])
        self.my_model.set_hop(hopping[470],8,9,[0,2,0])
        self.my_model.set_hop(hopping[471],8,13,[-2,0,0])
        self.my_model.set_hop(hopping[472],9,12,[2,0,0])
        self.my_model.set_hop(hopping[473],12,13,[-2,0,0])
        self.my_model.set_hop(hopping[474],12,13,[0,2,0])
        self.my_model.set_hop(hopping[475],5,8,[0,-2,-1])
        self.my_model.set_hop(hopping[476],4,9,[0,2,1])
        self.my_model.set_hop(hopping[477],8,13,[-2,0,1])
        self.my_model.set_hop(hopping[478],9,12,[2,0,-1])
        self.my_model.set_hop(hopping[479],4,13,[-2,0,1])
        self.my_model.set_hop(hopping[480],5,12,[0,-2,-1])
        self.my_model.set_hop(hopping[481],4,13,[0,2,1])
        self.my_model.set_hop(hopping[482],5,12,[2,0,-1])
        self.my_model.set_hop(hopping[483],12,13,[-2,0,1])
        self.my_model.set_hop(hopping[484],12,13,[0,2,1])
        self.my_model.set_hop(hopping[485],9,12,[0,-2,-1])
        self.my_model.set_hop(hopping[486],8,13,[0,2,1])
        self.my_model.set_hop(hopping[487],4,5,[-2,0,1])
        self.my_model.set_hop(hopping[488],4,5,[0,2,1])
        self.my_model.set_hop(hopping[489],4,9,[-2,0,1])
        self.my_model.set_hop(hopping[490],5,8,[2,0,-1])
        self.my_model.set_hop(hopping[491],8,9,[-2,0,1])
        self.my_model.set_hop(hopping[492],8,9,[0,2,1])
        self.my_model.set_hop(hopping[493],6,15,[0,0,0])
        self.my_model.set_hop(hopping[494],7,14,[0,0,0])
        self.my_model.set_hop(hopping[495],10,11,[0,0,0])
        self.my_model.set_hop(hopping[496],6,11,[0,0,0])
        self.my_model.set_hop(hopping[497],7,10,[0,0,0])
        self.my_model.set_hop(hopping[498],10,15,[0,0,0])
        self.my_model.set_hop(hopping[499],11,14,[0,0,0])
        self.my_model.set_hop(hopping[500],14,15,[0,0,0])
        self.my_model.set_hop(hopping[501],6,7,[0,0,0])
        self.my_model.set_hop(hopping[502],6,15,[-1,0,1])
        self.my_model.set_hop(hopping[503],7,14,[0,-1,-1])
        self.my_model.set_hop(hopping[504],6,15,[0,1,1])
        self.my_model.set_hop(hopping[505],7,14,[1,0,-1])
        self.my_model.set_hop(hopping[506],10,15,[-1,0,1])
        self.my_model.set_hop(hopping[507],11,14,[1,0,-1])
        self.my_model.set_hop(hopping[508],7,10,[0,-1,-1])
        self.my_model.set_hop(hopping[509],6,11,[0,1,1])
        self.my_model.set_hop(hopping[510],10,11,[-1,0,1])
        self.my_model.set_hop(hopping[511],10,11,[0,1,1])
        self.my_model.set_hop(hopping[512],6,11,[-1,0,1])
        self.my_model.set_hop(hopping[513],7,10,[1,0,-1])
        self.my_model.set_hop(hopping[514],11,14,[0,-1,-1])
        self.my_model.set_hop(hopping[515],10,15,[0,1,1])
        self.my_model.set_hop(hopping[516],6,7,[-1,0,1])
        self.my_model.set_hop(hopping[517],6,7,[0,1,1])
        self.my_model.set_hop(hopping[518],14,15,[-1,0,1])
        self.my_model.set_hop(hopping[519],14,15,[0,1,1])
        self.my_model.set_hop(hopping[520],8,16,[-1,-1,0])
        self.my_model.set_hop(hopping[521],9,17,[-1,-1,0])
        self.my_model.set_hop(hopping[522],10,18,[-1,-1,0])
        self.my_model.set_hop(hopping[523],11,19,[-1,-1,0])
        self.my_model.set_hop(hopping[524],4,16,[-1,-1,0])
        self.my_model.set_hop(hopping[525],5,17,[-1,-1,0])
        self.my_model.set_hop(hopping[526],6,18,[-1,-1,0])
        self.my_model.set_hop(hopping[527],7,19,[-1,-1,0])
        self.my_model.set_hop(hopping[528],12,16,[-1,-1,0])
        self.my_model.set_hop(hopping[529],13,17,[-1,-1,0])
        self.my_model.set_hop(hopping[530],14,18,[-1,-1,0])
        self.my_model.set_hop(hopping[531],15,19,[-1,-1,0])
        self.my_model.set_hop(hopping[532],14,19,[-1,0,0])
        self.my_model.set_hop(hopping[533],14,19,[0,1,0])
        self.my_model.set_hop(hopping[534],6,19,[-1,0,0])
        self.my_model.set_hop(hopping[535],6,19,[0,1,0])
        self.my_model.set_hop(hopping[536],10,19,[-1,0,0])
        self.my_model.set_hop(hopping[537],11,18,[0,-1,0])
        self.my_model.set_hop(hopping[538],10,19,[0,1,0])
        self.my_model.set_hop(hopping[539],11,18,[1,0,0])
        self.my_model.set_hop(hopping[540],7,18,[0,-1,0])
        self.my_model.set_hop(hopping[541],7,18,[1,0,0])
        self.my_model.set_hop(hopping[542],15,18,[0,-1,0])
        self.my_model.set_hop(hopping[543],15,18,[1,0,0])
        self.my_model.set_hop(hopping[544],6,16,[0,0,0])
        self.my_model.set_hop(hopping[545],12,18,[0,0,0])
        self.my_model.set_hop(hopping[546],8,18,[0,0,0])
        self.my_model.set_hop(hopping[547],10,16,[0,0,0])
        self.my_model.set_hop(hopping[548],14,16,[0,0,0])
        self.my_model.set_hop(hopping[549],4,18,[0,0,0])
        self.my_model.set_hop(hopping[550],5,19,[0,0,0])
        self.my_model.set_hop(hopping[551],15,17,[0,0,0])
        self.my_model.set_hop(hopping[552],9,19,[0,0,0])
        self.my_model.set_hop(hopping[553],11,17,[0,0,0])
        self.my_model.set_hop(hopping[554],13,19,[0,0,0])
        self.my_model.set_hop(hopping[555],7,17,[0,0,0])
        self.my_model.set_hop(hopping[556],13,18,[0,-1,-1])
        self.my_model.set_hop(hopping[557],13,18,[1,0,-1])
        self.my_model.set_hop(hopping[558],10,17,[-1,0,1])
        self.my_model.set_hop(hopping[559],9,18,[0,-1,-1])
        self.my_model.set_hop(hopping[560],5,18,[0,-1,-1])
        self.my_model.set_hop(hopping[561],5,18,[1,0,-1])
        self.my_model.set_hop(hopping[562],6,17,[-1,0,1])
        self.my_model.set_hop(hopping[563],6,17,[0,1,1])
        self.my_model.set_hop(hopping[564],10,17,[0,1,1])
        self.my_model.set_hop(hopping[565],9,18,[1,0,-1])
        self.my_model.set_hop(hopping[566],14,17,[-1,0,1])
        self.my_model.set_hop(hopping[567],14,17,[0,1,1])
        self.my_model.set_hop(hopping[568],15,16,[0,-1,-1])
        self.my_model.set_hop(hopping[569],15,16,[1,0,-1])
        self.my_model.set_hop(hopping[570],8,19,[-1,0,1])
        self.my_model.set_hop(hopping[571],11,16,[0,-1,-1])
        self.my_model.set_hop(hopping[572],7,16,[0,-1,-1])
        self.my_model.set_hop(hopping[573],7,16,[1,0,-1])
        self.my_model.set_hop(hopping[574],4,19,[-1,0,1])
        self.my_model.set_hop(hopping[575],4,19,[0,1,1])
        self.my_model.set_hop(hopping[576],8,19,[0,1,1])
        self.my_model.set_hop(hopping[577],11,16,[1,0,-1])
        self.my_model.set_hop(hopping[578],12,19,[-1,0,1])
        self.my_model.set_hop(hopping[579],12,19,[0,1,1])
        self.my_model.set_hop(hopping[580],7,17,[-1,0,0])
        self.my_model.set_hop(hopping[581],7,17,[0,1,0])
        self.my_model.set_hop(hopping[582],11,17,[-1,0,0])
        self.my_model.set_hop(hopping[583],9,19,[0,-1,0])
        self.my_model.set_hop(hopping[584],15,17,[-1,0,0])
        self.my_model.set_hop(hopping[585],15,17,[0,1,0])
        self.my_model.set_hop(hopping[586],13,19,[0,-1,0])
        self.my_model.set_hop(hopping[587],13,19,[1,0,0])
        self.my_model.set_hop(hopping[588],11,17,[0,1,0])
        self.my_model.set_hop(hopping[589],9,19,[1,0,0])
        self.my_model.set_hop(hopping[590],5,19,[0,-1,0])
        self.my_model.set_hop(hopping[591],5,19,[1,0,0])
        self.my_model.set_hop(hopping[592],4,18,[-1,0,0])
        self.my_model.set_hop(hopping[593],4,18,[0,1,0])
        self.my_model.set_hop(hopping[594],8,18,[-1,0,0])
        self.my_model.set_hop(hopping[595],10,16,[0,-1,0])
        self.my_model.set_hop(hopping[596],12,18,[-1,0,0])
        self.my_model.set_hop(hopping[597],12,18,[0,1,0])
        self.my_model.set_hop(hopping[598],14,16,[0,-1,0])
        self.my_model.set_hop(hopping[599],14,16,[1,0,0])
        self.my_model.set_hop(hopping[600],8,18,[0,1,0])
        self.my_model.set_hop(hopping[601],10,16,[1,0,0])
        self.my_model.set_hop(hopping[602],6,16,[0,-1,0])
        self.my_model.set_hop(hopping[603],6,16,[1,0,0])
        self.my_model.set_hop(hopping[604],14,17,[-1,0,0])
        self.my_model.set_hop(hopping[605],14,17,[0,1,0])
        self.my_model.set_hop(hopping[606],12,19,[-1,0,0])
        self.my_model.set_hop(hopping[607],12,19,[0,1,0])
        self.my_model.set_hop(hopping[608],8,19,[-1,0,0])
        self.my_model.set_hop(hopping[609],10,17,[-1,0,0])
        self.my_model.set_hop(hopping[610],9,18,[0,-1,0])
        self.my_model.set_hop(hopping[611],11,16,[0,-1,0])
        self.my_model.set_hop(hopping[612],5,18,[0,-1,0])
        self.my_model.set_hop(hopping[613],7,16,[0,-1,0])
        self.my_model.set_hop(hopping[614],5,18,[1,0,0])
        self.my_model.set_hop(hopping[615],7,16,[1,0,0])
        self.my_model.set_hop(hopping[616],4,19,[-1,0,0])
        self.my_model.set_hop(hopping[617],6,17,[-1,0,0])
        self.my_model.set_hop(hopping[618],4,19,[0,1,0])
        self.my_model.set_hop(hopping[619],6,17,[0,1,0])
        self.my_model.set_hop(hopping[620],8,19,[0,1,0])
        self.my_model.set_hop(hopping[621],10,17,[0,1,0])
        self.my_model.set_hop(hopping[622],9,18,[1,0,0])
        self.my_model.set_hop(hopping[623],11,16,[1,0,0])
        self.my_model.set_hop(hopping[624],15,16,[0,-1,0])
        self.my_model.set_hop(hopping[625],15,16,[1,0,0])
        self.my_model.set_hop(hopping[626],13,18,[0,-1,0])
        self.my_model.set_hop(hopping[627],13,18,[1,0,0])
        self.my_model.set_hop(hopping[628],12,17,[-1,1,0])
        self.my_model.set_hop(hopping[629],4,17,[-1,1,0])
        self.my_model.set_hop(hopping[630],8,17,[-1,1,0])
        self.my_model.set_hop(hopping[631],9,16,[1,-1,0])
        self.my_model.set_hop(hopping[632],5,16,[1,-1,0])
        self.my_model.set_hop(hopping[633],13,16,[1,-1,0])
        self.my_model.set_hop(hopping[634],4,17,[-1,1,1])
        self.my_model.set_hop(hopping[635],13,16,[1,-1,-1])
        self.my_model.set_hop(hopping[636],8,17,[-1,1,1])
        self.my_model.set_hop(hopping[637],9,16,[1,-1,-1])
        self.my_model.set_hop(hopping[638],12,17,[-1,1,1])
        self.my_model.set_hop(hopping[639],5,16,[1,-1,-1])
        self.my_model.set_hop(hopping[640],5,16,[0,-1,-1])
        self.my_model.set_hop(hopping[641],5,16,[1,0,-1])
        self.my_model.set_hop(hopping[642],13,16,[0,-1,-1])
        self.my_model.set_hop(hopping[643],13,16,[1,0,-1])
        self.my_model.set_hop(hopping[644],8,17,[-1,0,1])
        self.my_model.set_hop(hopping[645],9,16,[0,-1,-1])
        self.my_model.set_hop(hopping[646],8,17,[0,1,1])
        self.my_model.set_hop(hopping[647],9,16,[1,0,-1])
        self.my_model.set_hop(hopping[648],12,17,[-1,0,1])
        self.my_model.set_hop(hopping[649],12,17,[0,1,1])
        self.my_model.set_hop(hopping[650],4,17,[-1,0,1])
        self.my_model.set_hop(hopping[651],4,17,[0,1,1])
        self.my_model.set_hop(hopping[652],5,19,[-1,-1,0])
        self.my_model.set_hop(hopping[653],6,16,[-1,-1,0])
        self.my_model.set_hop(hopping[654],8,18,[-1,-1,0])
        self.my_model.set_hop(hopping[655],9,19,[-1,-1,0])
        self.my_model.set_hop(hopping[656],10,16,[-1,-1,0])
        self.my_model.set_hop(hopping[657],11,17,[-1,-1,0])
        self.my_model.set_hop(hopping[658],15,17,[-1,-1,0])
        self.my_model.set_hop(hopping[659],12,18,[-1,-1,0])
        self.my_model.set_hop(hopping[660],14,16,[-1,-1,0])
        self.my_model.set_hop(hopping[661],13,19,[-1,-1,0])
        self.my_model.set_hop(hopping[662],4,18,[-1,-1,0])
        self.my_model.set_hop(hopping[663],7,17,[-1,-1,0])
        self.my_model.set_hop(hopping[664],15,18,[0,0,-1])
        self.my_model.set_hop(hopping[665],7,18,[0,0,-1])
        self.my_model.set_hop(hopping[666],11,18,[0,0,-1])
        self.my_model.set_hop(hopping[667],10,19,[0,0,1])
        self.my_model.set_hop(hopping[668],6,19,[0,0,1])
        self.my_model.set_hop(hopping[669],14,19,[0,0,1])
        self.my_model.set_hop(hopping[670],12,17,[-2,0,0])
        self.my_model.set_hop(hopping[671],12,17,[0,2,0])
        self.my_model.set_hop(hopping[672],8,17,[-2,0,0])
        self.my_model.set_hop(hopping[673],9,16,[0,-2,0])
        self.my_model.set_hop(hopping[674],4,17,[-2,0,0])
        self.my_model.set_hop(hopping[675],4,17,[0,2,0])
        self.my_model.set_hop(hopping[676],5,16,[0,-2,0])
        self.my_model.set_hop(hopping[677],5,16,[2,0,0])
        self.my_model.set_hop(hopping[678],8,17,[0,2,0])
        self.my_model.set_hop(hopping[679],9,16,[2,0,0])
        self.my_model.set_hop(hopping[680],13,16,[0,-2,0])
        self.my_model.set_hop(hopping[681],13,16,[2,0,0])
        self.my_model.set_hop(hopping[682],8,17,[-2,0,1])
        self.my_model.set_hop(hopping[683],9,16,[0,-2,-1])
        self.my_model.set_hop(hopping[684],4,17,[-2,0,1])
        self.my_model.set_hop(hopping[685],4,17,[0,2,1])
        self.my_model.set_hop(hopping[686],13,16,[0,-2,-1])
        self.my_model.set_hop(hopping[687],13,16,[2,0,-1])
        self.my_model.set_hop(hopping[688],12,17,[-2,0,1])
        self.my_model.set_hop(hopping[689],12,17,[0,2,1])
        self.my_model.set_hop(hopping[690],5,16,[0,-2,-1])
        self.my_model.set_hop(hopping[691],5,16,[2,0,-1])
        self.my_model.set_hop(hopping[692],8,17,[0,2,1])
        self.my_model.set_hop(hopping[693],9,16,[2,0,-1])
        self.my_model.set_hop(hopping[694],7,18,[0,0,0])
        self.my_model.set_hop(hopping[695],14,19,[0,0,0])
        self.my_model.set_hop(hopping[696],10,19,[0,0,0])
        self.my_model.set_hop(hopping[697],11,18,[0,0,0])
        self.my_model.set_hop(hopping[698],15,18,[0,0,0])
        self.my_model.set_hop(hopping[699],6,19,[0,0,0])
        self.my_model.set_hop(hopping[700],15,18,[0,-1,-1])
        self.my_model.set_hop(hopping[701],15,18,[1,0,-1])
        self.my_model.set_hop(hopping[702],6,19,[-1,0,1])
        self.my_model.set_hop(hopping[703],6,19,[0,1,1])
        self.my_model.set_hop(hopping[704],10,19,[-1,0,1])
        self.my_model.set_hop(hopping[705],11,18,[0,-1,-1])
        self.my_model.set_hop(hopping[706],10,19,[0,1,1])
        self.my_model.set_hop(hopping[707],11,18,[1,0,-1])
        self.my_model.set_hop(hopping[708],7,18,[0,-1,-1])
        self.my_model.set_hop(hopping[709],7,18,[1,0,-1])
        self.my_model.set_hop(hopping[710],14,19,[-1,0,1])
        self.my_model.set_hop(hopping[711],14,19,[0,1,1])
        #endregion

        # print tight-binding model
        # my_model.display()

        self.deltak = 0.02
        self.k_vec = []
        # sample inside the FBZ
        for i in np.arange(-0.5, 0.5, self.deltak):
            for j in np.arange(-0.5, 0.5, self.deltak):
                for k in np.arange(-0.5, 0.5, self.deltak):
                    inFBZ = 0
                    if (np.sqrt(i**2 + j**2 + k**2) > np.sqrt(0.5**2 + 0.5**2)):# L[0 0.5 0.5]
                        inFBZ = inFBZ + 1
                    if (np.sqrt(i**2 + j**2 + k**2) > np.sqrt(0.5**2 + 0.5**2)):# Y[0.5 0.5 0]
                        inFBZ = inFBZ + 1
                    if (np.sqrt(i**2 + j**2 + k**2) > np.sqrt(0.5**2 + 0.5**2 + 0.5**2)):# M[0.5 0.5 0.5]
                        inFBZ = inFBZ + 1
                    if(inFBZ == 0):
                        self.k_vec.append([i, j, k])

        # total number of points
        self.npoints = len(self.k_vec)
        # 20x volume of the 1BZ is the total state density
        # 2pi is already left off in our k definition
        # the unit here is cm-3
        self.total_states = self.npoints * (self.deltak*1e10/1e2)**3 * np.abs(np.dot(np.cross(lat[0], lat[1]), lat[2]))* 20

        # print('---------------------------------------')
        # print('starting calculation')
        # print('---------------------------------------')
        # print('Calculating bands...')

        # obtain eigenvalues to be plotted
        self.evals = self.my_model.solve_all(self.k_vec)
        # set the botton of the conduction band to 0
        self.evals = self.evals - np.min(self.evals[0])
        np.save('evals.npy', self.evals)
        np.save('k_vec.npy', self.k_vec)

        # step of numerical DOS
        self.deltaE = 0.005
        # calculate ODS
        self.E = np.arange(0, np.amax(self.evals), self.deltaE)
        self.DOS = np.zeros(self.E.shape)
        for band in self.evals:
            for e in band:
                i = int(e / self.deltaE)
                self.DOS[i] = self.DOS[i] + 1
        # use cm-3eV-2 unit
        # self.DOS = self.DOS / self.npoints * self.total_states
        # use arb. unit, nomalized to the # of bands
        self.DOS = self.DOS / np.sum(self.DOS) * 20

    def plotDOS(self, name):
        fig, ax = plt.subplots()
        ax.set_xlim(0, np.max(self.E))
        ax.plot(self.E, self.DOS)
        ax.set_xlabel("Energy (eV)")
        # ax.set_ylabel("DOS (cm-3*eV-1)")
        ax.set_ylabel("DOS (arb. unit)")
        fig.tight_layout()
        fig.savefig(name)
        plt.close(fig)

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