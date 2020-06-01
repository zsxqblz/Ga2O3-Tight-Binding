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
        my_model.set_hop(hopping[22],0,4,[-1,-1,0])
        my_model.set_hop(hopping[23],0,12,[-1,-1,0])
        my_model.set_hop(hopping[24],1,5,[-1,-1,0])
        my_model.set_hop(hopping[25],1,13,[-1,-1,0])
        my_model.set_hop(hopping[26],2,6,[-1,-1,0])
        my_model.set_hop(hopping[27],2,14,[-1,-1,0])
        my_model.set_hop(hopping[28],3,7,[-1,-1,0])
        my_model.set_hop(hopping[29],3,15,[-1,-1,0])
        my_model.set_hop(hopping[30],0,8,[-1,-1,0])
        my_model.set_hop(hopping[31],1,9,[-1,-1,0])
        my_model.set_hop(hopping[32],2,10,[-1,-1,0])
        my_model.set_hop(hopping[33],3,11,[-1,-1,0])
        my_model.set_hop(hopping[34],3,14,[0,-1,0])
        my_model.set_hop(hopping[35],3,14,[1,0,0])
        my_model.set_hop(hopping[36],3,6,[0,-1,0])
        my_model.set_hop(hopping[37],3,6,[1,0,0])
        my_model.set_hop(hopping[38],2,11,[0,1,0])
        my_model.set_hop(hopping[39],3,10,[1,0,0])
        my_model.set_hop(hopping[40],2,11,[-1,0,0])
        my_model.set_hop(hopping[41],3,10,[0,-1,0])
        my_model.set_hop(hopping[42],2,7,[-1,0,0])
        my_model.set_hop(hopping[43],2,7,[0,1,0])
        my_model.set_hop(hopping[44],2,15,[-1,0,0])
        my_model.set_hop(hopping[45],2,15,[0,1,0])
        my_model.set_hop(hopping[46],0,6,[0,0,0])
        my_model.set_hop(hopping[47],2,12,[0,0,0])
        my_model.set_hop(hopping[48],0,10,[0,0,0])
        my_model.set_hop(hopping[49],2,8,[0,0,0])
        my_model.set_hop(hopping[50],0,14,[0,0,0])
        my_model.set_hop(hopping[51],2,4,[0,0,0])
        my_model.set_hop(hopping[52],3,5,[0,0,0])
        my_model.set_hop(hopping[53],1,15,[0,0,0])
        my_model.set_hop(hopping[54],1,11,[0,0,0])
        my_model.set_hop(hopping[55],3,9,[0,0,0])
        my_model.set_hop(hopping[56],3,13,[0,0,0])
        my_model.set_hop(hopping[57],1,7,[0,0,0])
        my_model.set_hop(hopping[58],2,13,[-1,0,1])
        my_model.set_hop(hopping[59],2,13,[0,1,1])
        my_model.set_hop(hopping[60],2,9,[0,1,1])
        my_model.set_hop(hopping[61],1,10,[1,0,-1])
        my_model.set_hop(hopping[62],2,5,[-1,0,1])
        my_model.set_hop(hopping[63],2,5,[0,1,1])
        my_model.set_hop(hopping[64],1,6,[0,-1,-1])
        my_model.set_hop(hopping[65],1,6,[1,0,-1])
        my_model.set_hop(hopping[66],2,9,[-1,0,1])
        my_model.set_hop(hopping[67],1,10,[0,-1,-1])
        my_model.set_hop(hopping[68],1,14,[0,-1,-1])
        my_model.set_hop(hopping[69],1,14,[1,0,-1])
        my_model.set_hop(hopping[70],0,15,[-1,0,1])
        my_model.set_hop(hopping[71],0,15,[0,1,1])
        my_model.set_hop(hopping[72],0,11,[0,1,1])
        my_model.set_hop(hopping[73],3,8,[1,0,-1])
        my_model.set_hop(hopping[74],0,7,[-1,0,1])
        my_model.set_hop(hopping[75],0,7,[0,1,1])
        my_model.set_hop(hopping[76],3,4,[0,-1,-1])
        my_model.set_hop(hopping[77],3,4,[1,0,-1])
        my_model.set_hop(hopping[78],0,11,[-1,0,1])
        my_model.set_hop(hopping[79],3,8,[0,-1,-1])
        my_model.set_hop(hopping[80],3,12,[0,-1,-1])
        my_model.set_hop(hopping[81],3,12,[1,0,-1])
        my_model.set_hop(hopping[82],1,7,[0,-1,0])
        my_model.set_hop(hopping[83],1,7,[1,0,0])
        my_model.set_hop(hopping[84],3,9,[0,1,0])
        my_model.set_hop(hopping[85],1,11,[1,0,0])
        my_model.set_hop(hopping[86],1,15,[0,-1,0])
        my_model.set_hop(hopping[87],1,15,[1,0,0])
        my_model.set_hop(hopping[88],3,13,[-1,0,0])
        my_model.set_hop(hopping[89],3,13,[0,1,0])
        my_model.set_hop(hopping[90],3,9,[-1,0,0])
        my_model.set_hop(hopping[91],1,11,[0,-1,0])
        my_model.set_hop(hopping[92],3,5,[-1,0,0])
        my_model.set_hop(hopping[93],3,5,[0,1,0])
        my_model.set_hop(hopping[94],2,4,[0,-1,0])
        my_model.set_hop(hopping[95],2,4,[1,0,0])
        my_model.set_hop(hopping[96],0,10,[0,1,0])
        my_model.set_hop(hopping[97],2,8,[1,0,0])
        my_model.set_hop(hopping[98],2,12,[0,-1,0])
        my_model.set_hop(hopping[99],2,12,[1,0,0])
        my_model.set_hop(hopping[100],0,14,[-1,0,0])
        my_model.set_hop(hopping[101],0,14,[0,1,0])
        my_model.set_hop(hopping[102],0,10,[-1,0,0])
        my_model.set_hop(hopping[103],2,8,[0,-1,0])
        my_model.set_hop(hopping[104],0,6,[-1,0,0])
        my_model.set_hop(hopping[105],0,6,[0,1,0])
        my_model.set_hop(hopping[106],1,14,[0,-1,0])
        my_model.set_hop(hopping[107],1,14,[1,0,0])
        my_model.set_hop(hopping[108],3,12,[0,-1,0])
        my_model.set_hop(hopping[109],3,12,[1,0,0])
        my_model.set_hop(hopping[110],0,11,[0,1,0])
        my_model.set_hop(hopping[111],2,9,[0,1,0])
        my_model.set_hop(hopping[112],1,10,[1,0,0])
        my_model.set_hop(hopping[113],3,8,[1,0,0])
        my_model.set_hop(hopping[114],0,7,[-1,0,0])
        my_model.set_hop(hopping[115],2,5,[-1,0,0])
        my_model.set_hop(hopping[116],0,7,[0,1,0])
        my_model.set_hop(hopping[117],2,5,[0,1,0])
        my_model.set_hop(hopping[118],1,6,[0,-1,0])
        my_model.set_hop(hopping[119],3,4,[0,-1,0])
        my_model.set_hop(hopping[120],1,6,[1,0,0])
        my_model.set_hop(hopping[121],3,4,[1,0,0])
        my_model.set_hop(hopping[122],0,11,[-1,0,0])
        my_model.set_hop(hopping[123],2,9,[-1,0,0])
        my_model.set_hop(hopping[124],1,10,[0,-1,0])
        my_model.set_hop(hopping[125],3,8,[0,-1,0])
        my_model.set_hop(hopping[126],0,15,[-1,0,0])
        my_model.set_hop(hopping[127],0,15,[0,1,0])
        my_model.set_hop(hopping[128],2,13,[-1,0,0])
        my_model.set_hop(hopping[129],2,13,[0,1,0])
        my_model.set_hop(hopping[130],1,12,[1,-1,0])
        my_model.set_hop(hopping[131],1,4,[1,-1,0])
        my_model.set_hop(hopping[132],0,9,[-1,1,0])
        my_model.set_hop(hopping[133],1,8,[1,-1,0])
        my_model.set_hop(hopping[134],0,5,[-1,1,0])
        my_model.set_hop(hopping[135],0,13,[-1,1,0])
        my_model.set_hop(hopping[136],1,4,[1,-1,-1])
        my_model.set_hop(hopping[137],0,13,[-1,1,1])
        my_model.set_hop(hopping[138],0,9,[-1,1,1])
        my_model.set_hop(hopping[139],1,8,[1,-1,-1])
        my_model.set_hop(hopping[140],1,12,[1,-1,-1])
        my_model.set_hop(hopping[141],0,5,[-1,1,1])
        my_model.set_hop(hopping[142],4,8,[-1,-1,0])
        my_model.set_hop(hopping[143],4,12,[-1,-1,0])
        my_model.set_hop(hopping[144],5,9,[-1,-1,0])
        my_model.set_hop(hopping[145],5,13,[-1,-1,0])
        my_model.set_hop(hopping[146],6,10,[-1,-1,0])
        my_model.set_hop(hopping[147],6,14,[-1,-1,0])
        my_model.set_hop(hopping[148],7,11,[-1,-1,0])
        my_model.set_hop(hopping[149],7,15,[-1,-1,0])
        my_model.set_hop(hopping[150],8,12,[-1,-1,0])
        my_model.set_hop(hopping[151],9,13,[-1,-1,0])
        my_model.set_hop(hopping[152],10,14,[-1,-1,0])
        my_model.set_hop(hopping[153],11,15,[-1,-1,0])
        my_model.set_hop(hopping[154],4,4,[-1,-1,0])
        my_model.set_hop(hopping[155],5,5,[-1,-1,0])
        my_model.set_hop(hopping[156],6,6,[-1,-1,0])
        my_model.set_hop(hopping[157],7,7,[-1,-1,0])
        my_model.set_hop(hopping[158],12,12,[-1,-1,0])
        my_model.set_hop(hopping[159],13,13,[-1,-1,0])
        my_model.set_hop(hopping[160],14,14,[-1,-1,0])
        my_model.set_hop(hopping[161],15,15,[-1,-1,0])
        my_model.set_hop(hopping[162],8,8,[-1,-1,0])
        my_model.set_hop(hopping[163],9,9,[-1,-1,0])
        my_model.set_hop(hopping[164],10,10,[-1,-1,0])
        my_model.set_hop(hopping[165],11,11,[-1,-1,0])
        my_model.set_hop(hopping[166],11,14,[0,-1,0])
        my_model.set_hop(hopping[167],10,15,[0,1,0])
        my_model.set_hop(hopping[168],7,10,[0,-1,0])
        my_model.set_hop(hopping[169],6,11,[0,1,0])
        my_model.set_hop(hopping[170],10,11,[-1,0,0])
        my_model.set_hop(hopping[171],10,11,[0,1,0])
        my_model.set_hop(hopping[172],6,11,[-1,0,0])
        my_model.set_hop(hopping[173],7,10,[1,0,0])
        my_model.set_hop(hopping[174],6,7,[-1,0,0])
        my_model.set_hop(hopping[175],6,7,[0,1,0])
        my_model.set_hop(hopping[176],10,15,[-1,0,0])
        my_model.set_hop(hopping[177],11,14,[1,0,0])
        my_model.set_hop(hopping[178],6,15,[-1,0,0])
        my_model.set_hop(hopping[179],7,14,[0,-1,0])
        my_model.set_hop(hopping[180],6,15,[0,1,0])
        my_model.set_hop(hopping[181],7,14,[1,0,0])
        my_model.set_hop(hopping[182],14,15,[-1,0,0])
        my_model.set_hop(hopping[183],14,15,[0,1,0])
        my_model.set_hop(hopping[184],4,14,[0,0,0])
        my_model.set_hop(hopping[185],6,12,[0,0,0])
        my_model.set_hop(hopping[186],4,10,[0,0,0])
        my_model.set_hop(hopping[187],6,8,[0,0,0])
        my_model.set_hop(hopping[188],8,14,[0,0,0])
        my_model.set_hop(hopping[189],10,12,[0,0,0])
        my_model.set_hop(hopping[190],8,10,[0,0,0])
        my_model.set_hop(hopping[191],12,14,[0,0,0])
        my_model.set_hop(hopping[192],4,6,[0,0,0])
        my_model.set_hop(hopping[193],5,15,[0,0,0])
        my_model.set_hop(hopping[194],7,13,[0,0,0])
        my_model.set_hop(hopping[195],5,11,[0,0,0])
        my_model.set_hop(hopping[196],7,9,[0,0,0])
        my_model.set_hop(hopping[197],9,15,[0,0,0])
        my_model.set_hop(hopping[198],11,13,[0,0,0])
        my_model.set_hop(hopping[199],9,11,[0,0,0])
        my_model.set_hop(hopping[200],13,15,[0,0,0])
        my_model.set_hop(hopping[201],5,7,[0,0,0])
        my_model.set_hop(hopping[202],10,13,[-1,0,1])
        my_model.set_hop(hopping[203],9,14,[1,0,-1])
        my_model.set_hop(hopping[204],6,9,[-1,0,1])
        my_model.set_hop(hopping[205],5,10,[1,0,-1])
        my_model.set_hop(hopping[206],5,6,[0,-1,-1])
        my_model.set_hop(hopping[207],5,6,[1,0,-1])
        my_model.set_hop(hopping[208],5,10,[0,-1,-1])
        my_model.set_hop(hopping[209],6,9,[0,1,1])
        my_model.set_hop(hopping[210],6,13,[-1,0,1])
        my_model.set_hop(hopping[211],5,14,[0,-1,-1])
        my_model.set_hop(hopping[212],6,13,[0,1,1])
        my_model.set_hop(hopping[213],5,14,[1,0,-1])
        my_model.set_hop(hopping[214],9,10,[0,-1,-1])
        my_model.set_hop(hopping[215],9,10,[1,0,-1])
        my_model.set_hop(hopping[216],9,14,[0,-1,-1])
        my_model.set_hop(hopping[217],10,13,[0,1,1])
        my_model.set_hop(hopping[218],13,14,[0,-1,-1])
        my_model.set_hop(hopping[219],13,14,[1,0,-1])
        my_model.set_hop(hopping[220],8,15,[-1,0,1])
        my_model.set_hop(hopping[221],11,12,[1,0,-1])
        my_model.set_hop(hopping[222],4,11,[-1,0,1])
        my_model.set_hop(hopping[223],7,8,[1,0,-1])
        my_model.set_hop(hopping[224],4,7,[-1,0,1])
        my_model.set_hop(hopping[225],4,7,[0,1,1])
        my_model.set_hop(hopping[226],7,8,[0,-1,-1])
        my_model.set_hop(hopping[227],4,11,[0,1,1])
        my_model.set_hop(hopping[228],4,15,[-1,0,1])
        my_model.set_hop(hopping[229],7,12,[0,-1,-1])
        my_model.set_hop(hopping[230],4,15,[0,1,1])
        my_model.set_hop(hopping[231],7,12,[1,0,-1])
        my_model.set_hop(hopping[232],8,11,[-1,0,1])
        my_model.set_hop(hopping[233],8,11,[0,1,1])
        my_model.set_hop(hopping[234],11,12,[0,-1,-1])
        my_model.set_hop(hopping[235],8,15,[0,1,1])
        my_model.set_hop(hopping[236],12,15,[-1,0,1])
        my_model.set_hop(hopping[237],12,15,[0,1,1])
        my_model.set_hop(hopping[238],5,11,[0,-1,0])
        my_model.set_hop(hopping[239],7,9,[0,1,0])
        my_model.set_hop(hopping[240],9,15,[0,-1,0])
        my_model.set_hop(hopping[241],11,13,[0,1,0])
        my_model.set_hop(hopping[242],13,15,[0,-1,0])
        my_model.set_hop(hopping[243],13,15,[1,0,0])
        my_model.set_hop(hopping[244],11,13,[-1,0,0])
        my_model.set_hop(hopping[245],9,15,[1,0,0])
        my_model.set_hop(hopping[246],7,13,[-1,0,0])
        my_model.set_hop(hopping[247],5,15,[0,-1,0])
        my_model.set_hop(hopping[248],7,13,[0,1,0])
        my_model.set_hop(hopping[249],5,15,[1,0,0])
        my_model.set_hop(hopping[250],9,11,[0,-1,0])
        my_model.set_hop(hopping[251],9,11,[1,0,0])
        my_model.set_hop(hopping[252],7,9,[-1,0,0])
        my_model.set_hop(hopping[253],5,11,[1,0,0])
        my_model.set_hop(hopping[254],5,7,[0,-1,0])
        my_model.set_hop(hopping[255],5,7,[1,0,0])
        my_model.set_hop(hopping[256],6,8,[0,-1,0])
        my_model.set_hop(hopping[257],4,10,[0,1,0])
        my_model.set_hop(hopping[258],10,12,[0,-1,0])
        my_model.set_hop(hopping[259],8,14,[0,1,0])
        my_model.set_hop(hopping[260],12,14,[-1,0,0])
        my_model.set_hop(hopping[261],12,14,[0,1,0])
        my_model.set_hop(hopping[262],8,14,[-1,0,0])
        my_model.set_hop(hopping[263],10,12,[1,0,0])
        my_model.set_hop(hopping[264],4,14,[-1,0,0])
        my_model.set_hop(hopping[265],6,12,[0,-1,0])
        my_model.set_hop(hopping[266],4,14,[0,1,0])
        my_model.set_hop(hopping[267],6,12,[1,0,0])
        my_model.set_hop(hopping[268],8,10,[-1,0,0])
        my_model.set_hop(hopping[269],8,10,[0,1,0])
        my_model.set_hop(hopping[270],4,10,[-1,0,0])
        my_model.set_hop(hopping[271],6,8,[1,0,0])
        my_model.set_hop(hopping[272],4,6,[-1,0,0])
        my_model.set_hop(hopping[273],4,6,[0,1,0])
        my_model.set_hop(hopping[274],9,14,[0,-1,0])
        my_model.set_hop(hopping[275],11,12,[0,-1,0])
        my_model.set_hop(hopping[276],8,15,[0,1,0])
        my_model.set_hop(hopping[277],10,13,[0,1,0])
        my_model.set_hop(hopping[278],4,15,[-1,0,0])
        my_model.set_hop(hopping[279],6,13,[-1,0,0])
        my_model.set_hop(hopping[280],5,14,[0,-1,0])
        my_model.set_hop(hopping[281],7,12,[0,-1,0])
        my_model.set_hop(hopping[282],4,15,[0,1,0])
        my_model.set_hop(hopping[283],6,13,[0,1,0])
        my_model.set_hop(hopping[284],5,14,[1,0,0])
        my_model.set_hop(hopping[285],7,12,[1,0,0])
        my_model.set_hop(hopping[286],4,11,[-1,0,0])
        my_model.set_hop(hopping[287],6,9,[-1,0,0])
        my_model.set_hop(hopping[288],5,10,[1,0,0])
        my_model.set_hop(hopping[289],7,8,[1,0,0])
        my_model.set_hop(hopping[290],4,7,[-1,0,0])
        my_model.set_hop(hopping[291],5,6,[0,-1,0])
        my_model.set_hop(hopping[292],4,7,[0,1,0])
        my_model.set_hop(hopping[293],5,6,[1,0,0])
        my_model.set_hop(hopping[294],5,10,[0,-1,0])
        my_model.set_hop(hopping[295],7,8,[0,-1,0])
        my_model.set_hop(hopping[296],4,11,[0,1,0])
        my_model.set_hop(hopping[297],6,9,[0,1,0])
        my_model.set_hop(hopping[298],8,11,[-1,0,0])
        my_model.set_hop(hopping[299],9,10,[0,-1,0])
        my_model.set_hop(hopping[300],8,11,[0,1,0])
        my_model.set_hop(hopping[301],9,10,[1,0,0])
        my_model.set_hop(hopping[302],8,15,[-1,0,0])
        my_model.set_hop(hopping[303],10,13,[-1,0,0])
        my_model.set_hop(hopping[304],9,14,[1,0,0])
        my_model.set_hop(hopping[305],11,12,[1,0,0])
        my_model.set_hop(hopping[306],12,15,[-1,0,0])
        my_model.set_hop(hopping[307],12,15,[0,1,0])
        my_model.set_hop(hopping[308],13,14,[0,-1,0])
        my_model.set_hop(hopping[309],13,14,[1,0,0])
        my_model.set_hop(hopping[310],4,9,[-1,1,0])
        my_model.set_hop(hopping[311],8,13,[-1,1,0])
        my_model.set_hop(hopping[312],5,8,[1,-1,0])
        my_model.set_hop(hopping[313],9,12,[1,-1,0])
        my_model.set_hop(hopping[314],8,9,[-1,1,0])
        my_model.set_hop(hopping[315],4,5,[-1,1,0])
        my_model.set_hop(hopping[316],4,13,[-1,1,0])
        my_model.set_hop(hopping[317],5,12,[1,-1,0])
        my_model.set_hop(hopping[318],12,13,[-1,1,0])
        my_model.set_hop(hopping[319],4,13,[-1,1,1])
        my_model.set_hop(hopping[320],5,12,[1,-1,-1])
        my_model.set_hop(hopping[321],4,9,[-1,1,1])
        my_model.set_hop(hopping[322],8,13,[-1,1,1])
        my_model.set_hop(hopping[323],5,8,[1,-1,-1])
        my_model.set_hop(hopping[324],9,12,[1,-1,-1])
        my_model.set_hop(hopping[325],8,9,[-1,1,1])
        my_model.set_hop(hopping[326],12,13,[-1,1,1])
        my_model.set_hop(hopping[327],4,5,[-1,1,1])
        my_model.set_hop(hopping[328],8,16,[-1,-1,0])
        my_model.set_hop(hopping[329],9,17,[-1,-1,0])
        my_model.set_hop(hopping[330],10,18,[-1,-1,0])
        my_model.set_hop(hopping[331],11,19,[-1,-1,0])
        my_model.set_hop(hopping[332],4,16,[-1,-1,0])
        my_model.set_hop(hopping[333],5,17,[-1,-1,0])
        my_model.set_hop(hopping[334],6,18,[-1,-1,0])
        my_model.set_hop(hopping[335],7,19,[-1,-1,0])
        my_model.set_hop(hopping[336],12,16,[-1,-1,0])
        my_model.set_hop(hopping[337],13,17,[-1,-1,0])
        my_model.set_hop(hopping[338],14,18,[-1,-1,0])
        my_model.set_hop(hopping[339],15,19,[-1,-1,0])
        my_model.set_hop(hopping[340],14,19,[-1,0,0])
        my_model.set_hop(hopping[341],14,19,[0,1,0])
        my_model.set_hop(hopping[342],6,19,[-1,0,0])
        my_model.set_hop(hopping[343],6,19,[0,1,0])
        my_model.set_hop(hopping[344],10,19,[-1,0,0])
        my_model.set_hop(hopping[345],11,18,[0,-1,0])
        my_model.set_hop(hopping[346],10,19,[0,1,0])
        my_model.set_hop(hopping[347],11,18,[1,0,0])
        my_model.set_hop(hopping[348],7,18,[0,-1,0])
        my_model.set_hop(hopping[349],7,18,[1,0,0])
        my_model.set_hop(hopping[350],15,18,[0,-1,0])
        my_model.set_hop(hopping[351],15,18,[1,0,0])
        my_model.set_hop(hopping[352],6,16,[0,0,0])
        my_model.set_hop(hopping[353],12,18,[0,0,0])
        my_model.set_hop(hopping[354],8,18,[0,0,0])
        my_model.set_hop(hopping[355],10,16,[0,0,0])
        my_model.set_hop(hopping[356],14,16,[0,0,0])
        my_model.set_hop(hopping[357],4,18,[0,0,0])
        my_model.set_hop(hopping[358],5,19,[0,0,0])
        my_model.set_hop(hopping[359],15,17,[0,0,0])
        my_model.set_hop(hopping[360],9,19,[0,0,0])
        my_model.set_hop(hopping[361],11,17,[0,0,0])
        my_model.set_hop(hopping[362],13,19,[0,0,0])
        my_model.set_hop(hopping[363],7,17,[0,0,0])
        my_model.set_hop(hopping[364],13,18,[0,-1,-1])
        my_model.set_hop(hopping[365],13,18,[1,0,-1])
        my_model.set_hop(hopping[366],10,17,[-1,0,1])
        my_model.set_hop(hopping[367],9,18,[0,-1,-1])
        my_model.set_hop(hopping[368],5,18,[0,-1,-1])
        my_model.set_hop(hopping[369],5,18,[1,0,-1])
        my_model.set_hop(hopping[370],6,17,[-1,0,1])
        my_model.set_hop(hopping[371],6,17,[0,1,1])
        my_model.set_hop(hopping[372],10,17,[0,1,1])
        my_model.set_hop(hopping[373],9,18,[1,0,-1])
        my_model.set_hop(hopping[374],14,17,[-1,0,1])
        my_model.set_hop(hopping[375],14,17,[0,1,1])
        my_model.set_hop(hopping[376],15,16,[0,-1,-1])
        my_model.set_hop(hopping[377],15,16,[1,0,-1])
        my_model.set_hop(hopping[378],8,19,[-1,0,1])
        my_model.set_hop(hopping[379],11,16,[0,-1,-1])
        my_model.set_hop(hopping[380],7,16,[0,-1,-1])
        my_model.set_hop(hopping[381],7,16,[1,0,-1])
        my_model.set_hop(hopping[382],4,19,[-1,0,1])
        my_model.set_hop(hopping[383],4,19,[0,1,1])
        my_model.set_hop(hopping[384],8,19,[0,1,1])
        my_model.set_hop(hopping[385],11,16,[1,0,-1])
        my_model.set_hop(hopping[386],12,19,[-1,0,1])
        my_model.set_hop(hopping[387],12,19,[0,1,1])
        my_model.set_hop(hopping[388],7,17,[-1,0,0])
        my_model.set_hop(hopping[389],7,17,[0,1,0])
        my_model.set_hop(hopping[390],11,17,[-1,0,0])
        my_model.set_hop(hopping[391],9,19,[0,-1,0])
        my_model.set_hop(hopping[392],15,17,[-1,0,0])
        my_model.set_hop(hopping[393],15,17,[0,1,0])
        my_model.set_hop(hopping[394],13,19,[0,-1,0])
        my_model.set_hop(hopping[395],13,19,[1,0,0])
        my_model.set_hop(hopping[396],11,17,[0,1,0])
        my_model.set_hop(hopping[397],9,19,[1,0,0])
        my_model.set_hop(hopping[398],5,19,[0,-1,0])
        my_model.set_hop(hopping[399],5,19,[1,0,0])
        my_model.set_hop(hopping[400],4,18,[-1,0,0])
        my_model.set_hop(hopping[401],4,18,[0,1,0])
        my_model.set_hop(hopping[402],8,18,[-1,0,0])
        my_model.set_hop(hopping[403],10,16,[0,-1,0])
        my_model.set_hop(hopping[404],12,18,[-1,0,0])
        my_model.set_hop(hopping[405],12,18,[0,1,0])
        my_model.set_hop(hopping[406],14,16,[0,-1,0])
        my_model.set_hop(hopping[407],14,16,[1,0,0])
        my_model.set_hop(hopping[408],8,18,[0,1,0])
        my_model.set_hop(hopping[409],10,16,[1,0,0])
        my_model.set_hop(hopping[410],6,16,[0,-1,0])
        my_model.set_hop(hopping[411],6,16,[1,0,0])
        my_model.set_hop(hopping[412],14,17,[-1,0,0])
        my_model.set_hop(hopping[413],14,17,[0,1,0])
        my_model.set_hop(hopping[414],12,19,[-1,0,0])
        my_model.set_hop(hopping[415],12,19,[0,1,0])
        my_model.set_hop(hopping[416],8,19,[-1,0,0])
        my_model.set_hop(hopping[417],10,17,[-1,0,0])
        my_model.set_hop(hopping[418],9,18,[0,-1,0])
        my_model.set_hop(hopping[419],11,16,[0,-1,0])
        my_model.set_hop(hopping[420],5,18,[0,-1,0])
        my_model.set_hop(hopping[421],7,16,[0,-1,0])
        my_model.set_hop(hopping[422],5,18,[1,0,0])
        my_model.set_hop(hopping[423],7,16,[1,0,0])
        my_model.set_hop(hopping[424],4,19,[-1,0,0])
        my_model.set_hop(hopping[425],6,17,[-1,0,0])
        my_model.set_hop(hopping[426],4,19,[0,1,0])
        my_model.set_hop(hopping[427],6,17,[0,1,0])
        my_model.set_hop(hopping[428],8,19,[0,1,0])
        my_model.set_hop(hopping[429],10,17,[0,1,0])
        my_model.set_hop(hopping[430],9,18,[1,0,0])
        my_model.set_hop(hopping[431],11,16,[1,0,0])
        my_model.set_hop(hopping[432],15,16,[0,-1,0])
        my_model.set_hop(hopping[433],15,16,[1,0,0])
        my_model.set_hop(hopping[434],13,18,[0,-1,0])
        my_model.set_hop(hopping[435],13,18,[1,0,0])
        my_model.set_hop(hopping[436],12,17,[-1,1,0])
        my_model.set_hop(hopping[437],4,17,[-1,1,0])
        my_model.set_hop(hopping[438],8,17,[-1,1,0])
        my_model.set_hop(hopping[439],9,16,[1,-1,0])
        my_model.set_hop(hopping[440],5,16,[1,-1,0])
        my_model.set_hop(hopping[441],13,16,[1,-1,0])
        my_model.set_hop(hopping[442],4,17,[-1,1,1])
        my_model.set_hop(hopping[443],13,16,[1,-1,-1])
        my_model.set_hop(hopping[444],8,17,[-1,1,1])
        my_model.set_hop(hopping[445],9,16,[1,-1,-1])
        my_model.set_hop(hopping[446],12,17,[-1,1,1])
        my_model.set_hop(hopping[447],5,16,[1,-1,-1])
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