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
        lat = [[2.4912500000, 1.4383238581, 4.4776666667], 
               [-2.4912500000, 1.4383238581, 4.4776666667], 
               [0.0000000000, -2.8766477162, 4.4776666667]]
        # define coordinates of orbitals
        # the four groups are s, px, py, pz in order
        orb = [[0.1446000000, 0.1446000000, 0.1446000000], [0.6446000000, 0.6446000000, -0.3554000000],
                [0.8554000000,-0.1446000000,-0.1446000000], [0.3554000000,0.3554000000,-0.6446000000], 
                [0.1446000000, 0.1446000000, 0.1446000000], [0.6446000000, 0.6446000000, -0.3554000000],
                [0.8554000000,-0.1446000000,-0.1446000000], [0.3554000000,0.3554000000,-0.6446000000],
                [0.1446000000, 0.1446000000, 0.1446000000], [0.6446000000, 0.6446000000, -0.3554000000],
                [0.8554000000,-0.1446000000,-0.1446000000], [0.3554000000,0.3554000000,-0.6446000000],
                [0.1446000000, 0.1446000000, 0.1446000000], [0.6446000000, 0.6446000000, -0.3554000000],
                [0.8554000000,-0.1446000000,-0.1446000000], [0.3554000000,0.3554000000,-0.6446000000],
                [0.1446000000, 0.1446000000, 0.1446000000], [0.6446000000, 0.6446000000, -0.3554000000],
                [0.8554000000,-0.1446000000,-0.1446000000], [0.3554000000,0.3554000000,-0.6446000000]]

        # make three-dimensional tight-binding model
        my_model = tb_model(3, 3, lat, orb)

        # set on-site energies
        my_model.set_onsite(onsite)
        # set hoppings (one for each connected pair of orbitals)
        # (amplitude, i, j, [lattice vector to cell containing j])
        #region
        # ss
        my_model.set_hop(hopping[0],0,3,[0,0,1])
        my_model.set_hop(hopping[1],1,2,[0,1,0])  
        my_model.set_hop(hopping[2],0,2,[-1,0,1]) 
        my_model.set_hop(hopping[3],1,3,[0,0,1])  
        my_model.set_hop(hopping[4],0,2,[-1,1,0]) 
        my_model.set_hop(hopping[5],1,3,[0,1,0])  
        my_model.set_hop(hopping[6],1,3,[1,0,0])  
        my_model.set_hop(hopping[7],0,2,[0,0,0])  
        my_model.set_hop(hopping[8],0,3,[0,0,0])  
        my_model.set_hop(hopping[9],1,2,[0,1,-1]) 
        my_model.set_hop(hopping[10],1,2,[-1,1,0])
        my_model.set_hop(hopping[11],1,2,[0,0,0]) 
        my_model.set_hop(hopping[12],0,3,[-1,0,1])
        my_model.set_hop(hopping[13],0,3,[0,-1,1])
        my_model.set_hop(hopping[14],0,1,[0,0,0])
        my_model.set_hop(hopping[15],0,1,[-1,-1,1])
        my_model.set_hop(hopping[16],2,3,[0,-1,1])
        my_model.set_hop(hopping[17],2,3,[1,0,0])
        my_model.set_hop(hopping[18],0,1,[-1,0,1])
        my_model.set_hop(hopping[19],0,1,[0,-1,1])
        my_model.set_hop(hopping[20],2,3,[0,0,0])
        my_model.set_hop(hopping[21],2,3,[1,-1,0])
        my_model.set_hop(hopping[22],0,1,[-1,0,0])
        my_model.set_hop(hopping[23],0,1,[0,-1,0])
        my_model.set_hop(hopping[24],2,3,[0,0,1])
        my_model.set_hop(hopping[25],2,3,[1,-1,1])
        my_model.set_hop(hopping[26],0,2,[-1,0,0])
        my_model.set_hop(hopping[27],1,3,[0,0,0])
        my_model.set_hop(hopping[28],0,0,[-1,0,1])
        my_model.set_hop(hopping[29],1,1,[-1,0,1])
        my_model.set_hop(hopping[30],2,2,[-1,0,1])
        my_model.set_hop(hopping[31],3,3,[-1,0,1])
        my_model.set_hop(hopping[32],0,0,[0,-1,1])
        my_model.set_hop(hopping[33],1,1,[0,-1,1])
        my_model.set_hop(hopping[34],2,2,[0,-1,1])
        my_model.set_hop(hopping[35],3,3,[0,-1,1])
        my_model.set_hop(hopping[36],0,0,[-1,1,0])
        my_model.set_hop(hopping[37],1,1,[-1,1,0])
        my_model.set_hop(hopping[38],2,2,[-1,1,0])
        my_model.set_hop(hopping[39],3,3,[-1,1,0])
        my_model.set_hop(hopping[40],0,15,[0,0,1])
        my_model.set_hop(hopping[41],3,8,[0,0,-1])
        my_model.set_hop(hopping[42],3,4,[0,0,-1])
        my_model.set_hop(hopping[43],0,7,[0,0,1])
        my_model.set_hop(hopping[44],0,11,[0,0,1])
        my_model.set_hop(hopping[45],3,12,[0,0,-1])
        my_model.set_hop(hopping[46],1,14,[0,1,0])
        my_model.set_hop(hopping[47],1,10,[0,1,0])
        my_model.set_hop(hopping[48],2,5,[0,-1,0])
        my_model.set_hop(hopping[49],1,6,[0,1,0])
        my_model.set_hop(hopping[50],2,9,[0,-1,0])
        my_model.set_hop(hopping[51],2,13,[0,-1,0])
        my_model.set_hop(hopping[52],3,9,[0,0,-1])
        my_model.set_hop(hopping[53],2,8,[1,0,-1])
        my_model.set_hop(hopping[54],1,15,[0,0,1])
        my_model.set_hop(hopping[55],0,14,[-1,0,1])
        my_model.set_hop(hopping[56],0,6,[-1,0,1])
        my_model.set_hop(hopping[57],3,5,[0,0,-1])
        my_model.set_hop(hopping[58],1,7,[0,0,1])
        my_model.set_hop(hopping[59],2,4,[1,0,-1])
        my_model.set_hop(hopping[60],2,12,[1,0,-1])
        my_model.set_hop(hopping[61],3,13,[0,0,-1])
        my_model.set_hop(hopping[62],0,10,[-1,0,1])
        my_model.set_hop(hopping[63],1,11,[0,0,1])
        my_model.set_hop(hopping[64],2,4,[1,-1,0])
        my_model.set_hop(hopping[65],0,10,[-1,1,0])
        my_model.set_hop(hopping[66],0,14,[-1,1,0])
        my_model.set_hop(hopping[67],2,12,[1,-1,0])
        my_model.set_hop(hopping[68],2,8,[1,-1,0])
        my_model.set_hop(hopping[69],0,6,[-1,1,0])
        my_model.set_hop(hopping[70],3,5,[0,-1,0])
        my_model.set_hop(hopping[71],1,7,[1,0,0])
        my_model.set_hop(hopping[72],1,11,[0,1,0])
        my_model.set_hop(hopping[73],1,11,[1,0,0])
        my_model.set_hop(hopping[74],1,15,[0,1,0])
        my_model.set_hop(hopping[75],1,15,[1,0,0])
        my_model.set_hop(hopping[76],3,13,[-1,0,0])
        my_model.set_hop(hopping[77],3,13,[0,-1,0])
        my_model.set_hop(hopping[78],3,9,[-1,0,0])
        my_model.set_hop(hopping[79],3,9,[0,-1,0])
        my_model.set_hop(hopping[80],3,5,[-1,0,0])
        my_model.set_hop(hopping[81],1,7,[0,1,0])
        my_model.set_hop(hopping[82],0,6,[0,0,0])
        my_model.set_hop(hopping[83],0,10,[0,0,0])
        my_model.set_hop(hopping[84],0,14,[0,0,0])
        my_model.set_hop(hopping[85],2,12,[0,0,0])
        my_model.set_hop(hopping[86],2,8,[0,0,0])
        my_model.set_hop(hopping[87],2,4,[0,0,0])
        my_model.set_hop(hopping[88],0,11,[0,0,0])
        my_model.set_hop(hopping[89],3,12,[0,0,0])
        my_model.set_hop(hopping[90],0,7,[0,0,0])
        my_model.set_hop(hopping[91],3,4,[0,0,0])
        my_model.set_hop(hopping[92],0,15,[0,0,0])
        my_model.set_hop(hopping[93],3,8,[0,0,0])
        my_model.set_hop(hopping[94],1,10,[0,1,-1])
        my_model.set_hop(hopping[95],2,13,[0,-1,1])
        my_model.set_hop(hopping[96],2,5,[0,-1,1])
        my_model.set_hop(hopping[97],1,6,[0,1,-1])
        my_model.set_hop(hopping[98],1,14,[0,1,-1])
        my_model.set_hop(hopping[99],2,9,[0,-1,1])
        my_model.set_hop(hopping[100],1,6,[0,0,0])
        my_model.set_hop(hopping[101],2,5,[1,-1,0])
        my_model.set_hop(hopping[102],2,13,[0,0,0])
        my_model.set_hop(hopping[103],2,13,[1,-1,0])
        my_model.set_hop(hopping[104],2,9,[0,0,0])
        my_model.set_hop(hopping[105],2,9,[1,-1,0])
        my_model.set_hop(hopping[106],1,10,[-1,1,0])
        my_model.set_hop(hopping[107],1,10,[0,0,0])
        my_model.set_hop(hopping[108],1,14,[-1,1,0])
        my_model.set_hop(hopping[109],1,14,[0,0,0])
        my_model.set_hop(hopping[110],1,6,[-1,1,0])
        my_model.set_hop(hopping[111],2,5,[0,0,0])
        my_model.set_hop(hopping[112],0,7,[0,-1,1])
        my_model.set_hop(hopping[113],3,4,[1,0,-1])
        my_model.set_hop(hopping[114],3,12,[0,1,-1])
        my_model.set_hop(hopping[115],3,12,[1,0,-1])
        my_model.set_hop(hopping[116],3,8,[0,1,-1])
        my_model.set_hop(hopping[117],3,8,[1,0,-1])
        my_model.set_hop(hopping[118],0,11,[-1,0,1])
        my_model.set_hop(hopping[119],0,11,[0,-1,1])
        my_model.set_hop(hopping[120],0,15,[-1,0,1])
        my_model.set_hop(hopping[121],0,15,[0,-1,1])
        my_model.set_hop(hopping[122],0,7,[-1,0,1])
        my_model.set_hop(hopping[123],3,4,[0,1,-1])
        my_model.set_hop(hopping[124],0,9,[0,0,0])
        my_model.set_hop(hopping[125],0,13,[0,0,0])
        my_model.set_hop(hopping[126],0,5,[0,0,0])
        my_model.set_hop(hopping[127],1,4,[0,0,0])
        my_model.set_hop(hopping[128],1,12,[0,0,0])
        my_model.set_hop(hopping[129],1,8,[0,0,0])
        my_model.set_hop(hopping[130],3,10,[0,1,-1])
        my_model.set_hop(hopping[131],2,11,[1,0,0])
        my_model.set_hop(hopping[132],1,8,[1,1,-1])
        my_model.set_hop(hopping[133],2,15,[1,0,0])
        my_model.set_hop(hopping[134],3,14,[0,1,-1])
        my_model.set_hop(hopping[135],1,12,[1,1,-1])
        my_model.set_hop(hopping[136],0,5,[-1,-1,1])
        my_model.set_hop(hopping[137],3,6,[-1,0,0])
        my_model.set_hop(hopping[138],2,7,[0,-1,1])
        my_model.set_hop(hopping[139],3,6,[0,1,-1])
        my_model.set_hop(hopping[140],2,7,[1,0,0])
        my_model.set_hop(hopping[141],1,4,[1,1,-1])
        my_model.set_hop(hopping[142],0,13,[-1,-1,1])
        my_model.set_hop(hopping[143],2,15,[0,-1,1])
        my_model.set_hop(hopping[144],3,14,[-1,0,0])
        my_model.set_hop(hopping[145],0,9,[-1,-1,1])
        my_model.set_hop(hopping[146],3,10,[-1,0,0])
        my_model.set_hop(hopping[147],2,11,[0,-1,1])
        my_model.set_hop(hopping[148],0,5,[0,-1,1])
        my_model.set_hop(hopping[149],3,6,[0,0,0])
        my_model.set_hop(hopping[150],2,7,[1,-1,0])
        my_model.set_hop(hopping[151],1,4,[1,0,-1])
        my_model.set_hop(hopping[152],0,13,[-1,0,1])
        my_model.set_hop(hopping[153],3,14,[-1,1,0])
        my_model.set_hop(hopping[154],0,13,[0,-1,1])
        my_model.set_hop(hopping[155],3,14,[0,0,0])
        my_model.set_hop(hopping[156],1,8,[0,1,-1])
        my_model.set_hop(hopping[157],1,8,[1,0,-1])
        my_model.set_hop(hopping[158],2,11,[0,0,0])
        my_model.set_hop(hopping[159],2,11,[1,-1,0])
        my_model.set_hop(hopping[160],3,10,[-1,1,0])
        my_model.set_hop(hopping[161],3,10,[0,0,0])
        my_model.set_hop(hopping[162],0,9,[-1,0,1])
        my_model.set_hop(hopping[163],0,9,[0,-1,1])
        my_model.set_hop(hopping[164],2,15,[0,0,0])
        my_model.set_hop(hopping[165],1,12,[0,1,-1])
        my_model.set_hop(hopping[166],2,15,[1,-1,0])
        my_model.set_hop(hopping[167],1,12,[1,0,-1])
        my_model.set_hop(hopping[168],0,5,[-1,0,1])
        my_model.set_hop(hopping[169],3,6,[-1,1,0])
        my_model.set_hop(hopping[170],2,7,[0,0,0])
        my_model.set_hop(hopping[171],1,4,[0,1,-1])
        my_model.set_hop(hopping[172],0,5,[0,-1,0])
        my_model.set_hop(hopping[173],3,6,[0,0,-1])
        my_model.set_hop(hopping[174],2,7,[1,-1,1])
        my_model.set_hop(hopping[175],1,4,[1,0,0])
        my_model.set_hop(hopping[176],1,12,[0,1,0])
        my_model.set_hop(hopping[177],1,12,[1,0,0])
        my_model.set_hop(hopping[178],2,15,[0,0,1])
        my_model.set_hop(hopping[179],2,15,[1,-1,1])
        my_model.set_hop(hopping[180],0,9,[-1,0,0])
        my_model.set_hop(hopping[181],3,10,[-1,1,-1])
        my_model.set_hop(hopping[182],0,9,[0,-1,0])
        my_model.set_hop(hopping[183],3,10,[0,0,-1])
        my_model.set_hop(hopping[184],2,11,[0,0,1])
        my_model.set_hop(hopping[185],1,8,[0,1,0])
        my_model.set_hop(hopping[186],2,11,[1,-1,1])
        my_model.set_hop(hopping[187],1,8,[1,0,0])
        my_model.set_hop(hopping[188],3,14,[-1,1,-1])
        my_model.set_hop(hopping[189],3,14,[0,0,-1])
        my_model.set_hop(hopping[190],0,13,[-1,0,0])
        my_model.set_hop(hopping[191],0,13,[0,-1,0])
        my_model.set_hop(hopping[192],0,5,[-1,0,0])
        my_model.set_hop(hopping[193],3,6,[-1,1,-1])
        my_model.set_hop(hopping[194],2,7,[0,0,1])
        my_model.set_hop(hopping[195],1,4,[0,1,0])
        my_model.set_hop(hopping[196],3,13,[0,0,0])
        my_model.set_hop(hopping[197],2,12,[1,0,0])
        my_model.set_hop(hopping[198],0,6,[-1,0,0])
        my_model.set_hop(hopping[199],0,10,[-1,0,0])
        my_model.set_hop(hopping[200],1,7,[0,0,0])
        my_model.set_hop(hopping[201],1,11,[0,0,0])
        my_model.set_hop(hopping[202],3,5,[0,0,0])
        my_model.set_hop(hopping[203],3,9,[0,0,0])
        my_model.set_hop(hopping[204],2,4,[1,0,0])
        my_model.set_hop(hopping[205],2,8,[1,0,0])
        my_model.set_hop(hopping[206],0,14,[-1,0,0])
        my_model.set_hop(hopping[207],1,15,[0,0,0])
        my_model.set_hop(hopping[208],0,4,[0,-1,1])
        my_model.set_hop(hopping[209],1,5,[0,-1,1])
        my_model.set_hop(hopping[210],2,6,[0,-1,1])
        my_model.set_hop(hopping[211],3,7,[0,-1,1])
        my_model.set_hop(hopping[212],0,12,[-1,0,1])
        my_model.set_hop(hopping[213],1,13,[-1,0,1])
        my_model.set_hop(hopping[214],2,14,[-1,0,1])
        my_model.set_hop(hopping[215],3,15,[-1,0,1])
        my_model.set_hop(hopping[216],0,12,[0,-1,1])
        my_model.set_hop(hopping[217],1,13,[0,-1,1])
        my_model.set_hop(hopping[218],2,14,[0,-1,1])
        my_model.set_hop(hopping[219],3,15,[0,-1,1])
        my_model.set_hop(hopping[220],0,4,[-1,0,1])
        my_model.set_hop(hopping[221],1,5,[-1,0,1])
        my_model.set_hop(hopping[222],2,6,[-1,0,1])
        my_model.set_hop(hopping[223],3,7,[-1,0,1])
        my_model.set_hop(hopping[224],0,8,[-1,0,1])
        my_model.set_hop(hopping[225],1,9,[-1,0,1])
        my_model.set_hop(hopping[226],2,10,[-1,0,1])
        my_model.set_hop(hopping[227],3,11,[-1,0,1])
        my_model.set_hop(hopping[228],0,8,[0,-1,1])
        my_model.set_hop(hopping[229],1,9,[0,-1,1])
        my_model.set_hop(hopping[230],2,10,[0,-1,1])
        my_model.set_hop(hopping[231],3,11,[0,-1,1])
        my_model.set_hop(hopping[232],0,8,[-1,1,0])
        my_model.set_hop(hopping[233],0,12,[-1,1,0])
        my_model.set_hop(hopping[234],1,9,[-1,1,0])
        my_model.set_hop(hopping[235],1,13,[-1,1,0])
        my_model.set_hop(hopping[236],2,10,[-1,1,0])
        my_model.set_hop(hopping[237],2,14,[-1,1,0])
        my_model.set_hop(hopping[238],3,11,[-1,1,0])
        my_model.set_hop(hopping[239],3,15,[-1,1,0])
        my_model.set_hop(hopping[240],0,4,[-1,1,0])
        my_model.set_hop(hopping[241],1,5,[-1,1,0])
        my_model.set_hop(hopping[242],2,6,[-1,1,0])
        my_model.set_hop(hopping[243],3,7,[-1,1,0])
        my_model.set_hop(hopping[244],11,12,[0,0,-1])
        my_model.set_hop(hopping[245],8,15,[0,0,1])
        my_model.set_hop(hopping[246],7,8,[0,0,-1])
        my_model.set_hop(hopping[247],7,12,[0,0,-1])
        my_model.set_hop(hopping[248],4,11,[0,0,1])
        my_model.set_hop(hopping[249],4,15,[0,0,1])
        my_model.set_hop(hopping[250],4,7,[0,0,1])
        my_model.set_hop(hopping[251],8,11,[0,0,1])
        my_model.set_hop(hopping[252],12,15,[0,0,1])
        my_model.set_hop(hopping[253],6,9,[0,-1,0])
        my_model.set_hop(hopping[254],6,13,[0,-1,0])
        my_model.set_hop(hopping[255],5,10,[0,1,0])
        my_model.set_hop(hopping[256],5,14,[0,1,0])
        my_model.set_hop(hopping[257],5,6,[0,1,0])
        my_model.set_hop(hopping[258],9,10,[0,1,0])
        my_model.set_hop(hopping[259],10,13,[0,-1,0])
        my_model.set_hop(hopping[260],9,14,[0,1,0])
        my_model.set_hop(hopping[261],13,14,[0,1,0])
        my_model.set_hop(hopping[262],11,13,[0,0,-1])
        my_model.set_hop(hopping[263],9,15,[0,0,1])
        my_model.set_hop(hopping[264],8,14,[-1,0,1])
        my_model.set_hop(hopping[265],10,12,[1,0,-1])
        my_model.set_hop(hopping[266],4,10,[-1,0,1])
        my_model.set_hop(hopping[267],6,8,[1,0,-1])
        my_model.set_hop(hopping[268],7,9,[0,0,-1])
        my_model.set_hop(hopping[269],7,13,[0,0,-1])
        my_model.set_hop(hopping[270],5,11,[0,0,1])
        my_model.set_hop(hopping[271],5,15,[0,0,1])
        my_model.set_hop(hopping[272],5,7,[0,0,1])
        my_model.set_hop(hopping[273],4,6,[-1,0,1])
        my_model.set_hop(hopping[274],4,14,[-1,0,1])
        my_model.set_hop(hopping[275],6,12,[1,0,-1])
        my_model.set_hop(hopping[276],12,14,[-1,0,1])
        my_model.set_hop(hopping[277],13,15,[0,0,1])
        my_model.set_hop(hopping[278],8,10,[-1,0,1])
        my_model.set_hop(hopping[279],9,11,[0,0,1])
        my_model.set_hop(hopping[280],4,10,[-1,1,0])
        my_model.set_hop(hopping[281],6,8,[1,-1,0])
        my_model.set_hop(hopping[282],4,14,[-1,1,0])
        my_model.set_hop(hopping[283],6,12,[1,-1,0])
        my_model.set_hop(hopping[284],12,14,[-1,1,0])
        my_model.set_hop(hopping[285],8,14,[-1,1,0])
        my_model.set_hop(hopping[286],10,12,[1,-1,0])
        my_model.set_hop(hopping[287],8,10,[-1,1,0])
        my_model.set_hop(hopping[288],4,6,[-1,1,0])
        my_model.set_hop(hopping[289],7,9,[0,-1,0])
        my_model.set_hop(hopping[290],5,11,[0,1,0])
        my_model.set_hop(hopping[291],7,13,[0,-1,0])
        my_model.set_hop(hopping[292],5,15,[0,1,0])
        my_model.set_hop(hopping[293],13,15,[0,1,0])
        my_model.set_hop(hopping[294],13,15,[1,0,0])
        my_model.set_hop(hopping[295],11,13,[-1,0,0])
        my_model.set_hop(hopping[296],11,13,[0,-1,0])
        my_model.set_hop(hopping[297],9,15,[0,1,0])
        my_model.set_hop(hopping[298],9,15,[1,0,0])
        my_model.set_hop(hopping[299],7,13,[-1,0,0])
        my_model.set_hop(hopping[300],5,15,[1,0,0])
        my_model.set_hop(hopping[301],9,11,[0,1,0])
        my_model.set_hop(hopping[302],9,11,[1,0,0])
        my_model.set_hop(hopping[303],7,9,[-1,0,0])
        my_model.set_hop(hopping[304],5,11,[1,0,0])
        my_model.set_hop(hopping[305],5,7,[0,1,0])
        my_model.set_hop(hopping[306],5,7,[1,0,0])
        my_model.set_hop(hopping[307],12,14,[0,0,0])
        my_model.set_hop(hopping[308],8,14,[0,0,0])
        my_model.set_hop(hopping[309],10,12,[0,0,0])
        my_model.set_hop(hopping[310],4,14,[0,0,0])
        my_model.set_hop(hopping[311],6,12,[0,0,0])
        my_model.set_hop(hopping[312],8,10,[0,0,0])
        my_model.set_hop(hopping[313],4,10,[0,0,0])
        my_model.set_hop(hopping[314],6,8,[0,0,0])
        my_model.set_hop(hopping[315],4,6,[0,0,0])
        my_model.set_hop(hopping[316],8,15,[0,0,0])
        my_model.set_hop(hopping[317],11,12,[0,0,0])
        my_model.set_hop(hopping[318],4,11,[0,0,0])
        my_model.set_hop(hopping[319],4,15,[0,0,0])
        my_model.set_hop(hopping[320],7,8,[0,0,0])
        my_model.set_hop(hopping[321],7,12,[0,0,0])
        my_model.set_hop(hopping[322],4,7,[0,0,0])
        my_model.set_hop(hopping[323],12,15,[0,0,0])
        my_model.set_hop(hopping[324],8,11,[0,0,0])
        my_model.set_hop(hopping[325],10,13,[0,-1,1])
        my_model.set_hop(hopping[326],9,14,[0,1,-1])
        my_model.set_hop(hopping[327],6,9,[0,-1,1])
        my_model.set_hop(hopping[328],6,13,[0,-1,1])
        my_model.set_hop(hopping[329],5,10,[0,1,-1])
        my_model.set_hop(hopping[330],5,14,[0,1,-1])
        my_model.set_hop(hopping[331],5,6,[0,1,-1])
        my_model.set_hop(hopping[332],13,14,[0,1,-1])
        my_model.set_hop(hopping[333],9,10,[0,1,-1])
        my_model.set_hop(hopping[334],5,14,[0,0,0])
        my_model.set_hop(hopping[335],6,13,[0,0,0])
        my_model.set_hop(hopping[336],5,10,[0,0,0])
        my_model.set_hop(hopping[337],6,9,[0,0,0])
        my_model.set_hop(hopping[338],9,10,[-1,1,0])
        my_model.set_hop(hopping[339],9,10,[0,0,0])
        my_model.set_hop(hopping[340],9,14,[-1,1,0])
        my_model.set_hop(hopping[341],9,14,[0,0,0])
        my_model.set_hop(hopping[342],10,13,[0,0,0])
        my_model.set_hop(hopping[343],10,13,[1,-1,0])
        my_model.set_hop(hopping[344],13,14,[-1,1,0])
        my_model.set_hop(hopping[345],13,14,[0,0,0])
        my_model.set_hop(hopping[346],5,10,[-1,1,0])
        my_model.set_hop(hopping[347],6,9,[1,-1,0])
        my_model.set_hop(hopping[348],5,14,[-1,1,0])
        my_model.set_hop(hopping[349],6,13,[1,-1,0])
        my_model.set_hop(hopping[350],5,6,[-1,1,0])
        my_model.set_hop(hopping[351],5,6,[0,0,0])
        my_model.set_hop(hopping[352],4,15,[0,-1,1])
        my_model.set_hop(hopping[353],7,12,[0,1,-1])
        my_model.set_hop(hopping[354],4,11,[0,-1,1])
        my_model.set_hop(hopping[355],7,8,[0,1,-1])
        my_model.set_hop(hopping[356],8,11,[-1,0,1])
        my_model.set_hop(hopping[357],8,11,[0,-1,1])
        my_model.set_hop(hopping[358],8,15,[-1,0,1])
        my_model.set_hop(hopping[359],8,15,[0,-1,1])
        my_model.set_hop(hopping[360],11,12,[0,1,-1])
        my_model.set_hop(hopping[361],11,12,[1,0,-1])
        my_model.set_hop(hopping[362],12,15,[-1,0,1])
        my_model.set_hop(hopping[363],12,15,[0,-1,1])
        my_model.set_hop(hopping[364],4,11,[-1,0,1])
        my_model.set_hop(hopping[365],7,8,[1,0,-1])
        my_model.set_hop(hopping[366],4,15,[-1,0,1])
        my_model.set_hop(hopping[367],7,12,[1,0,-1])
        my_model.set_hop(hopping[368],4,7,[-1,0,1])
        my_model.set_hop(hopping[369],4,7,[0,-1,1])
        my_model.set_hop(hopping[370],4,9,[0,0,0])
        my_model.set_hop(hopping[371],4,13,[0,0,0])
        my_model.set_hop(hopping[372],5,8,[0,0,0])
        my_model.set_hop(hopping[373],5,12,[0,0,0])
        my_model.set_hop(hopping[374],4,5,[0,0,0])
        my_model.set_hop(hopping[375],12,13,[0,0,0])
        my_model.set_hop(hopping[376],8,13,[0,0,0])
        my_model.set_hop(hopping[377],9,12,[0,0,0])
        my_model.set_hop(hopping[378],8,9,[0,0,0])
        my_model.set_hop(hopping[379],4,9,[-1,-1,1])
        my_model.set_hop(hopping[380],4,13,[-1,-1,1])
        my_model.set_hop(hopping[381],7,10,[-1,0,0])
        my_model.set_hop(hopping[382],7,14,[-1,0,0])
        my_model.set_hop(hopping[383],6,11,[0,-1,1])
        my_model.set_hop(hopping[384],6,15,[0,-1,1])
        my_model.set_hop(hopping[385],7,10,[0,1,-1])
        my_model.set_hop(hopping[386],7,14,[0,1,-1])
        my_model.set_hop(hopping[387],6,11,[1,0,0])
        my_model.set_hop(hopping[388],6,15,[1,0,0])
        my_model.set_hop(hopping[389],5,8,[1,1,-1])
        my_model.set_hop(hopping[390],5,12,[1,1,-1])
        my_model.set_hop(hopping[391],4,5,[-1,-1,1])
        my_model.set_hop(hopping[392],6,7,[0,-1,1])
        my_model.set_hop(hopping[393],6,7,[1,0,0])
        my_model.set_hop(hopping[394],12,13,[-1,-1,1])
        my_model.set_hop(hopping[395],14,15,[0,-1,1])
        my_model.set_hop(hopping[396],14,15,[1,0,0])
        my_model.set_hop(hopping[397],8,13,[-1,-1,1])
        my_model.set_hop(hopping[398],10,15,[0,-1,1])
        my_model.set_hop(hopping[399],11,14,[0,1,-1])
        my_model.set_hop(hopping[400],9,12,[1,1,-1])
        my_model.set_hop(hopping[401],11,14,[-1,0,0])
        my_model.set_hop(hopping[402],10,15,[1,0,0])
        my_model.set_hop(hopping[403],8,9,[-1,-1,1])
        my_model.set_hop(hopping[404],10,11,[0,-1,1])
        my_model.set_hop(hopping[405],10,11,[1,0,0])
        my_model.set_hop(hopping[406],4,13,[-1,0,1])
        my_model.set_hop(hopping[407],7,14,[-1,1,0])
        my_model.set_hop(hopping[408],6,15,[1,-1,0])
        my_model.set_hop(hopping[409],5,12,[1,0,-1])
        my_model.set_hop(hopping[410],4,9,[0,-1,1])
        my_model.set_hop(hopping[411],6,11,[0,0,0])
        my_model.set_hop(hopping[412],7,10,[0,0,0])
        my_model.set_hop(hopping[413],5,8,[0,1,-1])
        my_model.set_hop(hopping[414],8,13,[-1,0,1])
        my_model.set_hop(hopping[415],8,13,[0,-1,1])
        my_model.set_hop(hopping[416],9,12,[0,1,-1])
        my_model.set_hop(hopping[417],9,12,[1,0,-1])
        my_model.set_hop(hopping[418],11,14,[-1,1,0])
        my_model.set_hop(hopping[419],10,15,[0,0,0])
        my_model.set_hop(hopping[420],11,14,[0,0,0])
        my_model.set_hop(hopping[421],10,15,[1,-1,0])
        my_model.set_hop(hopping[422],10,11,[0,0,0])
        my_model.set_hop(hopping[423],10,11,[1,-1,0])
        my_model.set_hop(hopping[424],8,9,[-1,0,1])
        my_model.set_hop(hopping[425],8,9,[0,-1,1])
        my_model.set_hop(hopping[426],4,9,[-1,0,1])
        my_model.set_hop(hopping[427],7,10,[-1,1,0])
        my_model.set_hop(hopping[428],6,11,[1,-1,0])
        my_model.set_hop(hopping[429],5,8,[1,0,-1])
        my_model.set_hop(hopping[430],12,13,[-1,0,1])
        my_model.set_hop(hopping[431],12,13,[0,-1,1])
        my_model.set_hop(hopping[432],14,15,[0,0,0])
        my_model.set_hop(hopping[433],14,15,[1,-1,0])
        my_model.set_hop(hopping[434],4,13,[0,-1,1])
        my_model.set_hop(hopping[435],6,15,[0,0,0])
        my_model.set_hop(hopping[436],7,14,[0,0,0])
        my_model.set_hop(hopping[437],5,12,[0,1,-1])
        my_model.set_hop(hopping[438],4,5,[-1,0,1])
        my_model.set_hop(hopping[439],4,5,[0,-1,1])
        my_model.set_hop(hopping[440],6,7,[0,0,0])
        my_model.set_hop(hopping[441],6,7,[1,-1,0])
        my_model.set_hop(hopping[442],4,13,[0,-1,0])
        my_model.set_hop(hopping[443],5,12,[0,1,0])
        my_model.set_hop(hopping[444],7,14,[0,0,-1])
        my_model.set_hop(hopping[445],6,15,[0,0,1])
        my_model.set_hop(hopping[446],4,9,[-1,0,0])
        my_model.set_hop(hopping[447],7,10,[-1,1,-1])
        my_model.set_hop(hopping[448],6,11,[1,-1,1])
        my_model.set_hop(hopping[449],5,8,[1,0,0])
        my_model.set_hop(hopping[450],8,13,[-1,0,0])
        my_model.set_hop(hopping[451],8,13,[0,-1,0])
        my_model.set_hop(hopping[452],9,12,[0,1,0])
        my_model.set_hop(hopping[453],9,12,[1,0,0])
        my_model.set_hop(hopping[454],11,14,[-1,1,-1])
        my_model.set_hop(hopping[455],11,14,[0,0,-1])
        my_model.set_hop(hopping[456],10,15,[0,0,1])
        my_model.set_hop(hopping[457],10,15,[1,-1,1])
        my_model.set_hop(hopping[458],8,9,[-1,0,0])
        my_model.set_hop(hopping[459],8,9,[0,-1,0])
        my_model.set_hop(hopping[460],10,11,[0,0,1])
        my_model.set_hop(hopping[461],10,11,[1,-1,1])
        my_model.set_hop(hopping[462],4,9,[0,-1,0])
        my_model.set_hop(hopping[463],7,10,[0,0,-1])
        my_model.set_hop(hopping[464],6,11,[0,0,1])
        my_model.set_hop(hopping[465],5,8,[0,1,0])
        my_model.set_hop(hopping[466],14,15,[0,0,1])
        my_model.set_hop(hopping[467],14,15,[1,-1,1])
        my_model.set_hop(hopping[468],12,13,[-1,0,0])
        my_model.set_hop(hopping[469],12,13,[0,-1,0])
        my_model.set_hop(hopping[470],7,14,[-1,1,-1])
        my_model.set_hop(hopping[471],6,15,[1,-1,1])
        my_model.set_hop(hopping[472],4,13,[-1,0,0])
        my_model.set_hop(hopping[473],5,12,[1,0,0])
        my_model.set_hop(hopping[474],4,5,[-1,0,0])
        my_model.set_hop(hopping[475],4,5,[0,-1,0])
        my_model.set_hop(hopping[476],6,7,[0,0,1])
        my_model.set_hop(hopping[477],6,7,[1,-1,1])
        my_model.set_hop(hopping[478],4,14,[-1,0,0])
        my_model.set_hop(hopping[479],8,14,[-1,0,0])
        my_model.set_hop(hopping[480],6,12,[1,0,0])
        my_model.set_hop(hopping[481],10,12,[1,0,0])
        my_model.set_hop(hopping[482],5,11,[0,0,0])
        my_model.set_hop(hopping[483],5,15,[0,0,0])
        my_model.set_hop(hopping[484],7,9,[0,0,0])
        my_model.set_hop(hopping[485],7,13,[0,0,0])
        my_model.set_hop(hopping[486],9,15,[0,0,0])
        my_model.set_hop(hopping[487],11,13,[0,0,0])
        my_model.set_hop(hopping[488],5,7,[0,0,0])
        my_model.set_hop(hopping[489],9,11,[0,0,0])
        my_model.set_hop(hopping[490],4,10,[-1,0,0])
        my_model.set_hop(hopping[491],6,8,[1,0,0])
        my_model.set_hop(hopping[492],4,6,[-1,0,0])
        my_model.set_hop(hopping[493],8,10,[-1,0,0])
        my_model.set_hop(hopping[494],12,14,[-1,0,0])
        my_model.set_hop(hopping[495],13,15,[0,0,0])
        my_model.set_hop(hopping[496],4,8,[0,-1,1])
        my_model.set_hop(hopping[497],5,9,[0,-1,1])
        my_model.set_hop(hopping[498],6,10,[0,-1,1])
        my_model.set_hop(hopping[499],7,11,[0,-1,1])
        my_model.set_hop(hopping[500],4,12,[-1,0,1])
        my_model.set_hop(hopping[501],5,13,[-1,0,1])
        my_model.set_hop(hopping[502],6,14,[-1,0,1])
        my_model.set_hop(hopping[503],7,15,[-1,0,1])
        my_model.set_hop(hopping[504],8,12,[-1,0,1])
        my_model.set_hop(hopping[505],9,13,[-1,0,1])
        my_model.set_hop(hopping[506],10,14,[-1,0,1])
        my_model.set_hop(hopping[507],11,15,[-1,0,1])
        my_model.set_hop(hopping[508],4,12,[0,-1,1])
        my_model.set_hop(hopping[509],5,13,[0,-1,1])
        my_model.set_hop(hopping[510],6,14,[0,-1,1])
        my_model.set_hop(hopping[511],7,15,[0,-1,1])
        my_model.set_hop(hopping[512],8,12,[0,-1,1])
        my_model.set_hop(hopping[513],9,13,[0,-1,1])
        my_model.set_hop(hopping[514],10,14,[0,-1,1])
        my_model.set_hop(hopping[515],11,15,[0,-1,1])
        my_model.set_hop(hopping[516],12,12,[-1,0,1])
        my_model.set_hop(hopping[517],13,13,[-1,0,1])
        my_model.set_hop(hopping[518],14,14,[-1,0,1])
        my_model.set_hop(hopping[519],15,15,[-1,0,1])
        my_model.set_hop(hopping[520],12,12,[0,-1,1])
        my_model.set_hop(hopping[521],13,13,[0,-1,1])
        my_model.set_hop(hopping[522],14,14,[0,-1,1])
        my_model.set_hop(hopping[523],15,15,[0,-1,1])
        my_model.set_hop(hopping[524],4,4,[-1,0,1])
        my_model.set_hop(hopping[525],5,5,[-1,0,1])
        my_model.set_hop(hopping[526],6,6,[-1,0,1])
        my_model.set_hop(hopping[527],7,7,[-1,0,1])
        my_model.set_hop(hopping[528],4,4,[0,-1,1])
        my_model.set_hop(hopping[529],5,5,[0,-1,1])
        my_model.set_hop(hopping[530],6,6,[0,-1,1])
        my_model.set_hop(hopping[531],7,7,[0,-1,1])
        my_model.set_hop(hopping[532],4,8,[-1,0,1])
        my_model.set_hop(hopping[533],5,9,[-1,0,1])
        my_model.set_hop(hopping[534],6,10,[-1,0,1])
        my_model.set_hop(hopping[535],7,11,[-1,0,1])
        my_model.set_hop(hopping[536],8,8,[-1,0,1])
        my_model.set_hop(hopping[537],9,9,[-1,0,1])
        my_model.set_hop(hopping[538],10,10,[-1,0,1])
        my_model.set_hop(hopping[539],11,11,[-1,0,1])
        my_model.set_hop(hopping[540],8,8,[0,-1,1])
        my_model.set_hop(hopping[541],9,9,[0,-1,1])
        my_model.set_hop(hopping[542],10,10,[0,-1,1])
        my_model.set_hop(hopping[543],11,11,[0,-1,1])
        my_model.set_hop(hopping[544],4,8,[-1,1,0])
        my_model.set_hop(hopping[545],4,12,[-1,1,0])
        my_model.set_hop(hopping[546],5,9,[-1,1,0])
        my_model.set_hop(hopping[547],5,13,[-1,1,0])
        my_model.set_hop(hopping[548],6,10,[-1,1,0])
        my_model.set_hop(hopping[549],6,14,[-1,1,0])
        my_model.set_hop(hopping[550],7,11,[-1,1,0])
        my_model.set_hop(hopping[551],7,15,[-1,1,0])
        my_model.set_hop(hopping[552],8,12,[-1,1,0])
        my_model.set_hop(hopping[553],9,13,[-1,1,0])
        my_model.set_hop(hopping[554],10,14,[-1,1,0])
        my_model.set_hop(hopping[555],11,15,[-1,1,0])
        my_model.set_hop(hopping[556],8,8,[-1,1,0])
        my_model.set_hop(hopping[557],9,9,[-1,1,0])
        my_model.set_hop(hopping[558],10,10,[-1,1,0])
        my_model.set_hop(hopping[559],11,11,[-1,1,0])
        my_model.set_hop(hopping[560],12,12,[-1,1,0])
        my_model.set_hop(hopping[561],13,13,[-1,1,0])
        my_model.set_hop(hopping[562],14,14,[-1,1,0])
        my_model.set_hop(hopping[563],15,15,[-1,1,0])
        my_model.set_hop(hopping[564],4,4,[-1,1,0])
        my_model.set_hop(hopping[565],5,5,[-1,1,0])
        my_model.set_hop(hopping[566],6,6,[-1,1,0])
        my_model.set_hop(hopping[567],7,7,[-1,1,0])
        my_model.set_hop(hopping[568],15,16,[0,0,-1])
        my_model.set_hop(hopping[569],8,19,[0,0,1])
        my_model.set_hop(hopping[570],7,16,[0,0,-1])
        my_model.set_hop(hopping[571],4,19,[0,0,1])
        my_model.set_hop(hopping[572],11,16,[0,0,-1])
        my_model.set_hop(hopping[573],12,19,[0,0,1])
        my_model.set_hop(hopping[574],14,17,[0,-1,0])
        my_model.set_hop(hopping[575],10,17,[0,-1,0])
        my_model.set_hop(hopping[576],6,17,[0,-1,0])
        my_model.set_hop(hopping[577],5,18,[0,1,0])
        my_model.set_hop(hopping[578],9,18,[0,1,0])
        my_model.set_hop(hopping[579],13,18,[0,1,0])
        my_model.set_hop(hopping[580],8,18,[-1,0,1])
        my_model.set_hop(hopping[581],9,19,[0,0,1])
        my_model.set_hop(hopping[582],15,17,[0,0,-1])
        my_model.set_hop(hopping[583],14,16,[1,0,-1])
        my_model.set_hop(hopping[584],6,16,[1,0,-1])
        my_model.set_hop(hopping[585],7,17,[0,0,-1])
        my_model.set_hop(hopping[586],5,19,[0,0,1])
        my_model.set_hop(hopping[587],4,18,[-1,0,1])
        my_model.set_hop(hopping[588],12,18,[-1,0,1])
        my_model.set_hop(hopping[589],13,19,[0,0,1])
        my_model.set_hop(hopping[590],11,17,[0,0,-1])
        my_model.set_hop(hopping[591],10,16,[1,0,-1])
        my_model.set_hop(hopping[592],4,18,[-1,1,0])
        my_model.set_hop(hopping[593],10,16,[1,-1,0])
        my_model.set_hop(hopping[594],14,16,[1,-1,0])
        my_model.set_hop(hopping[595],12,18,[-1,1,0])
        my_model.set_hop(hopping[596],8,18,[-1,1,0])
        my_model.set_hop(hopping[597],6,16,[1,-1,0])
        my_model.set_hop(hopping[598],7,17,[-1,0,0])
        my_model.set_hop(hopping[599],5,19,[0,1,0])
        my_model.set_hop(hopping[600],11,17,[-1,0,0])
        my_model.set_hop(hopping[601],11,17,[0,-1,0])
        my_model.set_hop(hopping[602],15,17,[-1,0,0])
        my_model.set_hop(hopping[603],15,17,[0,-1,0])
        my_model.set_hop(hopping[604],13,19,[0,1,0])
        my_model.set_hop(hopping[605],13,19,[1,0,0])
        my_model.set_hop(hopping[606],9,19,[0,1,0])
        my_model.set_hop(hopping[607],9,19,[1,0,0])
        my_model.set_hop(hopping[608],7,17,[0,-1,0])
        my_model.set_hop(hopping[609],5,19,[1,0,0])
        my_model.set_hop(hopping[610],6,16,[0,0,0])
        my_model.set_hop(hopping[611],10,16,[0,0,0])
        my_model.set_hop(hopping[612],14,16,[0,0,0])
        my_model.set_hop(hopping[613],12,18,[0,0,0])
        my_model.set_hop(hopping[614],8,18,[0,0,0])
        my_model.set_hop(hopping[615],4,18,[0,0,0])
        my_model.set_hop(hopping[616],11,16,[0,0,0])
        my_model.set_hop(hopping[617],12,19,[0,0,0])
        my_model.set_hop(hopping[618],4,19,[0,0,0])
        my_model.set_hop(hopping[619],7,16,[0,0,0])
        my_model.set_hop(hopping[620],15,16,[0,0,0])
        my_model.set_hop(hopping[621],8,19,[0,0,0])
        my_model.set_hop(hopping[622],10,17,[0,-1,1])
        my_model.set_hop(hopping[623],13,18,[0,1,-1])
        my_model.set_hop(hopping[624],6,17,[0,-1,1])
        my_model.set_hop(hopping[625],5,18,[0,1,-1])
        my_model.set_hop(hopping[626],14,17,[0,-1,1])
        my_model.set_hop(hopping[627],9,18,[0,1,-1])
        my_model.set_hop(hopping[628],5,18,[-1,1,0])
        my_model.set_hop(hopping[629],6,17,[0,0,0])
        my_model.set_hop(hopping[630],13,18,[-1,1,0])
        my_model.set_hop(hopping[631],13,18,[0,0,0])
        my_model.set_hop(hopping[632],9,18,[-1,1,0])
        my_model.set_hop(hopping[633],9,18,[0,0,0])
        my_model.set_hop(hopping[634],10,17,[0,0,0])
        my_model.set_hop(hopping[635],10,17,[1,-1,0])
        my_model.set_hop(hopping[636],14,17,[0,0,0])
        my_model.set_hop(hopping[637],14,17,[1,-1,0])
        my_model.set_hop(hopping[638],5,18,[0,0,0])
        my_model.set_hop(hopping[639],6,17,[1,-1,0])
        my_model.set_hop(hopping[640],4,19,[-1,0,1])
        my_model.set_hop(hopping[641],7,16,[0,1,-1])
        my_model.set_hop(hopping[642],12,19,[-1,0,1])
        my_model.set_hop(hopping[643],12,19,[0,-1,1])
        my_model.set_hop(hopping[644],8,19,[-1,0,1])
        my_model.set_hop(hopping[645],8,19,[0,-1,1])
        my_model.set_hop(hopping[646],11,16,[0,1,-1])
        my_model.set_hop(hopping[647],11,16,[1,0,-1])
        my_model.set_hop(hopping[648],15,16,[0,1,-1])
        my_model.set_hop(hopping[649],15,16,[1,0,-1])
        my_model.set_hop(hopping[650],4,19,[0,-1,1])
        my_model.set_hop(hopping[651],7,16,[1,0,-1])
        my_model.set_hop(hopping[652],9,16,[0,0,0])
        my_model.set_hop(hopping[653],13,16,[0,0,0])
        my_model.set_hop(hopping[654],4,17,[0,0,0])
        my_model.set_hop(hopping[655],5,16,[0,0,0])
        my_model.set_hop(hopping[656],12,17,[0,0,0])
        my_model.set_hop(hopping[657],8,17,[0,0,0])
        my_model.set_hop(hopping[658],8,17,[-1,-1,1])
        my_model.set_hop(hopping[659],11,18,[-1,0,0])
        my_model.set_hop(hopping[660],10,19,[0,-1,1])
        my_model.set_hop(hopping[661],15,18,[-1,0,0])
        my_model.set_hop(hopping[662],12,17,[-1,-1,1])
        my_model.set_hop(hopping[663],14,19,[0,-1,1])
        my_model.set_hop(hopping[664],4,17,[-1,-1,1])
        my_model.set_hop(hopping[665],7,18,[-1,0,0])
        my_model.set_hop(hopping[666],6,19,[0,-1,1])
        my_model.set_hop(hopping[667],7,18,[0,1,-1])
        my_model.set_hop(hopping[668],6,19,[1,0,0])
        my_model.set_hop(hopping[669],5,16,[1,1,-1])
        my_model.set_hop(hopping[670],15,18,[0,1,-1])
        my_model.set_hop(hopping[671],13,16,[1,1,-1])
        my_model.set_hop(hopping[672],14,19,[1,0,0])
        my_model.set_hop(hopping[673],11,18,[0,1,-1])
        my_model.set_hop(hopping[674],10,19,[1,0,0])
        my_model.set_hop(hopping[675],9,16,[1,1,-1])
        my_model.set_hop(hopping[676],4,17,[-1,0,1])
        my_model.set_hop(hopping[677],7,18,[-1,1,0])
        my_model.set_hop(hopping[678],6,19,[0,0,0])
        my_model.set_hop(hopping[679],5,16,[0,1,-1])
        my_model.set_hop(hopping[680],14,19,[0,0,0])
        my_model.set_hop(hopping[681],13,16,[0,1,-1])
        my_model.set_hop(hopping[682],14,19,[1,-1,0])
        my_model.set_hop(hopping[683],13,16,[1,0,-1])
        my_model.set_hop(hopping[684],8,17,[-1,0,1])
        my_model.set_hop(hopping[685],8,17,[0,-1,1])
        my_model.set_hop(hopping[686],11,18,[-1,1,0])
        my_model.set_hop(hopping[687],11,18,[0,0,0])
        my_model.set_hop(hopping[688],10,19,[0,0,0])
        my_model.set_hop(hopping[689],10,19,[1,-1,0])
        my_model.set_hop(hopping[690],9,16,[0,1,-1])
        my_model.set_hop(hopping[691],9,16,[1,0,-1])
        my_model.set_hop(hopping[692],12,17,[-1,0,1])
        my_model.set_hop(hopping[693],15,18,[-1,1,0])
        my_model.set_hop(hopping[694],12,17,[0,-1,1])
        my_model.set_hop(hopping[695],15,18,[0,0,0])
        my_model.set_hop(hopping[696],4,17,[0,-1,1])
        my_model.set_hop(hopping[697],7,18,[0,0,0])
        my_model.set_hop(hopping[698],6,19,[1,-1,0])
        my_model.set_hop(hopping[699],5,16,[1,0,-1])
        my_model.set_hop(hopping[700],4,17,[-1,0,0])
        my_model.set_hop(hopping[701],7,18,[-1,1,-1])
        my_model.set_hop(hopping[702],6,19,[0,0,1])
        my_model.set_hop(hopping[703],5,16,[0,1,0])
        my_model.set_hop(hopping[704],12,17,[-1,0,0])
        my_model.set_hop(hopping[705],12,17,[0,-1,0])
        my_model.set_hop(hopping[706],15,18,[-1,1,-1])
        my_model.set_hop(hopping[707],15,18,[0,0,-1])
        my_model.set_hop(hopping[708],10,19,[0,0,1])
        my_model.set_hop(hopping[709],9,16,[0,1,0])
        my_model.set_hop(hopping[710],10,19,[1,-1,1])
        my_model.set_hop(hopping[711],9,16,[1,0,0])
        my_model.set_hop(hopping[712],8,17,[-1,0,0])
        my_model.set_hop(hopping[713],11,18,[-1,1,-1])
        my_model.set_hop(hopping[714],8,17,[0,-1,0])
        my_model.set_hop(hopping[715],11,18,[0,0,-1])
        my_model.set_hop(hopping[716],14,19,[0,0,1])
        my_model.set_hop(hopping[717],14,19,[1,-1,1])
        my_model.set_hop(hopping[718],13,16,[0,1,0])
        my_model.set_hop(hopping[719],13,16,[1,0,0])
        my_model.set_hop(hopping[720],4,17,[0,-1,0])
        my_model.set_hop(hopping[721],7,18,[0,0,-1])
        my_model.set_hop(hopping[722],6,19,[1,-1,1])
        my_model.set_hop(hopping[723],5,16,[1,0,0])
        my_model.set_hop(hopping[724],12,18,[-1,0,0])
        my_model.set_hop(hopping[725],13,19,[0,0,0])
        my_model.set_hop(hopping[726],6,16,[1,0,0])
        my_model.set_hop(hopping[727],10,16,[1,0,0])
        my_model.set_hop(hopping[728],5,19,[0,0,0])
        my_model.set_hop(hopping[729],7,17,[0,0,0])
        my_model.set_hop(hopping[730],9,19,[0,0,0])
        my_model.set_hop(hopping[731],11,17,[0,0,0])
        my_model.set_hop(hopping[732],4,18,[-1,0,0])
        my_model.set_hop(hopping[733],8,18,[-1,0,0])
        my_model.set_hop(hopping[734],15,17,[0,0,0])
        my_model.set_hop(hopping[735],14,16,[1,0,0])
        my_model.set_hop(hopping[736],8,16,[-1,0,1])
        my_model.set_hop(hopping[737],9,17,[-1,0,1])
        my_model.set_hop(hopping[738],10,18,[-1,0,1])
        my_model.set_hop(hopping[739],11,19,[-1,0,1])
        my_model.set_hop(hopping[740],8,16,[0,-1,1])
        my_model.set_hop(hopping[741],9,17,[0,-1,1])
        my_model.set_hop(hopping[742],10,18,[0,-1,1])
        my_model.set_hop(hopping[743],11,19,[0,-1,1])
        my_model.set_hop(hopping[744],4,16,[-1,0,1])
        my_model.set_hop(hopping[745],5,17,[-1,0,1])
        my_model.set_hop(hopping[746],6,18,[-1,0,1])
        my_model.set_hop(hopping[747],7,19,[-1,0,1])
        my_model.set_hop(hopping[748],12,16,[-1,0,1])
        my_model.set_hop(hopping[749],13,17,[-1,0,1])
        my_model.set_hop(hopping[750],14,18,[-1,0,1])
        my_model.set_hop(hopping[751],15,19,[-1,0,1])
        my_model.set_hop(hopping[752],12,16,[0,-1,1])
        my_model.set_hop(hopping[753],13,17,[0,-1,1])
        my_model.set_hop(hopping[754],14,18,[0,-1,1])
        my_model.set_hop(hopping[755],15,19,[0,-1,1])
        my_model.set_hop(hopping[756],4,16,[0,-1,1])
        my_model.set_hop(hopping[757],5,17,[0,-1,1])
        my_model.set_hop(hopping[758],6,18,[0,-1,1])
        my_model.set_hop(hopping[759],7,19,[0,-1,1])
        my_model.set_hop(hopping[760],4,16,[-1,1,0])
        my_model.set_hop(hopping[761],5,17,[-1,1,0])
        my_model.set_hop(hopping[762],6,18,[-1,1,0])
        my_model.set_hop(hopping[763],7,19,[-1,1,0])
        my_model.set_hop(hopping[764],8,16,[-1,1,0])
        my_model.set_hop(hopping[765],9,17,[-1,1,0])
        my_model.set_hop(hopping[766],10,18,[-1,1,0])
        my_model.set_hop(hopping[767],11,19,[-1,1,0])
        my_model.set_hop(hopping[768],12,16,[-1,1,0])
        my_model.set_hop(hopping[769],13,17,[-1,1,0])
        my_model.set_hop(hopping[770],14,18,[-1,1,0])
        my_model.set_hop(hopping[771],15,19,[-1,1,0])
        #endregion

        # print tight-binding model
        # my_model.display()

        # generate list of k-points following a segmented path in the BZ
        # list of nodes (high-symmetry points) that will be connected
        path = [[0, 0, 0], [2/3,1/3,1/2], [
            0, 0, 0], [1/2, 0, 0], [2/3,1/3,1/2]]
        # labels of the nodes
        self.label = ( r'$\Gamma $', r'$H $', r'$\Gamma $', r'$M $', r'$H $')
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
        Gamma1 = np.where(self.k_dist == self.k_node[0])[0][0] # the 1st Gamma point
        Gamma2 = np.where(self.k_dist == self.k_node[2])[0][0] # the 2nd Gamma point
        interval = 10 # interval length for calculating the derivative

        # calculate band mass & effective mass for all bands
        self.bmass_0_t = findBMass(self.evals[0], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_0_l = findBMass(self.evals[0], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_0 = np.power(self.bmass_0_t*self.bmass_0_l, 1/2)

        self.bmass_1_t = findBMass(self.evals[1], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_1_l = findBMass(self.evals[1], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_1 = np.power(self.bmass_1_t*self.bmass_1_l, 1/2)

        self.bmass_2_t = findBMass(self.evals[2], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_2_l = findBMass(self.evals[2], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_2 = np.power(self.bmass_2_t*self.bmass_2_l, 1/2)

        self.bmass_3_t = findBMass(self.evals[3], self.k_dist, Gamma1, Gamma1+interval)
        self.bmass_3_l = findBMass(self.evals[3], self.k_dist, Gamma2, Gamma2+interval)
        self.emass_3 = np.power(self.bmass_3_t*self.bmass_3_l, 1/2)


    def getMassString(self):
        """
        return the string containing all band mass and
        effective mass. The order is:
        bmass_0_f, bmass_0_t, bmass_0_l, emass_0,
        bmass_1_f, bmass_1_t, bmass_1_l, emass_1,
        bmass_2_f, bmass_2_t, bmass_2_l, emass_2,
        bmass_3_f, bmass_3_t, bmass_3_l, emass_3
        """
        return ('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' 
        %(
          self.bmass_0_t, self.bmass_0_l, self.emass_0,
          self.bmass_1_t, self.bmass_1_l, self.emass_1,
          self.bmass_2_t, self.bmass_2_l, self.emass_2,
          self.bmass_3_t, self.bmass_3_l, self.emass_3))
        

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