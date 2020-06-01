#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_stp_O_sstar_Class import *

lat = 3.293087727
ss_bond_len = [3.040000,
    3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 3.277766, 
    3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 3.327418,
    3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 3.445864,
    3.445864, 3.605804, 3.612253, 4.337574, 4.337574, 4.470498, 
    4.470498, 4.652131, 4.716293, 4.716293, 4.721225, 4.721225,
    4.862916, 4.945890, 4.945890
]
sp_len_cos = [[3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000],
    [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, -0.000000],
    [3.040000, -0.000000], [3.040000, -0.000000], [3.040000, 1.000000],
    [3.040000, 1.000000], [3.040000, 1.000000], [3.040000, 1.000000],
    [3.109274, -0.673094], [3.109274, -0.673094], [3.109274, -0.554942],
    [3.109274, -0.554942], [3.109274, -0.488860], [3.109274, -0.488860],
    [3.109274, 0.488860], [3.109274, 0.488860], [3.109274, 0.554942],
    [3.109274, 0.554942], [3.109274, 0.673094], [3.109274, 0.673094],
    [3.277766, -0.982253], [3.277766, -0.187560], [3.277766, -0.000000],
    [3.277766, 0.000000], [3.277766, 0.187560], [3.277766, 0.982253],
    [3.277766, -0.982253], [3.277766, -0.187560], [3.277766, -0.000000],
    [3.277766, -0.000000], [3.277766, 0.187560], [3.277766, 0.982253], 
    [3.300672, -0.886903], [3.300672, -0.886903], [3.300672, -0.886903],
    [3.300672, -0.886903], [3.300672, -0.460512], [3.300672, -0.460512],
    [3.300672, -0.460512], [3.300672, -0.460512], [3.300672, -0.036498],
    [3.300672, -0.036498], [3.300672, -0.036498], [3.300672, -0.036498],
    [3.300672, 0.036498], [3.300672, 0.036498], [3.300672, 0.036498],
    [3.300672, 0.036498], [3.300672, 0.460512], [3.300672, 0.460512],
    [3.300672, 0.460512], [3.300672, 0.460512], [3.300672, 0.886903],
    [3.300672, 0.886903], [3.300672, 0.886903], [3.300672, 0.886903],
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811],
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761], 
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811],
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165],
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811],
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761],
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811],
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165],
    [3.445864, -0.785757], [3.445864, -0.785757], [3.445864, -0.441109],
    [3.445864, -0.441109], [3.445864, -0.433601], [3.445864, -0.433601],
    [3.445864, 0.433601], [3.445864, 0.433601], [3.445864, 0.441109],
    [3.445864, 0.441109], [3.445864, 0.785757], [3.445864, 0.785757],
    [3.445864, -0.785757], [3.445864, -0.785757], [3.445864, -0.441109],
    [3.445864, -0.441109], [3.445864, -0.433601], [3.445864, -0.433601],
    [3.445864, 0.433601], [3.445864, 0.433601], [3.445864, 0.441109],
    [3.445864, 0.441109], [3.445864, 0.785757], [3.445864, 0.785757], 
    [3.605804, -0.921400], [3.605804, -0.388616], [3.605804, -0.000000],
    [3.605804, -0.000000], [3.605804, 0.388616], [3.605804, 0.921400],
    [3.612253, -0.768201], [3.612253, -0.640209], [3.612253, -0.000000], 
    [3.612253, -0.000000], [3.612253, 0.640209], [3.612253, 0.768201],
    [4.337574, -0.770031], [4.337574, -0.770031], [4.337574, -0.533155],
    [4.337574, -0.533155], [4.337574, -0.350426], [4.337574, -0.350426],
    [4.337574, 0.350426], [4.337574, 0.350426], [4.337574, 0.533155], 
    [4.337574, 0.533155], [4.337574, 0.770031], [4.337574, 0.770031],
    [4.470498, -0.720187], [4.470498, -0.720187], [4.470498, -0.137519],
    [4.470498, -0.137519], [4.470498, 0.137519], [4.470498, 0.137519],
    [4.470498, 0.680014], [4.470498, 0.680014], [4.470498, 0.680014],
    [4.470498, 0.680014], [4.470498, 0.720187], [4.470498, 0.720187],
    [4.652131, -0.761404], [4.652131, -0.648278], [4.652131, -0.000000],
    [4.652131, 0.000000], [4.652131, 0.648278], [4.652131, 0.761404],
    [4.716293, -0.704449], [4.716293, -0.704449], [4.716293, -0.644574],
    [4.716293, -0.644574], [4.716293, -0.297113], [4.716293, -0.297113],
    [4.716293, 0.297113], [4.716293, 0.297113], [4.716293, 0.644574],
    [4.716293, 0.644574], [4.716293, 0.704449], [4.716293, 0.704449], 
    [4.721225, -0.643901], [4.721225, -0.643901], [4.721225, -0.587757],
    [4.721225, -0.587757], [4.721225, -0.489830], [4.721225, -0.489830],
    [4.721225, 0.489830], [4.721225, 0.489830], [4.721225, 0.587757],
    [4.721225, 0.587757], [4.721225, 0.643901], [4.721225, 0.643901],
    [4.862916, -0.902655], [4.862916, -0.430366], [4.862916, -0.000000],
    [4.862916, 0.000000], [4.862916, 0.430366], [4.862916, 0.902655],
    [4.945890, -0.716181], [4.945890, -0.626607], [4.945890, -0.307326],
    [4.945890, 0.307326], [4.945890, 0.626607], [4.945890, 0.716181],
    [4.945890, -0.716181], [4.945890, -0.626607], [4.945890, -0.307326],
    [4.945890, 0.307326], [4.945890, 0.626607], [4.945890, 0.716181]
]
pp_len_cos = [[3.040000, 0.000000, -1.000000], [3.040000, 0.000000, -1.000000], [3.040000, 0.000000, -1.000000],
    [3.040000, 0.000000, -1.000000], [3.040000, 0.000000, -1.000000], [3.040000, 0.000000, -1.000000],
    [3.040000, 0.000000, -1.000000], [3.040000, 0.000000, -1.000000], [3.040000, -0.000000, 0.000000],
    [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000],
    [3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, -0.000000, 0.000000], [3.040000, -0.000000, 0.000000], [3.040000, 1.000000, 0.000000],
    [3.040000, 1.000000, 0.000000], [3.040000, 1.000000, 0.000000], [3.040000, 1.000000, 0.000000],
    [3.109274, -0.329049, 0.329049], [3.109274, -0.329049, 0.329049], [3.109274, -0.271289, 0.271289],
    [3.109274, -0.271289, 0.271289], [3.109274, 0.238984, -0.761016], [3.109274, 0.238984, -0.761016], 
    [3.109274, 0.271289, -0.271289], [3.109274, 0.271289, -0.271289], [3.109274, 0.307960, -0.692040],
    [3.109274, 0.307960, -0.692040], [3.109274, 0.329049, -0.329049], [3.109274, 0.329049, -0.329049],
    [3.109274, 0.373528, -0.373528], [3.109274, 0.373528, -0.373528], [3.109274, 0.373528, -0.373528], 
    [3.109274, 0.373528, -0.373528], [3.109274, 0.453055, -0.546945], [3.109274, 0.453055, -0.546945],
    [3.277766, -0.184231, 0.184231], [3.277766, -0.184231, 0.184231], [3.277766, -0.000000, 0.000000],
    [3.277766, -0.000000, 0.000000], [3.277766, 0.000000, -1.000000], [3.277766, 0.000000, -0.000000], 
    [3.277766, 0.000000, -0.000000], [3.277766, 0.035179, -0.964821], [3.277766, 0.964821, -0.035179],
    [3.277766, -0.184231, 0.184231], [3.277766, -0.184231, 0.184231], [3.277766, 0.000000, -1.000000], 
    [3.277766, -0.000000, 0.000000], [3.277766, 0.000000, -0.000000], [3.277766, 0.000000, -0.000000],
    [3.277766, -0.000000, 0.000000], [3.277766, 0.035179, -0.964821], [3.277766, 0.964821, -0.035179],
    [3.300672, -0.408430, 0.408430], [3.300672, -0.408430, 0.408430], [3.300672, -0.408430, 0.408430],
    [3.300672, -0.408430, 0.408430], [3.300672, -0.016808, 0.016808], [3.300672, -0.016808, 0.016808],
    [3.300672, -0.016808, 0.016808], [3.300672, -0.016808, 0.016808], [3.300672, 0.001332, -0.998668],
    [3.300672, 0.001332, -0.998668], [3.300672, 0.001332, -0.998668], [3.300672, 0.001332, -0.998668],
    [3.300672, 0.016808, -0.016808], [3.300672, 0.016808, -0.016808], [3.300672, 0.016808, -0.016808],
    [3.300672, 0.016808, -0.016808], [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370],
    [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370],
    [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.212072, -0.787928], [3.300672, 0.212072, -0.787928], [3.300672, 0.212072, -0.787928],
    [3.300672, 0.212072, -0.787928], [3.300672, 0.408430, -0.408430], [3.300672, 0.408430, -0.408430],
    [3.300672, 0.408430, -0.408430], [3.300672, 0.408430, -0.408430], [3.300672, 0.786596, -0.213404],
    [3.300672, 0.786596, -0.213404], [3.300672, 0.786596, -0.213404], [3.300672, 0.786596, -0.213404],
    [3.327418, -0.397501, 0.397501], [3.327418, -0.397501, 0.397501], [3.327418, -0.084401, 0.084401],
    [3.327418, -0.084401, 0.084401], [3.327418, 0.034137, -0.965863], [3.327418, 0.034137, -0.965863],
    [3.327418, 0.084401, -0.084401], [3.327418, 0.084401, -0.084401], [3.327418, 0.160773, -0.160773],
    [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773],
    [3.327418, 0.208676, -0.791324], [3.327418, 0.208676, -0.791324], [3.327418, 0.397501, -0.397501],
    [3.327418, 0.397501, -0.397501], [3.327418, 0.757187, -0.242813], [3.327418, 0.757187, -0.242813],
    [3.327418, -0.397501, 0.397501], [3.327418, -0.397501, 0.397501], [3.327418, -0.084401, 0.084401],
    [3.327418, -0.084401, 0.084401], [3.327418, 0.034137, -0.965863], [3.327418, 0.034137, -0.965863], 
    [3.327418, 0.084401, -0.084401], [3.327418, 0.084401, -0.084401], [3.327418, 0.160773, -0.160773],
    [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773],
    [3.327418, 0.208676, -0.791324], [3.327418, 0.208676, -0.791324], [3.327418, 0.397501, -0.397501],
    [3.327418, 0.397501, -0.397501], [3.327418, 0.757187, -0.242813], [3.327418, 0.757187, -0.242813],
    [3.445864, -0.346604, 0.346604], [3.445864, -0.346604, 0.346604], [3.445864, -0.340705, 0.340705],
    [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705],
    [3.445864, -0.191265, 0.191265], [3.445864, -0.191265, 0.191265], [3.445864, 0.188010, -0.811990], 
    [3.445864, 0.188010, -0.811990], [3.445864, 0.191265, -0.191265], [3.445864, 0.191265, -0.191265],
    [3.445864, 0.194577, -0.805423], [3.445864, 0.194577, -0.805423], [3.445864, 0.346604, -0.346604],
    [3.445864, 0.346604, -0.346604], [3.445864, 0.617413, -0.382587], [3.445864, 0.617413, -0.382587],
    [3.445864, -0.346604, 0.346604], [3.445864, -0.346604, 0.346604], [3.445864, -0.340705, 0.340705],
    [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], 
    [3.445864, -0.191265, 0.191265], [3.445864, -0.191265, 0.191265], [3.445864, 0.188010, -0.811990],
    [3.445864, 0.188010, -0.811990], [3.445864, 0.191265, -0.191265], [3.445864, 0.191265, -0.191265],
    [3.445864, 0.194577, -0.805423], [3.445864, 0.194577, -0.805423], [3.445864, 0.346604, -0.346604], 
    [3.445864, 0.346604, -0.346604], [3.445864, 0.617413, -0.382587], [3.445864, 0.617413, -0.382587], 
    [3.605804, 0.000000, -1.000000], [3.605804, 0.000000, -0.000000], [3.605804, 0.000000, -0.000000],
    [3.605804, -0.000000, 0.000000], [3.605804, -0.000000, 0.000000], [3.605804, 0.151022, -0.848978],
    [3.605804, 0.358071, -0.358071], [3.605804, 0.358071, -0.358071], [3.605804, 0.848978, -0.151022],
    [3.612253, -0.491809, 0.491809], [3.612253, -0.491809, 0.491809], [3.612253, 0.000000, -1.000000],
    [3.612253, 0.000000, -0.000000], [3.612253, -0.000000, 0.000000], [3.612253, -0.000000, 0.000000],
    [3.612253, 0.000000, -0.000000], [3.612253, 0.409868, -0.590132], [3.612253, 0.590132, -0.409868],
    [4.337574, -0.269839, 0.269839], [4.337574, -0.269839, 0.269839], [4.337574, -0.186831, 0.186831],
    [4.337574, -0.186831, 0.186831], [4.337574, 0.122799, -0.877201], [4.337574, 0.122799, -0.877201],
    [4.337574, 0.186831, -0.186831], [4.337574, 0.186831, -0.186831], [4.337574, 0.269839, -0.269839],
    [4.337574, 0.269839, -0.269839], [4.337574, 0.284254, -0.715746], [4.337574, 0.284254, -0.715746],
    [4.337574, 0.410546, -0.410546], [4.337574, 0.410546, -0.410546], [4.337574, 0.410546, -0.410546],
    [4.337574, 0.410546, -0.410546], [4.337574, 0.592947, -0.407053], [4.337574, 0.592947, -0.407053],
    [4.470498, -0.489737, 0.489737], [4.470498, -0.489737, 0.489737], [4.470498, -0.099039, 0.099039],
    [4.470498, -0.099039, 0.099039], [4.470498, -0.099039, 0.099039], [4.470498, -0.099039, 0.099039],
    [4.470498, -0.093515, 0.093515], [4.470498, -0.093515, 0.093515], [4.470498, 0.018911, -0.981089],
    [4.470498, 0.018911, -0.981089], [4.470498, 0.093515, -0.093515], [4.470498, 0.093515, -0.093515], 
    [4.470498, 0.462419, -0.537581], [4.470498, 0.462419, -0.537581], [4.470498, 0.489737, -0.489737],
    [4.470498, 0.489737, -0.489737], [4.470498, 0.518670, -0.481330], [4.470498, 0.518670, -0.481330],
    [4.652131, 0.000000, -1.000000], [4.652131, 0.000000, -0.000000], [4.652131, 0.000000, -0.000000],
    [4.652131, 0.000000, -0.000000], [4.652131, 0.000000, -0.000000], [4.652131, 0.420264, -0.579736],
    [4.652131, 0.493601, -0.493601], [4.652131, 0.493601, -0.493601], [4.652131, 0.579736, -0.420264],
    [4.716293, -0.454069, 0.454069], [4.716293, -0.454069, 0.454069], [4.716293, -0.191511, 0.191511],
    [4.716293, -0.191511, 0.191511], [4.716293, 0.088276, -0.911724], [4.716293, 0.088276, -0.911724],
    [4.716293, 0.191511, -0.191511], [4.716293, 0.191511, -0.191511], [4.716293, 0.209301, -0.209301],
    [4.716293, 0.209301, -0.209301], [4.716293, 0.209301, -0.209301], [4.716293, 0.209301, -0.209301],
    [4.716293, 0.415476, -0.584524], [4.716293, 0.415476, -0.584524], [4.716293, 0.454069, -0.454069],
    [4.716293, 0.454069, -0.454069], [4.716293, 0.496248, -0.503752], [4.716293, 0.496248, -0.503752],
    [4.721225, -0.378457, 0.378457], [4.721225, -0.378457, 0.378457], [4.721225, -0.315402, 0.315402],
    [4.721225, -0.315402, 0.315402], [4.721225, -0.287901, 0.287901], [4.721225, -0.287901, 0.287901],
    [4.721225, -0.287901, 0.287901], [4.721225, -0.287901, 0.287901], [4.721225, 0.239933, -0.760067], 
    [4.721225, 0.239933, -0.760067], [4.721225, 0.315402, -0.315402], [4.721225, 0.315402, -0.315402],
    [4.721225, 0.345459, -0.654541], [4.721225, 0.345459, -0.654541], [4.721225, 0.378457, -0.378457],
    [4.721225, 0.378457, -0.378457], [4.721225, 0.414608, -0.585392], [4.721225, 0.414608, -0.585392],
    [4.862916, -0.388472, 0.388472], [4.862916, -0.388472, 0.388472], [4.862916, -0.000000, 0.000000],
    [4.862916, -0.000000, 0.000000], [4.862916, 0.000000, -1.000000], [4.862916, 0.000000, -0.000000],
    [4.862916, 0.000000, -0.000000], [4.862916, 0.185215, -0.814785], [4.862916, 0.814785, -0.185215], 
    [4.945890, -0.448764, 0.448764], [4.945890, -0.448764, 0.448764], [4.945890, -0.220101, 0.220101],
    [4.945890, -0.220101, 0.220101], [4.945890, 0.094449, -0.905551], [4.945890, 0.192572, -0.192572],
    [4.945890, 0.192572, -0.192572], [4.945890, 0.392636, -0.607364], [4.945890, 0.512915, -0.487085], 
    [4.945890, -0.448764, 0.448764], [4.945890, -0.448764, 0.448764], [4.945890, -0.192572, 0.192572],
    [4.945890, -0.192572, 0.192572], [4.945890, 0.094449, -0.905551], [4.945890, 0.220101, -0.220101],
    [4.945890, 0.220101, -0.220101], [4.945890, 0.392636, -0.607364], [4.945890, 0.512915, -0.487085]
]

stp_len_cos = [
    [1.802903, -0.957343], [1.802903, -0.288955], [1.802903, -0.000000], 
    [1.802903, 0.000000], [1.802903, 0.288955], [1.802903, 0.957343], 
    [1.832680, -0.829386], [1.832680, -0.537018], [1.832680, -0.154044], 
    [1.832680, -0.829386], [1.832680, 0.154044], [1.832680, 0.154044], 
    [1.832680, 0.537018], [1.832680, 0.537018], [1.832680, 0.829386], 
    [1.852850, -0.703442], [1.852850, -0.000000], [1.852850, 0.710753], 
    [1.852850, -0.710753], [1.852850, 0.000000], [1.852850, 0.703442], 
    [1.939922, -0.985205], [1.939922, -0.171380], [1.939922, 0.000000], 
    [1.939922, 0.000000], [1.939922, 0.171380], [1.939922, 0.985205], 
    [1.978409, -0.768294], [1.978409, -0.202430], [1.978409, 0.607245], 
    [1.978409, -0.768294], [1.978409, -0.607245], [1.978409, -0.607245], 
    [1.978409, 0.202430], [1.978409, 0.202430], [1.978409, 0.768294], 
    [2.023449, -0.940314], [2.023449, -0.340308], [2.023449, -0.000000], 
    [2.023449, 0.000000], [2.023449, 0.340308], [2.023449, 0.940314], 
    [2.076942, -0.731845], [2.076942, -0.676109], [2.076942, -0.085323], 
    [2.076942, 0.085323], [2.076942, 0.676109], [2.076942, 0.731845], 
    [2.076942, -0.731845], [2.076942, -0.085323], [2.076942, 0.676109], 
    [3.256051, -0.865806], [3.256051, -0.865806], [3.256051, -0.466823], 
    [3.256051, -0.466823], [3.256051, -0.180157], [3.256051, -0.180157], 
    [3.256051, 0.180157], [3.256051, 0.180157], [3.256051, 0.466823], 
    [3.256051, 0.466823], [3.256051, 0.865806], [3.256051, 0.865806], 
    [3.347687, -0.984528], [3.347687, -0.175226], [3.347687, 0.000000], 
    [3.347687, 0.000000], [3.347687, 0.175226], [3.347687, 0.984528], 
    [3.385999, -0.000000], [3.385999, 0.596283], [3.385999, 0.802774], 
    [3.385999, -0.802774], [3.385999, -0.596283], [3.385999, 0.000000], 
    [3.394459, -0.945567], [3.394459, -0.325428], [3.394459, 0.000000], 
    [3.394459, 0.000000], [3.394459, 0.325428], [3.394459, 0.945567], 
    [3.424386, -0.887772], [3.424386, -0.887772], [3.424386, -0.443875], 
    [3.424386, -0.121800], [3.424386, -0.121800], [3.424386, 0.443875], 
    [3.424386, -0.443875], [3.424386, 0.121800], [3.424386, 0.121800], 
    [3.424386, 0.443875], [3.424386, 0.887772], [3.424386, 0.887772], 
    [3.500108, -0.741380], [3.500108, -0.741380], [3.500108, -0.434272], 
    [3.500108, 0.434272], [3.500108, 0.511628], [3.500108, 0.511628], 
    [3.500108, -0.511628], [3.500108, -0.511628], [3.500108, -0.434272], 
    [3.500108, 0.434272], [3.500108, 0.741380], [3.500108, 0.741380], 
    [3.534411, -0.860115], [3.534411, -0.860115], [3.534411, -0.488341], 
    [3.534411, -0.488341], [3.534411, -0.147396], [3.534411, -0.147396], 
    [3.534411, 0.147396], [3.534411, 0.147396], [3.534411, 0.488341], 
    [3.534411, 0.488341], [3.534411, 0.860115], [3.534411, 0.860115], 
    [3.550690, -0.927767], [3.550690, -0.373161], [3.550690, 0.000000], 
    [3.550690, 0.000000], [3.550690, 0.373161], [3.550690, 0.927767], 
    [3.560148, -0.853897], [3.560148, -0.853897], [3.560148, -0.369906], 
    [3.560148, -0.366101], [3.560148, 0.366101], [3.560148, 0.369906], 
    [3.572376, -0.755418], [3.572376, 0.000000], [3.572376, 0.655243], 
    [3.572376, -0.655243], [3.572376, 0.000000], [3.572376, 0.755418], 
    [3.606230, -0.842986], [3.606230, -0.842986], [3.606230, -0.529977], 
    [3.606230, -0.092192], [3.606230, 0.092192], [3.606230, 0.092192], 
    [3.606230, 0.529977], [3.606230, 0.529977], [3.606230, 0.842986], 
    [3.606230, -0.529977], [3.606230, -0.092192], [3.606230, 0.842986], 
    [3.651841, -0.832457], [3.651841, -0.832457], [3.651841, -0.521019], 
    [3.651841, -0.188561], [3.651841, 0.188561], [3.651841, 0.521019], 
    [4.233807, -0.476879], [4.233807, 0.359015], [4.233807, 0.802306], 
    [4.233807, -0.802306], [4.233807, -0.359015], [4.233807, 0.476879], 
    [4.331985, -0.999914], [4.331985, -0.013098], [4.331985, -0.000000], 
    [4.331985, 0.000000], [4.331985, 0.013098], [4.331985, 0.999914], 
    [4.411478, -0.774425], [4.411478, -0.774425], [4.411478, -0.530610], 
    [4.411478, -0.530610], [4.411478, -0.344556], [4.411478, 0.344556], 
    [4.411478, -0.344556], [4.411478, 0.344556], [4.411478, 0.530610], 
    [4.411478, 0.530610], [4.411478, 0.774425], [4.411478, 0.774425], 
    [4.413475, -0.819302], [4.413475, -0.819302], [4.413475, -0.458402], 
    [4.413475, -0.458402], [4.413475, -0.344400], [4.413475, -0.344400], 
    [4.413475, 0.344400], [4.413475, 0.344400], [4.413475, 0.458402], 
    [4.413475, 0.458402], [4.413475, 0.819302], [4.413475, 0.819302], 
    [4.482487, -0.937823], [4.482487, 0.074170], [4.482487, 0.339097], 
    [4.482487, -0.937823], [4.482487, -0.339097], [4.482487, -0.339097], 
    [4.482487, -0.074170], [4.482487, -0.074170], [4.482487, 0.074170], 
    [4.482487, 0.339097], [4.482487, 0.937823], [4.482487, 0.937823], 
    [4.522013, -0.728855], [4.522013, -0.728855], [4.522013, -0.672267], 
    [4.522013, -0.672267], [4.522013, -0.129721], [4.522013, -0.129721], 
    [4.522013, 0.129721], [4.522013, 0.129721], [4.522013, 0.672267], 
    [4.522013, 0.672267], [4.522013, 0.728855], [4.522013, 0.728855], 
    [4.530808, -0.929707], [4.530808, -0.929707], [4.530808, -0.335481], 
    [4.530808, -0.335481], [4.530808, -0.151981], [4.530808, -0.151981], 
    [4.530808, 0.151981], [4.530808, 0.151981], [4.530808, 0.335481], 
    [4.530808, 0.335481], [4.530808, 0.929707], [4.530808, 0.929707], 
    [4.550449, 0.443696], [4.550449, 0.597346], [4.550449, 0.668066], 
    [4.550449, -0.668066], [4.550449, -0.668066], [4.550449, -0.597346], 
    [4.550449, -0.597346], [4.550449, -0.443696], [4.550449, -0.443696], 
    [4.550449, 0.443696], [4.550449, 0.597346], [4.550449, 0.668066], 
    [4.556748, -0.704381], [4.556748, -0.704381], [4.556748, -0.667142], 
    [4.556748, -0.667142], [4.556748, -0.242421], [4.556748, -0.242421], 
    [4.556748, 0.242421], [4.556748, 0.242421], [4.556748, 0.667142], 
    [4.556748, 0.667142], [4.556748, 0.704381], [4.556748, 0.704381], 
    [4.611475, -0.659225], [4.611475, -0.659225], [4.611475, -0.537423], 
    [4.611475, -0.525927], [4.611475, 0.525927], [4.611475, 0.537423], 
    [4.627954, -0.785809], [4.627954, -0.524054], [4.627954, -0.328439], 
    [4.627954, -0.785809], [4.627954, -0.524054], [4.627954, -0.328439], 
    [4.627954, 0.328439], [4.627954, 0.328439], [4.627954, 0.524054], 
    [4.627954, 0.524054], [4.627954, 0.785809], [4.627954, 0.785809], 
    [4.652001, -0.909447], [4.652001, -0.909447], [4.652001, -0.326741], 
    [4.652001, -0.326741], [4.652001, -0.257190], [4.652001, -0.257190], 
    [4.652001, 0.257190], [4.652001, 0.257190], [4.652001, 0.326741], 
    [4.652001, 0.326741], [4.652001, 0.909447], [4.652001, 0.909447], 
    [4.673533, -0.975707], [4.673533, -0.975707], [4.673533, -0.210586], 
    [4.673533, -0.210586], [4.673533, -0.060407], [4.673533, -0.060407], 
    [4.673533, 0.060407], [4.673533, 0.210586], [4.673533, 0.975707], 
    [4.674291, -0.704751], [4.674291, -0.650366], [4.674291, -0.650366], 
    [4.674291, -0.283461], [4.674291, 0.283461], [4.674291, 0.704751], 
    [4.690785, -0.648079], [4.690785, -0.575306], [4.690785, -0.575306], 
    [4.690785, 0.499015], [4.690785, 0.499015], [4.690785, 0.648079], 
    [4.690785, -0.648079], [4.690785, -0.499015], [4.690785, -0.499015], 
    [4.690785, 0.575306], [4.690785, 0.575306], [4.690785, 0.648079], 
    [4.732579, -0.963534], [4.732579, -0.963534], [4.732579, -0.253853], 
    [4.732579, -0.084624], [4.732579, -0.084624], [4.732579, 0.084624], 
    [4.732579, 0.253853], [4.732579, 0.253853], [4.732579, 0.963534], 
    [4.774609, -0.955052], [4.774609, -0.955052], [4.774609, -0.294105], 
    [4.774609, -0.294105], [4.774609, -0.037115], [4.774609, 0.037115], 
    [4.774609, 0.037115], [4.774609, 0.294105], [4.774609, 0.955052], 
    [4.786891, -0.926197], [4.786891, -0.926197], [4.786891, -0.317534], 
    [4.786891, -0.317534], [4.786891, -0.203299], [4.786891, -0.203299], 
    [4.786891, 0.203299], [4.786891, 0.203299], [4.786891, 0.317534], 
    [4.786891, 0.317534], [4.786891, 0.926197], [4.786891, 0.926197], 
]
# convert to numpy array for tuple slicing
ss_bond_len = np.array(ss_bond_len)
sp_len_cos = np.array(sp_len_cos)
pp_len_cos = np.array(pp_len_cos)
stp_len_cos = np.array(stp_len_cos)

# s: -1eV, p: 1eV, st: 3eV. The value is set arbitarily 
# here but follwing the incresing physical trend.
onsite = [
    0, 0, 0, 0, 
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4]
hopping = np.zeros((856,))
unit_conv = 3.80998212 # hbar/2mA to eV

vss_start = -1
vss_end = 0
vss_step = 0.2

vsp_start = 0
vsp_end = 2.5
vsp_step = 0.5

vpp_s_start = 2
vpp_s_end = 4.5
vpp_s_step = 0.5

vpp_p_start = 0
vpp_p_end = 2.5
vpp_p_step = 0.5

vstp_start = 1
vstp_end = 3.5
vstp_step = 0.5

n = 4

vss_list = np.arange(vss_start, vss_end, vss_step)
vsp_list = np.arange(vsp_start, vsp_end, vsp_step)
vpp_s_list = np.arange(vpp_s_start, vpp_s_end, vpp_s_step)
vpp_p_list = np.arange(vpp_p_start, vpp_p_end, vpp_p_step)
vstp_list = np.arange(vstp_start, vstp_end, vstp_step)

target_emass = 0.31
threshold = 0.05

outfile = open("Emass_scan.csv", "w+")
good_scan_outfile = open("good_Emass_scan.csv", "w+")

for vss in vss_list:
    for vsp in vsp_list:
        for vpp_s in vpp_s_list:
            for vpp_p in vpp_p_list:
                for vstp in vstp_list:
                    vss = -0.68
                    vsp = 0.62
                    vpp_s = 1.5
                    vpp_p = 0.2
                    vstp = 0.9
                    n = 4
                    print("vss=%.2f vsp=%.2f vpp_s=%.2f vpp_p=%.2f vstp=%.2f started\n" %(vss, vsp, vpp_s, vpp_p, vstp))
                    # calculate ss hopping
                    hopping[0:34] = vss * pow(np.divide(lat, ss_bond_len), n)
                    # calculate sp hopping
                    hopping[34:226] = vsp * sp_len_cos[:,1] * pow(np.divide(lat, sp_len_cos[:,0]), n)
                    # calculate pp hopping
                    hopping[226:520] = (vpp_s * pp_len_cos[:,1] + vpp_p * pp_len_cos[:,2]) * pow(np.divide(lat, pp_len_cos[:,0]), n)
                    # calculate stp hopping
                    hopping[520:856] = vstp * stp_len_cos[:,1] * pow(np.divide(lat, stp_len_cos[:,0]), n)
                    my_class = Ga2O3_Ga_s_Class(onsite, hopping)
                    outfile.writelines('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t' %(vss, vsp, vpp_s, vpp_p, vstp) + my_class.getMassString())
                    my_class.savePlot('Emass_pdf/%.3f_%.3f_%.3f_%.3f_%.3f.png' %(vss, vsp, vpp_s, vpp_p, vstp))

                    if(np.abs(my_class.emass_0 - target_emass) < threshold):
                        good_scan_outfile.writelines('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t' %(vss, vsp, vpp_s, vpp_p, vstp) + my_class.getMassString())
                    del my_class
