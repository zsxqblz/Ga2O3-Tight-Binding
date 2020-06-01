#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_s_Class import *

lat = 3.293087727
ss_bond_len = [3.040000, 3.040000, 3.040000, 3.040000, 3.109274, 3.109274, 3.277766, 
    3.277766, 3.300672, 3.300672, 3.300672, 3.300672, 3.327418, 
    3.327418, 3.327418, 3.327418, 3.445864, 3.445864, 3.445864, 
    3.445864, 3.605804, 3.612253, 4.337574, 4.337574, 4.470498, 
    4.470498, 4.652131, 4.716293, 4.716293, 4.721225, 4.721225, 
    4.862916, 4.945890, 4.945890,
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
    [3.277766, -0.000000], [3.277766, 0.187560], [3.277766, 0.982253], 
    [3.277766, -0.982253], [3.277766, -0.187560], [3.277766, -0.000000], 
    [3.277766, -0.000000], [3.277766, 0.187560], [3.277766, 0.982253], 
    [3.300672, -0.886903], [3.300672, -0.886903], [3.300672, -0.460512], 
    [3.300672, -0.460512], [3.300672, -0.036498], [3.300672, -0.036498], 
    [3.300672, 0.036498], [3.300672, 0.036498], [3.300672, 0.460512], 
    [3.300672, 0.460512], [3.300672, 0.886903], [3.300672, 0.886903], 
    [3.300672, -0.886903], [3.300672, -0.886903], [3.300672, -0.460512], 
    [3.300672, -0.460512], [3.300672, -0.036498], [3.300672, -0.036498], 
    [3.300672, 0.036498], [3.300672, 0.036498], [3.300672, 0.460512], 
    [3.300672, 0.460512], [3.300672, 0.886903], [3.300672, 0.886903], 
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811], 
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761], 
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811], 
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165], 
    [3.327418, -0.870165], [3.327418, -0.870165], [3.327418, -0.456811], 
    [3.327418, -0.456811], [3.327418, -0.184761], [3.327418, -0.184761], 
    [3.327418, 0.184761], [3.327418, 0.184761], [3.327418, 0.456811], 
    [3.327418, 0.456811], [3.327418, 0.870165], [3.327418, 0.870165], 
    [3.445864, -0.785757], [3.445864, -0.785757], [3.445864, -0.785757], 
    [3.445864, -0.785757], [3.445864, -0.441109], [3.445864, -0.441109], 
    [3.445864, -0.441109], [3.445864, -0.441109], [3.445864, -0.433601], 
    [3.445864, -0.433601], [3.445864, -0.433601], [3.445864, -0.433601], 
    [3.445864, 0.433601], [3.445864, 0.433601], [3.445864, 0.433601], 
    [3.445864, 0.433601], [3.445864, 0.441109], [3.445864, 0.441109], 
    [3.445864, 0.441109], [3.445864, 0.441109], [3.445864, 0.785757], 
    [3.445864, 0.785757], [3.445864, 0.785757], [3.445864, 0.785757], 
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
    [4.652131, -0.000000], [4.652131, 0.648278], [4.652131, 0.761404], 
    [4.716293, -0.704449], [4.716293, -0.704449], [4.716293, -0.644574], 
    [4.716293, -0.644574], [4.716293, -0.297113], [4.716293, -0.297113], 
    [4.716293, 0.297113], [4.716293, 0.297113], [4.716293, 0.644574], 
    [4.716293, 0.644574], [4.716293, 0.704449], [4.716293, 0.704449], 
    [4.721225, -0.643901], [4.721225, -0.643901], [4.721225, -0.587757], 
    [4.721225, -0.587757], [4.721225, -0.489830], [4.721225, -0.489830], 
    [4.721225, 0.489830], [4.721225, 0.489830], [4.721225, 0.587757], 
    [4.721225, 0.587757], [4.721225, 0.643901], [4.721225, 0.643901], 
    [4.862916, -0.902655], [4.862916, -0.430366], [4.862916, -0.000000], 
    [4.862916, -0.000000], [4.862916, 0.430366], [4.862916, 0.902655], 
    [4.945890, -0.716181], [4.945890, -0.716181], [4.945890, -0.626607], 
    [4.945890, -0.626607], [4.945890, -0.307326], [4.945890, -0.307326], 
    [4.945890, 0.307326], [4.945890, 0.307326], [4.945890, 0.626607], 
    [4.945890, 0.626607], [4.945890, 0.716181], [4.945890, 0.716181],
]
pp_len_cos = [[3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000], 
    [3.040000, -0.000000, 0.000000], [3.040000, 0.000000, -0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, -0.000000, 0.000000], [3.040000, -0.000000, 0.000000], [3.040000, -0.000000, 0.000000], 
    [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], 
    [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], 
    [3.040000, 0.000000, 1.000000], [3.040000, 0.000000, 1.000000], [3.040000, 1.000000, 0.000000], 
    [3.040000, 1.000000, 0.000000], [3.040000, 1.000000, 0.000000], [3.040000, 1.000000, 0.000000], 
    [3.109274, -0.329049, 0.329049], [3.109274, -0.329049, 0.329049], [3.109274, -0.271289, 0.271289], 
    [3.109274, -0.271289, 0.271289], [3.109274, 0.238984, 0.761016], [3.109274, 0.238984, 0.761016], 
    [3.109274, 0.271289, -0.271289], [3.109274, 0.271289, -0.271289], [3.109274, 0.307960, 0.692040], 
    [3.109274, 0.307960, 0.692040], [3.109274, 0.329049, -0.329049], [3.109274, 0.329049, -0.329049], 
    [3.109274, 0.373528, -0.373528], [3.109274, 0.373528, -0.373528], [3.109274, 0.373528, -0.373528], 
    [3.109274, 0.373528, -0.373528], [3.109274, 0.453055, 0.546945], [3.109274, 0.453055, 0.546945], 
    [3.277766, -0.184231, 0.184231], [3.277766, -0.184231, 0.184231], [3.277766, 0.000000, -0.000000], 
    [3.277766, -0.000000, 0.000000], [3.277766, -0.000000, 0.000000], [3.277766, 0.000000, -0.000000], 
    [3.277766, 0.000000, 1.000000], [3.277766, 0.035179, 0.964821], [3.277766, 0.964821, 0.035179], 
    [3.277766, -0.184231, 0.184231], [3.277766, -0.184231, 0.184231], [3.277766, -0.000000, 0.000000], 
    [3.277766, 0.000000, -0.000000], [3.277766, 0.000000, -0.000000], [3.277766, -0.000000, 0.000000], 
    [3.277766, 0.000000, 1.000000], [3.277766, 0.035179, 0.964821], [3.277766, 0.964821, 0.035179], 
    [3.300672, -0.408430, 0.408430], [3.300672, -0.408430, 0.408430], [3.300672, -0.016808, 0.016808], 
    [3.300672, -0.016808, 0.016808], [3.300672, 0.001332, 0.998668], [3.300672, 0.001332, 0.998668], 
    [3.300672, 0.016808, -0.016808], [3.300672, 0.016808, -0.016808], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.212072, 0.787928], [3.300672, 0.212072, 0.787928], [3.300672, 0.408430, -0.408430], 
    [3.300672, 0.408430, -0.408430], [3.300672, 0.786596, 0.213404], [3.300672, 0.786596, 0.213404], 
    [3.300672, -0.408430, 0.408430], [3.300672, -0.408430, 0.408430], [3.300672, -0.016808, 0.016808], 
    [3.300672, -0.016808, 0.016808], [3.300672, 0.001332, 0.998668], [3.300672, 0.001332, 0.998668], 
    [3.300672, 0.016808, -0.016808], [3.300672, 0.016808, -0.016808], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], [3.300672, 0.032370, -0.032370], 
    [3.300672, 0.212072, 0.787928], [3.300672, 0.212072, 0.787928], [3.300672, 0.408430, -0.408430], 
    [3.300672, 0.408430, -0.408430], [3.300672, 0.786596, 0.213404], [3.300672, 0.786596, 0.213404], 
    [3.327418, -0.397501, 0.397501], [3.327418, -0.397501, 0.397501], [3.327418, -0.084401, 0.084401], 
    [3.327418, -0.084401, 0.084401], [3.327418, 0.034137, 0.965863], [3.327418, 0.034137, 0.965863], 
    [3.327418, 0.084401, -0.084401], [3.327418, 0.084401, -0.084401], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.208676, 0.791324], [3.327418, 0.208676, 0.791324], [3.327418, 0.397501, -0.397501], 
    [3.327418, 0.397501, -0.397501], [3.327418, 0.757187, 0.242813], [3.327418, 0.757187, 0.242813], 
    [3.327418, -0.397501, 0.397501], [3.327418, -0.397501, 0.397501], [3.327418, -0.084401, 0.084401], 
    [3.327418, -0.084401, 0.084401], [3.327418, 0.034137, 0.965863], [3.327418, 0.034137, 0.965863], 
    [3.327418, 0.084401, -0.084401], [3.327418, 0.084401, -0.084401], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], [3.327418, 0.160773, -0.160773], 
    [3.327418, 0.208676, 0.791324], [3.327418, 0.208676, 0.791324], [3.327418, 0.397501, -0.397501], 
    [3.327418, 0.397501, -0.397501], [3.327418, 0.757187, 0.242813], [3.327418, 0.757187, 0.242813], 
    [3.445864, -0.346604, 0.346604], [3.445864, -0.346604, 0.346604], [3.445864, -0.346604, 0.346604], 
    [3.445864, -0.346604, 0.346604], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], 
    [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], 
    [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], [3.445864, -0.340705, 0.340705], 
    [3.445864, -0.191265, 0.191265], [3.445864, -0.191265, 0.191265], [3.445864, -0.191265, 0.191265], 
    [3.445864, -0.191265, 0.191265], [3.445864, 0.188010, 0.811990], [3.445864, 0.188010, 0.811990], 
    [3.445864, 0.188010, 0.811990], [3.445864, 0.188010, 0.811990], [3.445864, 0.191265, -0.191265], 
    [3.445864, 0.191265, -0.191265], [3.445864, 0.191265, -0.191265], [3.445864, 0.191265, -0.191265], 
    [3.445864, 0.194577, 0.805423], [3.445864, 0.194577, 0.805423], [3.445864, 0.194577, 0.805423], 
    [3.445864, 0.194577, 0.805423], [3.445864, 0.346604, -0.346604], [3.445864, 0.346604, -0.346604], 
    [3.445864, 0.346604, -0.346604], [3.445864, 0.346604, -0.346604], [3.445864, 0.617413, 0.382587], 
    [3.445864, 0.617413, 0.382587], [3.445864, 0.617413, 0.382587], [3.445864, 0.617413, 0.382587], 
    [3.605804, -0.000000, 0.000000], [3.605804, -0.000000, 0.000000], [3.605804, 0.000000, -0.000000], 
    [3.605804, 0.000000, -0.000000], [3.605804, 0.000000, 1.000000], [3.605804, 0.151022, 0.848978], 
    [3.605804, 0.358071, -0.358071], [3.605804, 0.358071, -0.358071], [3.605804, 0.848978, 0.151022], 
    [3.612253, -0.491809, 0.491809], [3.612253, -0.491809, 0.491809], [3.612253, -0.000000, 0.000000], 
    [3.612253, 0.000000, -0.000000], [3.612253, 0.000000, -0.000000], [3.612253, -0.000000, 0.000000], 
    [3.612253, 0.000000, 1.000000], [3.612253, 0.409868, 0.590132], [3.612253, 0.590132, 0.409868], 
    [4.337574, -0.269839, 0.269839], [4.337574, -0.269839, 0.269839], [4.337574, -0.186831, 0.186831], 
    [4.337574, -0.186831, 0.186831], [4.337574, 0.122799, 0.877201], [4.337574, 0.122799, 0.877201], 
    [4.337574, 0.186831, -0.186831], [4.337574, 0.186831, -0.186831], [4.337574, 0.269839, -0.269839], 
    [4.337574, 0.269839, -0.269839], [4.337574, 0.284254, 0.715746], [4.337574, 0.284254, 0.715746], 
    [4.337574, 0.410546, -0.410546], [4.337574, 0.410546, -0.410546], [4.337574, 0.410546, -0.410546], 
    [4.337574, 0.410546, -0.410546], [4.337574, 0.592947, 0.407053], [4.337574, 0.592947, 0.407053], 
    [4.470498, -0.489737, 0.489737], [4.470498, -0.489737, 0.489737], [4.470498, -0.099039, 0.099039], 
    [4.470498, -0.099039, 0.099039], [4.470498, -0.099039, 0.099039], [4.470498, -0.099039, 0.099039], 
    [4.470498, -0.093515, 0.093515], [4.470498, -0.093515, 0.093515], [4.470498, 0.018911, 0.981089], 
    [4.470498, 0.018911, 0.981089], [4.470498, 0.093515, -0.093515], [4.470498, 0.093515, -0.093515], 
    [4.470498, 0.462419, 0.537581], [4.470498, 0.462419, 0.537581], [4.470498, 0.489737, -0.489737], 
    [4.470498, 0.489737, -0.489737], [4.470498, 0.518670, 0.481330], [4.470498, 0.518670, 0.481330], 
    [4.652131, -0.000000, 0.000000], [4.652131, -0.000000, 0.000000], [4.652131, 0.000000, -0.000000], 
    [4.652131, 0.000000, -0.000000], [4.652131, 0.000000, 1.000000], [4.652131, 0.420264, 0.579736], 
    [4.652131, 0.493601, -0.493601], [4.652131, 0.493601, -0.493601], [4.652131, 0.579736, 0.420264], 
    [4.716293, -0.454069, 0.454069], [4.716293, -0.454069, 0.454069], [4.716293, -0.191511, 0.191511], 
    [4.716293, -0.191511, 0.191511], [4.716293, 0.088276, 0.911724], [4.716293, 0.088276, 0.911724], 
    [4.716293, 0.191511, -0.191511], [4.716293, 0.191511, -0.191511], [4.716293, 0.209301, -0.209301], 
    [4.716293, 0.209301, -0.209301], [4.716293, 0.209301, -0.209301], [4.716293, 0.209301, -0.209301], 
    [4.716293, 0.415476, 0.584524], [4.716293, 0.415476, 0.584524], [4.716293, 0.454069, -0.454069], 
    [4.716293, 0.454069, -0.454069], [4.716293, 0.496248, 0.503752], [4.716293, 0.496248, 0.503752], 
    [4.721225, -0.378457, 0.378457], [4.721225, -0.378457, 0.378457], [4.721225, -0.315402, 0.315402], 
    [4.721225, -0.315402, 0.315402], [4.721225, -0.287901, 0.287901], [4.721225, -0.287901, 0.287901], 
    [4.721225, -0.287901, 0.287901], [4.721225, -0.287901, 0.287901], [4.721225, 0.239933, 0.760067], 
    [4.721225, 0.239933, 0.760067], [4.721225, 0.315402, -0.315402], [4.721225, 0.315402, -0.315402], 
    [4.721225, 0.345459, 0.654541], [4.721225, 0.345459, 0.654541], [4.721225, 0.378457, -0.378457], 
    [4.721225, 0.378457, -0.378457], [4.721225, 0.414608, 0.585392], [4.721225, 0.414608, 0.585392], 
    [4.862916, -0.388472, 0.388472], [4.862916, -0.388472, 0.388472], [4.862916, 0.000000, -0.000000], 
    [4.862916, -0.000000, 0.000000], [4.862916, -0.000000, 0.000000], [4.862916, 0.000000, -0.000000], 
    [4.862916, 0.000000, 1.000000], [4.862916, 0.185215, 0.814785], [4.862916, 0.814785, 0.185215], 
    [4.945890, -0.448764, 0.448764], [4.945890, -0.448764, 0.448764], [4.945890, -0.448764, 0.448764], 
    [4.945890, -0.448764, 0.448764], [4.945890, -0.220101, 0.220101], [4.945890, -0.220101, 0.220101], 
    [4.945890, -0.192572, 0.192572], [4.945890, -0.192572, 0.192572], [4.945890, 0.094449, 0.905551], 
    [4.945890, 0.094449, 0.905551], [4.945890, 0.192572, -0.192572], [4.945890, 0.192572, -0.192572], 
    [4.945890, 0.220101, -0.220101], [4.945890, 0.220101, -0.220101], [4.945890, 0.392636, 0.607364], 
    [4.945890, 0.392636, 0.607364], [4.945890, 0.512915, 0.487085], [4.945890, 0.512915, 0.487085],
]
# convert to numpy array for tuple slicing
ss_bond_len = np.array(ss_bond_len)
sp_len_cos = np.array(sp_len_cos)
pp_len_cos = np.array(pp_len_cos)

wannier_band_raw = np.loadtxt('wannier_band.dat', dtype=float)
# shift minimum to 0
wannier_band_raw[:,1] = wannier_band_raw[:,1] - np.min(wannier_band_raw[:,1])
wannier_band = []
# store different bands into different dimensions
wannier_band.append(wannier_band_raw[0:532,1])
wannier_band.append(wannier_band_raw[533:1065,1])
wannier_band.append(wannier_band_raw[1066:1598,1])
wannier_band.append(wannier_band_raw[1599:2131,1])

s_onite = 0
p_onsite = 2
onsite = [
    s_onite, s_onite, s_onite, s_onite, 
    p_onsite, p_onsite, p_onsite, p_onsite,
    p_onsite, p_onsite, p_onsite, p_onsite,
    p_onsite, p_onsite, p_onsite, p_onsite,
    ]
hopping = np.zeros((532,))
unit_conv = 3.80998212 # hbar/2mA to eV

vss = -0.75
vsp = 0.75
vpp_s = 1
vpp_p = 0
n = 4

print("vss=%.2f vsp=%.2f vpp_s=%.2f vpp_p=%.2f started\n" %(vss, vsp, vpp_s, vpp_p))
# calculate ss hopping
hopping[0:34] = vss * pow(np.divide(lat, ss_bond_len), n)
# calculate sp hopping
hopping[34:226] = vsp * sp_len_cos[:,1] * pow(np.divide(lat, sp_len_cos[:,0]), n)
# calculate pp hopping
hopping[226:520] = (vpp_s * pp_len_cos[:,1] + vpp_p * pp_len_cos[:,2]) * pow(np.divide(lat, pp_len_cos[:,0]), n)
my_class = Ga2O3_Ga_s_Class(onsite, hopping, vss, n)

fig1, ax1 = plt.subplots()
# specify horizontal axis details
# set range of horizontal axis
ax1.set_xlim(my_class.k_node[0], my_class.k_node[-1])
ax1.set_ylim(0, 12)
# put tickmarks and labels at node positions
ax1.set_xticks(my_class.k_node)
ax1.set_xticklabels(my_class.label)
# add vertical lines at node positions
for n in range(len(my_class.k_node)):
    ax1.axvline(x=my_class.k_node[n], linewidth=0.5, color='k')
# put title
ax1.set_xlabel("Path in k-space")
ax1.set_ylabel("Band energy (eV)")

band_num = range(4)
# plot bands specified by band_num
for i in band_num:
    if(i == 3):
        ax1.plot(my_class.k_dist, my_class.evals[i], 'b-', label='TB')
    else:
        ax1.plot(my_class.k_dist, my_class.evals[i], 'b-')
    if(i == 3):
        ax1.plot(my_class.k_dist, wannier_band[i], 'g--', label='DFT')
    else:
        ax1.plot(my_class.k_dist, wannier_band[i], 'g--')
ax1.legend()

# make an PDF figure of a plot
fig1.tight_layout()
fig1.savefig('200522/sp_TB_wannier.png')
plt.close(fig1)

fig1, ax1 = plt.subplots()
# specify horizontal axis details
# set range of horizontal axis
ax1.set_xlim(my_class.k_node[0], my_class.k_node[-1])
ax1.set_ylim(0, 20)
# put tickmarks and labels at node positions
ax1.set_xticks(my_class.k_node)
ax1.set_xticklabels(my_class.label)
# add vertical lines at node positions
for n in range(len(my_class.k_node)):
    ax1.axvline(x=my_class.k_node[n], linewidth=0.5, color='k')
# put title
ax1.set_xlabel("Path in k-space")
ax1.set_ylabel("Band energy (eV)")

band_num = range(16)
# plot bands specified by band_num
for i in band_num:
    if(i < 4):
        ax1.plot(my_class.k_dist, my_class.evals[i], 'm-')
    else:
        ax1.plot(my_class.k_dist, my_class.evals[i], 'b-')

# make an PDF figure of a plot
fig1.tight_layout()
fig1.savefig('200522/spt_TB.png')
plt.close(fig1)

del my_class
