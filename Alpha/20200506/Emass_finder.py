#!/usr/bin/env python

from __future__ import division
import numpy as np
from Ga2O3_Ga_stp_Class import *

lat = 3.293087727
ss_bond_len = [2.831676,
                2.831676, 2.937101, 2.937101, 2.937101, 2.937101, 2.937101,
                2.937101, 3.314270, 3.314270, 3.314270, 3.314270, 3.314270, 
                3.314270, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 
                3.645199, 3.645199, 3.645199, 3.645199, 3.645199, 3.645199,
                3.645199, 3.884824, 3.884824, 4.982500, 4.982500, 4.982500,
                4.982500, 4.982500, 4.982500, 4.982500, 4.982500, 4.982500,
                4.982500, 4.982500, 4.982500]

sp_len_cos = [[2.831676, -1.000000], [2.831676, -0.000000], [2.831676, -0.000000],
                [2.831676, -0.000000], [2.831676, 0.000000], [2.831676, 1.000000],
                [2.831676, -1.000000], [2.831676, -0.000000], [2.831676, -0.000000],
                [2.831676, -0.000000], [2.831676, 0.000000], [2.831676, 1.000000],
                [2.937101, -0.979417], [2.937101, -0.979417], [2.937101, -0.201846],
                [2.937101, -0.201846], [2.937101, -0.000000], [2.937101, -0.000000],
                [2.937101, -0.000000], [2.937101, 0.000000], [2.937101, 0.201846],
                [2.937101, 0.201846], [2.937101, 0.979417], [2.937101, 0.979417],
                [2.937101, -0.848200], [2.937101, -0.489709], [2.937101, -0.201846],
                [2.937101, 0.201846], [2.937101, 0.489709], [2.937101, 0.848200],
                [2.937101, -0.848200], [2.937101, -0.848200], [2.937101, -0.489709],
                [2.937101, -0.489709], [2.937101, -0.201846], [2.937101, -0.201846],
                [2.937101, 0.201846], [2.937101, 0.201846], [2.937101, 0.489709], 
                [2.937101, 0.489709], [2.937101, 0.848200], [2.937101, 0.848200], 
                [2.937101, -0.848200], [2.937101, -0.489709], [2.937101, -0.201846], 
                [2.937101, 0.201846], [2.937101, 0.489709], [2.937101, 0.848200],
                [3.314270, -0.867958], [3.314270, -0.496637], [3.314270, -0.000000],
                [3.314270, -0.000000], [3.314270, 0.496637], [3.314270, 0.867958],
                [3.314270, -0.867958], [3.314270, -0.496637], [3.314270, -0.000000],
                [3.314270, -0.000000], [3.314270, 0.496637], [3.314270, 0.867958], 
                [3.314270, -0.751674], [3.314270, -0.751674], [3.314270, -0.496637],
                [3.314270, -0.496637], [3.314270, -0.433979], [3.314270, -0.433979],
                [3.314270, 0.433979], [3.314270, 0.433979], [3.314270, 0.496637],
                [3.314270, 0.496637], [3.314270, 0.751674], [3.314270, 0.751674],
                [3.314270, -0.751674], [3.314270, -0.751674], [3.314270, -0.496637],
                [3.314270, -0.496637], [3.314270, -0.433979], [3.314270, -0.433979],
                [3.314270, 0.433979], [3.314270, 0.433979], [3.314270, 0.496637],
                [3.314270, 0.496637], [3.314270, 0.751674], [3.314270, 0.751674],
                [3.645199, -0.789161], [3.645199, -0.614187], [3.645199, -0.000000],
                [3.645199, -0.000000], [3.645199, 0.614187], [3.645199, 0.789161],
                [3.645199, -0.789161], [3.645199, -0.789161], [3.645199, -0.789161],
                [3.645199, -0.614187], [3.645199, -0.614187], [3.645199, -0.614187], 
                [3.645199, -0.000000], [3.645199, -0.000000], [3.645199, -0.000000],
                [3.645199, -0.000000], [3.645199, -0.000000], [3.645199, -0.000000], 
                [3.645199, 0.614187], [3.645199, 0.614187], [3.645199, 0.614187],
                [3.645199, 0.789161], [3.645199, 0.789161], [3.645199, 0.789161],
                [3.645199, -0.683433], [3.645199, -0.683433], [3.645199, -0.683433],
                [3.645199, -0.683433], [3.645199, -0.614187], [3.645199, -0.614187],
                [3.645199, -0.614187], [3.645199, -0.614187], [3.645199, -0.394580],
                [3.645199, -0.394580], [3.645199, -0.394580], [3.645199, -0.394580],
                [3.645199, 0.394580], [3.645199, 0.394580], [3.645199, 0.394580],
                [3.645199, 0.394580], [3.645199, 0.614187], [3.645199, 0.614187],
                [3.645199, 0.614187], [3.645199, 0.614187], [3.645199, 0.683433],
                [3.645199, 0.683433], [3.645199, 0.683433], [3.645199, 0.683433],
                [3.645199, -0.683433], [3.645199, -0.683433], [3.645199, -0.683433], 
                [3.645199, -0.683433], [3.645199, -0.614187], [3.645199, -0.614187],
                [3.645199, -0.614187], [3.645199, -0.614187], [3.645199, -0.394580],
                [3.645199, -0.394580], [3.645199, -0.394580], [3.645199, -0.394580],
                [3.645199, 0.394580], [3.645199, 0.394580], [3.645199, 0.394580],
                [3.645199, 0.394580], [3.645199, 0.614187], [3.645199, 0.614187],
                [3.645199, 0.614187], [3.645199, 0.614187], [3.645199, 0.683433],
                [3.645199, 0.683433], [3.645199, 0.683433], [3.645199, 0.683433],
                [3.884824, -1.000000], [3.884824, -1.000000], [3.884824, -0.000000],
                [3.884824, -0.000000], [3.884824, -0.000000], [3.884824, -0.000000],
                [3.884824, -0.000000], [3.884824, -0.000000], [3.884824, 0.000000], 
                [3.884824, 0.000000], [3.884824, 1.000000], [3.884824, 1.000000],
                [4.982500, -0.500000], [4.982500, -0.500000], [4.982500, -0.500000],
                [4.982500, -0.500000], [4.982500, -0.000000], [4.982500, -0.000000],
                [4.982500, -0.000000], [4.982500, -0.000000], [4.982500, -0.000000],
                [4.982500, -0.000000], [4.982500, -0.000000], [4.982500, -0.000000],
                [4.982500, 0.500000], [4.982500, 0.500000], [4.982500, 0.500000],
                [4.982500, 0.500000], [4.982500, 0.866025], [4.982500, 0.866025],
                [4.982500, 0.866025], [4.982500, 0.866025], [4.982500, 0.866025],
                [4.982500, 0.866025], [4.982500, 0.866025], [4.982500, 0.866025], 
                [4.982500, -0.000000], [4.982500, -0.000000], [4.982500, -0.000000],
                [4.982500, -0.000000], [4.982500, -0.000000], [4.982500, -0.000000],
                [4.982500, -0.000000], [4.982500, -0.000000], [4.982500, 1.000000],
                [4.982500, 1.000000], [4.982500, 1.000000], [4.982500, 1.000000]
]
pp_len_cos = [[2.831676, -0.000000, 0.000000], [2.831676, -0.000000, 0.000000], [2.831676, 0.000000, -0.000000],
                [2.831676, -0.000000, 0.000000], [2.831676, -0.000000, 0.000000], [2.831676, 0.000000, -0.000000],
                [2.831676, 0.000000, 1.000000], [2.831676, 0.000000, 1.000000], [2.831676, 1.000000, 0.000000],
                [2.831676, -0.000000, 0.000000], [2.831676, -0.000000, 0.000000], [2.831676, 0.000000, -0.000000],
                [2.831676, 0.000000, -0.000000], [2.831676, 0.000000, 1.000000], [2.831676, 0.000000, 1.000000],
                [2.831676, 0.000000, -0.000000], [2.831676, 0.000000, -0.000000], [2.831676, 1.000000, 0.000000],
                [2.937101, -0.197692, 0.197692], [2.937101, -0.197692, 0.197692], [2.937101, -0.197692, 0.197692],
                [2.937101, -0.197692, 0.197692], [2.937101, -0.000000, 0.000000], [2.937101, -0.000000, 0.000000],
                [2.937101, 0.000000, -0.000000], [2.937101, -0.000000, 0.000000], [2.937101, -0.000000, 0.000000],
                [2.937101, 0.000000, -0.000000], [2.937101, 0.000000, 1.000000], [2.937101, 0.000000, 1.000000], 
                [2.937101, 0.000000, -0.000000], [2.937101, 0.000000, -0.000000], [2.937101, 0.040742, 0.959258],
                [2.937101, 0.040742, 0.959258], [2.937101, 0.959258, 0.040742], [2.937101, 0.959258, 0.040742],
                [2.937101, -0.415371, 0.415371], [2.937101, -0.415371, 0.415371], [2.937101, -0.171206, 0.171206],
                [2.937101, -0.171206, 0.171206], [2.937101, 0.040742, 0.959258], [2.937101, 0.098846, -0.098846], 
                [2.937101, 0.098846, -0.098846], [2.937101, 0.239815, 0.760185], [2.937101, 0.719444, 0.280556],
                [2.937101, -0.415371, 0.415371], [2.937101, -0.415371, 0.415371], [2.937101, -0.171206, 0.171206],
                [2.937101, -0.171206, 0.171206], [2.937101, 0.040742, 0.959258], [2.937101, 0.040742, 0.959258],
                [2.937101, 0.098846, -0.098846], [2.937101, 0.098846, -0.098846], [2.937101, 0.098846, -0.098846],
                [2.937101, 0.098846, -0.098846], [2.937101, 0.171206, -0.171206], [2.937101, 0.171206, -0.171206],
                [2.937101, 0.239815, 0.760185], [2.937101, 0.239815, 0.760185], [2.937101, 0.415371, -0.415371],
                [2.937101, 0.415371, -0.415371], [2.937101, 0.719444, 0.280556], [2.937101, 0.719444, 0.280556],
                [2.937101, 0.040742, 0.959258], [2.937101, 0.098846, -0.098846], [2.937101, 0.098846, -0.098846],
                [2.937101, 0.171206, -0.171206], [2.937101, 0.171206, -0.171206], [2.937101, 0.239815, 0.760185], 
                [2.937101, 0.415371, -0.415371], [2.937101, 0.415371, -0.415371], [2.937101, 0.719444, 0.280556],
                [3.314270, -0.431060, 0.431060], [3.314270, -0.431060, 0.431060], [3.314270, 0.000000, -0.000000],
                [3.314270, -0.000000, 0.000000], [3.314270, -0.000000, 0.000000], [3.314270, 0.000000, -0.000000], 
                [3.314270, 0.000000, 1.000000], [3.314270, 0.246649, 0.753351], [3.314270, 0.753351, 0.246649],
                [3.314270, -0.431060, 0.431060], [3.314270, -0.431060, 0.431060], [3.314270, -0.000000, 0.000000],
                [3.314270, 0.000000, -0.000000], [3.314270, 0.000000, -0.000000], [3.314270, -0.000000, 0.000000],
                [3.314270, 0.000000, 1.000000], [3.314270, 0.246649, 0.753351], [3.314270, 0.753351, 0.246649],
                [3.314270, -0.373309, 0.373309], [3.314270, -0.373309, 0.373309], [3.314270, -0.326211, 0.326211],
                [3.314270, -0.326211, 0.326211], [3.314270, 0.188338, 0.811662], [3.314270, 0.188338, 0.811662],
                [3.314270, 0.215530, -0.215530], [3.314270, 0.215530, -0.215530], [3.314270, 0.215530, -0.215530],
                [3.314270, 0.215530, -0.215530], [3.314270, 0.246649, 0.753351], [3.314270, 0.246649, 0.753351],
                [3.314270, 0.326211, -0.326211], [3.314270, 0.326211, -0.326211], [3.314270, 0.373309, -0.373309],
                [3.314270, 0.373309, -0.373309], [3.314270, 0.565014, 0.434986], [3.314270, 0.565014, 0.434986], 
                [3.314270, -0.373309, 0.373309], [3.314270, -0.373309, 0.373309], [3.314270, -0.326211, 0.326211],
                [3.314270, -0.326211, 0.326211], [3.314270, 0.188338, 0.811662], [3.314270, 0.188338, 0.811662],
                [3.314270, 0.215530, -0.215530], [3.314270, 0.215530, -0.215530], [3.314270, 0.215530, -0.215530],
                [3.314270, 0.215530, -0.215530], [3.314270, 0.246649, 0.753351], [3.314270, 0.246649, 0.753351],
                [3.314270, 0.326211, -0.326211], [3.314270, 0.326211, -0.326211], [3.314270, 0.373309, -0.373309],
                [3.314270, 0.373309, -0.373309], [3.314270, 0.565014, 0.434986], [3.314270, 0.565014, 0.434986],
                [3.645199, 0.000000, -0.000000], [3.645199, 0.000000, -0.000000], [3.645199, -0.000000, 0.000000],
                [3.645199, -0.000000, 0.000000], [3.645199, 0.000000, 1.000000], [3.645199, 0.377225, 0.622775],
                [3.645199, 0.484692, -0.484692], [3.645199, 0.484692, -0.484692], [3.645199, 0.622775, 0.377225],
                [3.645199, -0.000000, 0.000000], [3.645199, -0.000000, 0.000000], [3.645199, -0.000000, 0.000000],
                [3.645199, -0.000000, 0.000000], [3.645199, -0.000000, 0.000000], [3.645199, -0.000000, 0.000000],
                [3.645199, 0.000000, -0.000000], [3.645199, 0.000000, -0.000000], [3.645199, 0.000000, -0.000000],
                [3.645199, 0.000000, -0.000000], [3.645199, 0.000000, -0.000000], [3.645199, 0.000000, -0.000000], 
                [3.645199, 0.000000, 1.000000], [3.645199, 0.000000, 1.000000], [3.645199, 0.000000, 1.000000], 
                [3.645199, 0.377225, 0.622775], [3.645199, 0.377225, 0.622775], [3.645199, 0.377225, 0.622775],
                [3.645199, 0.484692, -0.484692], [3.645199, 0.484692, -0.484692], [3.645199, 0.484692, -0.484692],
                [3.645199, 0.484692, -0.484692], [3.645199, 0.484692, -0.484692], [3.645199, 0.484692, -0.484692],
                [3.645199, 0.622775, 0.377225], [3.645199, 0.622775, 0.377225], [3.645199, 0.622775, 0.377225],
                [3.645199, -0.419756, 0.419756], [3.645199, -0.419756, 0.419756], [3.645199, -0.419756, 0.419756],
                [3.645199, -0.419756, 0.419756], [3.645199, -0.269669, 0.269669], [3.645199, -0.269669, 0.269669],
                [3.645199, -0.269669, 0.269669], [3.645199, -0.269669, 0.269669], [3.645199, -0.242346, 0.242346],
                [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346],
                [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346],
                [3.645199, -0.242346, 0.242346], [3.645199, 0.155694, 0.844306], [3.645199, 0.155694, 0.844306],
                [3.645199, 0.155694, 0.844306], [3.645199, 0.155694, 0.844306], [3.645199, 0.269669, -0.269669],
                [3.645199, 0.269669, -0.269669], [3.645199, 0.269669, -0.269669], [3.645199, 0.269669, -0.269669], 
                [3.645199, 0.377225, 0.622775], [3.645199, 0.377225, 0.622775], [3.645199, 0.377225, 0.622775],
                [3.645199, 0.377225, 0.622775], [3.645199, 0.419756, -0.419756], [3.645199, 0.419756, -0.419756],
                [3.645199, 0.419756, -0.419756], [3.645199, 0.419756, -0.419756], [3.645199, 0.467081, 0.532919],
                [3.645199, 0.467081, 0.532919], [3.645199, 0.467081, 0.532919], [3.645199, 0.467081, 0.532919],
                [3.645199, -0.419756, 0.419756], [3.645199, -0.419756, 0.419756], [3.645199, -0.419756, 0.419756],
                [3.645199, -0.419756, 0.419756], [3.645199, -0.269669, 0.269669], [3.645199, -0.269669, 0.269669], 
                [3.645199, -0.269669, 0.269669], [3.645199, -0.269669, 0.269669], [3.645199, -0.242346, 0.242346],
                [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346],
                [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346], [3.645199, -0.242346, 0.242346],
                [3.645199, -0.242346, 0.242346], [3.645199, 0.155694, 0.844306], [3.645199, 0.155694, 0.844306],
                [3.645199, 0.155694, 0.844306], [3.645199, 0.155694, 0.844306], [3.645199, 0.269669, -0.269669],
                [3.645199, 0.269669, -0.269669], [3.645199, 0.269669, -0.269669], [3.645199, 0.269669, -0.269669],
                [3.645199, 0.377225, 0.622775], [3.645199, 0.377225, 0.622775], [3.645199, 0.377225, 0.622775],
                [3.645199, 0.377225, 0.622775], [3.645199, 0.419756, -0.419756], [3.645199, 0.419756, -0.419756],
                [3.645199, 0.419756, -0.419756], [3.645199, 0.419756, -0.419756], [3.645199, 0.467081, 0.532919],
                [3.645199, 0.467081, 0.532919], [3.645199, 0.467081, 0.532919], [3.645199, 0.467081, 0.532919], 
                [3.884824, -0.000000, 0.000000], [3.884824, -0.000000, 0.000000], [3.884824, -0.000000, 0.000000],
                [3.884824, -0.000000, 0.000000], [3.884824, 0.000000, -0.000000], [3.884824, -0.000000, 0.000000],
                [3.884824, 0.000000, -0.000000], [3.884824, 0.000000, -0.000000], [3.884824, -0.000000, 0.000000],
                [3.884824, 0.000000, -0.000000], [3.884824, 0.000000, 1.000000], [3.884824, 0.000000, 1.000000],
                [3.884824, 0.000000, -0.000000], [3.884824, 0.000000, -0.000000], [3.884824, 0.000000, 1.000000],
                [3.884824, 0.000000, 1.000000], [3.884824, 1.000000, 0.000000], [3.884824, 1.000000, 0.000000],
                [4.982500, -0.433013, 0.433013], [4.982500, -0.433013, 0.433013], [4.982500, -0.433013, 0.433013], 
                [4.982500, -0.433013, 0.433013], [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000],
                [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000],
                [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000],
                [4.982500, 0.000000, -0.000000], [4.982500, 0.000000, -0.000000], [4.982500, 0.000000, -0.000000],
                [4.982500, 0.000000, -0.000000], [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000],
                [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000], [4.982500, 0.000000, 1.000000],
                [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000],
                [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000],
                [4.982500, 0.000000, 1.000000], [4.982500, 0.250000, 0.750000], [4.982500, 0.250000, 0.750000],
                [4.982500, 0.250000, 0.750000], [4.982500, 0.250000, 0.750000], [4.982500, 0.250000, 0.750000],
                [4.982500, 0.250000, 0.750000], [4.982500, 0.250000, 0.750000], [4.982500, 0.250000, 0.750000],
                [4.982500, 0.433013, -0.433013], [4.982500, 0.433013, -0.433013], [4.982500, 0.433013, -0.433013],
                [4.982500, 0.433013, -0.433013], [4.982500, 0.750000, 0.250000], [4.982500, 0.750000, 0.250000],
                [4.982500, 0.750000, 0.250000], [4.982500, 0.750000, 0.250000], [4.982500, 0.750000, 0.250000],
                [4.982500, 0.750000, 0.250000], [4.982500, 0.750000, 0.250000], [4.982500, 0.750000, 0.250000], 
                [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000],
                [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000],
                [4.982500, -0.000000, 0.000000], [4.982500, -0.000000, 0.000000], [4.982500, 0.000000, -0.000000],
                [4.982500, 0.000000, -0.000000], [4.982500, 0.000000, -0.000000], [4.982500, 0.000000, -0.000000],
                [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000],
                [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000],
                [4.982500, 0.000000, 1.000000], [4.982500, 0.000000, 1.000000], [4.982500, 1.000000, 0.000000], 
                [4.982500, 1.000000, 0.000000], [4.982500, 1.000000, 0.000000], [4.982500, 1.000000, 0.000000]
]

stp_len_cos = [
    [2.831676, -1.000000], [2.831676, -0.000000], [2.831676, 0.000000],
    [2.831676, 0.000000], [2.831676, 0.000000], [2.831676, 1.000000],
    [2.831676, -1.000000], [2.831676, -0.000000], [2.831676, 0.000000],
    [2.831676, 0.000000], [2.831676, 0.000000], [2.831676, 1.000000],
    [2.937101, -0.979417], [2.937101, -0.979417], [2.937101, -0.201846],
    [2.937101, -0.201846], [2.937101, -0.000000], [2.937101, 0.000000], 
    [2.937101, 0.000000], [2.937101, 0.000000], [2.937101, 0.201846],
    [2.937101, 0.201846], [2.937101, 0.979417], [2.937101, 0.979417],
    [2.937101, -0.848200], [2.937101, -0.489709], [2.937101, -0.201846],
    [2.937101, 0.201846], [2.937101, 0.489709], [2.937101, 0.848200],
    [2.937101, -0.848200], [2.937101, -0.848200], [2.937101, -0.489709],
    [2.937101, -0.489709], [2.937101, -0.201846], [2.937101, -0.201846],
    [2.937101, 0.201846], [2.937101, 0.201846], [2.937101, 0.489709],
    [2.937101, 0.489709], [2.937101, 0.848200], [2.937101, 0.848200],
    [2.937101, -0.848200], [2.937101, -0.489709], [2.937101, -0.201846],
    [2.937101, 0.201846], [2.937101, 0.489709], [2.937101, 0.848200],
    [3.314270, -0.867958], [3.314270, -0.496637], [3.314270, 0.000000],
    [3.314270, 0.000000], [3.314270, 0.496637], [3.314270, 0.867958],
    [3.314270, -0.867958], [3.314270, -0.496637], [3.314270, 0.000000],
    [3.314270, 0.000000], [3.314270, 0.496637], [3.314270, 0.867958],
    [3.314270, -0.751674], [3.314270, -0.751674], [3.314270, -0.496637],
    [3.314270, -0.496637], [3.314270, -0.433979], [3.314270, -0.433979],
    [3.314270, 0.433979], [3.314270, 0.433979], [3.314270, 0.496637],
    [3.314270, 0.496637], [3.314270, 0.751674], [3.314270, 0.751674],
    [3.314270, -0.751674], [3.314270, -0.751674], [3.314270, -0.496637],
    [3.314270, -0.496637], [3.314270, -0.433979], [3.314270, -0.433979], 
    [3.314270, 0.433979], [3.314270, 0.433979], [3.314270, 0.496637],
    [3.314270, 0.496637], [3.314270, 0.751674], [3.314270, 0.751674],
    [3.645199, -0.789161], [3.645199, -0.614187], [3.645199, 0.000000],
    [3.645199, 0.000000], [3.645199, 0.614187], [3.645199, 0.789161],
    [3.645199, -0.789161], [3.645199, -0.789161], [3.645199, -0.789161],
    [3.314270, 0.433979], [3.314270, 0.433979], [3.314270, 0.496637],
    [3.314270, 0.496637], [3.314270, 0.751674], [3.314270, 0.751674],
    [3.645199, -0.789161], [3.645199, -0.614187], [3.645199, 0.000000],
    [3.645199, 0.000000], [3.645199, 0.614187], [3.645199, 0.789161],
    [3.645199, -0.789161], [3.645199, -0.789161], [3.645199, -0.789161],
    [3.645199, -0.614187], [3.645199, -0.614187], [3.645199, -0.614187],
    [3.645199, 0.000000], [3.645199, 0.000000], [3.645199, 0.000000],
    [3.645199, 0.000000], [3.645199, 0.000000], [3.645199, 0.000000],
    [3.645199, 0.614187], [3.645199, 0.614187], [3.645199, 0.614187],
    [3.645199, 0.789161], [3.645199, 0.789161], [3.645199, 0.789161],
    [3.645199, -0.683433], [3.645199, -0.683433], [3.645199, -0.683433],
    [3.645199, -0.683433], [3.645199, -0.614187], [3.645199, -0.614187],
    [3.645199, -0.614187], [3.645199, -0.614187], [3.645199, -0.394580],
    [3.645199, -0.394580], [3.645199, -0.394580], [3.645199, -0.394580],
    [3.645199, 0.394580], [3.645199, 0.394580], [3.645199, 0.394580],
    [3.645199, 0.394580], [3.645199, 0.614187], [3.645199, 0.614187],
    [3.645199, 0.614187], [3.645199, 0.614187], [3.645199, 0.683433],
    [3.645199, 0.683433], [3.645199, 0.683433], [3.645199, 0.683433],
    [3.645199, -0.683433], [3.645199, -0.683433], [3.645199, -0.683433],
    [3.645199, -0.683433], [3.645199, -0.614187], [3.645199, -0.614187],
    [3.645199, -0.614187], [3.645199, -0.614187], [3.645199, -0.394580],
    [3.645199, -0.394580], [3.645199, -0.394580], [3.645199, -0.394580],
    [3.645199, 0.394580], [3.645199, 0.394580], [3.645199, 0.394580],
    [3.645199, 0.394580], [3.645199, 0.614187], [3.645199, 0.614187],
    [3.645199, 0.614187], [3.645199, 0.614187], [3.645199, 0.683433],
    [3.645199, 0.683433], [3.645199, 0.683433], [3.645199, 0.683433],
    [3.884824, -1.000000], [3.884824, -1.000000], [3.884824, -0.000000],
    [3.884824, -0.000000], [3.884824, 0.000000], [3.884824, 0.000000],
    [3.884824, 0.000000], [3.884824, 0.000000], [3.884824, 0.000000],
    [3.884824, 0.000000], [3.884824, 1.000000], [3.884824, 1.000000],
    [4.982500, -0.866025], [4.982500, -0.866025], [4.982500, -0.866025],
    [4.982500, -0.866025], [4.982500, -0.866025], [4.982500, -0.866025],
    [4.982500, -0.866025], [4.982500, -0.866025], [4.982500, -0.500000],
    [4.982500, -0.500000], [4.982500, -0.500000], [4.982500, -0.500000],
    [4.982500, 0.000000], [4.982500, 0.000000], [4.982500, 0.000000],
    [4.982500, 0.000000], [4.982500, 0.000000], [4.982500, 0.000000],
    [4.982500, 0.000000], [4.982500, 0.000000], [4.982500, 0.500000],
    [4.982500, 0.500000], [4.982500, 0.500000], [4.982500, 0.500000],
    [4.982500, -1.000000], [4.982500, -1.000000], [4.982500, -1.000000],
    [4.982500, -1.000000], [4.982500, 0.000000], [4.982500, 0.000000],
    [4.982500, 0.000000], [4.982500, 0.000000], [4.982500, 0.000000],
    [4.982500, 0.000000], [4.982500, 0.000000], [4.982500, 0.000000]
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
    4, 4, 4, 4,]
hopping = np.zeros((787,))
unit_conv = 3.80998212 # hbar/2mA to eV

vss_start = -1
vss_end = 0
vss_step = 0.5

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

target_emass = 0.282
threshold = 0.03

outfile = open("Emass_scan.csv", "w+")
good_scan_outfile = open("good_Emass_scan.csv", "w+")

for vss in vss_list:
    for vsp in vsp_list:
        for vpp_s in vpp_s_list:
            for vpp_p in vpp_p_list:
                for vstp in vstp_list:
                    print("vss=%.2f vsp=%.2f vpp_s=%.2f vpp_p=%.2f vstp=%.2f started\n" %(vss, vsp, vpp_s, vpp_p, vstp))
                    # calculate ss hopping
                    hopping[0:40] = vss * pow(np.divide(lat, ss_bond_len), n)
                    # calculate sp hopping
                    hopping[40:244] = vsp * sp_len_cos[:,1] * pow(np.divide(lat, sp_len_cos[:,0]), n)
                    # calculate pp hopping
                    hopping[244:568] = (vpp_s * pp_len_cos[:,1] + vpp_p * pp_len_cos[:,2]) * pow(np.divide(lat, pp_len_cos[:,0]), n)
                    # calculate stp hopping
                    hopping[568:787] = vstp * stp_len_cos[:,1] * pow(np.divide(lat, stp_len_cos[:,0]), n)
                    my_class = Ga2O3_Ga_s_Class(onsite, hopping)
                    outfile.writelines('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t' %(vss, vsp, vpp_s, vpp_p, vstp) + my_class.getMassString())
                    my_class.savePlot('Emass_pdf/%.3f_%.3f_%.3f_%.3f_%.3f.png' %(vss, vsp, vpp_s, vpp_p, vstp))

                    if(np.abs(my_class.emass_0 - target_emass) < threshold):
                        good_scan_outfile.writelines('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t' %(vss, vsp, vpp_s, vpp_p, vstp) + my_class.getMassString())
                        my_class.savePlot('good_Emass_pdf/%.3f_%.3f_%.3f_%.3f_%.3f.png' %(vss, vsp, vpp_s, vpp_p, vstp))
                    del my_class
