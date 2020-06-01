import numpy as np

# define lattice vectors
lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
        [-6.1149997711,1.5199999809, 0.0000000000], 
        [-1.3736609922, 0.0000000000, 5.6349851545]]

# define coordinate of sites under lattice vectors 
Ga_orb = [[0.9096000000, -0.9096000000, 0.2052000000], [1.0904000000, -0.0904000000, 0.7948000000],
        [0.6586000000, -0.6586000000, 0.3143000000], [1.3414000000, -0.3414000000, 0.6857000000]]
        
O_orb = [[0.8326000000, -0.8326000000, 0.8989000000], [1.1674000000, -0.1674000000, 0.1011000000],
        [0.5043000000, -0.5043000000, 0.7447000000], [1.4957000000, -0.4957000000, 0.2553000000],
        [1.1721000000, -0.1721000000, 0.5635000000], [0.8279000000, -0.8279000000, 0.4365000000]]

# a class containing all information about a site
# and a method to calculate distance
class site:
    def __init__(self, lat_vec, lat_trans):
        """
        construct a site
        parameters:
        lat_vec: the coordinate of sites under lattice vectors.
            This can be obtained from orb
        lat_trans: lattice vector to cell containing this site
        """
        self.lat_vec = lat_vec
        self.lat_trans = lat_trans

    def findDistVec(self, target_site):
        """
        calclate the distance vector between this site and another site
        parameters:
        target_site: the second site to find distance to.
        return: the distance vector in the unit of A
        """
        lat_trans_diff = np.subtract(target_site.lat_trans, self.lat_trans) # difference in lattice translaiton
        site_diff = np.subtract(target_site.lat_vec, self.lat_vec) # difference in site
        diff = lat_trans_diff + site_diff # sum then up
        diff = np.multiply(diff[0], lat[0]) + np.multiply(diff[1], lat[1]) + np.multiply(diff[2], lat[2]) # convert to angstrom
        return diff

    def findDist(self, target_site):
        """
        calclate the distance between this site and another site
        parameters:
        target_site: the second site to find distance to.
        return: the euclidian distance in the unit of A
        """
        return np.linalg.norm(self.findDistVec(target_site))

    def findDistCos(self, target_site_1, target_site_2):
        """
        calclate the direction cosine:
        target_site: the second site to find distance to.
        return: the euclidian distance in the unit of A
        """
        diff1 = self.findDistVec(target_site_1)
        diff2 = self.findDistVec(target_site_2)
        return np.dot(diff1/np.linalg.norm(diff1), diff2/np.linalg.norm(diff2))

    def findDistProj(self, taget_site_1, target_site_2):
        dist_cos =  self.findDistCos(taget_site_1, target_site_2)
        diff2 = self.findDistVec(target_site_2)
        return dist_cos*np.linalg.norm(diff2)

    def findDistOrth(self, taget_site_1, target_site_2):
        dist_cos =  self.findDistCos(taget_site_1, target_site_2)
        diff2 = self.findDistVec(target_site_2)
        return np.sqrt(1-dist_cos**2)*np.linalg.norm(diff2)



start_cell = []    

start_cell.append(site(Ga_orb[0], [0, 0, 0]))
start_cell.append(site(Ga_orb[1], [0, 0, 0]))
start_cell.append(site(Ga_orb[2], [0, 0, 0]))
start_cell.append(site(Ga_orb[3], [0, 0, 0]))

# set the threshold distance
threshold = 6
# range of the translation
trans_range = 2
# a dict to store pairs within threshold
dist_dict = {}

for x in range(-trans_range, trans_range+1):
    for y in range(-trans_range, trans_range+1):
        for z in range(-trans_range, trans_range+1):
            # define target cell
            target_cell = []    
            target_cell.append(site(Ga_orb[0], [x, y, z]))
            target_cell.append(site(Ga_orb[1], [x, y, z]))
            target_cell.append(site(Ga_orb[2], [x, y, z]))
            target_cell.append(site(Ga_orb[3], [x, y, z]))
            
            # compate all orbits in the cell
            for i in range(4):
                for j in range(i, 4):
                    dist = start_cell[i].findDist(target_cell[j])
                    # if they are within threshold and two orbits aren't the same
                    if(dist < threshold) and (x != 0 or y != 0 or z != 0 or i != j):
                        # <i|H|j+R> is equivalent to <i|H|j-R>
                        if (i, j, -x, -y, -z) not in dist_dict:
                            dist_dict[(i, j, x, y, z)] = dist # add to the dict

dist_dict_sorted = {k: v for k, v in sorted(dist_dict.items(), key=lambda item: item[1])}

# # set a counter
# cnt = 0
# for k, v in dist_dict_sorted.items():
#     print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
#     cnt = cnt + 1

# print('\n\n\n\n\n')

                        
# for k, v in dist_dict_sorted.items():
#     print('%f' %(v), end=', ')


print('Search for O proximity\n\n\n\n\n')
trans_range = 1

O_prox_dict = {}
orth_threshold = 1

for k, v in dist_dict_sorted.items():
    Ga_1 = site(k[0], [0, 0, 0])
    Ga_2 = site(k[1], [k[2], k[3], k[4]])

    O_list = []
    for x in range(-trans_range, trans_range+1):
        for y in range(-trans_range, trans_range+1):
            for z in range(-trans_range, trans_range+1):
                target_cell = []
                target_cell.append(site(O_orb[0], [x, y, z]))
                target_cell.append(site(O_orb[1], [x, y, z]))
                target_cell.append(site(O_orb[2], [x, y, z]))
                target_cell.append(site(O_orb[3], [x, y, z]))

                for i in range(4):
                    dir_orth = Ga_1.findDistOrth(Ga_2, target_cell[i])
                    dir_proj = Ga_1.findDistProj(Ga_2, target_cell[i])
                    if(dir_proj < v and dir_orth < orth_threshold):
                        O_list.append(target_cell[i])
    
    O_prox_dict[k] = O_list

print('List oxygen proximity\n\n\n')
for k, v in O_prox_dict.items():
    print(k, len(v))
print('\n\n\n')

# construct groups
no_oxygen = {}
has_oxygen = {}
for k, v in O_prox_dict.items():
    if(len(v) > 0):
        has_oxygen[k] = dist_dict_sorted[k]
    else:
        no_oxygen[k] = dist_dict_sorted[k]

# print results
cnt = 0
for k, v in no_oxygen.items():
    print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1

for k, v in has_oxygen.items():
    print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1

print ('\n\n---------no oxygen----------\n\n')

for k, v in no_oxygen.items():
    print('%f' %(v), end=', ')

print ('\n\n---------has oxygen----------\n\n')

for k, v in has_oxygen.items():
    print('%f' %(v), end=', ')