import numpy as np

# define lattice vectors
lat = [[2.4912500000, 1.4383238581, 4.4776666667], 
       [-2.4912500000, 1.4383238581, 4.4776666667], 
       [0.0000000000, -2.8766477162, 4.4776666667]]

# define coordinate of sites under lattice vectors 
orb = [[0.1446000000, 0.1446000000, 0.1446000000], [0.6446000000, 0.6446000000, -0.3554000000],
        [0.8554000000,-0.1446000000,-0.1446000000], [0.3554000000,0.3554000000,-0.6446000000]]
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

    def findDist(self, target_site):
        """
        calclate the distance between this site and another site
        parameters:
        target_site: the second site to find distance to.
        return: the euclidian distance in the unit of A
        """
        lat_trans_diff = np.subtract(target_site.lat_trans, self.lat_trans) # difference in lattice translaiton
        site_diff = np.subtract(target_site.lat_vec, self.lat_vec) # difference in site
        diff = lat_trans_diff + site_diff # sum then up
        diff = np.multiply(diff[0], lat[0]) + np.multiply(diff[1], lat[1]) + np.multiply(diff[2], lat[2]) # convert to angstrom
        return np.linalg.norm(diff)



start_cell = []    

start_cell.append(site(orb[0], [0, 0, 0]))
start_cell.append(site(orb[1], [0, 0, 0]))
start_cell.append(site(orb[2], [0, 0, 0]))
start_cell.append(site(orb[3], [0, 0, 0]))

# set the threshold distance
threshold = 5
# range of the translation
trans_range = 2
# a dict to store pairs within threshold
dist_dict = {}

for x in range(-trans_range, trans_range+1):
    for y in range(-trans_range, trans_range+1):
        for z in range(-trans_range, trans_range+1):
            # define target cell
            target_cell = []    
            target_cell.append(site(orb[0], [x, y, z]))
            target_cell.append(site(orb[1], [x, y, z]))
            target_cell.append(site(orb[2], [x, y, z]))
            target_cell.append(site(orb[3], [x, y, z]))
            
            # compute all orbits in the cell
            for i in range(4):
                for j in range(i, 4):
                    dist = start_cell[i].findDist(target_cell[j])
                    # if they are within threshold and two orbits aren't the same
                    if(dist < threshold) and (x != 0 or y != 0 or z != 0 or i != j):
                        # <i|H|j+R> is equivalent to <i|H|j-R>
                        if (i, j, -x, -y, -z) not in dist_dict:
                            dist_dict[(i, j, x, y, z)] = dist # add to the dict

dist_dict_sorted = {k: v for k, v in sorted(dist_dict.items(), key=lambda item: item[1])}

# set a counter
cnt = 0
for k, v in dist_dict_sorted.items():
    print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1

print('\n\n\n\n\n')

                        
for k, v in dist_dict_sorted.items():
    print('%f' %(v), end=', ')
