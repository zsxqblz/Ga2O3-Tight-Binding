import numpy as np

# define lattice vectors
lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
        [-6.1149997711,1.5199999809, 0.0000000000], 
        [-1.3736609922, 0.0000000000, 5.6349851545]]

# define coordinate of sites under lattice vectors 
orb = [[0.9096000000, 	-0.9096000000, 	0.2052000000], 
        [1.0904000000, 	-0.0904000000, 	0.7948000000],
        [0.6586000000, 	-0.6586000000, 	0.3143000000], 
        [1.3414000000, 	-0.3414000000, 	0.6857000000],
        [0.8326000000, 	-0.8326000000, 	0.8989000000],
        [1.1674000000, 	-0.1674000000, 	0.1011000000], 
        [0.5043000000, 	-0.5043000000, 	0.7447000000],
        [1.4957000000, 	-0.4957000000, 	0.2553000000],
        [1.1721000000, 	-0.1721000000, 	0.5635000000], 
        [0.8279000000, 	-0.8279000000, 	0.4365000000]]

# a class containing all information about a site
# and a method to calculate distance
class site:
    def __init__(self, lat_vec, lat_trans, orb_type, num):
        """
        construct a site
        parameters:
        lat_vec: the coordinate of sites under lattice vectors.
            This can be obtained from orb
        lat_trans: lattice vector to cell containing this site
        orb_type: type of orbital. can be s, px, py, and pz
        num: the number coding of the site
        """
        self.lat_vec = lat_vec
        self.lat_trans = lat_trans
        self.type = orb_type
        self.num = num

    def findDistVec(self, target_site):
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
        return diff

    def findDist(self, target_site):
        return np.linalg.norm(self.findDistVec(target_site))

    def findDistCos(self, target_site_1, target_site_2):
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

    def findDirCos(self, target_site):
        diff = self.findDistVec(target_site)
        return diff/np.linalg.norm(diff)



start_cell = []    

start_cell.append(site(orb[0], [0, 0, 0], 's', 1))
start_cell.append(site(orb[1], [0, 0, 0], 's', 2))
start_cell.append(site(orb[2], [0, 0, 0], 's', 3))
start_cell.append(site(orb[3], [0, 0, 0], 's', 4))
# start_cell.append(site(orb[0], [0, 0, 0], 'px', 1))
# start_cell.append(site(orb[1], [0, 0, 0], 'px', 2))
# start_cell.append(site(orb[2], [0, 0, 0], 'px', 3))
# start_cell.append(site(orb[3], [0, 0, 0], 'px', 4))
# start_cell.append(site(orb[0], [0, 0, 0], 'py', 1))
# start_cell.append(site(orb[1], [0, 0, 0], 'py', 2))
# start_cell.append(site(orb[2], [0, 0, 0], 'py', 3))
# start_cell.append(site(orb[3], [0, 0, 0], 'py', 4))
# start_cell.append(site(orb[0], [0, 0, 0], 'pz', 1))
# start_cell.append(site(orb[1], [0, 0, 0], 'pz', 2))
# start_cell.append(site(orb[2], [0, 0, 0], 'pz', 3))
# start_cell.append(site(orb[3], [0, 0, 0], 'pz', 4))
# an artificial excited s state. see P. Voul et al. (1981)
# s* orbitals are on the oxygen site
start_cell.append(site(orb[4], [0, 0, 0], 'px', 1))
start_cell.append(site(orb[5], [0, 0, 0], 'px', 2))
start_cell.append(site(orb[6], [0, 0, 0], 'px', 3))
start_cell.append(site(orb[7], [0, 0, 0], 'px', 4))
start_cell.append(site(orb[8], [0, 0, 0], 'px', 5))
start_cell.append(site(orb[9], [0, 0, 0], 'px', 6))

start_cell.append(site(orb[4], [0, 0, 0], 'py', 1))
start_cell.append(site(orb[5], [0, 0, 0], 'py', 2))
start_cell.append(site(orb[6], [0, 0, 0], 'py', 3))
start_cell.append(site(orb[7], [0, 0, 0], 'py', 4))
start_cell.append(site(orb[8], [0, 0, 0], 'py', 5))
start_cell.append(site(orb[9], [0, 0, 0], 'py', 6))

start_cell.append(site(orb[4], [0, 0, 0], 'pz', 1))
start_cell.append(site(orb[5], [0, 0, 0], 'pz', 2))
start_cell.append(site(orb[6], [0, 0, 0], 'pz', 3))
start_cell.append(site(orb[7], [0, 0, 0], 'pz', 4))
start_cell.append(site(orb[8], [0, 0, 0], 'pz', 5))
start_cell.append(site(orb[9], [0, 0, 0], 'pz', 6))

# set the threshold distance
threshold = 5
# range of the translation
trans_range = 2
# dict to store pairs within threshold
# ss: {start_site, target_site, [translation vector]} = dist
dist_dict_ss = {}
# sp: {start_site, target_site, [translation vector], start_orbit_code, target_orbit_code} = (dist, cos)
# for orbit code, s=0, px=1, py=2, pz=3
dist_dict_ps = {}
for x in range(-trans_range, trans_range+1):
    for y in range(-trans_range, trans_range+1):
        for z in range(-trans_range, trans_range+1):
            # define target cell
            target_cell = []
            target_cell.append(site(orb[0], [x, y, z], 's', 1))
            target_cell.append(site(orb[1], [x, y, z], 's', 2))
            target_cell.append(site(orb[2], [x, y, z], 's', 3))
            target_cell.append(site(orb[3], [x, y, z], 's', 4))

            target_cell.append(site(orb[4], [x, y, z], 'px', 1))
            target_cell.append(site(orb[5], [x, y, z], 'px', 2))
            target_cell.append(site(orb[6], [x, y, z], 'px', 3))
            target_cell.append(site(orb[7], [x, y, z], 'px', 4))
            target_cell.append(site(orb[8], [x, y, z], 'px', 5))
            target_cell.append(site(orb[9], [x, y, z], 'px', 6))

            target_cell.append(site(orb[4], [x, y, z], 'py', 1))
            target_cell.append(site(orb[5], [x, y, z], 'py', 2))
            target_cell.append(site(orb[6], [x, y, z], 'py', 3))
            target_cell.append(site(orb[7], [x, y, z], 'py', 4))
            target_cell.append(site(orb[8], [x, y, z], 'py', 5))
            target_cell.append(site(orb[9], [x, y, z], 'py', 6))

            target_cell.append(site(orb[4], [x, y, z], 'pz', 1))
            target_cell.append(site(orb[5], [x, y, z], 'pz', 2))
            target_cell.append(site(orb[6], [x, y, z], 'pz', 3))
            target_cell.append(site(orb[7], [x, y, z], 'pz', 4))
            target_cell.append(site(orb[8], [x, y, z], 'pz', 5))
            target_cell.append(site(orb[9], [x, y, z], 'pz', 6))

            for i in range(22):
                for j in range(i, 22):
                    dist = start_cell[i].findDist(target_cell[j])
                    dir_cos = start_cell[i].findDirCos(target_cell[j]);
                    # if they are within threshold and two sites aren't the same
                    if(dist < threshold) and (x != 0 or y != 0 or z != 0 or (i-j)%4 != 0):

                        # ss sigma bonds
                        if(start_cell[i].type=='s' and target_cell[j].type=='s'):
                            # <i|H|j+R> is equivalent to <j|H|i-R>
                            if (j, i, -x, -y, -z) not in dist_dict_ss:
                                dist_dict_ss[(i, j, x, y, z)] = dist # add to the dict

                        # sp sigma bonds
                        elif(start_cell[i].type=='s' and target_cell[j].type=='px'):
                            # add distance and direction cosion to the dict
                            dist_dict_ps[(i, j, x, y, z, 0, 1)] = (dist, -dir_cos[0])
                        elif(start_cell[i].type=='s' and target_cell[j].type=='py'):
                            # add distance and direction cosion to the dict
                            dist_dict_ps[(i, j, x, y, z, 0, 2)] = (dist, -dir_cos[1])
                        elif(start_cell[i].type=='s' and target_cell[j].type=='pz'):
                            # add distance and direction cosion to the dict
                            dist_dict_ps[(i, j, x, y, z, 0, 3)] = (dist, -dir_cos[2])



dist_dict_ss_sorted = {k: v for k, v in sorted(dist_dict_ss.items(), key=lambda item: item[1])}
dist_dict_ps_sorted = {k: v for k, v in sorted(dist_dict_ps.items(), key=lambda item: item[1])}


# set a counter
cnt = 0
for k, v in dist_dict_ss_sorted.items():
    print('self.my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1
# print('\n', cnt, '\n') # print the counter to check with the distance list

for k, v in dist_dict_ps_sorted.items():
    # the orbit types are coded for calculating the site number in the program
    print('self.my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1
# print('\n', cnt, '\n')



# reset the counter for double check
cnt = 0                     
for k, v in dist_dict_ss_sorted.items():
    print('%f' %(v), end=', ')
    cnt = cnt + 1
    if(cnt%6 == 1):
        print('')
print('\n', cnt, '\n')

for k, v in dist_dict_ps_sorted.items():
    print('[%f, %f]' %(v[0], v[1]), end=', ')
    cnt = cnt + 1
    if(cnt%3 == 1):
        print('')
print('\n', cnt, '\n')