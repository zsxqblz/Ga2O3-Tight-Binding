import numpy as np

# define lattice vectors
lat = [[6.1149997711, 1.5199999809, 0.0000000000], 
        [-6.1149997711,1.5199999809, 0.0000000000], 
        [-1.3736609922, 0.0000000000, 5.6349851545]]

# define coordinate of sites under lattice vectors 
orb = [[0.0000, 0.0000, 0.0000], [0.8192, -0.8192, -0.5896],
        [0.2510, -0.2510, -0.1091], [0.5682, -0.5682, -0.4805]]

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
start_cell.append(site(orb[0], [0, 0, 0], 'px', 1))
start_cell.append(site(orb[1], [0, 0, 0], 'px', 2))
start_cell.append(site(orb[2], [0, 0, 0], 'px', 3))
start_cell.append(site(orb[3], [0, 0, 0], 'px', 4))
start_cell.append(site(orb[0], [0, 0, 0], 'py', 1))
start_cell.append(site(orb[1], [0, 0, 0], 'py', 2))
start_cell.append(site(orb[2], [0, 0, 0], 'py', 3))
start_cell.append(site(orb[3], [0, 0, 0], 'py', 4))
start_cell.append(site(orb[0], [0, 0, 0], 'pz', 1))
start_cell.append(site(orb[1], [0, 0, 0], 'pz', 2))
start_cell.append(site(orb[2], [0, 0, 0], 'pz', 3))
start_cell.append(site(orb[3], [0, 0, 0], 'pz', 4))
# an artificial excited s state. see P. Voul et al. (1981)
start_cell.append(site(orb[0], [0, 0, 0], 'st', 1))
start_cell.append(site(orb[1], [0, 0, 0], 'st', 2))
start_cell.append(site(orb[2], [0, 0, 0], 'st', 3))
start_cell.append(site(orb[3], [0, 0, 0], 'st', 4))

# set the threshold distance
threshold = 5
# range of the translation
trans_range = 2
# dict to store pairs within threshold
# ss: {start_site, target_site, [translation vector]} = dist
dist_dict_ss = {}
# sp: {start_site, target_site, [translation vector], start_orbit_code, target_orbit_code} = (dist, cos)
# for orbit code, s=0, px=1, py=2, pz=3
dist_dict_sp = {}
# pp: {start_site, target_site, [translation vector], start_orbit_code, target_orbit_code} = (dist, cos1, cos2)
# for orbit code, s=0, px=1, py=2, pz=3, the two cos are in the form given in wikipaedia
dist_dict_pp = {}
# stp: {start_site, target_site, [translation vector], start_orbit_code, target_orbit_code} = (dist, cos)
# for orbit code, s=0, px=1, py=2, pz=3, st=4
dist_dict_stp = {}

for x in range(-trans_range, trans_range+1):
    for y in range(-trans_range, trans_range+1):
        for z in range(-trans_range, trans_range+1):
            # define target cell
            target_cell = []    
            target_cell.append(site(orb[0], [x, y, z], 's', 1))
            target_cell.append(site(orb[1], [x, y, z], 's', 2))
            target_cell.append(site(orb[2], [x, y, z], 's', 3))
            target_cell.append(site(orb[3], [x, y, z], 's', 4))
            target_cell.append(site(orb[0], [x, y, z], 'px', 1))
            target_cell.append(site(orb[1], [x, y, z], 'px', 2))
            target_cell.append(site(orb[2], [x, y, z], 'px', 3))
            target_cell.append(site(orb[3], [x, y, z], 'px', 4))
            target_cell.append(site(orb[0], [x, y, z], 'py', 1))
            target_cell.append(site(orb[1], [x, y, z], 'py', 2))
            target_cell.append(site(orb[2], [x, y, z], 'py', 3))
            target_cell.append(site(orb[3], [x, y, z], 'py', 4))
            target_cell.append(site(orb[0], [x, y, z], 'pz', 1))
            target_cell.append(site(orb[1], [x, y, z], 'pz', 2))
            target_cell.append(site(orb[2], [x, y, z], 'pz', 3))
            target_cell.append(site(orb[3], [x, y, z], 'pz', 4))
            target_cell.append(site(orb[0], [x, y, z], 'st', 1))
            target_cell.append(site(orb[1], [x, y, z], 'st', 2))
            target_cell.append(site(orb[2], [x, y, z], 'st', 3))
            target_cell.append(site(orb[3], [x, y, z], 'st', 4))
            
            # compate all orbits in the cell
            for i in range(20):
                for j in range(i, 20):
                    dist = start_cell[i].findDist(target_cell[j])
                    dir_cos = start_cell[i].findDirCos(target_cell[j]);
                    # if they are within threshold and two sites aren't the same
                    if(dist < threshold) and (x != 0 or y != 0 or z != 0 or (i-j)%4 != 0):

                        # ss sigma bonds
                        if(start_cell[i].type=='s' and target_cell[j].type=='s'):
                            # <i|H|j+R> is equivalent to <i|H|j-R>
                            if (i, j, -x, -y, -z) not in dist_dict_ss:
                                dist_dict_ss[(i, j, x, y, z)] = dist # add to the dict

                        ## sp sigma bonds
                        elif(start_cell[i].type=='s' and target_cell[j].type=='px'):
                            if (i, j, -x, -y, -z, 0, 1) not in dist_dict_sp:
                                # add distance and direction cosion to the dict
                                dist_dict_sp[(i, j, x, y, z, 0, 1)] = (dist, -dir_cos[0])
                        elif(start_cell[i].type=='s' and target_cell[j].type=='py'):
                            if (i, j, -x, -y, -z, 0, 2) not in dist_dict_sp:
                                # add distance and direction cosion to the dict
                                dist_dict_sp[(i, j, x, y, z, 0, 2)] = (dist, -dir_cos[1])
                        elif(start_cell[i].type=='s' and target_cell[j].type=='pz'):
                            if (i, j, -x, -y, -z, 0, 3) not in dist_dict_sp:
                                # add distance and direction cosion to the dict
                                dist_dict_sp[(i, j, x, y, z, 0, 3)] = (dist, -dir_cos[2])

                        # pp pi bonds
                        elif(start_cell[i].type=='px' and target_cell[j].type=='px'):
                            if (i, j, -x, -y, -z, 1, 1) not in dist_dict_pp:
                                # add distance and direction cosion to the dict
                                dist_dict_pp[(i, j, x, y, z, 1, 1)] = (dist, dir_cos[0]**2, 1-dir_cos[0]**2)
                        elif(start_cell[i].type=='px' and target_cell[j].type=='py'):
                            if (i, j, -x, -y, -z, 1, 2) not in dist_dict_pp:
                                # add distance and direction cosion to the dict
                                dist_dict_pp[(i, j, x, y, z, 1, 2)] = (dist, dir_cos[0]*dir_cos[1], -dir_cos[0]*dir_cos[1])
                        elif(start_cell[i].type=='px' and target_cell[j].type=='pz'):
                            if (i, j, -x, -y, -z, 1, 3) not in dist_dict_pp:
                                # add distance and direction cosion to the dict
                                dist_dict_pp[(i, j, x, y, z, 1, 3)] = (dist, dir_cos[0]*dir_cos[2], -dir_cos[0]*dir_cos[2])
                        elif(start_cell[i].type=='py' and target_cell[j].type=='py'):
                            if (i, j, -x, -y, -z, 2, 2) not in dist_dict_pp:
                                # add distance and direction cosion to the dict
                                dist_dict_pp[(i, j, x, y, z, 2, 2)] = (dist, dir_cos[1]**2, 1-dir_cos[1]**2)
                        elif(start_cell[i].type=='py' and target_cell[j].type=='pz'):
                            if (i, j, -x, -y, -z, 2, 3) not in dist_dict_pp:
                                # add distance and direction cosion to the dict
                                dist_dict_pp[(i, j, x, y, z, 2, 3)] = (dist, dir_cos[1]*dir_cos[2], -dir_cos[1]*dir_cos[2])
                        elif(start_cell[i].type=='pz' and target_cell[j].type=='pz'):
                            if (i, j, -x, -y, -z, 3, 3) not in dist_dict_pp:
                                # add distance and direction cosion to the dict
                                dist_dict_pp[(i, j, x, y, z, 3, 3)] = (dist, dir_cos[2]**2, 1-dir_cos[2]**2)

                        # stp sigma bond
                        elif(start_cell[i].type=='px' and target_cell[j].type=='st'):
                            if (i, j, -x, -y, -z, 1, 4) not in dist_dict_stp:
                                # add distance and direction cosion to the dict
                                dist_dict_stp[(i, j, x, y, z, 1, 4)] = (dist, dir_cos[0])
                        elif(start_cell[i].type=='py' and target_cell[j].type=='st'):
                            if (i, j, -x, -y, -z, 2, 4) not in dist_dict_stp:
                                # add distance and direction cosion to the dict
                                dist_dict_stp[(i, j, x, y, z, 2, 4)] = (dist, dir_cos[1])
                        elif(start_cell[i].type=='pz' and target_cell[j].type=='st'):
                            if (i, j, -x, -y, -z, 3, 4) not in dist_dict_stp:
                                # add distance and direction cosion to the dict
                                dist_dict_stp[(i, j, x, y, z, 3, 4)] = (dist, dir_cos[2])


dist_dict_ss_sorted = {k: v for k, v in sorted(dist_dict_ss.items(), key=lambda item: item[1])}
dist_dict_sp_sorted = {k: v for k, v in sorted(dist_dict_sp.items(), key=lambda item: item[1])}
dist_dict_pp_sorted = {k: v for k, v in sorted(dist_dict_pp.items(), key=lambda item: item[1])}
dist_dict_stp_sorted = {k: v for k, v in sorted(dist_dict_stp.items(), key=lambda item: item[1])}


# set a counter
cnt = 0
for k, v in dist_dict_ss_sorted.items():
    print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1
# print('\n', cnt, '\n') # print the counter to check with the distance list

for k, v in dist_dict_sp_sorted.items():
    # the orbit types are coded for calculating the site number in the program
    print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1
# print('\n', cnt, '\n')

for k, v in dist_dict_pp_sorted.items():
    # the orbit types are coded for calculating the site number in the program
    print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1
# print('\n', cnt, '\n')

for k, v in dist_dict_stp_sorted.items():
    # the orbit types are coded for calculating the site number in the program
    print('my_model.set_hop(hopping[%d],%d,%d,[%d,%d,%d])' %(cnt, k[0], k[1], k[2], k[3], k[4]))
    cnt = cnt + 1
print('\n', cnt, '\n\n\n\n\n\n')

# reset the counter for double check
cnt = 0                     
for k, v in dist_dict_ss_sorted.items():
    print('%f' %(v), end=', ')
    cnt = cnt + 1
    if(cnt%6 == 1):
        print('')
print('\n', cnt, '\n')

for k, v in dist_dict_sp_sorted.items():
    print('[%f, %f]' %(v[0], v[1]), end=', ')
    cnt = cnt + 1
    if(cnt%3 == 1):
        print('')
print('\n', cnt, '\n')

for k, v in dist_dict_pp_sorted.items():
    print('[%f, %f, %f]' %(v[0], v[1], v[2]), end=', ')
    cnt = cnt + 1
    if(cnt%3 == 1):
        print('')
print('\n', cnt, '\n')

for k, v in dist_dict_stp_sorted.items():
    print('[%f, %f]' %(v[0], v[1]), end=', ')
    cnt = cnt + 1
    if(cnt%3 == 1):
        print('')
print('\n', cnt, '\n')