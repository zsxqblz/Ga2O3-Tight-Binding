import numpy as np

# read data
data = np.loadtxt('wannier_band.dat')
k = data[:, 0]
E = data[:, 1]

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
    k_norm = np.multiply(k_norm, 1e10)  # 1/Ang to 1/m
    E = E*1.60218e-19  # eV to J
    hbar = 1.05457e-34  # reduced planck constant
    me = 9.1094e-31  # electron mass
    deriv1 = np.divide(np.gradient(
        E[start:end+1]), np.gradient(k_norm[start:end+1]))
    deriv2 = np.divide(np.gradient(deriv1), np.gradient(k_norm[start:end+1]))
    return hbar**2/np.mean(deriv2)/me


bmass_f = findBMass(E, k, 90, 100)
bmass_t = findBMass(E, k, 101, 111)
bmass_l = findBMass(E, k, 448, 458)
emass = np.power(bmass_f*bmass_t*bmass_l, 1/3)

print('bmass_f: %f \nbmass_t: %f \nbmass_l: %f \nemass: %f \n' %
      (bmass_f, bmass_t, bmass_l, emass))
