import numpy as np

def sol_ham_eff(ham,norb, eig_vectors=False, roi=None,):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        ham_use=ham

        # check that matrix is hermitian
        if np.max(ham_use-ham_use.T.conj())>1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix
        # subspace we are interested in
        ham_ul = ham_use[0:roi,0:roi]
        ham_lr = ham_use[roi:norb,roi:norb]
        ham_ur = ham_use[0:roi,roi:norb]
        (ul_eval, ul_eig) = np.linalg.eigh(ham_ul)
        (lr_eval, lr_eig) = np.linalg.eigh(ham_lr)
        ul_eig = ul_eig.T
        lr_eig = lr_eig.T

        ham_eff = ham_ul
        for i in range(ham_ul.shape[0]):
            for j in range(ham_ul.shape[0]):
                for k in range(ham_lr.shape[0]):
                    # Han
                    a = ul_eig[i]
                    n = lr_eig[k]
                    temp = np.matmul(a.T, ham_ur)
                    pur = np.matmul(temp, n)
                    # Hnb
                    b = ul_eig[j]
                    n = n.T
                    temp = np.matmul(n, ham_ur.T)
                    pur = pur * np.matmul(temp, b)
                    # /(En - (Ea+Eb)/2)
                    pur = pur / (lr_eval[k] - (ul_eval[i]+ul_eval[j])/2)
                    # add the term with outer product
                    out_prod = np.outer(a, b)
                    ham_eff = ham_eff + pur*out_prod

        if eig_vectors==False: # only find eigenvalues
            eval=np.linalg.eigvalsh(ham_eff)
            # sort eigenvalues and convert to real numbers
            return np.array(eval,dtype=float)
        else: # find eigenvalues and eigenvectors
            (eval,eig)=np.linalg.eigh(ham_eff)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig=eig.T
            # sort evectors, eigenvalues and convert to real numbers
            # reshape eigenvectors if doing a spinfull calculation
            return (eval,eig)

H = np.zeros((10,10))
for i in range(4):
    H[i,i] = 5+i
for i in range(4,10):
    H[i,i] = 1

(eval, eig) = sol_ham_eff(H, 10, eig_vectors=True, roi=4)
print(eval)