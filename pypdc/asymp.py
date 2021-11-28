# -*- coding:utf-8 -*-


# In this file we calculate the asymptotic statistics for all measures, including
# first and second order asymptotic aproximations.

from numpy import *
import scipy.stats as st
#from scipy.stats import cov as cov
from numpy import cov as cov
from numpy.linalg import cholesky
from numpy.linalg import eigh
from numpy.linalg import inv
from numpy.linalg import pinv

from scipy.sparse.linalg import inv as sinv

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import dia_matrix
from scipy import sparse 
from matplotlib.pyplot import xcorr
from pypdc.ar_fit import *
from numpy.linalg import eig
from numpy.linalg.linalg import LinAlgError 
import sys
import numpy as np
from numba import jit

from timeit import default_timer as timer

# These functions are used to make code more readable.

#def vec(x): return np.array(x.ravel('F')).T
def vec(x): return csr_matrix.transpose(csr_matrix(x.ravel('F')))

def vec3(x): return csr_matrix(x.ravel('F'))

def vec2(x): return np.transpose(x.ravel('F'))
# def vec2(x): return (np.ravel(x, order='F')).T


def O(n): return np.zeros([n, n], dtype=float)


def I(n): return np.array(identity(n, dtype=float))


def cat(a, b, ax): return concatenate((a, b), axis=ax)


def mdiag(a): return np.array(diag(diag(a)))


# def diagtom(a): return diagonal(csr_matrix(a.reshape(-1)))
# def diagtom(a): return csr_matrix(np.diag(np.array(a).reshape(-1)))
def diagtom(a): 
    m,m =  a.shape
    return csr_matrix(dia_matrix((a, 0), shape = (m, m)))


def diagtom2(a):
    c,m = a.shape
    print("***> diagtom c ", c, " m ", m)
    data = np.diag(a)
    ind = [(m+1)*k for k in np.arange(m)]
    b = lil_matrix((m*m, m*m), dtype=np.float64)
    for k in np.arange(m):
        index = (m+1)*k
        b[index,index] = a[k,k]
    return b.tocsr()


def A_to_f(A, nf=64):
    '''Calculates A(f), in the frequency domain.

    Input:
        A(n, n, r) - recurrence matrix (n - number of signals, r - model order)
        nf - frequency resolution

    Output:
        AL(nf, n, n)
    '''

    n, n, r = A.shape
    A = np.array(A)

    # Exponents contains the array of the fft exponents, with all frequencies for all lags
    expoents = (-1j * pi * np.kron(np.arange(nf), (np.arange(r) + 1.0)) / nf).reshape(nf, r)
 
    # Af performs the multiplications of exp(ar) by the matrix A, for all frequencies
    # the repeat and transpose functions are tricks to make possible the vector calculation 
    Af = (A.reshape(n, n, 1, r).repeat(nf, axis=2) \
          * np.exp(expoents)).transpose([2, 0, 1, 3])
    # fft sums the value for all lags
    AL = np.eye(n) - np.sum(Af, axis=3)

    return AL


def vech(a):
    ''' Returns vech(v) '''
    n = a.shape[0]
    v = np.empty((n * (n + 1)) / 2)
    cont = 0
    for j in np.arange(n):
        for i in np.arange(n):
            if i >= j:
                v[cont] = a[i, j]
                cont = cont + 1
    return v


def Dup(n):
    '''D*vech(A) = vec(A), with symmetric A'''
    d = np.zeros([n * n, round((n * (n + 1)) / 2)])
    count = 0
    for j in np.arange(n):
        for i in np.arange(n):
            if i >= j:
                d[j * n + i, count] = 1
                count = count + 1
            else:
                d[j * n + i, :] = d[i * n + j, :]
    return d


def TT(a, b):
    ''' TT(a,b)*vec(B) = vec(B.T), where B is (a x b).'''
    t = O(a * b)
    for i in np.arange(a):
        for j in np.arange(b):
            t[i * b + j, j * a + i] = 1
    return coo_matrix(t, dtype = np.int8)


def fdebig_de(n):
    '''Derivative of kron(I(2n), A) by A'''
    a = sparse.kron(TT(2 * n, n), I(n * 2 * n))
    b = sparse.kron(vec(I(2 * n)), I(n))
    #c = coo_matrix.transpose(sparse.kron(I(n), b ))
    c = sparse.kron(I(n), b )
    return  coo_matrix(a.dot(c), dtype = int8)


def fdebig_de_small(n):
    '''Derivative of kron(I(2), A) by A'''
    return dot(sparse.kron(TT(2, n), I(n * 2)),
               sparse.kron(I(n), sparse.kron(vec(I(2)), I(n))))


def xlag(x, lag):
    if(lag == 0):
        return x.copy()
    xl = np.zeros(x.shape)
    xl[:, lag:] = x[:, :-lag]
    return xl


# def bigautocorr_old(x, p):
#     '''Autocorrelation. Data in rows. From order 0 to p-1.
#     Output: nxn blocks of autocorr of lags i. (Nuttall Strand matrix)'''
#     y = x[:]
#     nd = x.shape[1]
#     for i in arange(1, p):
#         y = concatenate((y, xlag(x, i)), axis=0)
#     return dot(y, y.T) / nd
#     # return cov(y.T)


# Older version
def bigautocorr(x, p):
    '''Autocorrelation. Data in rows. From order 0 to p-1.
    Output: nxn blocks of autocorr of lags i. (Nuttall Strand matrix)'''
    n, nd = x.shape
    gamma = np.zeros([n * p, n * p])
    for i in arange(p):
        for j in arange(p):
            gamma[i * n:i * n + n, j * n:j * n + n] = dot(xlag(x, i), xlag(x, j).T) / nd
    return gamma

# New version
# def bigautocorr(x, p):
#     '''Autocorrelation. Data in rows. From order 0 to p-1.
#     Output: nxn blocks of autocorr of lags i. (Nuttall Strand matrix)'''
#     n, nd = x.shape
#     gamma = np.zeros([n * p, n * p])
#     for i in arange(p):
#         g = dot(x, xlag(x, i).T) / (nd - i)
#         for j in arange(p - i):
#             gamma[(i + j) * n:(i + j) * n + n, j * n:j * n + n] = g.T
#             gamma[j * n:j * n + n, (i + j) * n:(i + j) * n + n] = g
#     return gamma


def fdh_da(Af, n):
    '''Derivative of vec(H) by vec(A), with H = A^-1 and complex A.'''
    ha = Af.I
    h = -kron(ha.T, ha)

    h1 = cat(h.real, -h.imag, 1)
    h2 = cat(h.imag, h.real, 1)
    hh = cat(h1, h2, 0)
    return hh


def fIij(i, j, n):
    '''Returns Iij of the formula'''
    Iij = np.zeros(n**2)
    Iij[n * j + i] = 1
    Iij = diag(Iij)
    return np.array(kron(I(2), Iij))
    #return csr_matrix(np.kron(I(2), Iij))   # ### sparse


def fIj(j, n):
    '''Returns Ij of the formula'''
    Ij = np.zeros(n)
    Ij[j] = 1
    Ij = np.diag(Ij)
    Ij = np.kron(Ij, I(n))
    return np.kron(I(2), Ij)


def fIi(i, n):
    '''Returns Ii of the formula'''
    Ii = np.zeros(n)
    Ii[i] = 1
    Ii = np.diag(Ii)
    Ii = np.kron(I(n), Ii)
    return np.kron(I(2), Ii)


def fCa(f, p, n):
    '''Returns C* of the formula'''
    C1 = cos(-2 * pi * f * np.arange(1, p + 1))
    S1 = sin(-2 * pi * f * np.arange(1, p + 1))
    C2 = cat(C1.reshape(1, -1), S1.reshape(1, -1), 0)
    return np.kron(C2, identity(n**2))


def fChol(omega):
    # Try Cholesky factorization
    try:
#       L = np.linalg.cholesky(omega, lower=1)
        L = np.linalg.cholesky(omega)
    # If there's a small negative eigenvalue, diagonalize
    except LinAlgError:
        val, vec = eigh(omega)
        print('non-positive eig. in omega:', val[val <= 0])
        L = np.zeros(vec.shape)
        for i in np.arange(len(val)):
            if val[i] < 0.:
                val[i] = 0.
            L[:, i] = vec[:, i] * sqrt(val[i])
        # print 'L', L

    return L


def fEig(L, G2):
    '''Returns the eigenvalues'''
    L = np.array(L)
    G2 = np.array(G2)
    D = L.T @ G2 @ L

    u, s, v = np.linalg.svd(D, full_matrices=True) 
    # s, v = np.linalg.eigh(D) 
    # previous version 
    # d = np.linalg.eigh(D, eigvals_only=True)
    # print("d ", s.shape)
    # the two biggest eigenvalues no matter which values are non negative by
    # construction
    d1 = np.sort(s)
    d = d1[-2:]  # two largest elements
    # d = d[abs(d) > 1E-8]
    if (d.size > 2):
        print('more than two chi-square in the sum:')
        print(d)
    return d


# @jit(nopython=True)
def asymp_pdc(x, A, nf, e_var, p, metric='info', alpha=0.01):
    '''Asymptotic statistics for the three PDC formulations
        x -> data
        A -> autoregressive matrix
        nf -> number of frequency
        e_var -> residues
        p -> autoregressive model order
        metric -> witch PDC (iPDC = 'info', gPDC = 'diag', PDC = 'euc')
        alpha -> confidence margin
    '''
        
    print(A)
    
    A = np.array(A)
    #x = np.array(x)
    #e_var = np.array(e_var)
    Af = A_to_f(A, nf)
    n, nd = x.shape   # n: number of channels; nd: data points.

    pdc2 = np.empty([n, n, nf])
    
    if alpha != 0:
        th = np.empty([n, n, nf])
        ic1 = np.empty([n, n, nf])
        ic2 = np.empty([n, n, nf])
        varass1 = np.empty([n, n, nf])
        varass2 = np.empty([n, n, nf])
        patdfr = np.empty([n, n, nf])
        patdenr = np.empty([n, n, nf])
        pvalues = np.empty([n, n, nf])

    start0 = timer()
    # dpdc_dev = np.zeros(int((n*(n + 1))/2))
    if metric == 'euc':
        dpdc_dev = np.zeros(int((n*(n + 1))/2))
    elif metric == 'diag':
        evar_d = mdiag(e_var)
        evar_d_big = np.kron(I(2*n), evar_d)
        # inv_ed = evar_d_big.I
        pinv_evar_d_big = np.linalg.pinv(evar_d_big)

        #'derivative of vec(Ed-1) by vecE'
        de_deh = Dup(n)
        debig_de = fdebig_de(n)  #sparsed
        tmp = -pinv_evar_d_big @ pinv_evar_d_big
        tmp3 = vec3(tmp)
        #dedinv_dev = diagtom(vec(-inv_ed * inv_ed))
        dedinv_dev = diagtom(tmp3.toarray())
        dedinv_deh = dedinv_dev @ debig_de @ de_deh
    elif metric == 'info':
        evar_d = mdiag(e_var)   # (n, n)
        evar_d_big = np.kron(I(2 * n), evar_d)    # (2n^2, 2n^2)

        #pinv_evar_d_big = coo_matrix(np.linalg.pinv(evar_d_big)) #
        pinv_evar_d_big = np.linalg.pinv(evar_d_big) # (2n^2, 2n^2)

        #inv_ed = evar_d_big
        evar_big = np.kron(I(2 * n), e_var) # (2n^2, 2n^2)
        pinv_evar_big = np.linalg.pinv(evar_big)  #  (2n^2, 2n^2)

        'derivada de vec(Ed-1) por vecE'
        de_deh = Dup(n)  # (n^2, sum(1:n))

        debig_de = fdebig_de(n)  # sparsed (4n^4, n^2)
        tmp = -pinv_evar_d_big @ pinv_evar_d_big

        pinv_evar_big = csr_matrix(pinv_evar_big)
        # tmp2 = vec(tmp).toarray()   # (2n^2, 2n^2)
        tmp3 = vec3(tmp)
        dedinv_devd = diagtom(tmp3.toarray()) #sparse # (4n^4, 4n^4) for n=8, duration = 24.34 sec

        dedinv_dehd = dedinv_devd @ (debig_de @ de_deh) #sparse (4n^4, sum(1:n))

        dedinv_dev = csr_matrix(-sparse.kron(pinv_evar_big.T, pinv_evar_big))  # (4n^4, 4n^4)
 
        dedinv_deh = coo_matrix(dedinv_dev @ debig_de @ de_deh) #sparse
        # (4n^4, sum(1:n)) = (4n^4, 4n^4) @ (4n^4, n^2) @ (n^2, sum(1:n))

    else:
        pass
        #raise SystemExit("Error: metric should be either 'euc', 'diag' or 'info'.")

    duration1 = timer() - start0
    start1 = timer()
    
    gamma = bigautocorr(x, p)
    omega = np.kron(inv(gamma), e_var)
    omega_evar = 2 * np.linalg.pinv(Dup(n)) @ np.kron(e_var, e_var) \
                                            @ np.linalg.pinv(Dup(n)).T

    icdf_norm_alpha = st.norm.ppf(1 - alpha / 2.0)

    duration2 = timer() - start1

    start2 = timer()

    print('alpha = ', alpha)

    # pinv_evar_d_big = csr_matrix(pinv_evar_d_big)
    duration_ff = 0
    for ff in range(nf): # Frequency loop
        start_ff = timer()

        f = ff / (2.0 * nf)
        Ca = fCa(f, p, n)

        a = vec2(Af[ff, :, :])
        a = cat(a.real, a.imag, 0)
        #a = vec(cat(I(n), O(n), 1)) - dot(Ca, al)

        omega2 = Ca @ omega @ Ca.T
        L = fChol(omega2)

        for j in range(n): # Channel loop j (columnwise)
            Ij = csr_matrix(fIj(j, n))

            if metric == 'euc':
                Ije = Ij
            elif metric == 'diag':
                Ije = csr_matrix(Ij @ pinv_evar_d_big)
            elif metric == 'info':
                Ije = csr_matrix(Ij @ pinv_evar_big @ Ij)
            else:
                pass

            for i in range(n):  # Channel loop i (rowwise)

                # t0 = timer()
                Iij = fIij(i, j, n)

                # For diag or info case, add evar in the formula'

                if metric == 'euc':
                    Iije = Iij  # = Iij.toarray()  # if fIij returns csr_matrix
                else:  # metric == 'diag' or 'info'
                    Iije = Iij @ pinv_evar_d_big

                # t0 = timer()
                num = a.T @ Iije @ a
                den = a.T @ Ije @ a
                pdc2[i, j, ff] = num / den

                # 'Add evar derivation'
                if metric == 'diag':
                    #'derivada do num por vecE'
                    dnum_dev = sparse.kron((Iij @ a.T).T, a) @ dedinv_deh
                    #'derivada do den por vecE'
                    dden_dev = sparse.kron((Ij @ a).T, a) @ dedinv_deh
                    dpdc_dev = (den * dnum_dev - num * dden_dev) / (den**2)
                elif metric == 'info':
                    #'derivada do num por vecE'
                    dnum_dev = sparse.kron((Iij @ a.T).T, a) @ dedinv_dehd
                    #'derivada do den por vecE'
                    dden_dev = sparse.kron((Ij @ a.T).T, a @ Ij) @ dedinv_deh
                    dpdc_dev = (den * dnum_dev - num * dden_dev) / (den**2)
                else: # metric == 'euc'
                    pass

                t0 = timer()
                G1a = 2 * a.T @ Iije / den - 2 * num * a.T @ Ije / (den**2)
                G1 = -G1a @ Ca

                varalpha = G1 @ omega @ G1.T
                varevar = dpdc_dev @ omega_evar @ (dpdc_dev.T)
                varass1[i, j, ff] = (varalpha + varevar) / nd

                ic1[i, j, ff] = pdc2[i, j, ff] \
                                - sqrt(varass1[i, j, ff]) * icdf_norm_alpha
                ic2[i, j, ff] = pdc2[i, j, ff] \
                                + sqrt(varass1[i, j, ff]) * icdf_norm_alpha
                
                G2a = Iije / den
                #G2 = Ca.T*G2a*Ca

                d = fEig(L.real, G2a.real)

                patdf = sum(d)**2 / sum(d**2)
                patden = sum(d) / sum(d**2)
                th[i, j, ff] = st.chi2.ppf(1 - alpha, patdf) / (patden * nd)
                pvalues[i, j, ff] = 1 - st.chi2.cdf(pdc2[i, j, ff] * patden * nd, patdf)

                varass2[i, j, ff] = patdf / (patden * nd)**2
                patdfr[i, j, ff] = patdf
                patdenr[i, j, ff] = patden
                # print("Duração 4 = ", n*n*nf*(timer()-t0))
        duration_ff = timer() - start_ff

    duration3 = timer() - start2
    print("Duração 1 = ", duration1)
    print("Duração 2 = ", duration2)
    print("Duração 3 = ", duration3)

    #ss = ss_alg(A, e_var, nf)
    ss = []
    #coh = coh_alg(ss)
    coh2 = []

    return pdc2, ss, coh2, th, ic1, ic2, patdfr, patdenr
          #mes, ss, coh2, th, ic1, ic2, patdf, patden 

def asymp_dtf(x, A, nf, e_var, p, alpha=0.05):
    '''Asymptotic statistics for the DTF
        x -> data
        A -> autoregressive matrix
        nf -> number of frequencies
        e_var -> residues
        p -> model order
        alpha -> confidence margin
    '''

    #x = np.array(x)
    #e_var = np.array(e_var)
    Af = A_to_f(A, nf)

    n, nd = x.shape

    th = np.empty([n, n, nf])
    ic1 = np.empty([n, n, nf])
    ic2 = np.empty([n, n, nf])
    dtf2 = np.empty([n, n, nf])
    varass1 = np.empty([n, n, nf])
    varass2 = np.empty([n, n, nf])

    gammai = inv(bigautocorr(x, p))
    omega = kron(gammai, e_var)

    for ff in range(nf):
        f = ff / (2.0 * nf)
        Ca = fCa(f, p, n)

        #Hf = np.array(Af[ff, :, :]).I
        Hf = Af[ff, :, :].I
        h = vec(Hf)
        h = cat(h.real, h.imag, 0)

        #dhda = fdh_da(np.array(Af[ff, :, :]), n)
        dhda = fdh_da(Af[ff, :, :], n)

        #L = fChol(omega)

        omega2 = dhda * Ca * omega * Ca.T * dhda.T
        L = fChol(omega2)

        for i in range(n):
            for j in range(n):

                Iij = fIij(i, j, n)


                Ii = fIi(i, n)

                num = h.T * Iij * h
                den = h.T * Ii * h
                dtf2[i, j, ff] = num / den

                G1a = 2 * h.T * Iij / den - 2 * num * h.T * Ii / (den**2)
                G1 = -G1a * dhda * Ca

                varass1[i, j, ff] = G1 * omega * G1.T / nd
                ic1[i, j, ff] = dtf2[i, j, ff] - sqrt(varass1[i, j, ff]) * st.norm.ppf(1 - alpha / 2)
                ic2[i, j, ff] = dtf2[i, j, ff] + sqrt(varass1[i, j, ff]) * st.norm.ppf(1 - alpha / 2)

                G2a = 2 * Iij / den
                #G2 = Ca.T*dhda.T*G2a*dhda*Ca
                G2 = G2a

                d = fEig(L, G2)
                patdf = sum(d)**2 / sum(d**2)
                patden = sum(d) / sum(d**2)

                th[i, j, ff] = st.chi2.ppf(1 - alpha, patdf) / (patden * 2 * nd)
                varass2[i, j, ff] = 2 * patdf / (patden * 2 * nd)**2

    return dtf2, th, ic1, ic2, patdf, patden


def fc(i, n):
    vi = np.zeros(n)
    vi[i] = 1
    vi.resize(n, 1)
    return np.kron(vi, I(n))


def fk1(e_var_inv, i, j, n):
    ci = fc(i, n)
    cj = fc(j, n)

    return np.kron(I(2), ci) @ e_var_inv @ np.transpose(np.kron(I(2), cj))


def fk2(e_var_inv, i, j, n):
    ci = fc(i, n)
    cj = fc(j, n)

    return np.kron(I(2), ci) @ e_var_inv @ np.transpose(np.kron(np.array([[0, 1], [-1, 0]]), cj))


def fl(i, n):
    vi = np.zeros(n)
    vi[i] = 1
    vi.resize(n, 1)
    return np.kron(I(n), vi)


def fkl1(evar_big, i, j, n):
    li = fl(i, n)
    lj = fl(j, n)

    return np.kron(I(2), li) @ evar_big @ np.transpose(np.kron(I(2), lj))


def fkl2(evar_big, i, j, n):
    li = fl(i, n)
    lj = fl(j, n)

    return np.kron(I(2), li) @ evar_big @ np.transpose(np.kron(array([[0, 1], [-1, 0]]), lj))


def ss_alg(A, e_cov, nf):
    '''Calculates the Spectral density (SS)
         A -> autoregressive matrix
         e_cov -> residues
         nf -> number of frequencies
    '''
    n, n, r = A.shape
    AL = A_to_f(A, nf)
    ss = np.empty(AL.shape)
    for i in range(n):
        #H = inv(reshape(AL[i,:,:],n,n))
        H = inv(AL[i,:,:].reshape(n, n, order='F').copy())
        ss[i,:,:] = H * e_cov * H.T
 
    #print ss[5]
    #SS=permute(ss,[2,3,1])
    SS = ss[:,:,None].transpose([1, 2, 0]) # or equivalently
    # SS = np.transpose( np.expand_dims(ss, axis=2), (1, 2, 0) )
    return SS   #MATLAB    return ss.transpose(1,2,0)


def coh_alg(SS):
    """Coh2 = coh_alg(SS)"""
    m, n, nf = SS.shape
    Coh2 = np.empty([m, n, nf])
    if m == n: 
        nChannels = m
    else:
        print('Wrong SS dimension.')

    for k in range(nf):
        for iu in range(nChannels):
            for ju in range(nChannels):
                Coh2[iu, ju, k] = SS[iu, ju, k] / sqrt(SS[iu,iu,k] * SS[ju, ju, k])

    return Coh2


def fCij(i, j, n, p):
    '''Returns Cij of the formula'''
    Cij = np.zeros([1, n * n])
    Cij[0, (j * n + i)] = 1
    Cij = np.kron(I(p), Cij)
    return Cij


def asymp_gct(x, A, e_var):
    '''Asymptotic statistics for Wald statistic of the GC in time
        x -> data
        A
        e_var
    '''

    x = np.array(x)
    e_var = np.array(e_var)

    n, nd = x.shape
    n, n, p = A.shape

    wt = np.empty([n, n])

    gammai = inv(bigautocorr(x, p))
    omega = kron(gammai, e_var)

    a = vec(A)

    for i in range(n):
        for j in range(n):
            Cij = fCij(i, j, n, p)
            wt[i, j] = (nd * 1.0 / p) * (Cij * a).T * ((Cij * omega * Cij.T).I) * (Cij * a)

    pv = 1 - st.f.cdf(wt, p, nd - n * p - 1)

    return pv, wt


def asymp_igct(e_var, nd):
    '''Asymptotic statistics for Wald statistic of instantaneous GC
        e_var
        nd
    '''

    e_var = np.array(e_var)

    n, n = e_var.shape

    wt = np.zeros([n, n])

    for i in np.arange(n):
        for j in np.arange(n):
            Cij = np.zeros([n, n])
            Cij[i, j] = 1
            Cij[j, i] = 1
            Cij = np.array(vech(Cij))
            eij = e_var[i, j]
            Di = Dup(n).I
            wt[i, j] = nd * eij * (2 * Cij * Di * kron(e_var, e_var) * Di.T * Cij.T).I * eij

    pv = 1 - st.chi2.cdf(wt, 1)

    return pv, wt


def asymp_white(x, res, p, h=20):

    n, nd = res.shape

    x = np.empty([n, n, h + 1])

    for i in np.arange(n):
        for j in np.arange(n):
            dum, aux, dum, dum = xcorr(res[i], res[j], maxlags=h, normed=False)
            x[i, j] = aux[h:]

    s0 = np.array(inv(x[:, :, 0]))
    s = 0
    for i in arange(1, h + 1):
        s = s + (1.0 / (nd - i)) * trace(x[:, :, i].T * s0 * x[:, :, i] * s0)
    s = s * nd**2
    pv = 1 - st.chi2.cdf(s, (h - p) * n**2)

    return pv, s
