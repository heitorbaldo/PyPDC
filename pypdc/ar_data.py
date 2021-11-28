# -*- coding:utf-8 -*-
"""
Created on 06/11/2009

@author: Carlos Stein
"""

from numpy import *
from numpy.random import randn
from numpy.random import multivariate_normal as mnorm

import time

# from scipy import weavee
# from scipy.weave import converters

y = array([[1936, 1.0, 0.9,  40],
           [1937, 0.8, 0.8, 115],
           [1938, 0.8, 0.8, 100],
           [1939, 1.4, 1.3,  80],
           [1940, 1.2, 1.4,  60],
           [1941, 1.0, 1.2,  40],
           [1942, 1.5, 1.7,  23],
           [1943, 1.9, 1.8,  10],
           [1944, 1.5, 1.6,  10],
           [1945, 1.5, 1.5,  25],
           [1946, 1.5, 1.5,  75],
           [1947, 1.6, 2.0, 145],
           [1948, 1.8, 2.5, 130],
           [1949, 2.8, 2.7, 130],
           [1950, 2.5, 2.9,  80],
           [1951, 2.5, 2.5,  65],
           [1952, 2.4, 3.1,  20],
           [1953, 2.1, 2.4,  10],
           [1954, 1.9, 2.2,   5],
           [1955, 2.4, 2.9,  10],
           [1956, 2.4, 2.5,  60],
           [1957, 2.6, 2.6, 190],
           [1958, 2.6, 3.2, 180],
           [1959, 4.4, 3.8, 175],
           [1960, 4.2, 4.2, 120],
           [1961, 3.8, 3.9,  50],
           [1962, 3.4, 3.7,  35],
           [1963, 3.6, 3.3,  20],
           [1964, 4.1, 3.7,  10],
           [1965, 3.7, 3.9,  15],
           [1966, 4.2, 4.1,  30],
           [1967, 4.1, 3.8,  60],
           [1968, 4.1, 4.7, 105],
           [1969, 4.0, 4.4, 105],
           [1970, 5.2, 4.8, 105],
           [1971, 5.3, 4.8,  80],
           [1972, 5.3, 4.8,  65]])
sun = y[:, [3, 2]].transpose()

''' Some pre-specified models for ready use.
    ar_models(2) is the sunspot-melanoma data.
    lam is the free parameter is models that have one.'''


def ar_models(id, lam=0.0):

    models = [
        # 0
        [array([[[0.2, 0], [0, 0], [0.3, -0.2]],
                [[0, 0], [0.8, -0.1], [0.4, -0.1]],
                [[0, 0], [-0.1, 0.2], [0.4, 0.1]]], dtype=float),
         identity(3)],
        # 1
        [array([[[4, -4], [3, 3]], [[0, -2], [2, -3]]], dtype=float).reshape(2, 2, 2) / 20,
            array([[0.7, 0], [0, 2]], dtype=float)],
        # 2 sunspot melanoma
        sun,
        # 3
        [array([[[4, 3, -2], [-2, -5, 3]], [[4, -2, 1], [-4, 0, 3]]], dtype=float).reshape(2, 2, 3) / 20,
            array([[0.7, 0.3], [0.3, 2]], dtype=float)],
        # 4 JAS Daniel (12)
        [array([[[0.2, 0], [-0.4, -0.2], [0.3, 0]],
                [[lam, 0], [0.8, -0.1], [0.4, 0.0]],
                [[0, 0.5], [-0.1, 0.2], [0.4, 0.1]]], dtype=float),
         identity(3)],

    ]
    return models[id]

# def ar_data_R(A, er = None, m = 1000):
# #   from rpy2.robjects import r as r_
# #   import rpy2.rinterface as ri_
# #   import rpy2.robjects as ro_
#
#    '''Simulate ar-model from A matrix
#
#      Input:
#        A(n, n, p) - AR model (n - number of signals, p - model order)
#        er(n) - variance of innovations
#        m - length of simulated time-series
#
#      Output:
#        data(n, m) - simulated time-series
#    '''
#    if er == None:
#        er = ones(A.shape[0])
#
#    ri_.initr()
#    r_('library(dse1)')
#
#    n = A.shape[0]
#    A = concatenate((eye(n).reshape(n,n,1), -A), axis = 2)
#    ri_.globalEnv["A"] = ri_.FloatSexpVector(A.ravel())
#    ri_.globalEnv["dim"] = ri_.IntSexpVector(A.shape[::-1])
#    ri_.globalEnv["er"] = ri_.FloatSexpVector(er)
#    ro_.globalEnv["m"] = m
#    ro_.globalEnv["n"] = n
#    return array(r_('simulate(ARMA(A = array(A, dim), B = diag(1,n)), sd = sqrt(er), sampleT = m)$output')).T

# def ar_data_old(A, er=None, m=1000, dummy=100):


def ar_data(A, er=None, m=1000, dummy=100):
    '''Simulate ar-model from A matrix

      Input:
        A(n, n, p) - AR model (n - number of signals, p - model order)
        er(n) - variance of innovations
        m - length of simulated time-series

      Output:
        data(n, m) - simulated time-series
    '''

    if (A.ndim == 2):
        A.resize(A.shape[0], A.shape[1], 1)

    n = A.shape[0]
    p = A.shape[2]
    if er.any() == None:
        er = identity(n)
    if er.ndim == 1:
        er = diag(er)

    print(time.clock())
    w = mnorm(zeros(n), er, m + dummy - p)
    print(time.clock())
    data = zeros([n, m + dummy])
    for i in arange(p, m + dummy):
        for j in arange(p):
            data[:, i] = data[:, i] + dot(A[:, :, j], data[:, i - j - 1])
        data[:, i] = data[:, i] + w[i - p]
    print(time.clock())

    return data[:, dummy:]

# def ar_data(A, er = None, m = 1000, dummy = 100):
#     '''Simulate ar-model from A matrix
#
#       Input:
#         A(n, n, p) - AR model (n - number of signals, p - model order)
#         er(n) - variance of innovations
#         m - length of simulated time-series
#
#       Output:
#         data(n, m) - simulated time-series
#     '''
#
#     if (A.ndim == 2):
#         A.resize(A.shape[0], A.shape[1], 1)
#
#     n = A.shape[0]
#     p = A.shape[2]
#     if er is None:
#         er = identity(n)
#     if er.ndim == 1:
#         er = diag(er)
#
#     w = mnorm(zeros(n), er, m+dummy-p)
#     data = zeros([n, m+dummy])
#     code = '''
#         for (int i = p; i < m+dummy; i++) {
#             for (int j = 0; j < p; j++) {
#                 for (int k = 0; k < n; k++) {
#                     double s = 0;
#                     for (int t = 0; t < n; t++) {
#                         data[t,i];
#                         s += A(k,t,j)*data(t,i-j-1);
#                     }
#                     data(k,i) = data(k,i) + s;
#                 }
#             }
#             for (int k = 0; k < n; k++) {
#                 data(k,i) += w(i-p,k);
#             }
#         }
#     '''
#     weave.inline(code, ['data', 'p', 'm', 'dummy', 'n', 'A', 'w'],
#                        type_converters=converters.blitz, compiler = 'gcc')
#
#
#     return data[:,dummy:]
