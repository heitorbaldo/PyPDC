# -*- coding:utf-8 -*-
"""
Created on 06/11/2009

@author: Carlos Stein
"""

from numpy import *
import scipy.signal as sig
import time

from pypdc.analysis import pdc_alg, pdc_and_plot
from pypdc.plotting import pdc_plot
from matplotlib.pyplot import gcf

def aPDC(data, p, step = 50, se = 100, preproc = False):
    '''data(m,n,nd) -> data, m = #trials, n = #channels, nd = #time samples
       p -> VAR model order
       se -> efective sample memory of adaptative model'''

    m,n,nd = data.shape

    if preproc:
        data = pre_proc_ding_99(data)

    A, er = AMVAR(data, p, se)

    for i in arange(p, nd, step):
        pd = pdc_alg(A[i], er[i], metric = 'diag')
        pdc_plot(pd)
        canvas = gcf().canvas
        canvas.start_event_loop(timeout=0.2)

        #time.sleep(1)

    return A, er

def AMVAR (data, p, se = 100):
    '''data(m,n,nd) -> data, m = #trials, n = #channels, nd = #samples'''

    m,n,nd = data.shape

    A = zeros([nd,n,n*p])
    er = zeros([nd,n,n])
    era = zeros([nd,n,n])
    C = mat(identity(n*p))

    #Stein modification: (usar cf em torno de 0.02 para um AR(2,2,2))
    cf = 2.0*m/float64(se+m)
    print(cf)

    for i in arange(p,nd):
        C = (1.0/(1.0-cf))*C
        if i == p:
            Wt = mat(data[:,:,i-1::-1].transpose(0,2,1).reshape(m,-1))
        else:
            Wt = mat(data[:,:,i-1:i-p-1:-1].transpose(0,2,1).reshape(m,-1))

        for j in arange(m):
            C = C*(identity(n*p) - Wt[j].T*Wt[j]*C/(Wt[j]*C*Wt[j].T + 1.0))

        K = Wt*C
        #Yn = data(:,:,i)
        Z = data[:,:,i] - Wt*A[i-1].T
        A[i] = A[i-1] + Z.T*K

        if cf > 0:
            er[i] = (1-cf)*er[i-1] + (cf/float64(m))*Z.T*Z
            era[i] = er[i]/(1.0-(1.0-cf)**i)
        else:
            er[i] = (n/(n+1.0))*er[i-1] + (1.0/(m*(n+1.0)))*Z.T*Z
            era[i] = er[i]

        #if (i%(nd/10) == 0 and i*m > 200):
        #    print 'nd', i
        #    print 'C', C
        #    #print 'Z', Z
        #    print 'dA', Z.T*K


    er = era
    A = A.reshape(nd,n,p,n).transpose(0,1,3,2)

    return A, er


def pre_proc_ding_99(data):

    m,n,nd = data.shape

    #normalize per trial
    data = sig.detrend(data)
    data = data/std(data, axis = 2).reshape(m,n,1)

    #normalize per time
    if m > 1:
        data = data - mean(data, 0).reshape(1,n,nd)
        data = data/std(data, axis = 0).reshape(1,n,nd)

    return data
