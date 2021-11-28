# -*- coding:utf-8 -*-
"""
Created on 22/10/2009

@author: Carlos Stein
"""

from numpy import *
import time
from pypdc.ar_fit import nstrand

from pypdc.ar_data import ar_data

def bootstrap(method_func, nd, nm, A, er, 
              nf, alpha = 0.05, metric = None, thres = True):
    '''Bootstrap statistics'''
    
    n = A.shape[0]
    mes = empty([nm, n, n, nf])
    th  = zeros([nm, n, n, nf])
    #ic1 = empty([nm, n, n, nf])
    #ic2 = empty([nm, n, n, nf])
    maxp = A.shape[2]
    tbegin = time.clock()
    for i in range(nm):
        if (i%1 == 0):
            print('nm:', i, 'time:', time.clock()-tbegin)
        #Generate data from AR
        data = ar_data(A, er, nd)
        #Estimate AR parameters with Nuttall-Strand
        Aest, erest = nstrand(data, maxp = maxp)
        #Calculate the connectivity and statistics
        if (metric == None):
            mes[i] = abs(method_func(Aest, erest, nf = nf))**2
        else:
            mes[i] = abs(method_func(Aest, erest, nf = nf, metric = metric))**2
            
    if thres:
        for a in range(n):
            for b in range(n):
                
                if a == b:
                    continue
                
                Anew = A.copy()
                Anew[a,b,:] = 0
                
                for i in range(nm):
                    if (i%1 == 0):
                        print('nm:', i, 'time:', time.clock()-tbegin)
                    #Generate data from AR
                    data = ar_data(Anew, er, nd)
                    #Estimate AR parameters with Nuttall-Strand
                    Aest, erest = nstrand(data, maxp = maxp)
                    #Calculate the connectivity and statistics
                    if (metric == None):
                        aux = abs(method_func(Aest, erest, nf = nf))**2
                        th[i][a][b] = aux[a,b,:]
                    else:
                        aux = abs(method_func(Aest, erest, nf = nf, metric = metric))**2
                        th[i][a][b] = aux[a,b,:]
        

    so = sort(mes, axis = 0)
    ic1 = so[(alpha/2)*nm]
    ic2 = so[(1-alpha/2)*nm]
    #bvar = var(mes, axis = 0)

    if (metric == None):
        mes0 = abs(method_func(A, er, nf = nf))**2
    else:
        mes0 = abs(method_func(A, er, nf = nf, metric = metric))**2
    
    if thres:
        to = sort(th, axis = 0)
        th = to[(1-alpha)*nm]
    else:
        th = ones(ic1.shape)*0#0.5/sqrt(nd) #TODO: pensar melhor como fazer.
    
    return mes0, th, ic1, ic2
