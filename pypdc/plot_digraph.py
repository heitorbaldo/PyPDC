'''
This code plots the connectivity digraph such that the nodes positions are  
in 10-20 system style configuration.
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#fixed positions :
pos = {0: (0,15), 1: (2,16), 2: (4,16), 3: (6, 15), 
       4: (-2,13), 5: (0,12), 6: (2,12), 7: (4,12), 8: (6,12), 9: (8, 13),
       10: (-2,8), 11: (0, 8), 12: (2, 8), 13: (4, 8), 14: (6, 8), 15: (8, 8),
       16: (-2,3), 17: (0, 4), 18: (2, 4), 19: (4, 4), 20: (6, 4), 21: (8, 3),
       22: (2, 0), 23: (4, 0)}


#fixed positions:
pos_labels_1020 = {"F9": (0,15), "Fp1": (2,16), "Fp2": (4,16), "F10": (6, 15), 
       "T1": (-2,13), "F7": (0,12), "F3": (2,12), "F4": (4,12), "F8": (6,12), "T2": (8, 13),
       "T9": (-2,8), "T3": (0, 8), "C3": (2, 8), "C4": (4, 8), "T4": (6, 8), "T10": (8, 8),
       "P9": (-2,3), "T5": (0, 4), "P3": (2, 4), "P4": (4, 4), "T6": (6, 4), "P10": (8, 3),
       "O1": (2, 0), "O2": (4, 0)}


#Setting a 24x24 matrix (24 channels):
def set_matrix(M):
    A = np.zeros((24,24))
    arr = np.array([i for i in range(len(M))])
    A[arr[:, None], arr] += M
    return A


#Nullifies the principal diagonal
def diag_null(M):
    for i in range(len(M)):
        M[i,i]=0
        
    return M

#Sets an additional threshold to the weights:
def th_weigths(M, th):
    for i in range(len(M)):
        for j in range(len(M)):
            if(M[i, j] < th):
                M[i, j] = 0
            else:
                pass
    return M

#Plotting the weighted connectivity digraph:
def plot_digraph(M):
    A = set_matrix(M)
    B = diag_null(A)
    G = nx.from_numpy_matrix(B, create_using=nx.MultiDiGraph())
    weights = [12*G[u][v][0].get("weight") for u,v in G.edges()]
    
    labels_1020 = {0: "F9", 1: "Fp1", 2: "Fp2", 3: "F10", 4: "T1", 5: "F7", 6: "F3",
               7: "F4", 8: "F8", 9: "T2", 10: "T9", 11: "T3", 12: "C3", 13: "C4",
               14: "T4", 15: "T10", 16: "P9", 17: "T5", 18: "P3", 19: "P4", 20: "T6",
               21: "P10", 22: "O1", 23: "O2"}
    H = nx.relabel_nodes(G, labels_1020)

    plt.figure(3,figsize=(8,8)) 
    nx.draw(H, pos=pos_labels_1020, with_labels=True, node_color="#ffffff", node_size=580,
            connectionstyle='arc3, rad = 0.1', edgecolors='black', font_weight='bold', width=weights)
    plt.show()