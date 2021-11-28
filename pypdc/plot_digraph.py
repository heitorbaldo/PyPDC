'''
This code plots the connectivity digraph such that the node positions are  
in 10-20 system style configuration.
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#fixed positions:
pos = {0: (0,15), 1: (2,16), 2: (4,16), 3: (6, 15), 
       4: (-2,13), 5: (0,12), 6: (2,12), 7: (4,12), 8: (6,12), 9: (8, 13),
       10: (-2,8), 11: (0, 8), 12: (2, 8), 13: (4, 8), 14: (6, 8), 15: (8, 8),
       16: (-2,3), 17: (0, 4), 18: (2, 4), 19: (4, 4), 20: (6, 4), 21: (8, 3),
       22: (2, 0), 23: (4, 0)}

#Setting a 24x24 matrix (24 channels):
def set_matrix(M):
    A = np.zeros((24,24))
    arr = np.array([i for i in range(len(M))])
    A[arr[:, None], arr] += M
    return A

#Plotting the connectivity digraph:
def plot_digraph(M):
    A = set_matrix(M)
    G = nx.from_numpy_matrix(A, create_using=nx.MultiDiGraph())

    plt.figure(3,figsize=(8,8)) 
    nx.draw(G, pos=pos, with_labels=True, node_color="#ffffff", node_size=580,
            connectionstyle='arc3, rad = 0.1', edgecolors='black', font_weight='bold') #, width=weights)
    plt.show()
