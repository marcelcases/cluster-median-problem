# OTDM Lab 3: Minimum spanning tree
# Marcel, Mengxue
# Autumn 2021

import math, time
import numpy as np
import pylab as pl
import matplotlib.collections as mc

# Import K, M, N, and A from one of the datasets
# from data.s1 import *
from data.spiral import *
# from data.moons import *


# Calculate n-dimensional Euclidean distances matrix D
def dist():
    D = np.zeros((M,M))
    # print(D)
    for mi in range(M):
        for mj in range(M):
            partsum = 0
            for n in range(N):
                partsum += (A[mj][n] - A[mi][n])**2
            D[mi,mj] = math.sqrt(partsum)
    return D

# Prim's MST algorithm
def prims():
    # Calculate MST
    node = np.zeros((M)) # initial node
    node[0] = True
    print("Edge : Weight\n")
    segments = []
    weights = np.zeros((M))
    idx = 0
    while (idx < M-1):
        incumbent = 10**8 # current minimum value, initial set to +inf
        x1,x2 = 0,0
        for m1 in range(M):
            if node[m1]:
                for m2 in range(M):
                    if ((not node[m2]) and D[m1,m2]):  # node not in selected, edge present
                        if incumbent > D[m1,m2]:
                            incumbent = D[m1,m2]
                            x1,x2 = m1,m2
        segments.append([(A[x1,0],A[x1,1]),(A[x2,0],A[x2,1])])
        weights[idx] = D[x1,x2]
        print("("+str(A[x1,0])+","+str(A[x1,1])+") -> " + "("+str(A[x2,0])+","+str(A[x2,1])+")" + " : " + str(D[x1][x2]))
        node[x2] = True
        idx += 1
    # The k-1 arcs with largest distances will be removed to obtain k clusters form the MST
    largest_weights_idx = (np.argpartition(weights, -(K-1))[-(K-1):]).tolist()
    clustered_segments = segments.copy()
    for i in sorted(largest_weights_idx, reverse=True):
        del clustered_segments[i]
    return clustered_segments,weights

def plotMST():
    lc = mc.LineCollection(clustered_segments, linewidths=1.5,colors="dimgray") #linewidths=1.5
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.scatter(A[:, 0], A[:, 1],c="lightblue",linewidth=.7)
    ax.autoscale()
    ax.margins(0.1)
    pl.title("Moons dataset, 600 points, 2 clusters, MST (Prim's)")
    pl.show()

if __name__ == "__main__":
    D = dist()
    tick = time.time()
    clustered_segments,weights = prims()
    print("\nTime: %s seconds" % (time.time() - tick))
    print("Obj.: ", sum(weights))
    plotMST()
