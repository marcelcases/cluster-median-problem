# OTDM Lab 3: Minimum spanning tree
# Marcel, Mengxue
# Autumn 2021

# from amply import Amply
# from amplpy import AMPL, DataFrame, Environment
import math
import numpy as np


# Load data
# -cmp1

# with open('data/cmp1.dat','r') as f:
#     lines = f.read()
# print(lines)
#cmp1 = Amply(lines)

# data = Amply("")
# data.load_file(open('data/cmp1.dat'))
# data.load_file(open(lines))
# print(data.k)

# data = Amply("""
#     param k := 4;
#     param m := 100;
#     param n := 2;
#     """)

# ampl = AMPL(Environment("D:/Software/ampl_mswin64"))
# ampl.read('cmp.mod')
# ampl.read_data('data/cmp1.dat')

# # print(ampl.get_data("A"))
# vrb = ampl.get_data("A")
# vrb[1]

# s2:

# A = np.array(
#     [[1, 2, 3],
#     [4, 5, 6]], 
#     np.int32)

# Import K, M, N, and A from one of the instances
from data.s1 import *
# from data.s2 import *
# from data.s3 import *
# from data.s4 import *



# print(A[0])
# print(A[:,1])
# print(A)



# Preprocessing
# Calculate n-dimensional Euclidean distances matrix D
D = np.zeros((M,M))
# print(D)
for mi in range(M):
    for mj in range(M):
        partsum = 0
        for n in range(N):
            partsum += (A[mj][n] - A[mi][n])**2
        D[mi,mj] = math.sqrt(partsum)

# print(D[3,5])
# print(D[3][5])
# print(D[5][3])
# print(D)


INF = 9999999
# selected_node = [0, 0, 0, 0, 0]
selected_node = np.zeros((M))
# print(selected_node)
# print(len(selected_node))
selected_node[0] = True

# printing for edge and weight
# print("Edge : Weight\n")
# segments = np.zeros((M-1,5)) # x1 | x2 | y1 | y2 | weight
segments = []
weights = np.zeros((M))
idx = 0
while (idx < M - 1):
    minimum = INF
    a = 0
    b = 0
    for m in range(M):
        if selected_node[m]:
            for n in range(M):
                if ((not selected_node[n]) and D[m,n]):  
                    # not in selected and there is an edge
                    if minimum > D[m,n]:
                        minimum = D[m,n]
                        a = m
                        b = n
    segments.append([(A[a,0],A[a,1]),(A[b,0],A[b,1])])
    weights[idx] = D[a,b]
    # print("("+str(a)+"->"+str(b)+")" + " " + str(D[a][b]))
    selected_node[b] = True
    idx += 1

# print(segments)
# print(weights)



# The k-1 arcs with largest distances will be removed to obtain k clusters form the MST
largest_weights_idx = (np.argpartition(weights, -(K-1))[-(K-1):]).tolist()
print(largest_weights_idx)

# clustered_segments = segments.pop(largest_weights_idx)
clustered_segments = segments.copy()
for i in sorted(largest_weights_idx, reverse=True):
    del clustered_segments[i]
#print(segments)
# print(segments[0])
# print(len(clustered_segments))

import numpy as np
import pylab as pl
from matplotlib import collections  as mc
# lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
# c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

# lc = mc.LineCollection(segments, linewidths=.7,colors="grey")
lc = mc.LineCollection(clustered_segments, linewidths=.7,colors="grey")
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.scatter(A[:, 0], A[:, 1],c="lightblue",linewidth=.7)
ax.autoscale()
ax.margins(0.1)
pl.title("s2 cluster, 625 points, MST (Prim's)")
pl.show()






# from mst_clustering import MSTClustering
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# # create some data with four clusters
# X, y = make_blobs(200, centers=4, random_state=42)

# # predict the labels with the MST algorithm
# model = MSTClustering(cutoff_scale=2)
# labels = model.fit_predict(X)

# # plot the results
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()

# # matplotlib 1.4 + numpy 1.10 produces warnings; we'll filter these
# import warnings; warnings.filterwarnings('ignore', message='elementwise')

# def plot_mst(model, cmap='rainbow'):
#     """Utility code to visualize a minimum spanning tree"""
#     X = model.X_fit_
#     fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
#     for axi, full_graph, colors in zip(ax, [True, False], ['lightblue', model.labels_]):
#         segments = model.get_graph_segments(full_graph=full_graph)
#         axi.plot(segments[0], segments[1], '-k', zorder=1, lw=1)
#         axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2)
#         axi.axis('tight')
#         print(segments[0])
    
#     ax[0].set_title('Full Minimum Spanning Tree', size=16)
#     ax[1].set_title('Trimmed Minimum Spanning Tree', size=16)
#     plt.show()



# # from sklearn.datasets import make_blobs
# # X, y = make_blobs(200, centers=4, random_state=42)
# # plt.scatter(X[:, 0], X[:, 1], c='lightblue')
# # plt.scatter(A[:, 0], A[:, 1], c='lightblue')

# from mst_clustering import MSTClustering
# model = MSTClustering(cutoff_scale=2, approximate=False)
# labels = model.fit_predict(A)
# # print(model.X_fit_)
# # print(model.labels_)
# plot_mst(model)

# labels = model.fit_predict(A)
# print(labels)
# plt.scatter(A[:, 0], A[:, 1], c=labels, cmap='rainbow')
# plt.show()
# print(type(X))
# print(type(A))

# from mst_clustering import MSTClustering
# model = MSTClustering(cutoff_scale=2, approximate=False)
# labels = model.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')

# plot_mst(model)



# X, y = make_blobs(200, centers=4, random_state=42)
# print(X)
# print(y)

