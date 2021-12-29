# The cluster-median problem

The objective of this project is to implement different cluster analysis tools in order to separate data from three datasets. It is part of the course on integer optimization.

## About

**Course**  
Optimization Techniques for Data Mining (OTDM-MIRI)  
FIB - Universitat Politècnica de Catalunya. BarcelonaTech  
December 2021 

**Team**  
* Marcel Cases
&lt;marcel.cases@estudiantat.upc.edu&gt;
* Mengxue Wang
&lt;mengxue.wang@estudiantat.upc.edu&gt;

## Introduction

Cluster analysis or clustering is a method of unsupervised learning and a common technique for statistical data analysis widely used in many fields, including machine learning, data mining, pattern recognition, image analysis and bioinformatics. This project deals with two particular cases of clustering known as the Cluster-Median Problem (CMP) and the Minimum Spanning Tree (MST).

In this project, we implement the mathematical formulation of the Cluster-Median Problem (as a Mixed Integer Program) with AMPL. We also implement another solution for clustering, an heuristic approach of the Minimum Spanning Tree using Prim's algorithm developed with Python. We apply it to three different datasets of different sizes that contain data distributed in different shapes, and finally we compare the obtained results and make conclusions, both visually and analytically.

## Models

### Cluster-Median Problem

AMPL code of the preprocessing script that computes Euclidean distances matrix (contained in *cmp.run*):



AMPL code of the Cluster-Median Problem formulation *(cmp.mod)*:



### Minimum Spanning Tree



## Datasets

We use three different datasets to test the performance of the two rpeviously described clustering techniques. Although the models accept any n-dimensional dataset, we will use only two-dimensional data for visualization purposes and to obtain conclusions from the visual analysis.

The plots below show an overview of the datasets:

![Datasets](./img/raw.png)

### S1 dataset

S1 dataset was obtained from a [repository by the University of Eastern Finland - Joensuu](http://cs.joensuu.fi/sipu/datasets/) (credited to *P. Fränti and O. Virmajoki, "Iterative shrinking method for clustering problems", Pattern Recognition, 39 (5), 761-765, May 2006*). It consists of 15 Gaussian clusters with different degree of cluster overlap. Samples have been generated synthetically. For our project, we reduced the size from 5000 to 625 samples, and we applied rescaling (min-max normalization) to avoid working with extremely large objective function values.

### Moons dataset

Moons dataset consists of two interleaving half circles. It was generated using `sklearn` with the script below, and contains 600 samples with some noise. Each one of the half-moons makes up one cluster (two clusters in total). 

````python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples = 600, shuffle=True, noise = 0.09)
````

### Spiral dataset

Spiral dataset was also obtained from the repository of the University of Eastern Finland (credited to *H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203*). It contains three spirals (three clusters) that are also interleaved. It is made of 312 points.

### Datasets summary




## Results and analysis

### S1 dataset

![s1-cmp](./img/s1-cmp.png)

![s1-mst](./img/s1-mst.png)

### Moons dataset

![moons-cmp](./img/moons-cmp.png)

![moons-mst](./img/moons-mst.png)

### Spiral dataset

![spiral-cmp](./img/spiral-cmp.png)

![spiral-mst](./img/spiral-mst.png)

### Performance and stats

| Dataset | Samples (M) | Dimensions (N) | Clusters (K) | Algorithm | Obj. value | Time (s) |
|---------|-------------|----------------|--------------|-----------|------------|----------|
| S1      | 625         | 2              | 15           | CMP       | 24.08      | 97.20    |
| S1      | 625         | 2              | 15           | MST       | 9.61       | 26.58    |
| Moons   | 600         | 2              | 2            | CMP       | 364.31     | 192.98   |
| Moons   | 600         | 2              | 2            | MST       | 24.91      | 22.50    |
| Spiral  | 312         | 2              | 3            | CMP       | 1811.10    | 11.42    |
| Spiral  | 312         | 2              | 3            | MST       | 188.62     | 3.18     |

## Conclusions

MST more adaptive to data with determined shapes (worms,...). Given its greedy approach, tendency to leave clusters with a single element. Does not guarantee balanced clusters. Good for data discovery
CMP finds more fairly balanced clusters.

## Files in the project

| File            | Description                                                                                                                           |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `cmp.mod`       | AMPL code of the CMP model                                                                                                            |
| `cmp.run`       | AMPL runfile, with preprocessing, solving and post-processing (data reporting)                                                    |
| `mst.py`        | Prim's algorithm of the MSP in Python. Contains preprocessing, the algorithm, plotting and data reporting                                             |
| `data/*.dat`    | AMPL input data for each dataset                                                                                                      |
| `data/*.py`     | Python input data for each dataset                                                                                                    |
| `gen/gen.py`    | Moons dataset generator                                                                                                               |
| `gen/plot.r`    | R script to plot CMP (AMPL) results                                                                                                   |
| `gen/*.csv`     | Output data from CMP (AMPL) (coord. + cluster), adapted to be read by `gen/plot.r`                                                    |
| `img/*`         | Plots of the CMP and MST results                                                                                                      |
| `out/*cmp.txt` | AMPL output logs of the CMP. Contains which cluster each point belongs to, num. of MIP simplex iterations, exec. time and obj.f value |
| `out/*mst.txt` | Python output logs of the MST. Contains all computed segments, weights, exec. time and obj.f value                                    |

## References

Task statement  
Class slides  
AMPL documentation  
Nocedal, J.; Wright, S.J. *Numerical optimization*  

## Annex I: AMPL code of the Euclidean distances matrix computation

````AMPL
param partsum default 0;
for {i in {1..m}} {
	for {j in {1..m}} {
		for {dim in {1..n}} {
			let partsum := partsum + (A[j,dim] - A[i,dim])^2
		}
		let D[i,j] := sqrt(partsum);
		let partsum := 0;
	}
}
````

## Annex II: AMPL code of the Cluster-Median Problem formulation

````AMPL
param k;
param m;
param n;

param A{i in 1..m, j in 1..n};
param D{i in 1..m, j in 1..m};


var x{i in 1..m, j in 1..m} binary;

# Distance of all points to their cluster medians
minimize objf:
	sum{i in 1..m, j in 1..m} D[i,j]*x[i,j];
	
# Every point belongs to one cluster
subject to c1 {i in 1..m}:
	sum{j in 1..m} x[i,j] = 1;
	
# Exactly k clusters
subject to c2:
	sum{j in 1..m} x[j,j] = k;

# A point may belong to a cluster only if the cluster exists
subject to c3 {i in 1..m, j in 1..m}:
	x[j,j] >= x[i,j];
````

## Annex III: Python code of the Euclidean distances matrix computation

````python
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
````

## Annex IV: Python code of the Prim's Minimum Spanning Tree algorithm

````python
def prims():
    # Calculate MST
    INF = 9999999
    selected_node = np.zeros((M))
    selected_node[0] = True
    # print("Edge : Weight\n")
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
    # The k-1 arcs with largest distances will be removed to obtain k clusters form the MST
    largest_weights_idx = (np.argpartition(weights, -(K-1))[-(K-1):]).tolist()
    clustered_segments = segments.copy()
    for i in sorted(largest_weights_idx, reverse=True):
        del clustered_segments[i]
    return clustered_segments,weights
````

## Annex V: Python code of the Moons dataset generator

````python
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from matplotlib import style
from numpy import savetxt
 
X, y = make_moons(n_samples = 600, shuffle=True, noise = 0.09)
plt.scatter(X[:, 0], X[:, 1], s = 40, color ='g')
plt.xlabel("X")
plt.ylabel("Y")

print(X)

savetxt('moons.dat', X, delimiter=',', fmt='%1.6f')
 
plt.show()
plt.clf()
````

## Annex VI: R code of the CMP plotter

````r
library(ggplot2)

X <- read.csv("moons.csv",header = TRUE)
X[1:5,]

ggplot(X, aes(X[,1],X[,2],color=factor(X[,3])))+
  geom_point()+
  labs(title="Moons dataset, 600 points, 2 clusters, CMP (MIP)",x="x1",y="x2",colour="cluster")
````

## Annex VII: Python code of the MST plotter

````python
def plotMST():
    lc = mc.LineCollection(clustered_segments, linewidths=1.5,colors="dimgray") #linewidths=1.5
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.scatter(A[:, 0], A[:, 1],c="lightblue",linewidth=.7)
    ax.autoscale()
    ax.margins(0.1)
    pl.title("Moons dataset, 600 points, 2 clusters, MST (Prim's)")
    pl.show()
````




