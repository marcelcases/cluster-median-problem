# OTDM Lab 3: Formulation of the Cluster-Median Problem
# Marcel, Mengxue
# Autumn 2021


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

#subject to c3 {j in 1..m}: # less efficient formulation with less constraints
#	m*x[j,j] >= sum{i in 1..m} x[i,j];