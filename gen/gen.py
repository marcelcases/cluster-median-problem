# Creating Test DataSets using sklearn.datasets.make_moon
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