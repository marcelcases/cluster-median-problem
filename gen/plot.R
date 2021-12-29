####################################
# Lab3 dataset cluster plot script #
# Marcel, Mengxue                  #
# Autumn 2021                      #
####################################

library(ggplot2)

X <- read.csv("moons.csv",header = TRUE)
X[1:5,]

ggplot(X, aes(X[,1],X[,2],color=factor(X[,3])))+
  geom_point()+
  labs(title="Moons dataset, 600 points, 2 clusters, CMP (MIP)",x="x1",y="x2",colour="cluster")


# Raw datasets plot

library(patchwork)
                 
data1 <- read.csv("s1.csv",header = TRUE)
plot1 <- ggplot(data1, aes(data1[,1],data1[,2]))+
  geom_point(color='skyblue4')+
  labs(title="S1 dataset",x="x1",y="x2")
data2 <- read.csv("moons.csv",header = TRUE)
plot2 <- ggplot(data2, aes(data2[,1],data2[,2]))+
  geom_point(color='skyblue4')+
  labs(title="Moons dataset",x="x1",y="x2")
data3 <- read.csv("spiral.csv",header = TRUE)
plot3 <- ggplot(data3, aes(data3[,1],data3[,2]))+
  geom_point(color='skyblue4')+
  labs(title="Spiral dataset",x="x1",y="x2")
plot1 + plot2 + plot3



