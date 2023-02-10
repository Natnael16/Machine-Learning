
import numpy as np
from matplotlib import pyplot
def kmeans(data, k, max_iterations=100):
    
    n_samples, n_features = data.shape
    cluster_centers = np.random.rand(k, n_features)
    
    cluster_assignments = np.zeros(n_samples)
    
    for iteration in range(max_iterations):
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - cluster_centers[i, :], axis=1)
        
        new_cluster_assignments = np.argmin(distances, axis=1)
        if np.array_equal(new_cluster_assignments, cluster_assignments):
            break
        cluster_assignments = new_cluster_assignments
        
        for i in range(k):
            cluster_centers[i, :] = np.mean(data[cluster_assignments == i, :], axis=0)
    
    return cluster_assignments, cluster_centers

np.random.seed(0)
data = np.array([
    [21,65],[65, 40],[5,75],[25,70],[60,60],
    [15,70],[10,55],[60,50],[40,10],[35,5],
    [48,10],[20,75],[40,20],[25,79],[34,51],
    [22,53],[22,78],[33,59],[33,74],[31,73],
    [22,57],[35,69],[34,75],[67,51],[54,32],
    [57,40],[43,47],[50,53],[57,36],[59,35],
    [52,58],[65,59],[47,50],[49,25],[48,20],
    [35,14],[33,12],[44,20],[45,5],[38,29],
    [43,27],[51,8],[46,7],[7,65],[50,40],
    [15,60],[5,50],[30,30]
])

markers= ["^", "x", "8", "1","+", "p", "o", "s", "6"]

print("For K == 2")
cluster_assignments, cluster_centers = kmeans(data, k=2)

print("Cluster assignments:", cluster_assignments)
print("Cluster centers:", cluster_centers)


for i in range(2):
    pyplot.scatter( data[cluster_assignments == i, 0], data[cluster_assignments == i, 1], marker=markers[i])
    pyplot.scatter( cluster_centers[i][0], cluster_centers[i][1],c="green" )
pyplot.show()

print("For K == 3")
cluster_assignments, cluster_centers = kmeans(data, k=3)

print("Cluster assignments:", cluster_assignments)
print("Cluster centers:", cluster_centers)

for i in range(3):
    pyplot.scatter( data[cluster_assignments == i, 0], data[cluster_assignments == i, 1], marker=markers[i])
    pyplot.scatter( cluster_centers[i][0], cluster_centers[i][1],c="yellow" )
pyplot.show()


print("For K == 4")
cluster_assignments, cluster_centers = kmeans(data, k=4)

print("Cluster assignments:", cluster_assignments)
print("Cluster centers:", cluster_centers)

for i in range(4):
    pyplot.scatter( data[cluster_assignments == i, 0], data[cluster_assignments == i, 1], marker=markers[i])
    pyplot.scatter( cluster_centers[i][0], cluster_centers[i][1],c="red" )
pyplot.show()
