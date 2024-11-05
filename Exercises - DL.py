## Leetcode ML



#001 Matrix times vector
def dot_mat_vec(mat,vec):
    out = []
    if len(mat[0]) != len(vec):
        return -1
    for row in mat:
        curr_dot = 0
        for i in range(len(vec)):
            curr_dot += row[i] * vec[i]
        
        out.append(curr_dot)
    
    return out


dot_mat_vec([[1,2],[2,4]],[1,2])





#017 KMeans Clustering
def euclidean_distance(point_a,point_b):
    return np.sqrt(((point_a - point_b) ** 2).sum(axis=1)) #row-wise sum
    

import numpy as np
euclidean_distance(np.array([[1,2,3,4],[5,5,6,7]]),[1,2,3,4])



def kmeans_clustering(points,k,initial_centroids,max_iterations):
    """
    For each point, calculate the distances to each and every centroid
    Assign the point to the closest centroid
    Once finished, update centroids by calculate the means of clusters
    """

    points = np.array(points)
    centroids = np.array(initial_centroids)

    for _ in range(max_iterations):
        distances = [euclidean_distance(points,centroid) for centroid in centroids]
        assignments = np.argmin(distances,axis=0) #assign cluster for each point
        new_centroids = [points[assignments == i].mean(axis=0) if sum(assignments==i) > 0 else centroids[i] for i in range(k)]

        centroids = new_centroids
       
    return centroids


kmeans_clustering([(1, 1), (2, 2), (3, 3), (4, 4)], 1, [(0,0)], 10)