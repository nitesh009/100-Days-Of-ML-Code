### Clustering is similar to classification, but the basis is different. 

In Clustering you don’t know what you are looking for, and you are trying to identify ** some segments or clusters  **
in your data. 
When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, 
clusters and groupings you would have never thought of otherwise.

---
> In this part, you will understand and learn how to implement the following Machine Learning Clustering models:

* K-Means Clustering
* Hierarchical Clustering

---

> Steps

* Choose the number K of clusters
* Select at random K points, the centroids(not necessarily from your dataset)
* Assign each data point to the closest centroid - that forms K clusters
* Compute and place the new centroid of each cluster
* Reassign each data point  to the new closest centroid .
If reassignment took place, go to STEP 4, otherwise go to FINISH.
