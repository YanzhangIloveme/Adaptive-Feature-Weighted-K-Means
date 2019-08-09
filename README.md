# Adaptive-Feature-Weighted-K-Means

Afw_kmeans is an improvement of clustering method based on K-Means. This module is the algorithm implementation of the paper 'K-means Clustering Algorithm Based on Adaptive Feature Weighted'.

Compared with K-Means, Afw-Kmeans has the ablitity to learn the feature weights throught every iteration. The weighted loss function could generate more precise and compact clusters and we can also learn the feature importances based on this algorithm.

The improvement of Afw-Kmeans:

(1) The selection of initial centroids: rather than randomness, the initial centroids are selected based on the mean and standard deviation. The initial points could calculated as the formular below:

     C = {mean +- 2v/(k-1) *j, j = 1,2,3...,k/2} U {mean}, when k is odd and 
     C = {mean +- 2v/k * j ,   j = 1,2,3...,k/2}, when k is even  where v stands for standard deviation vector of the all features and mean stads for the average value of features vector; k stands for the cluster centroids.

(2) weighted loss fuction: In stead of use Euclid Distance, the loss fuction of Afw-KMeans is 

     d(m, n) = sqrt(sum( w * (x_m - x_n)**2))) where w stands for the vector of weights for features.
     
(3) Adaptive weights: The weights of features are calculated based on the value of custering process of this feature. The value(C) can be calculated as

     C =  sum((M_j - Mj)**2)/sum_k(sum_n((X_kj-M_kj)**2))  where M_kj stands for the average value of data within centroids k on feature j and M_j stands for the average value of feature j. Therefore the value C of all features stands for the (distance between gropus) devided by (distance within groups).
     
In terms of the test data,I deployed this moethod on Iris database to cluster the classes of the flower and compared this result with sklearn-kmeans. The accuracy of Kmeans using sklean is 89.33% while the accuracy of AFW-kmeans is 94.67%.

This shows the Afw-kmeans has improved the kmeans method.
As I am not a specialist of programmer, of course the Implementation process can be improved I think. I hope this module can help you in some case.

Author: Yan
