# -*- coding: utf-8 -*-
"""
Created on =2019-03-29

@author: wenshijie
"""
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle

cluster_center = [[1, 1], [-1, -1], [1, -1]]
X, y_true_label = make_blobs(n_samples=300, centers=cluster_center, cluster_std=0.5, random_state=0)

af = AffinityPropagation(preference=-50).fit(X)
cluster_center_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters = len(cluster_center_indices)

print('Estimated number of clusters:{0:d}'.format(n_clusters))
print("Homogeneity: {0:0.3f}".format(metrics.homogeneity_score(y_true_label, labels)))
print("Completeness: {0:0.3f}".format(metrics.completeness_score(y_true_label, labels)))
print("V-measure: %0.3f" % metrics.v_measure_score(y_true_label, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y_true_label, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y_true_label, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters), colors):
    class_members = labels == k
    cluster_center = X[cluster_center_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()
