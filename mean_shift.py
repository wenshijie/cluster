# -*- coding: utf-8 -*-
"""
Created on =2019-03-29

@author: wenshijie
"""
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

centers = [[1, 1], [-1, -1], [1, -1]]
X, y_true_label = make_blobs(n_samples=10000, cluster_std=0.6, centers=centers)

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
# #############################################################################
# Compute clustering with MeanShift
ms = MeanShift(bandwidth=bandwidth, bin_seeding= True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(cluster_centers)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()