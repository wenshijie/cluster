# -*- coding: utf-8 -*-
"""
Created on =2019-03-29

@author: wenshijie
"""
import time
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成样本数据
np.random.seed(0)
bitch_size = 45
centers = [[1, 1],[-1, -1], [1, -1]]
n_clusters = len(centers)
X, y_true_label = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

# #############################################################################
# Compute clustering with Means
k_means = KMeans(n_clusters=n_clusters)
t0 = time.time()
k_means.fit(X)
t_k_means = time.time()-t0

# #############################################################################
# Compute clustering with MiniBatchKMeans
mbk_means = MiniBatchKMeans(n_clusters=n_clusters, batch_size=bitch_size)
t0 = time.time()
mbk_means.fit(X)
t_mbk_means = time.time()-t0

# #############################################################################
# Plot result

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
mbk_means_cluster_centers = np.sort(mbk_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)  # 返回X每一行距离Y中最近行的行的index
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)  # 如X[n]距离Y[m]最近，返回m
order = pairwise_distances_argmin(k_means_cluster_centers,
                                  mbk_means_cluster_centers)

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k  # 第k类为true其余为false
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
    t_k_means, k_means.inertia_))  # # 运行时间和样本距离聚类中心的平方和

# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == order[k]
    cluster_center = mbk_means_cluster_centers[order[k]]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
         (t_mbk_means, mbk_means.inertia_))

# Initialise the different array to all False
different = (mbk_means_labels == 4)  # 全是False的numpy.ndarray
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    # 两种聚类不相同的变为true，其余还是False
    different += ((k_means_labels == k) != (mbk_means_labels == order[k]))  # k_means的中心与min_k_means距离近的类别相同

identic = np.logical_not(different)  # 取not 让聚类相同同的样本变为true
ax.plot(X[identic, 0], X[identic, 1], 'w',
        markerfacecolor='#bbbbbb', marker='.')  # 不同分类的点，相同分类的点
ax.plot(X[different, 0], X[different, 1], 'w',
        markerfacecolor='m', marker='.')
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

plt.show()


