# -*- coding: utf-8 -*-
"""
Created on =2019-03-28

@author: wenshijie
"""
from time import time
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy import ndimage
from sklearn import manifold, datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits(n_class=10)  # images是直接有data转化的
X = digits.data
y = digits.target
n_samples, n_features = X.shape

np.random.seed(0)


def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples ，扩展数据
    # 把x.reshape((8, 8)) 移动0.3*。。。个单位
    shift = lambda x: ndimage.shift(x.reshape((8, 8)), .3 * np.random.normal(size=2), mode='constant',).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])  # 把X沿着横轴切片，一行一行送入shift
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)  # 扩展数据
# Visualize the clustering


def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ----------------------------------------------------------------------
# 2D embedding of the digits dataset


print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")


for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s :\t%.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)


plt.show()