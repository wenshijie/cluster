# -*- coding: utf-8 -*-
"""
Created on =2019-03-28

@author: wenshijie
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

n_samples = 1500
random_state = 170
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(12, 12))
# 生成3个聚类中心的数据
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
# 错误的聚类中心
y_clusters1 = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_clusters1)
plt.title('错误的分类--本身三类分两类')

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_clusters2 = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_clusters2)
plt.title('有方向性分布数据分为三类')

# 数据类别不同方差
X_varied, y_varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
y_clusters3 = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_clusters3)
plt.title('正确分类不同方差的数据')

# 各类别数据不平衡
# 第一类取500个，第二类50个，第三类10个
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:50], X[y == 2][:10]))
y_clusters4 = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_clusters4)
plt.title('类别数据不平衡下正确分类')
plt.show()


if __name__ == '__main__':
    print('result')