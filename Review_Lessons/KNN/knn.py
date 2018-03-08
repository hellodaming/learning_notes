# -*- coding: utf-8 -*-

"""
关于KNN方法的一个粗略的实现
@version: 0.1
@author: daMing
@license: Apache Licence 2.0
@file: knn.py
@time: 18-3-8 上午10:07
"""
import numpy as np
from scipy.spatial import distance
from collections import Counter


class KNN:
    def __init__(self, train_set, label):
        self.train_set = train_set
        self.label = label

    @staticmethod
    def calculate_distance(array1, array2):
        return distance.euclidean(array1, array2)

    def classify(self, test_set, k):
        dist = np.zeros(len(self.label))

        # 计算test和trian的每一行的距离
        for index, ts in enumerate(self.train_set):
            dist[index] = self.calculate_distance(ts, test_set)

        # zip(距离, label) ,并且根据距离进行排序, 取前k个
        sorted_dist = sorted(zip(dist, self.label), key=lambda x: x[0])
        sorted_dist = sorted_dist[0:k]

        # 获取范围内的所属类别
        list_a, list_b = zip(*sorted_dist)
        counter = Counter(list_b).most_common(1)
        return counter[0][0]


if __name__ == "__main__":
    group = np.array([[1.1, 1.0], [3.1, 0.1], [5.3, 1.4], [0.3, 3.5], [1.3, 4.5], [1.1, 6.5], [3.1, 6.1]])
    labels = ['A', 'A', 'B', 'B', 'C', 'C', 'C']
    KNN_classifier = KNN(group, labels)

    test_set = [1.1, 0.3]
    result = KNN_classifier.classify([1.1, 0.3], 2)
    print('{}属于类别{}'.format(test_set, result))
