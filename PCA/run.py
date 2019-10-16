#encoding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData():
    path = '../data/pca_test.txt'
    x = []
    labels = []
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            s = line.strip().split(',')
            if s[-1] == 'Iris-setosa':
                labels.append((1, 0, 0))
            elif s[-1] == 'Iris-versicolor':
                labels.append((0, 1, 0))
            else:
                labels.append((0, 0, 1))
            # if s[-1] == '0':
            #     labels.append((1, 0, 0))
            # elif s[-1] == '1':
            #     labels.append((0, 1, 0))
            # else:
            #     labels.append((0, 0, 1))
            s.pop(-1)
            x.append(map(float, s))
    return x, labels

def PCA(x, k):
    x = np.mat(x)
    mean = np.mean(x, axis=0)
    x_hat = x - mean
    cov = (x_hat.T * x_hat) * 1.0 / (len(x)-1)
    vals, vectors = np.linalg.eig(cov)
    Map = {}
    for i in range(len(vals)):
        Map[vals[i]] = i
    vals = sorted(vals, reverse=True)
    vec = np.zeros(shape=(4, k), dtype=np.float)
    vec = np.mat(vec)
    for i in range(k):
        vec[:, i] = vectors[:, Map[vals[i]]]
    new = x * vec
    return new

if __name__ == '__main__':
    x, labels = loadData()
    new = PCA(x, 2)
    fig = plt.figure()
    for i in range(len(new)):
        plt.scatter(new[i, 0], new[i, 1], color=labels[i])
    # for i in range(len(new)):
    #     plt.scatter(new[i, 0], 0, color=labels[i])
    plt.show()