#encoding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt

def loadDataSet():
    path = '../data/ex0.txt'
    axis = []
    labels = []
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for item in lines:
            s = item.strip().split()
            axis.append([float(s[0]), float(s[1])])
            labels.append(float(s[2]))
    return np.mat(axis), np.mat(labels)

def getW(axis, labels):
    axis = axis
    labels = labels
    return (axis.T * axis).I * axis.T * labels.T

def Linear_run():
    axis, labels = loadDataSet()
    w = getW(axis, labels)
    flg = plt.figure()
    ax = flg.add_subplot(111)
    ax.scatter(axis[:, 1].flatten().A[0], labels.T[:, 0].flatten().A[0])
    x = axis
    x.sort(0)
    y = axis * w
    ax.plot(x[:,1], y)
    plt.show()


if __name__ == '__main__':
    Linear_run()