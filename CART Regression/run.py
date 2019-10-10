#encoding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self):
        self.ls = None
        self.rs = None
        self.score = None
        self.val = None
        self.index = None
        self.data = None

def createTree(head, data, k, mean):
    head.data = data
    if len(data) < k:
        print mean
        head.score = mean
    else:
        index, v, mean_1, mean_2, data_1, data_2 = calc(data)
        head.ls = Node()
        head.rs = Node()
        createTree(head.ls, data_1, k, mean_1)
        createTree(head.rs, data_2, k, mean_2)
        head.index = index
        head.val = v
    return

def calc(data):
    index = -1
    mean_1 = -1
    mean_2 = -1
    v = -1
    error = -1
    data_1 = []
    data_2 = []
    for i in range(len(data[0])-1):
        Max = np.max(data[:, i])
        Min = np.min(data[:, i])
        val = Min
        while val <= Max:
            l_1 = []
            l_2 = []
            sum_1 = 0
            c_1 = 0
            sum_2 = 0
            c_2 = 0
            e = 0
            for j in range(len(data)):
                if data[j, i] <= val:
                    sum_1 += data[j, -1]
                    c_1 += 1.0
                    l_1.append(list(data[j]))
                else:
                    sum_2 += data[j, -1]
                    c_2 += 1.0
                    l_2.append(list(data[j]))
            m_1 = sum_1 / c_1
            m_2 = sum_2 / c_2
            for item in l_1:
                e += (item[-1] - m_1) ** 2
            for item in l_2:
                e += (item[-1] - m_2) ** 2
            if error == -1 or e < error:
                error = e
                index = i
                v = val
                mean_1 = m_1
                mean_2 = m_2
                data_1 = np.array(l_1)
                data_2 = np.array(l_2)
            val += 0.02
    return index, v, mean_1, mean_2, data_1, data_2

def loadData():
    xx = []
    path = '../data/sine.txt'
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            s = line.strip().split()
            xx.append([float(s[0]), float(s[1])])
    return np.array(xx)

def predict(xx, head):
    labels = []
    for item in xx:
        tmp = head
        while tmp.val != None:
            if item[tmp.index] <= tmp.val:
                tmp = tmp.ls
            else:
                tmp = tmp.rs
        labels.append(tmp.score)
    return np.array(labels)


if __name__ == '__main__':
    xx = loadData()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.scatter(xx[:, 0], xx[:, 1])
    head = Node()
    createTree(head, xx, 5, -1)
    x = []
    f = 0.0
    while f < 1.0:
        x.append([f])
        f += 0.02
    labels = predict(x, head)
    sub.plot(np.array(x), labels, color=(1, 0, 0))
    plt.show()