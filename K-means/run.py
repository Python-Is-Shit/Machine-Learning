#encoding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import random

class center(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.Set = set()
        self.tmp = set()

def loadData():
    path = '../data/KMeans.txt'
    x = []
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            s = line.split()
            x.append((float(s[0]), float(s[1])))
    return x

class KMeans(object):
    def __init__(self, k, Min, Max):
        self.centers = []
        for i in range(k):
            x = random.random() * (Max - Min) + Min
            y = random.random() * (Max - Min) + Min
            self.centers.append(center(x, y))

    def run(self, xx, n):
        flag = True
        while n > 0 and flag == True:
            flag = False
            for item in self.centers:
                item.Set.clear()
            for item in xx:
                index = -1
                Min = 0x7fffffff
                for i in range(len(self.centers)):
                    dis = (item[0] - self.centers[i].x) ** 2 + (item[1] - self.centers[i].y) ** 2
                    if dis < Min:
                        Min = dis
                        index = i
                self.centers[index].Set.add(item)
            for item in self.centers:
                x, y = self.calc(item)
                if (x, y) != (item.x, item.y):
                    flag = True
                    item.x = x
                    item.y = y
            n -= 1
        return

    def calc(self, cen):
        x = 0.0
        y = 0.0
        if len(cen.Set) == 0:
            return 0.0, 0.0
        for item in cen.Set:
            x += item[0]
            y += item[1]
        x /= len(cen.Set)
        y /= len(cen.Set)
        return x, y

if __name__ == '__main__':
    x = loadData()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    for item in x:
        sub.scatter(item[0], item[1], color=(0, 0, 1))
    KM = KMeans(3, -5, 5)
    KM.run(x, 100)
    for item in KM.centers:
        sub.scatter(item.x, item.y, color=(1, 0, 0))
    plt.show()