#encoding=utf-8

import numpy as np
import os
import csv
import data

class Node(object):
    def __init__(self):
        self.flag = None
        self.index = -1
        self.children = {}
        self.features = []

class DT(object):
    def __init__(self, x):
        self.head = Node()
        self.head.features = x
        self.createTree(self.head, set())

    def createTree(self, head, Set):
        x = head.features
        c = 0
        for i in range(len(x)):
            if x[i][-1] == 0:
                c += 1
        if c < len(x) / 2:
            head.flag = 1
        else:
            head.flag = 0
        if c == 0 or c == len(x):
            return
        l = len(x[0])-1
        Max = -1
        index = -1
        for i in range(l):
            if i in Set:
                continue
            sum = 0
            xx, _ = self.split(x, i)
            for item in xx:
                sum += self.calc(item)
            if Max == -1 or sum < Max:
                Max = sum
                index = i
        if Max != -1:
            Set.add(index)
            head.index = index
            xx, labels = self.split(x, index)
            for i in range(len(labels)):
                head.children[labels[i]] = Node()
                head.children[labels[i]].features = xx[i]
                self.createTree(head.children[labels[i]], Set)
            Set.remove(index)
        return

    def split(self, x, index):
        Set = set()
        Map = {}
        for i in range(len(x)):
            if x[i][index] not in Set:
                Set.add(x[i][index])
                Map[x[i][index]] = [i]
            else:
                Map[x[i][index]].append(i)
        xx = []
        labels = []
        for item in Set:
            xx.append([])
            labels.append(item)
            for item2 in Map[item]:
                xx[-1].append(x[item2])
        return xx, labels

    def calc(self, x):
        num = len(x)
        c = 0
        for i in range(num):
            if x[i][-1] == 0:
                c += 1
        p = float(c) / num
        if p == 0 or p == 1:
            return 0
        else:
            ans = -(p * np.log2(p)) - (1 - p) * np.log2(1 - p)
            return ans

    def predict(self, x):
        head = self.head
        while len(head.children.keys()) != 0:
            if x[head.index] not in head.children.keys():
                return head.flag
            else:
                head = head.children[x[head.index]]
        return head.flag


if __name__ == '__main__':
    x = data.getTrain()
    dt = DT(x)
    x, id = data.getTest()
    path = './out.csv'
    with open(path, 'w') as f:
        fw = csv.writer(f)
        fw.writerow(['PassengerId', 'Survived'])
        for i in range(len(x)):
            fw.writerow([id[i], dt.predict(x[i])])

