#encoding=utf-8

import numpy as np
import os
import data
import csv

class Node(object):
    def __init__(self, index, val, weigth, dir):
        self.index = index
        self.val = val
        self.weigth = weigth
        self.dir = dir
        self.next = None

class AdaBoost(object):
    def __init__(self, k):
        self.k = k
        self.classifier = None
        self.MAX = [3, 1, 3, 1, 1, 3, 2]
        self.num_f = 7

    def train(self):
        xx = data.getTrain()
        weigth = np.ones(shape=(len(xx)), dtype=np.float) / np.float(len(xx))
        head = self.classifier
        for i in range(self.k):
            error, index, val, dir = self.calc(xx, weigth)
            w = 1.0 / 2 * np.log((1 - error) / error)
            if self.classifier == None:
                self.classifier = Node(index, val, w, dir)
                head = self.classifier
            else:
                head.next = Node(index, val, w, dir)
            flag = False
            for j in range(len(xx)):
                if self.predict(xx[j]) != xx[j][-1]:
                    flag = True
                    break
            if flag == False:
                break
            s = np.sum(weigth)
            for j in range(len(xx)):
                if dir == 0:
                    if (xx[j][index] < val and xx[j][-1] == 1) or (xx[j][index] >= val and xx[j][-1] == 0):
                        weigth[j] = weigth[j] * np.exp(w) / s
                    else:
                        weigth[j] = weigth[j] * np.exp(-w) / s
                else:
                    if (xx[j][index] < val and xx[j][-1] == 1) or (xx[j][index] >= val and xx[j][-1] == 0):
                        weigth[j] = weigth[j] * np.exp(-w)
                    else:
                        weigth[j] = weigth[j] * np.exp(w) / s

    def calc(self, xx, weigth):
        Min = -1
        index = -1
        v = -1
        dir = 0
        for i in range(self.num_f):
            for val in range(self.MAX[i]+1):
                tmp = 0
                for j in range(len(xx)):
                    if xx[j][i] < val and xx[j][-1] == 1:
                        tmp += weigth[j]
                    elif xx[j][i] >= val and xx[j][-1] == 0:
                        tmp += weigth[j]
                if Min == -1 or tmp < Min:
                    Min = tmp
                    index = i
                    dir = 0
                    v = val
                tmp = 0
                for j in range(len(xx)):
                    if xx[j][i] < val and xx[j][-1] == 0:
                        tmp += weigth[j]
                    elif xx[j][i] >= val and xx[j][-1] == 1:
                        tmp += weigth[j]
                if tmp < Min:
                    Min = tmp
                    index = i
                    dir = 1
                    v = val
        return Min, index, v, dir

    def predict(self, x):
        score = 0.0
        head = self.classifier
        while head != None:
            index = head.index
            w = head.weigth
            val = head.val
            dir = head.dir
            if dir == 0:
                if x[index] >= val:
                    score += w
            else:
                if x[index] < val:
                    score += w
            head = head.next
        if score > 0.5:
            return 1
        else:
            return 0

    def forward(self):
        xx, id = data.getTest()
        path = './out.csv'
        with open(path, 'w') as f:
            fw = csv.writer(f)
            fw.writerow(['PassengerId', 'Survived'])
            for i in range(len(xx)):
                fw.writerow([id[i], self.predict(xx[i])])



if __name__ == '__main__':
    AB = AdaBoost(10)
    AB.train()
    AB.forward()