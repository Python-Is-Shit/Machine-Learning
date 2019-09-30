#encoding=utf-8

import numpy as np
import os
import csv

test_id = [1, 3, 4, 5, 6, 8, 10]
train_id = [2, 4, 5, 6, 7, 9, 11]

def getTrainSet():
    path = '../data/titanic/train.csv'
    Max = [0] * 7
    labels = []
    xx = []
    with open(path, 'r') as fr:
        fr.readline()
        csvfile = csv.reader(fr)
        for s in csvfile:
            flag = True
            for item in train_id:
                if s[item] == '':
                    flag = False
                    break
            if flag == False:
                continue
            labels.append(float(s[1]))
            x = []
            for item in train_id:
                if s[item] == 'male':
                    x.append(1.0)
                elif s[item] == 'female':
                    x.append(0.0)
                elif s[item] == 'C':
                    x.append(0.0)
                elif s[item] == 'Q':
                    x.append(1.0)
                elif s[item] == 'S':
                    x.append(2.0)
                else:
                    x.append(float(s[item]))
                if x[-1] > Max[len(x)-1]:
                    Max[len(x) - 1] = x[-1]
            xx.append(x)
    x = np.array(xx)
    Max = np.array(Max)
    x /= Max
    return x, np.array(labels), Max

def getTest(Max):
    path = '../data/titanic/test.csv'
    xx = []
    id = []
    with open(path, 'r') as fr:
        fr.readline()
        csvfile = csv.reader(fr)
        for s in csvfile:
            id.append(s[0])
            x = []
            for item in test_id:
                if s[item] == '':
                    x.append(0.5 * Max[len(x)])
                elif s[item] == 'male':
                    x.append(1.0)
                elif s[item] == 'female':
                    x.append(0.0)
                elif s[item] == 'C':
                    x.append(0.0)
                elif s[item] == 'Q':
                    x.append(1.0)
                elif s[item] == 'S':
                    x.append(2.0)
                else:
                    x.append(float(s[item]))
            xx.append(x)
    x = np.array(xx)
    x /= Max
    return x, id

def calc(x, y):
    return np.sum(np.square(x - y))

def Mysort(a, b):
    if a[0] < b[0]:
        return -1
    return 1

def predict(x_test, x_train, labels, K):
    dis = []
    for i in range(len(x_train)):
        dis.append((calc(x_test, x_train[i]), labels[i]))
    dis = sorted(dis, cmp=Mysort)
    c = 0
    for i in range(K):
        if dis[i][1] == 1:
            c += 1
    if c > K / 2:
        return 1
    else:
        return 0

def KNN(K):
    x_train, labels, Max = getTrainSet()
    x_test, id = getTest(Max)
    path = './out.csv'
    with open(path, 'w') as f:
        fw = csv.writer(f)
        fw.writerow(['PassengerId', 'Survived'])
        for i in range(len(x_test)):
            fw.writerow([id[i], predict(x_test[i], x_train, labels, K)])




if __name__ == '__main__':
    KNN(25)