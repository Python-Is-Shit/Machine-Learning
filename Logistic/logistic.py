#encoding=utf-8

import numpy as np
import os
import csv

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def Grad(data, labels, out):
    return np.dot((labels - out), data.transpose())

def change(l):
    for i in range(len(l)):
        if l[i] == '':
            l[i] = -1
        else:
            l[i] = float(l[i])
    return l

def find_Max(x):
    return np.max(x, axis=1)

def getTrain():
    path = './train.txt'
    with open(path, 'r') as fr:
        s = fr.readlines()
    labels = np.zeros(shape=(len(s), 1), dtype=np.float32)
    x = np.zeros(shape=(len(s), 8), dtype=np.float32)
    for i in range(len(s)):
        l = s[i].strip().split(' ')
        l = change(l)
        x[i, 0:7] = l[0:7]
        x[i, 7] = 1
        labels[i, :] = l[-1]
    Max = np.max(x, axis=0)
    x /= Max
    return x.transpose(), labels.transpose(), Max

def getWeigths(num, alpha = 0.0001):
    path = './weigths'
    x, labels, Max = getTrain()
    weigths = np.ones(shape=(1, 8), dtype=np.float32)
    if os.path.exists(path):
        weigths = np.loadtxt(path)
    for i in range(num):
        h = sigmoid(np.dot(weigths, x))
        g = Grad(x, labels, h)
        weigths = weigths + alpha * g
    np.savetxt(fname=path, X=weigths)
    return weigths, Max

def getTest(Max):
    path = './test.txt'
    with open(path, 'r') as fr:
        s = fr.readlines()
    id = np.zeros(shape=(len(s)), dtype=np.int)
    x = np.zeros(shape=(len(s), 8), dtype=np.float32)
    for i in range(len(s)):
        l = s[i].strip().split(' ')
        l = change(l)
        id[i] = l[0]
        x[i, 0:7] = l[1:]
        x[i, 7] = 1
    x /= Max
    return x.transpose(), id

def run_forward(weigths, Max):
    x, id = getTest(Max)
    h = sigmoid(np.dot(weigths, x))
    h = np.array(h[0, :] + 0.5, dtype=np.int)
    return h, id


if __name__ == '__main__':
    weigths, Max = getWeigths(1000)
    h, id = run_forward(weigths, Max)
    path = './out.csv'
    with open(path, 'w') as f:
        fw = csv.writer(f)
        fw.writerow(['PassengerId', 'Survived'])
        for i in range(len(h)):
            fw.writerow([id[i], h[i]])

