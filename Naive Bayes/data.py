#encoding=utf-8

import numpy as np
import csv
import os

train_id = [2, 4, 5, 6, 7, 9, 11]

mid = [1, 0, 1, 0, 0, 1, 1]

test_id = [1, 3, 4, 5, 6, 8, 10]

def getTrain():
    path = '../data/titanic/train.csv'
    Map = {}
    Set = set()
    c_1 = 0.0
    c_0 = 0.0
    n = 0.0
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
            n += 1.0
            if s[1] == '0':
                c_0 += 1.0
            elif s[1] == '1':
                c_1 += 1.0
            for i in range(len(train_id)):
                item = train_id[i]
                tmp = 0
                if item == 5:
                    age = float(s[item])
                    if age < 15:
                        tmp = 0
                    elif age < 30:
                        tmp = 1
                    elif age < 45:
                        tmp = 2
                    else:
                        tmp = 3
                elif item == 9:
                    fare = float(s[item])
                    if fare < 15:
                        tmp = 0
                    elif fare < 30:
                        tmp = 1
                    elif fare < 45:
                        tmp = 2
                    else:
                        tmp = 3
                elif item == 6 or item == 7:
                    num = int(s[item])
                    if num < 3:
                        tmp = 0
                    else:
                        tmp = 1
                elif s[item] == 'male':
                    tmp = 0
                elif s[item] == 'female':
                    tmp = 1
                elif s[item] == 'C':
                    tmp = 0
                elif s[item] == 'Q':
                    tmp = 1
                elif s[item] == 'S':
                    tmp = 2
                else:
                    tmp = int(s[item])
                if (i, tmp, int(s[1])) not in Set:
                    Set.add((i, tmp, int(s[1])))
                    Map[(i, tmp, int(s[1]))] = 1.0
                else:
                    Map[(i, tmp, int(s[1]))] += 1.0
                if (i, tmp) not in Set:
                    Set.add((i, tmp))
                    Map[(i, tmp)] = 1.0
                else:
                    Map[(i, tmp)] += 1.0
        for item in Set:
            if len(item) == 2:
                Map[item] /= n
            else:
                if item[2] == 0:
                    Map[item] /= c_0
                else:
                    Map[item] /= c_1
        p_1 = c_1 / n
        p_0 = c_0 / n
    return Map, p_0, p_1

def getTest():
    path = '../data/titanic/test.csv'
    xx = []
    id = []
    with open(path, 'r') as fr:
        fr.readline()
        csvfile = csv.reader(fr)
        for s in csvfile:
            x = []
            id.append(s[0])
            for i in range(len(test_id)):
                item = test_id[i]
                if s[item] == '':
                    x.append(mid[i])
                    continue
                if item == 4:
                    age = float(s[item])
                    if age < 15:
                        x.append(0)
                    elif age < 30:
                        x.append(1)
                    elif age < 45:
                        x.append(2)
                    else:
                        x.append(3)
                elif item == 8:
                    fare = float(s[item])
                    if fare < 15:
                        x.append(0)
                    elif fare < 30:
                        x.append(1)
                    elif fare < 45:
                        x.append(2)
                    else:
                        x.append(3)
                elif item == 5 or item == 6:
                    num = int(s[item])
                    if num < 3:
                        x.append(0)
                    else:
                        x.append(1)
                elif s[item] == 'male':
                    x.append(0)
                elif s[item] == 'female':
                    x.append(1)
                elif s[item] == 'C':
                    x.append(0)
                elif s[item] == 'Q':
                    x.append(1)
                elif s[item] == 'S':
                    x.append(2)
                else:
                    x.append(int(s[item]))
            xx.append(x)
    return xx, id