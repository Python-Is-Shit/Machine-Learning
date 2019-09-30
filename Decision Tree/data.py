#encoding=utf-8

import numpy as np
import csv
import os

train_id = [2, 4, 5, 6, 7, 9, 11, 1]

mid = [1, 0, 1, 0, 0, 1, 1]

test_id = [1, 3, 4, 5, 6, 8, 10]

def getTrain():
    path = '../data/titanic/train.csv'
    xx = []
    with open(path, 'r') as fr:
        fr.readline()
        csvfile = csv.reader(fr)
        for s in csvfile:
            flag = True
            x = []
            for item in train_id:
                if s[item] == '':
                    flag = False
                    break
            if flag == False:
                continue
            for item in train_id:
                if item == 5:
                    age = float(s[item])
                    if age < 15:
                        x.append(0)
                    elif age < 30:
                        x.append(1)
                    elif age < 45:
                        x.append(2)
                    else:
                        x.append(3)
                elif item == 9:
                    fare = float(s[item])
                    if fare < 15:
                        x.append(0)
                    elif fare < 30:
                        x.append(1)
                    elif fare < 45:
                        x.append(2)
                    else:
                        x.append(3)
                elif item == 6 or item == 7:
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
    return xx

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