#encoding=utf-8

import numpy as np
import os
import csv
import sklearn.svm as svm

test_id = [1, 3, 4, 5, 6, 8, 10]
train_id = [2, 4, 5, 6, 7, 9, 11]

def getTrain():
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
            labels.append(int(s[1]))
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


if __name__ == '__main__':
    xx, labels, Max = getTrain()
    SVM = svm.SVC(C=10, kernel='linear', gamma='auto')
    SVM.fit(X=xx, y=labels)
    xx, id = getTest(Max)
    res = SVM.predict(xx)
    path = './out.csv'
    with open(path, 'w') as f:
        fw = csv.writer(f)
        fw.writerow(['PassengerId', 'Survived'])
        for i in range(len(xx)):
            fw.writerow([id[i], res[i]])