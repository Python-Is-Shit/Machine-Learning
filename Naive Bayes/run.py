#encoding=utf-8

import numpy as np
import csv
import os
import data

class NB(object):
    def __init__(self):
        self.Map = {}
        self.p_0 = 0
        self.p_1 = 0
        self.Map, self.p_0, self.p_1 = data.getTrain()
        print self.p_0, self.p_1

    def predict(self, x):
        p = self.p_0
        for i in range(len(x)):
            p = p * self.Map[(i, x[i], 0)] / self.Map[(i, x[i])]
        if p >= 0.5:
            return 0
        else:
            return 1

    def run(self):
        path = './out.csv'
        x, id = data.getTest()
        with open(path, 'w') as f:
            fw = csv.writer(f)
            fw.writerow(['PassengerId', 'Survived'])
            for i in range(len(x)):
                fw.writerow([id[i], self.predict(x[i])])


if __name__ == '__main__':
    nb = NB()
    nb.run()