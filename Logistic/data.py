#encoding=utf-8

import numpy as np
import os
import csv

class_id = [2, 4, 5, 6, 7, 9, 11, 1]

class_id_test = [0, 1, 3, 4, 5, 6, 8, 10]

def getTrain_Set():
    source_path = '../data/titanic/train.csv'
    out_path = './train.txt'
    fw = open(out_path, 'w')
    with open(source_path, 'r') as csvfile:
        csvfile.readline()
        fr = csv.reader(csvfile)
        for s in fr:
            print s
            ss = ''
            for id in class_id:
                if id == 2:
                    ss += s[id]
                    continue
                ss += ' '
                if s[id] == 'male':
                    ss += str(0)
                elif s[id] == 'female':
                    ss += str(1)
                elif s[id] == 'C':
                    ss += str(0)
                elif s[id] == 'Q':
                    ss += str(1)
                elif s[id] == 'S':
                    ss += str(2)
                else:
                    ss += s[id]
            ss += '\n'
            fw.write(ss)


def getTest_Set():
    source_path = '../data/titanic/test.csv'
    out_path = './test.txt'
    fw = open(out_path, 'w')
    with open(source_path, 'r') as csvfile:
        csvfile.readline()
        fr = csv.reader(csvfile)
        for s in fr:
            print s
            ss = ''
            for id in class_id_test:
                if id == 0:
                    ss += s[id]
                    continue
                ss += ' '
                if s[id] == 'male':
                    ss += str(0)
                elif s[id] == 'female':
                    ss += str(1)
                elif s[id] == 'C':
                    ss += str(0)
                elif s[id] == 'Q':
                    ss += str(1)
                elif s[id] == 'S':
                    ss += str(2)
                else:
                    ss += s[id]
            ss += '\n'
            fw.write(ss)


if __name__ == '__main__':
    getTest_Set()

