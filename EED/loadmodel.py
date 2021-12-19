#!/usr/bin/env python
# coding:utf-8

import numpy as np
import math
# from DE.initial import create_child, initialize
from EED.constraint import constraints



def inital_model(file):
    data = open(file, 'r').readlines()
    num = int(data[0].split()[-1])
    C = np.zeros([num, 5])
    E = np.zeros([num, 5])
    P = np.zeros([num, 2])
    B = np.zeros([num, num])
    parameters = len(data[3].split()) - 3
    for i in range(2, 2 + num):
        raw_data = data[i].split()[1:]
        for j in range(parameters):
            C[i - 2][j] = float(raw_data[j])
        P[i - 2][0] = float(raw_data[-2])
        P[i - 2][1] = float(raw_data[-1])
    length = len(data[2 + num].split()) - 1
    for i in range(3 + num, 3 + 2 * num):
        raw_data = data[i].split()[1:]
        for j in range(length):
            E[i - 3 - num][j] = float(raw_data[j])
    for i in range(4 + 2 * num, 4 + 3 * num):
        raw_data = data[i].split()
        B[i - 4 - 2 * num] = np.array(list(map(float, raw_data)))
    B_0 = np.array(list(map(float, data[5 + 3 * num].split())))
    B_00 = float(data[7 + 3 * num])
    return num, C, E, P, B, B_0, B_00


def costfun(uid, load, C, P=None):
    if P is not None:
        return load * (C[uid][2] * load + C[uid][1]) + C[uid][0] + math.fabs(C[uid][3] *
                                                                             math.sin(C[uid][4] * (P[uid][0] - load)))
    return load * (C[uid][2] * load + C[uid][1]) + C[uid][0]


def emission(uid, load, E, flag=True):
    if E[0][3] != 0 and flag:
        return (E[uid][0] + (E[uid][1] + E[uid][2] * load) * load) + E[uid][3] * math.exp(E[uid][4] * load)
    else:
        return load * (E[uid][2] * load + E[uid][1]) + E[uid][0]


class Model:
    def __init__(self, file):
        self.nGen, self.C, self.E, self.P, self.B, self.B_0, self.B_00 = inital_model(file)

    def constraint(self):
        return constraints


if __name__ == '__main__':
    demand = 2.834
    model = Model('../data.txt')
    print(model.E)
    exit(0)
    pop = np.array([0.1917, 0.3804, 0.5603, 0.7154, 0.6009, 0.3804])
    fuel = 0
    emis = 0
    for i in range(len(pop)):
        fuel += costfun(i, pop[i], model.C, model.P)
        emis += emission(i, pop[i], model.E, flag=True)
    print(fuel, emis)
    print(constraints(pop, model, demand))
    exit(0)

