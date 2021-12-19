#!/usr/bin/env python
# coding:utf-8
import time
from multiprocessing.pool import ThreadPool

import numpy as np
from DE.operation import clipping
from EED.constraint import constraints
from EED.loadmodel import costfun, emission


def create_child(model):
    nGen = model.nGen
    P_matrix = np.transpose(model.P)
    child = P_matrix[0] + (P_matrix[1] - P_matrix[0]) * np.random.rand(nGen)
    return child

def initialize(nPop, model, kind=0, demand=None):
    pops = []
    if kind == 0:
        for i in range(nPop):
            pops.append(create_child(model))
    if kind == 1:
        while len(pops) < nPop:
            pop = create_child(model)
            fuel = 0
            emis = 0
            for i in range(len(pop)):
                fuel += costfun(i, pop[i], model.C, model.P)
                emis += emission(i, pop[i], model.E, flag=True)
            if constraints(pop, model, demand) == 0:
                pops.append(pop)
        # create constraints-satisfied population

    if kind == 2:
        # opposition-based initialization
        pass
    if kind == 3:
        # chaotic initialization
        pass
    # clipping
    return np.array(pops)


if __name__ == '__main__':
    a = np.array([[1, 2], [1, 2], [2, 3]])
    print(set(a.tolist()))