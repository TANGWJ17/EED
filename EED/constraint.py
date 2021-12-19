#!/usr/bin/env python
# coding:utf-8
import time

import numpy as np


def constraints(Px, model, demand):
    loss = model.B_00
    for i in range(model.nGen):
        for j in range(model.nGen):
            loss += Px[i] * model.B[i][j] * Px[j]
    loss += sum(Px * model.B_0)
    return max(0, abs(sum(Px) - loss - demand) - 0.1)


if __name__ == '__main__':
    a = './{}.png'.format(time.ctime())
    print(a)
