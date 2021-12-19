#!/usr/bin/env python
# coding:utf-8
import time

from PyQt5 import QtCore
from PyQt5.QtCore import *
from matplotlib import pyplot as plt
from DE.initial import *
from DE.operation import *
from EED.constraint import constraints
from EED.loadmodel import *


class MMODE(object):
    def __init__(self, model, **kwargs):
        arguments = {'nIter': 100, 'nPop': 200, 'nArc': 50, 'nGen': 6, 'F': 0.6, 'CR': 0.8, 'init': 0, 'mutation': 0}
        arguments.update(kwargs)
        self.model = model
        self.nIter = arguments['nIter']
        self.nPop = arguments['nPop']
        self.nArc = arguments['nArc']
        self.nGen = arguments['nGen']
        self.F = arguments['F']
        self.CR = arguments['CR']
        self.init = arguments['init']
        self.mutation = arguments['mutation']
        # self.swarm = np.zeros([self.nPop, self.nGen])
        self.Archive_X = []  # elites
        self.Archive_Y = []
        self.bestC = []  # vary with iteration
        self.bestE = []
        self.finalX = None  # shape: [nArc, nGen]
        self.finalY = None  # shape: [nArc, 2]

    def Func(self, solution, model, demand):
        fuel = 0
        emis = 0
        a1 = 1000  # penalty factor
        a2 = 1
        for i in range(len(solution)):
            fuel += costfun(i, solution[i], self.model.C, self.model.P)
            emis += emission(i, solution[i], self.model.E, flag=True)
        fuel += a1 * constraints(solution, model, demand)
        emis += a2 * constraints(solution, model, demand)
        return np.array([fuel, emis])

    def fitness(self, pops, model, demand):
        results = []
        for pop in pops:
            results.append(self.Func(pop, model, demand))
        return np.array(results)

    def is_dominated(self, s1, s2):
        # if s1 is dominated by s2
        Fx = self.Func(s1)
        Fy = self.Func(s2)
        i = 0
        for xv, yv in zip(Fx, Fy):
            if xv > yv:
                i += 1
            if xv < yv:
                return False
        if i != 0:
            return True
        return False

    def solve(self, demand, GUI=None):
        # start iteration
        print('-----start iteration-----')
        itera = 1
        parPops = initialize(self.nPop, self.model, kind=self.init, demand=demand)
        parFits = self.fitness(parPops, self.model, demand)
        while itera <= self.nIter:
            if GUI is not None and itera % 10 == 0:
                GUI.ui.progressBar.setProperty("value", 100 * itera / self.nIter)
                GUI.show()
            print(parPops.shape)
            mutantPops = mutate(parPops, self.F, itera, self.nIter, self.model, kind=self.mutation,
                                time_varing=True, Fmax=self.F, Fmin=0.1,
                                Archive=self.Archive_X)  # generate mutated vectors
            trialPops = crossover(parPops, mutantPops, self.CR)
            trialFits = self.fitness(trialPops, self.model, demand)  # calculate fitness
            pops = np.concatenate((parPops, trialPops), axis=0)  # merge two groups
            fits = np.concatenate((parFits, trialFits), axis=0)
            front, rank = fast_non_dominated_sort(fits)
            front_pops = pops[front[0]]
            front_fits = fits[front[0]]
            self.Archive_X, self.Archive_Y = update_archive(front_pops, front_fits, self.nArc, self.Archive_X,
                                                            self.Archive_Y)
            self.Archive_X, self.Archive_Y = np.array(self.Archive_X), np.array(self.Archive_Y)
            new_pop_index = []
            k = 0
            while len(new_pop_index) < self.nPop:
                new_pop_index += front[k]
                k += 1
            C_min = min(min(fits[:, 0]), min(self.Archive_Y[:, 0]))
            E_min = min(min(fits[:, 1]), min(self.Archive_Y[:, 1]))
            print("itera:", itera, " C_min:", C_min, ", E_min:", E_min)
            self.bestC.append(C_min)
            self.bestE.append(E_min)
            parPops = pops[new_pop_index[:self.nPop]]
            parFits = fits[new_pop_index[:self.nPop]]
            if itera % 50 == 0:
                # plt.scatter(front_fits[:, 0], front_fits[:, 1])
                # plt.show()
                # plt.figure()
                # plt.scatter(self.Archive_Y[:, 0], self.Archive_Y[:, 1])
                # plt.show()
                pass
            itera += 1

        sorting = zip(self.Archive_Y, self.Archive_X)
        sorting = sorted(sorting, key=lambda x: x[0][0], reverse=True)
        self.Archive_Y, self.Archive_X = zip(*sorting)
        self.finalX, self.finalY = np.array(self.Archive_X), np.array(self.Archive_Y)

    def plot(self):
        data = np.transpose(self.finalY)
        plt.scatter(data[0], data[1])
        plt.xlabel('cost($/h)')
        plt.ylabel('emission(ton/h)')
        plt.title('Pareto Set')
        plt.show()

    def best(self, ratio=1):
        # get best compromise solution
        # ratio: indicates the attitude of decision maker, the bigger the ratio, the less the decision maker care about 'objective 0'
        l = 100
        best_X = None
        best_Y = None
        Archive = np.transpose(self.Archive_Y)
        x1 = (Archive[0] - min(Archive[0])) * ratio / (max(Archive[0]) - min(Archive[0]))
        x2 = (Archive[1] - min(Archive[1])) / (max(Archive[1]) - min(Archive[1]))
        for i in range(len(self.Archive_Y)):
            cnt = x1[i] ** 2 + x2[i] ** 2
            if cnt < l:
                l = cnt
                best_X, best_Y = self.Archive_X[i], self.Archive_Y[i]
        return best_X, best_Y


class MMODE_Thread(MMODE, QtCore.QObject):
    signal_1 = pyqtSignal(str)
    signal_2 = pyqtSignal(str)
    signal_3 = pyqtSignal(str)
    signal_4 = pyqtSignal(str)

    def __init__(self, model, demand, **kwargs):
        # super(MMODE_Thread, self).__init__(model, **kwargs)
        MMODE.__init__(self, model, **kwargs)
        QtCore.QObject.__init__(self)
        self.demand = demand

    def run(self):
        # start iteration
        self.flag = True
        while self.flag:
            # time.sleep(1)
            print('-----start iteration-----')
            itera = 1
            parPops = initialize(self.nPop, self.model, kind=self.init, demand=self.demand)
            parFits = self.fitness(parPops, self.model, self.demand)
            while itera <= self.nIter and self.flag:
                if itera % 10 == 0:
                    self.signal_1.emit(str(int(itera * 100 / self.nIter)))
                mutantPops = mutate(parPops, self.F, itera, self.nIter, self.model, kind=self.mutation,
                                    time_varing=True, Fmax=self.F, Fmin=0.1,
                                    Archive=self.Archive_X)  # generate mutated vectors
                trialPops = crossover(parPops, mutantPops, self.CR)
                trialFits = self.fitness(trialPops, self.model, self.demand)  # calculate fitness
                pops = np.concatenate((parPops, trialPops), axis=0)  # merge two groups
                fits = np.concatenate((parFits, trialFits), axis=0)
                front, rank = fast_non_dominated_sort(fits)
                front_pops = pops[front[0]]
                front_fits = fits[front[0]]
                self.Archive_X, self.Archive_Y = update_archive(front_pops, front_fits, self.nArc, self.Archive_X,
                                                                self.Archive_Y)
                self.Archive_X, self.Archive_Y = np.array(self.Archive_X), np.array(self.Archive_Y)
                sorting = zip(self.Archive_Y, self.Archive_X)
                sorting = sorted(sorting, key=lambda x: x[0][0], reverse=False)
                self.Archive_Y, self.Archive_X = zip(*sorting)
                self.Archive_X, self.Archive_Y = np.array(self.Archive_X), np.array(self.Archive_Y)
                new_pop_index = []
                k = 0
                while len(new_pop_index) < self.nPop:
                    new_pop_index += front[k]
                    k += 1

                # C_min = min(min(fits[:, 0]), min(self.Archive_Y[:, 0]))
                # E_min = min(min(fits[:, 1]), min(self.Archive_Y[:, 1]))
                C_min = self.Archive_Y[0, 0]
                E_min = self.Archive_Y[-1, 1]
                # print("itera:", itera, " C_min:", C_min, ", E_min:", E_min)
                self.bestC.append(C_min)
                self.bestE.append(E_min)
                parPops = pops[new_pop_index[:self.nPop]]
                parFits = fits[new_pop_index[:self.nPop]]
                if itera % 50 == 0:
                    # plt.scatter(front_fits[:, 0], front_fits[:, 1])
                    # plt.show()
                    plt.figure()
                    plt.scatter(self.Archive_Y[:, 0], self.Archive_Y[:, 1])
                    plt.xlabel('cost($/h)')
                    plt.ylabel('emission(ton/h)')
                    plt.title('Pareto Set({}-itrea)'.format(itera))
                    path = './img/{}.png'.format(time.ctime())
                    plt.savefig(path)
                    self.signal_2.emit(str(path))
                    path_table1 = './doc/{}_1.txt'.format(time.ctime())
                    path_table2 = './doc/{}_2.txt'.format(time.ctime())
                    with open(path_table1, 'w') as f:
                        a = ['economic']
                        a += list(map(str, (self.Archive_Y[0])))
                        a += list(map(str, (self.Archive_X[0])))
                        string_1 = ' '.join(a)
                        b = ['environmental']
                        b += list(map(str, (self.Archive_Y[-1])))
                        b += list(map(str, (self.Archive_X[-1])))
                        string_2 = ' '.join(b)
                        c = ['compromise']
                        best_X, best_Y = self.best(1)
                        c += list(map(str, best_Y))
                        c += list(map(str, best_X))
                        string_3 = ' '.join(c)
                        lines = [string_1, '\n', string_2, '\n', string_3]
                        f.writelines(lines)
                    with open(path_table2, 'w') as f:
                        lines = []
                        for i in range(len(self.Archive_Y)):
                            a = [str(i + 1)]
                            a += list(map(str, (self.Archive_Y[i])))
                            a += list(map(str, (self.Archive_X[i])))
                            string_1 = ' '.join(a)
                            lines.append(string_1)
                            lines.append('\n')
                        f.writelines(lines)
                    self.signal_3.emit(str(path_table1))
                    self.signal_4.emit(str(path_table2))
                    pass
                itera += 1

            sorting = zip(self.Archive_Y, self.Archive_X)
            sorting = sorted(sorting, key=lambda x: x[0][0], reverse=True)
            self.Archive_Y, self.Archive_X = zip(*sorting)
            self.finalX, self.finalY = np.array(self.Archive_X), np.array(self.Archive_Y)
            self.flag = False


if __name__ == '__main__':
    demand = 2.834
    model = Model('../data.txt')
    arguments = {'nIter': 1000, 'nPop': 150, 'nArc': 150, 'nGen': 6, 'F': 0.4, 'CR': 1, 'init': 0}
    DE = MMODE(model=model, **arguments)
    DE.solve(demand)
    print(DE.finalY.shape)
    DE.plot()
