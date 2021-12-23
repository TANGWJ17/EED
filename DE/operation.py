#!/usr/bin/env python
# coding:utf-8
import random
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np


def clipping(pop, model):
    mutantPops = np.transpose(np.array(pop))
    for i in range(model.nGen):
        mutantPops[i] = np.clip(mutantPops[i], model.P[i][0], model.P[i][1])
    mutantPops = np.transpose(mutantPops)
    return mutantPops


def mutate(parPops, F, generation, maxGeneration, model, kind=0, time_varing=False, Fmax=None, Fmin=None, Archive=None):
    mutantPops = []
    f = F
    if time_varing and Fmax and Fmin:
        f = Fmax - (Fmax - Fmin) * generation / maxGeneration
    if kind == 1:
        for j in range(len(parPops)):
            r1, r2, r3 = random.randint(0, len(parPops) - 1), random.randint(0, len(parPops) - 1), random.randint(0,
                                                                                                                  len(
                                                                                                                      parPops) - 1)
            new_pop = parPops[r1] + f * (parPops[r2] - parPops[r3])
            mutantPops.append(new_pop)
    if kind == 2:
        for j in range(len(parPops)):
            r2, r3 = random.randint(0, len(parPops) - 1), random.randint(0, len(parPops) - 1)
            new_pop = parPops[j] + f * (parPops[r2] - parPops[r3])
            mutantPops.append(new_pop)
    if kind == 0:
        for j in range(len(parPops)):
            if len(Archive) > 0:
                # best = 1.2 * Archive[random.randint(0, len(Archive) - 1)] - 0.2 * parPops[j]
                # best = 1.1 * Archive[random.randint(0, len(Archive) - 1)] - 0.1 * parPops[j]
                best = Archive[random.randint(0, len(Archive) - 1)]
            else:
                best = parPops[random.randint(0, len(parPops) - 1)]
            r2, r3 = random.randint(0, len(parPops) - 1), random.randint(0, len(parPops) - 1)
            new_pop = best + f * (parPops[r2] - parPops[r3])
            mutantPops.append(new_pop)
    # clip
    mutantPops = clipping(mutantPops, model)
    return mutantPops


def crossover(parPops, mutantPops, Cr):
    num_list = range(len(parPops))
    crossPops = parPops

    def basis(i):
        r = np.random.uniform()
        if r < Cr:
            return parPops[i]
        else:
            return mutantPops[i]
    def basic(i):
        new = parPops[i]
        for j in range(len(parPops[i])):
            r = np.random.uniform()
            if r < Cr:
                new[j] = mutantPops[i][j]
        return new

    for i in num_list:
        crossPops[i] = basis(i)
    return crossPops
    # pool = ThreadPool()
    # results = pool.map(basis, num_list)
    # pool.close()
    # pool.join()
    # return np.array(results)


def fast_non_dominated_sort(fits):
    trans = np.transpose(fits)
    values1 = trans[0]
    values2 = trans[1]
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (
                    values1[p] <= values1[q] and values2[p] < values2[q]) or (
                    values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (
                    values1[q] <= values1[p] and values2[q] < values2[p]) or (
                    values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front, rank


def update_archive(pops, fits, nArc, archive_x, archive_y):
    # if archive is empty, add pops into archive directly
    if len(archive_x) == 0:
        return pops, fits
    add_index = []
    archive_xx = archive_x
    archive_yy = archive_y
    for i, fit in enumerate(fits):
        if fit in archive_yy:
            continue
        dominated_index = []
        flag = True
        for j, elite in enumerate(archive_yy):
            # if anyone in archivie dominate trial vector, it's rejected
            if (elite[0] < fit[0] and elite[1] < fit[1]) or (elite[0] <= fit[0] and elite[1] < fit[1]) or (
                    elite[0] < fit[0] and elite[1] <= fit[1]):
                flag = False
                continue
            elif (fit[0] < elite[0] and fit[1] < elite[1]) or (fit[0] <= elite[0] and fit[1] < elite[1]) or (
                    fit[0] < elite[0] and fit[1] <= elite[1]):
                dominated_index.append(j)
        if flag:
            add_index.append(i)
            # todo: archive should be numpy type
            archive_xx = np.delete(archive_xx, dominated_index, 0)
            archive_yy = np.delete(archive_yy, dominated_index, 0)
    for index in add_index:
        archive_xx = np.append(archive_xx, [pops[index]], axis=0)
        archive_yy = np.append(archive_yy, [fits[index]], axis=0)

    if len(archive_xx) > nArc:
        # sort
        archive = zip(archive_xx, archive_yy)
        archive = sorted(archive, key=lambda x: x[1][0], reverse=False)
        archive_xx, archive_yy = zip(*archive)
        distance = crowdingDistance(archive_yy)
        sorting = zip(distance, archive_xx, archive_yy)
        sorting = sorted(sorting, key=lambda x: x[0], reverse=True)
        distance, archive_xx, archive_yy = zip(*sorting)

    return archive_xx[:nArc], archive_yy[:nArc]


def crowdingDistance(fronts, ranks=None):
    distance = [0 for i in range(0, len(fronts))]
    distance[0] = 4444444444444444
    distance[len(fronts) - 1] = 4444444444444444
    front = np.transpose(fronts)
    max_1, min_1, max_2, min_2 = max(front[0]), min(front[0]), max(front[1]), min(front[1])
    for k in range(1, len(fronts) - 1):
        distance[k] += abs(fronts[k + 1][0] - fronts[k - 1][0]) / (max_1 - min_1)
    for k in range(1, len(fronts) - 1):
        distance[k] += abs(fronts[k + 1][1] - fronts[k - 1][1]) / (max_2 - min_2)
    return distance


if __name__ == '__main__':
    a = np.array([[4, 2, 1], [2, 1, 1], [0.5, 3, 1], [3, 0.5, 1], [2.2, 1.1, 1], [1, 2.5, 1], [2.3, 2, 1]])
    b = np.array([[1, 2], [2, 1], [0.5, 3], [3, 0.5], [2.2, 1.1], [1, 2.5], [2.3, 2]])
    b = np.append(b, np.array([[4, 4]]), axis=0)
    print(b)
    exit(0)
    front, rank = fast_non_dominated_sort(b)
    print(front)
    print(rank)
