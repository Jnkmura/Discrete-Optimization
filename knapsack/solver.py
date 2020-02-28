#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time

def get_data(input_data):
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    weights = []
    values = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        weights.append(int(parts[1]))
        values.append(int(parts[0]))
    return values, weights, capacity

def solve_it(values, weights, capacity, max_cols=-1):
    O = defaultdict(lambda: defaultdict(lambda: 0))
    print('Getting Labels...')
    capacities = get_cap_possibilities(weights, capacity)

    print('Iterating over Capacity x Items...')
    for j in tqdm(range(len(values))):
        for k in capacities:
            item_weight = weights[j]
            item_value = values[j]

            if item_weight <= k:
                O[j][k] = max(
                    O[j - 1][k],
                    item_value + O[j - 1][k - item_weight])
            else:
                O[j][k] = O[j - 1][k]

        if j >= (max_cols - 1) and max_cols > -1:
            O.pop(list(O.keys())[0])
    return O
    
def get_cap_possibilities(weights, capacity):
    weights = [0] + weights
    results = []
    caps = set()

    for _, w in enumerate(tqdm(weights)):
        current_weight = w
        caps.add(current_weight)

        for r in results:
            previous_weight = r + current_weight
            caps.add(previous_weight)
    
        results = sorted(list(filter(
            lambda x: x <= capacity, list(caps))))
    return results

def get_solution(O, weights):
    df = pd.DataFrame(O)
    columns = list(df.columns)
    max_id = df[columns[-1]].idxmax()
    taken = []

    for i, cur_col in reversed(list(
        enumerate(columns))):
        previous_col = columns[i - 1]
        previous_score, cur_score = np.array(
            df[df.index == max_id
            ][[previous_col, cur_col]])[0]

        istaken = 0
        if cur_score != previous_score:
            cur_weight = weights[cur_col]
            max_id = max_id - cur_weight
            istaken = 1
        taken.append(istaken)
    return list(reversed(list(taken)))[1:], max_id

def orchestrate(columns):
    items = len(VALUES)
    values, weights, capacity = VALUES, WEIGHTS, CAPACITY
    itemsfound = []
    i = 1
    while True:
        solution = solve_it(values, weights, capacity, columns)
        taken, capacity = get_solution(solution, weights)
        itemsfound = taken + itemsfound

        if taken == []:
            max_key = min(max(solution[0].keys()), 1)
            return [max_key] + itemsfound
        
        items = items - columns + 1
        if items <= 0 or columns == -1:
            return itemsfound

        values = values[:-columns + 1]
        weights = weights[:-columns + 1]
        columns = min(columns, items)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        VALUES, WEIGHTS, CAPACITY = get_data(input_data)
        VALUES, WEIGHTS = zip(*sorted(zip(VALUES, WEIGHTS)))
        VALUES, WEIGHTS = list(VALUES), list(WEIGHTS)

        start = time()
        items = orchestrate(200)
        print(items)
        print(np.dot(np.array(items), np.array(VALUES)))
        end = time()
        print(end-start)


    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

