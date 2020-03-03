#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
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

def DP(
    values, weights, capacity, capacities, max_cols=-1):

    capacities = list(filter(
        lambda x: x <= capacity, capacities))

    O = defaultdict(lambda: defaultdict(lambda: 0))
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
    caps = set()

    for _, w in enumerate(tqdm(weights)):
        current_weight = w
        caps.add(current_weight)

        for r in list(caps).copy():
            previous_weight = r + current_weight
            if previous_weight <= capacity:
                caps.add(previous_weight)                 
    return list(caps)

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

def orchestrate(columns, values, weights, capacity):
    items = len(values)
    capacities = get_cap_possibilities(weights, capacity)
    itemsfound = []
    while True:
        solution = DP(
            values, weights, capacity, capacities, columns)
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

def get_item_with_best_ratio(values, weights, items=None):
    if items is None:
        items = len(values)

    ratio = np.array(values) / np.array(weights)
    ratio, values, weights = zip(*sorted(
        zip(ratio, values, weights), reverse=True))

    return list(values)[0:items], list(weights)[0:items]

def get_items_from_sample(items, values, weights):
    items = np.array(items)
    values, weights = np.array(values), np.array(weights)
    chosen_values = values[items == 1]
    chosen_weights = weights[items == 1]
    chosen_items = []
    chosen_idx = []
    for origvalue, origweight in zip(ORIGVALUES, ORIGWEIGHTS):
        idx_value = list(np.where(chosen_values == origvalue)[0])
        idx_weight = list(np.where(chosen_weights == origweight)[0])
        intersection = list(set(idx_value).intersection(idx_weight))
        if len(intersection) > 0 and intersection not in chosen_idx:
            chosen_items.append(1)
            chosen_idx += intersection
            continue
        chosen_items.append(0)
    return chosen_items       

def solve_it(input_data):
    global ORIGVALUES
    global  ORIGWEIGHTS
    global ORIGCAPACITY

    ORIGVALUES, ORIGWEIGHTS, ORIGCAPACITY = get_data(input_data)
    data_size = len(ORIGVALUES)
    values, weights, capacity = ORIGVALUES, ORIGWEIGHTS, ORIGCAPACITY

    if data_size == 400 or data_size > 1000:
        values, weights = get_item_with_best_ratio(
        ORIGVALUES, ORIGWEIGHTS, 100)
        
        items = orchestrate(50, values, weights, capacity)
        items = get_items_from_sample(items, values, weights)
    else:
        items = orchestrate(-1, values, weights, capacity)
        
    result = np.dot(np.array(items), np.array(ORIGVALUES))
    items = map(str, items)
    items = ' '.join(items)
    result = str(result) + ' 0\n' + items
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
       
        print(solve_it(input_data))

    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

