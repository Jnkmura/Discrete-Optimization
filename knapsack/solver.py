#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time


O = defaultdict(lambda: defaultdict(lambda: 0)) 
   
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


def solve_it(values, weights, capacity):
    capacities = get_cap_possibilities(weights)
    for j in range(len(values)):
        O['previous'] = O['last'].copy()

        for k in capacities:
            item_weight = weights[j]
            item_value = values[j]

            if item_weight <= k:
                O['last'][k] = max(
                    O['previous'][k],
                    item_value + O['previous'][k - item_weight]
                    )
            else:
                O['last'][k] = O['previous'][k]
       
    df = pd.DataFrame(O).sort_index()
    print(df)
    print(np.array(df))
    return df
    
def get_cap_possibilities(weights):
    weights = [0] + weights
    results = []
    caps = set()

    for _, w in enumerate(weights):
        current_weight = w
        caps.add(current_weight)

        for r in results:
            previous_weight = r + current_weight
            caps.add(previous_weight)
    
        results = sorted(list(filter(
            lambda x: x <= capacity, list(caps))))

    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        values, weights, capacity = get_data(input_data)

        start = time()
        solution = solve_it(values, weights, capacity)
        end = time()
        print(end-start)

    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

