import numpy as np
from const import *

# compute cost and service time for each cg
def getCost(dist, job, clusters):
    costs = []
    times = []
    for cluster in clusters:
        route = cluster[0]
        lv = cluster[1]

        # cost
        cost = 0
        if lv == 1:
            cost += LV1CG
        elif lv == 2:
            cost += LV2CG
        elif lv == 3:
            cost += LV3CG
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i + 1]] * TR
        costs.append(cost)

        #time
        serve = 0
        for i in range(len(route) - 1):
            serve += job[route[i]][3]
            serve += dist[route[i]][route[i + 1]]
        serve += job[route[-1]][3]
        times.append(int(serve))
    return [costs, times]

# change data type
def to_builtin(x):
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    elif isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    elif isinstance(x, (list, tuple)):
        return [to_builtin(i) for i in x]
    return x