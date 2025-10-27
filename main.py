import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import pygad
import numpy as np
import math
import copy

from GA import HHC_GA
from const import *

def main():
    config = {
        "population": 500, # 200, 500
        "generation": 300, # 100, 150, 300
        "num_par_mat": 500, # 200, 500
        "parent_selection_type": "tournament", # "rws" / "tournament"
        "K_tour": 5, # 2, 3, 5
        "mut_prob": 0.5, # 0.3, 0.5
        "keep_parents": -1, # -1
        "keep_elitism": 100, # 50, 100
    }

    # 10-1-dist.txt / 10-1-job.txt
    # S1-n75-ODwalk.txt / S1-n75-Jobs.txt

    instances = ["U1-n75",  "U2-n75",  "U3-n75",  "U4-n75",  "U5-n75",
                 "S1-n75",  "S2-n75",  "S3-n75",  "S4-n75",  "S5-n75",
                 "U1-n100", "U2-n100", "U3-n100", "U4-n100", "U5-n100",
                 "S1-n100", "S2-n100", "S3-n100", "S4-n100", "S5-n100", 
                 "U1-n125", "U2-n125", "U3-n125", "U4-n125", "U5-n125",
                 "S1-n125", "S2-n125", "S3-n125", "S4-n125", "S5-n125"]

    for instance in instances:
        
        dist = np.loadtxt(f"dataset/{instance}-ODwalk.txt")
        job = np.loadtxt(f"dataset/{instance}-Jobs.txt")

        test_times = 30 # 30

        all_record = []
        avg_all_record = {}
        key_to_avg = {"best_fit_val", "lv3", "lv2", "lv1"}
        for _ in range(test_times):
            record, ga_inst = one_iter(dist, job, config)
            # print("REC:", record)
            all_record.append(record)
        for key in key_to_avg:
            avg_all_record[key] = [sum([all_record[i][key][gen_i] for i in range(test_times)]) / test_times
                            for gen_i in range(config["generation"])]
        all_result_fits = [all_record[i]["best_fit_val"][config["generation"] - 1] for i in range(test_times)]
        best_fit_val = max(all_result_fits)
        best_idx = all_result_fits.index(best_fit_val)
        best_sol = all_record[best_idx]["best_sol"]
        best_schedule = ga_inst.make_schedule(best_sol)
        # print(instance, to_builtin(best_schedule))
        results = cal_cost(best_schedule, best_fit_val)
        with open(f"results.txt", "a") as f:
            print(instance, "best sol: hiring", results[0], "trans", results[1], "total", abs(best_fit_val), "m", results[2], "\n", file=f)

        # print("best sol: hiring", results[0], "trans", results[1], "total", abs(best_fit_val), "m", results[2])    
        plot_result(config, avg_all_record)
        # plt.show()
        plt.savefig(f"results/plot_{instance}.png", dpi=300, bbox_inches='tight')
        plt.close()
        # print("============================")

def cal_cost(schedule, fit):
    lv3 = 0
    lv2 = 0
    lv1 = 0
    for i in schedule:
        if i[1] == 3:
            lv3 += 1
        elif i[1] == 2:
            lv2 += 1
        elif i[1] == 1:
            lv1 += 1
    hiring = 0
    trans = 0
    hiring = (lv3 * LV3CG + lv2 * LV2CG + lv1 * LV1CG)
    trans = abs(fit) - hiring
    return [hiring, trans, lv3+lv2+lv1]


def to_builtin(x):
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    elif isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    elif isinstance(x, (list, tuple)):
        return [to_builtin(i) for i in x]
    return x

def one_iter(dist, job, config):
    ga_inst = HHC_GA(dist, job, config)
    ga_inst.run()
    records = copy.deepcopy(ga_inst.records)
    return records, ga_inst

def plot_result(config, records):
    plt.figure()
    plt.plot(list(range(config["generation"])), records["best_fit_val"])
    plt.title("AVG Best Fitness Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")

    plt.figure()
    plt.plot(list(range(config["generation"])), records["lv3"], label = "lv3")
    plt.plot(list(range(config["generation"])), records["lv2"], label = "lv2")
    plt.plot(list(range(config["generation"])), records["lv1"], label = "lv1")
    plt.title("AVG individual Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("individual num")
    plt.legend()

if __name__ == "__main__":
    main()