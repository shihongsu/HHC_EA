import matplotlib.pyplot as plt
import numpy as np
import copy

from GA import HHC_GA
from utils import getCost, to_builtin
from const import *

def main():
    # for small size cases (10, 20, 30, 40)
    config1 = {
        "population": 200,
        "generation": 100,
        "num_par_mat": 50,
        "parent_selection_type": "tournament", # "rws" / "tournament"
        "K_tour": 2,
        "mut_prob": 0.3,
        "keep_elitism": 20
    }

    # for large size cases (75, 100, 125)
    config2 = {
            "population": 500,
            "generation": 300,
            "num_par_mat": 100,
            "parent_selection_type": "tournament", # "rws" / "tournament"
            "K_tour": 5,
            "mut_prob": 0.5,
            "keep_elitism": 50
    }

    # instances1 = ["10-1"]
    instances1 = ["10-1", "10-2", "10-3", "10-4", "10-5", "10-6", "10-7", "10-8", "10-9", "10-10", 
                 "20-1", "20-2", "20-3", "20-4", "20-5", "20-6", "20-7", "20-8", "20-9", "20-10", 
                 "30-1", "30-2", "30-3", "30-4", "30-5", "30-6", "30-7", "30-8", "30-9", "30-10", 
                 "40-1", "40-2", "40-3", "40-4", "40-5", "40-6", "40-7", "40-8", "40-9", "40-10"]


    instances2 = ["U1-n75"]
    # instances2 = ["U1-n75",  "U2-n75",  "U3-n75",  "U4-n75",  "U5-n75",
    #              "S1-n75",  "S2-n75",  "S3-n75",  "S4-n75",  "S5-n75",
    #              "U1-n100", "U2-n100", "U3-n100", "U4-n100", "U5-n100",
    #              "S1-n100", "S2-n100", "S3-n100", "S4-n100", "S5-n100", 
    #              "U1-n125", "U2-n125", "U3-n125", "U4-n125", "U5-n125",
    #              "S1-n125", "S2-n125", "S3-n125", "S4-n125", "S5-n125"]

    
    for instance in instances1:
        dist = np.loadtxt(f"dataset/{instance}-dist.txt")
        job = np.loadtxt(f"dataset/{instance}-job.txt")

        test_times = 30 # 10, 30

        all_record = []
        best_fit_val = float("-inf")
        best_ga_inst = None
        best_sol = None

        for _ in range(test_times):
            record, ga_inst = one_iter(dist, job, config1)
            all_record.append(record)

            run_best_fit = record["best_fit_val"][-1]

            if run_best_fit > best_fit_val:
                best_fit_val = run_best_fit
                best_sol = record["best_sol"]
                best_ga_inst = ga_inst

        key_to_avg = ["best_fit_val", "lv3", "lv2", "lv1"]

        avg_all_record = {key: np.mean([record[key] for record in all_record], axis=0).tolist() for key in key_to_avg}

        best_schedule = best_ga_inst.make_schedule(best_sol)
        best_schedule = to_builtin(best_schedule)
        print(best_schedule)
        print(best_fit_val)

        results = getCost(dist, job, best_schedule)

        with open("results.txt", "a") as f:
            print(instance, "Num of cg:", len(best_schedule), ", total cost:", sum(results[0]), file = f)

        plot_result(config1, avg_all_record)
        plt.savefig(
            f"results/plot_{instance}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    for instance in instances2:
        dist = np.loadtxt(f"dataset/{instance}-ODwalk.txt")
        job = np.loadtxt(f"dataset/{instance}-Jobs.txt")

        test_times = 20 # 5, 20

        all_record = []
        best_fit_val = float("-inf")
        best_ga_inst = None
        best_sol = None

        for _ in range(test_times):
            record, ga_inst = one_iter(dist, job, config1)
            all_record.append(record)

            run_best_fit = record["best_fit_val"][-1]

            if run_best_fit > best_fit_val:
                best_fit_val = run_best_fit
                best_sol = record["best_sol"]
                best_ga_inst = ga_inst

        key_to_avg = ["best_fit_val", "lv3", "lv2", "lv1"]

        avg_all_record = {key: np.mean([record[key] for record in all_record], axis=0).tolist() for key in key_to_avg}

        best_schedule = best_ga_inst.make_schedule(best_sol)
        best_schedule = to_builtin(best_schedule)
        print(best_schedule, best_fit_val)
        results = getCost(dist, job, best_schedule)

        with open("results.txt", "a") as f:
            print(instance, "Num of cg:", len(best_schedule), ", total cost:", sum(results[0]), file = f)

        plot_result(config1, avg_all_record)
        plt.savefig(
            f"results/plot_{instance}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()    


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

if __name__ == "__main__":
    main()