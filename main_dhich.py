import numpy as np

from dhich import DHICH

def main():
    # instances1 = ["10-1", "10-2", "10-3", "10-4", "10-5", "10-6", "10-7", "10-8", "10-9", "10-10", 
    #               "20-1", "20-2", "20-3", "20-4", "20-5", "20-6", "20-7", "20-8", "20-9", "20-10", 
    #               "30-1", "30-2", "30-3", "30-4", "30-5", "30-6", "30-7", "30-8", "30-9", "30-10", 
    #               "40-1", "40-2", "40-3", "40-4", "40-5", "40-6", "40-7", "40-8", "40-9", "40-10"]

    # instances2 = ["U1-n75",  "U2-n75",  "U3-n75",  "U4-n75",  "U5-n75",
    #              "S1-n75",  "S2-n75",  "S3-n75",  "S4-n75",  "S5-n75",
    #              "U1-n100", "U2-n100", "U3-n100", "U4-n100", "U5-n100",
    #              "S1-n100", "S2-n100", "S3-n100", "S4-n100", "S5-n100", 
    #              "U1-n125", "U2-n125", "U3-n125", "U4-n125", "U5-n125",
    #              "S1-n125", "S2-n125", "S3-n125", "S4-n125", "S5-n125"]

    instances1 = ["30-1"]
    instances2 = []

    time = 0
    sumCost = []
    sumCg = []

    for instance in instances1:
        dist = np.loadtxt(f"dataset/{instance}-dist.txt")
        job = np.loadtxt(f"dataset/{instance}-job.txt")
        solver = DHICH(dist, job)
        clusters = solver.convexHull()
        solver.plotProcess(clusters)
        ct = solver.getCost(clusters)
        # print("All routes:", clusters, ". Num of cg", len(clusters), ". Total cost:", sum(ct[0]))
        print("Num of cg", len(clusters), ". Total cost:", sum(ct[0]))
        time += 1
        sumCost.append(sum(ct[0]))
        sumCg.append(len(clusters))
        if(time == 10):
            print("\n === \n Avg cost:", sum(sumCost) / len(sumCost), 
                  ". Avg cg:", sum(sumCg) / len(sumCg), "\n === \n")
            time = 0
            sumCost = []
            sumCg = []

    for instance in instances2:
        dist = np.loadtxt(f"dataset/{instance}-ODwalk.txt")
        job = np.loadtxt(f"dataset/{instance}-Jobs.txt")
        solver = DHICH(dist, job)
        clusters = solver.convexHull()
        ct = solver.getCost(clusters)
        # print("All routes:", clusters, ". Num of cg", len(clusters), ". Total cost:", sum(ct[0]))
        print("Num of cg", len(clusters), ". Total cost:", sum(ct[0]))
        
        time += 1
        sumCost.append(sum(ct[0]))
        sumCg.append(len(clusters))
        if(time == 10):
            print("\n === \n Avg cost:", sum(sumCost) / len(sumCost), 
                    ". Avg cg:", sum(sumCg) / len(sumCg), "\n === \n")
            time = 0
            sumCost = []
            sumCg = []


if __name__ == "__main__":
    main()