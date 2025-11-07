import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.manifold import MDS

from const import *
from dhich import *

# mapping
def sort_map(job):
    # add indices
    idx = list(range(len(job)))
    # sort by lv -> twS -> twE
    sorted_lv_tw = sorted(idx, key = lambda i: (-job[i][2], job[i][0], job[i][1]))
    return sorted_lv_tw

# FF
def listSeq(dist, job, notServed):
    cg = []
    for i in notServed:
        done = False
        j = 0
        while j < len(cg) and not done:
            if (job[i][0] <= cg[j][2] + dist[cg[j][0][-1]][i] and               # start
                cg[j][2] + dist[cg[j][0][-1]][i] + job[i][3] <= job[i][1] and   # end
                cg[j][3] + dist[cg[j][0][-1]][i] + job[i][3] <= WLUB and        # workload
                cg[j][1] >= job[i][2]):                                         # level
                
                cg[j][2] += (dist[cg[j][0][-1]][i] + job[i][3])
                cg[j][3] += (dist[cg[j][0][-1]][i] + job[i][3])
                cg[j][0].append(i)
                done = True
            j += 1
        if done:
            continue
        else:
            # [route, lv, current time, workload]
            newCg = [[i], job[i][2], job[i][0] + job[i][3], job[i][3]]
            cg.append(newCg)
    return cg

# plot FF
def plotff(pts, job, routeList):  
    SIZE = 500 # 200
    EPSIZE = 200
    visited = []    
    route = []
    for i in range(len(routeList)):
        for j in route:
            visited.append(j)
        route = routeList[i][0]

        plt.figure(figsize=(8, 8))
        for i, pt in enumerate(pts):
            x, y = pt[0], pt[1]
            if i in visited:
                color = "lightgray"
            else:
                level = job[i][2]
                if level == 1:
                    color = "lime"
                elif level == 2:
                    color = "cyan"
                elif level == 3:
                    color = "red"
            plt.scatter(x, y, color = color, marker = "o", s = SIZE, zorder = 2, edgecolors = "black")
            plt.text(x, y, str(i + 1), fontsize = 10, color = "black", ha = "center", va = "center", zorder = 3)
        
        # route
        route_x = [pts[i][0] for i in route]
        route_y = [pts[i][1] for i in route]
        plt.text(route_x[0], route_y[0] - 2.2, str("start"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
        plt.text(route_x[-1], route_y[-1] + 2.2, str("end"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
        plt.plot(route_x, route_y, zorder = 1, color = "black")

        plt.xticks([])
        plt.yticks([])
        plt.axis = False
        plt.show()

def main():
    # Read the text file into a 2-dimensional array.
    dist = np.loadtxt("30-1-dist.txt")
    job = np.loadtxt("30-1-job.txt")
    # Check symmetry
    makeSymm(dist)
    checkData(dist, job)

    # Apply MDS
    # Use walk to plot location and calculation
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    ptsLocation = mds.fit_transform(dist)
    # Check validity
    validOfCoords(ptsLocation, dist)
    
    numOfPat = len(dist)
    whoRlv1 = [i for i in range(numOfPat) if job[i][2] == 1]
    whoRlv2 = [i for i in range(numOfPat) if job[i][2] == 2]
    whoRlv3 = [i for i in range(numOfPat) if job[i][2] == 3]
    print(f"Lv1: {whoRlv1}, \nLv2: {whoRlv2}, \nLv3: {whoRlv3}")
    print(f"Number of patients: Lv3: {len(whoRlv3)}, Lv2: {len(whoRlv2)}, and Lv1: {len(whoRlv1)}.")

    # F.F.
    allPats = [i for i in range(numOfPat)]
    clusters = listSeq(dist, job, allPats)
    print("FF", clusters)
    calCost(dist, job, clusters)
    # plotff(ptsLocation, job, clusters)
    print("=====================================")

    # hiCH
    allPats = [i for i in range(numOfPat)]
    clusters = hullPivot(dist, job, allPats, ptsLocation)
    check = 1
    while(check):
        saveCost = oneLessCG(dist, job, clusters)
        if saveCost == -1:
            print("We were not able to reduce the amount of caregivers.")
            check = 0
        else:
            print("We are able to reduce one caregiver.")
            clusters.pop(-1)
            clusters.pop(saveCost[0])
            clusters.append(saveCost[1])
            pass
            print("One less cg:", clusters)
    print("hiCH:", clusters)
    calCost(dist, job, clusters)  
    print("=====================================")

    mapping = sort_map(job)
    # print(mapping)

    # DFF
    allPats = mapping
    clusters = listSeq(dist, job, allPats)
    print("DFF:", clusters)
    calCost(dist, job, clusters)
    print("=====================================")

    # DHICH
    clusters = hullPivot(dist, job, allPats, ptsLocation)
    print("DhiCH:", clusters)
    check = 1
    while(check):
        saveCost = oneLessCG(dist, job, clusters)
        if saveCost == -1:
            print("We were not able to reduce the amount of caregivers.")
            check = 0
        else:
            print("We are able to reduce one caregiver.")
            clusters.pop(-1)
            clusters.pop(saveCost[0])
            clusters.append(saveCost[1])
            print("One less cg:", clusters)
    calCost(dist, job, clusters)

main()