import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull

from const import *

# Check whether the matrix is symmetric
def makeSymm(dist):
    n = len(dist)
    for i in range(n):
        for j in range(n):
            # Fix the index to match the bigger one
            if dist[i][j] != dist[j][i]:
                if dist[i][j] > dist[j][i]:
                    dist[j][i] = dist[i][j]
                elif dist[i][j] < dist[j][i]:
                    dist[i][j] = dist[j][i]

# Check some basic properties of the data
def checkData(dist, job):
    allDist = 0
    allJob = 0
    # number of caregivers, their workload and upper bound
    cgWL = 0
    cgCount = 0
    n = len(dist)
    for i in range(n):
        allJob += job[i][3]
        for j in range(i):
            allDist += dist[i][j]
    #print("All job:", allJob)
    avgDist = allDist / ((n * (n-1)) / 2)
    avgJob = allJob / n
    #print("Avg dist:", round(avgDist + 0.5))
    #print("Avg job:", round(avgJob + 0.5))
    cgWL += avgJob
    oneShift = avgJob + avgDist
    while(n > 0):
        if(cgWL + oneShift <= WLUB):
            cgWL += oneShift
            n -= 1
        else:
            cgCount += 1
            cgWL = avgJob
    # print(f"Hopefully the system will only need around {cgCount} caregivers.")

# Check if the computed matrix is valid
def validOfCoords(pts, dist):
    n = len(pts)
    distOfPts = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            distOfPts[i][j] = (np.sqrt((pts[i][0] - pts[j][0])**2
                                        + (pts[i][1] - pts[j][1])**2))
    diffRateCount = 0
    varRate = []
    maxRate = -1
    for i in range(n):
        for j in range(n):
            if i < j:
                diffRate = distOfPts[i][j] / dist[i][j]
                varRate.append(distOfPts[i][j] / dist[i][j])
                if  diffRate > 0.5:
                    diffRateCount += 1
                if diffRate > maxRate:
                    maxRate = diffRate
    varRate = np.array(varRate)
    print("Distances fall out of desired rate:", diffRateCount, 
          ", with max rate:", round(maxRate, 2), 
          ". Coefficient of variation:", round(np.std(varRate) / np.mean(varRate), 2))
    # The coefficient of variation is a useful criterion when comparing the variance to the mean. 
    # It's the ratio of the standard deviation (square root of variance) to the mean. 
    # A higher CV indicates greater variability relative to the mean, which could signal that variance is too high.
    # CV < 1: The variance is relatively low; CV > 1: The variance is relatively high, meaning the spread is more than the mean.

# Find the serving cluster of the caregiver
def findCluster(dist, job, notServed, startPt, cgLv, worktime=WLUB):
    serveCluster = [startPt]
    servetime = job[startPt][3]
    currentTime = job[startPt][0] + job[startPt][3]
    startTime = job[startPt][0]
    notServed.remove(startPt)
    
    while True:
        if not notServed:
            break

        candidates = sorted(
            notServed,
            key=lambda i: dist[startPt][i]
        )
        
        found = False
        for i in candidates:
            travel = dist[startPt][i]
            arrival = currentTime + travel
            begin = arrival
            finish = begin + job[i][3]

            if (job[i][2] <= cgLv and
                job[i][0] <= begin and
                finish <= job[i][1] and
                servetime + travel + job[i][3] <= worktime):
                
                serveCluster.append(i)
                servetime += travel + job[i][3]
                currentTime = finish
                notServed.remove(i)
                startPt = i
                found = True
                break
        
        if not found:
            break

    return [serveCluster, cgLv, startTime, currentTime]

# Return the level and index of remaining patients
def findMaxLv(serve, job):
    maxLv = -1
    maxLvIdx = -1
    for i in serve:
        if job[i][2] > maxLv:
            maxLv = job[i][2]
            maxLvIdx = i
    return [int(maxLvIdx), int(maxLv)]

# Return the time cost of routes
def calTime(route, dist, job):
    n = len(route)
    travellingtime = 0
    servicetime = 0
    for i in range(n-1):
        servicetime += job[route[i]][3]
        travellingtime += dist[route[i]][route[i + 1]]
    servicetime += job[route[-1]][3]
    return [servicetime, travellingtime]

# Use the points on convex hull as pivots for clustering
def hullPivot(dist, job, notServed, pts):
    served = []
    # Less than 3 points cannot form a convex hull, so we make them a cluster
    while len(notServed) >= 3 :
        # Construct the convex hull for not yet served        
        ptsOfnS = [pts[i] for i in notServed]
        hullTemp = ConvexHull(ptsOfnS)
        hullLabel = hullTemp.vertices
        hullIdx = [notServed[i] for i in hullLabel]
        # print("HI:", hullIdx)

        #print(hullIdx)
        lv1OnHull = [i for i in hullIdx if job[i][2] == 1]
        lv2OnHull = [i for i in hullIdx if job[i][2] == 2]
        lv3OnHull = [i for i in hullIdx if job[i][2] == 3]

        cluster_found = False

        for lvHull, lv in [(lv3OnHull, 3), (lv2OnHull, 2), (lv1OnHull, 1)]:
            for i in lvHull:
                startPt = i
                serveCluster = findCluster(dist, job, notServed, startPt, lv)
                served.append(serveCluster)
                cluster_found = True
                break
            if cluster_found:
                break
    if notServed != []:
        while(notServed != []):
            cluster = findCluster(dist, job, notServed, findMaxLv(notServed, job)[0], findMaxLv(notServed, job)[1])
            served.append(cluster)
    return served

# Check if we can actually need one less caregiver
def oneLessCG(dist, job, serve):
    temp = copy.deepcopy(serve[-1])
    for idx, cg in enumerate(serve[:-1]):
        toBeAdd = copy.deepcopy(cg)
        toBeAdd[0] += temp[0]
        if(toBeAdd[1] < temp[1]):
            toBeAdd[1] = temp[1]
        cgCount = 0
        while toBeAdd[0] != []:
            cluster = findCluster(dist, job, toBeAdd[0], toBeAdd[0][0], toBeAdd[1])
            cgCount += 1 
            if (cgCount > 1):
                #print("We are not able to reduce the amount of this caregiver")
                continue
            if toBeAdd[0] == []:
                if cgCount == 1:
                    return [idx, cluster]
    return -1

# Calculate the cost given list of routes
def calCost(dist, job, routeList):
    # totalcost = [hiring, transportation]
    totalCost = [0, 0]
    allTime = []
    for i in routeList:
        route = i[0]
        level = i[1]

        cgTime = calTime(route, dist, job)
        # print(f"Time spent: {sum(cgTime)} with service time {cgTime[0]}, travelling time {cgTime[1]}.")
        totalCost[1] += round(cgTime[1] + 0.5) * TR
        if level == 1:
            totalCost[0] += LV1CG
        elif level == 2:
            totalCost[0] += LV2CG
        elif level == 3:
            totalCost[0] += LV3CG

        allTime.append(sum(cgTime))
    allTime = np.array(allTime)
    print("Total cost for the area:", round(sum(totalCost)), "with hiring cost", round(totalCost[0]), "and transportation cost", round(totalCost[1]))
    print("Time for cgs:", round(np.sum(allTime)), 
          ", max:", round(np.max(allTime)), 
          ", min:", round(np.min(allTime)), 
          ", avg:", round(np.mean(allTime), 1), 
          ", std:", round(np.std(allTime), 2),
          ",with a total of", len(routeList), "caregivers.")
    return totalCost

# mapping
def sort_map(job):
    # add indices
    idx = list(range(len(job)))
    # sort by lv -> twS -> twE
    sorted_lv_tw = sorted(idx, key = lambda i: (-job[i][2], job[i][0], job[i][1]))
    return sorted_lv_tw

# Plot the vertices as well as the convex hull
def plot(pts, job, hull, routeList):  
    SIZE = 500
    # Plot
    plt.figure(figsize=(8, 8))
    # Annotate points with levels
    for i, pt in enumerate(pts):
        x, y = pt[0], pt[1]
        level = job[i][2]
        if level == 1:
            color = "green"
        elif level == 2:
            color = "cyan"
        elif level == 3:
            color = "red"

        plt.scatter(x, y, color = color, marker = "o", s = SIZE, zorder = 2)
        plt.text(x, y, str(i + 1), fontsize = 10, color = "black", ha = "center", va = "center", zorder = 3)
    
    for simplex in hull.simplices:
        plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-')
    """
    for simplex in hullLv1.simplices:
        plt.plot(patLv1[simplex, 0], patLv1[simplex, 1], 'g-')
    for simplex in hullLv2.simplices:
        plt.plot(patLv2[simplex, 0], patLv2[simplex, 1], 'y-')
    for simplex in hullLv3.simplices:
        plt.plot(patLv3[simplex, 0], patLv3[simplex, 1], 'r-')
    """
    # Plot the route
    for j in routeList:
        route = j[0]
        route_x = [pts[i][0] for i in route]
        route_y = [pts[i][1] for i in route]
        plt.plot(route_x, route_y, marker=None)
        
        # Optionally, highlight start and end points of the route
        plt.scatter(route_x[0], route_y[0], color='purple', label='Start', zorder=1, marker = ".")
        plt.scatter(route_x[-1], route_y[-1], color='orange', label='End', zorder=1, marker = ".")

    # Set labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Plot of Points')
    plt.grid()
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.show()

def plotProcess(pts, job, hull, routes):  
    SIZE = 500 # 200
    EPSIZE = 200
    visited = []    
    route = []
    for i in routes:
        plt.figure(figsize=(8, 8))

        for j in route:
            visited.append(j)
        route = i[0]
        # pts
        unvisited = np.array([pt for i, pt in enumerate(pts) if i not in visited])
        unvisited_idx = [i for i in range(len(pts)) if i not in visited]
        for i, pt in enumerate(pts):
            x, y = pt[0], pt[1]
            if i in visited:
                color = "lightgray"
            else:
                level = job[i][2]
                if level == 1:
                    color = "lime" # "black"
                elif level == 2:
                    color = "cyan" # "black"
                elif level == 3:
                    color = "red" # "black"
            plt.scatter(x, y, color = color, marker = "o", s = SIZE, zorder = 2, edgecolors = "black")
            plt.text(x, y, str(i + 1), fontsize = 10, color = "black", ha = "center", va = "center", zorder = 3)
        # hull
        hull = ConvexHull(unvisited)
        for simplex in hull.simplices:
            plt.plot(unvisited[simplex, 0], unvisited[simplex, 1], dashes = [3, 6], color = "black", zorder = 1, linewidth = 1)
 
        plt.xticks([])
        plt.yticks([])
        plt.axis = False
        plt.show()

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
        for simplex in hull.simplices:
            plt.plot(unvisited[simplex, 0], unvisited[simplex, 1], dashes = [3, 6], color = "black", zorder = 1, linewidth = 1)
        # route
        route_x = [pts[i][0] for i in route]
        route_y = [pts[i][1] for i in route]
        plt.text(route_x[0] - 1, route_y[0] + 2.1, str("start"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
        plt.text(route_x[-1], route_y[-1] + 2.1, str("end"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
        plt.plot(route_x, route_y, zorder = 1, color = "black")

        plt.xticks([])
        plt.yticks([])
        plt.axis = False
        plt.show()

def plotWhole(pts, job, routes):
    SIZE = 500 # 200
    EPSIZE = 200
    visited = []    
    route = []
    
    plt.figure(figsize=(8, 8))

    for i, pt in enumerate(pts):
        x, y = pt[0], pt[1]
        level = job[i][2]
        if level == 1:
            color = "lime" # "black"
        elif level == 2:
            color = "cyan" # "black"
        elif level == 3:
            color = "red" # "black"
        plt.scatter(x, y, color = color, marker = "o", s = SIZE, zorder = 2, edgecolors = "black")
        plt.text(x, y, str(i + 1), fontsize = 10, color = "black", ha = "center", va = "center", zorder = 3)
    
    for route in routes:
        # route
        print(route)
        route_x = [pts[i][0] for i in route[0]]
        route_y = [pts[i][1] for i in route[0]]
        plt.text(route_x[0], route_y[0] + 1.2, str("start"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
        plt.text(route_x[-1], route_y[-1] + 1.2, str("end"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
        plt.plot(route_x, route_y, zorder = 1, color = "black")

    plt.xticks([])
    plt.yticks([])
    plt.axis = False
    plt.show()

def getInit(dist = None, job = None):
    
    if dist is None and job is None:
        # Read the text file into a 2-dimensional array.
        dist = np.loadtxt("10-1-dist.txt")
        job = np.loadtxt("10-1-job.txt")

    # Check symmetry
    makeSymm(dist)
    checkData(dist, job)

    # Apply MDS
    # Use to plot location
    mds = MDS(n_components = 2, dissimilarity = "precomputed", random_state = 42)
    ptsLocation = mds.fit_transform(dist)

    mapping = sort_map(job)
    allPats = mapping
    clusters = hullPivot(dist, job, allPats, ptsLocation)
    # print("DhiCH:", clusters)

    check = 1
    while(check):
        saveCost = oneLessCG(dist, job, clusters)
        if saveCost == -1:
            # print("We were not able to reduce the amount of caregivers.")
            check = 0
        else:
            # print("We are able to reduce one caregiver.")
            clusters.pop(-1)
            clusters.pop(saveCost[0])
            clusters.append(saveCost[1])
            pass
            # print("One less cg:", clusters)

    seq = []
    for i in clusters:
        for j in i[0]:
            seq.append(j)
    return seq
