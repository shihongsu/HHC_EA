import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull

from const import *

class DHICH:
    def __init__(self, dist, job):
        self.dist = dist
        self.job = job
        self.pts = None

    def preprocess(self):
        # sort base on (twS(des), twE(asc), serv(des), lv(des))
        idx = list(range(len(self.job)))
        sorted_job = sorted(idx, key = lambda i: (-self.job[i][0], self.job[i][1], -self.job[i][3], -self.job[i][2]))
        return sorted_job
    
    # Find the serving cluster of the caregiver
    def findCluster(self, notServed, st, worktime=WLUB):
        cluster = [st]
        curr = self.job[st][0] + self.job[st][3]
        stTime = self.job[st][0]
        notServed.remove(st)
        
        while notServed:
            candidates = sorted(notServed, key=lambda i: self.dist[st][i])
            
            for i in candidates:
                begin = curr + self.dist[st][i]
                finish = begin + self.job[i][3]

                # begin >= twS && end <= twE && WL
                if (begin >= self.job[i][0] and
                    finish <= self.job[i][1] and
                    finish - stTime <= worktime):

                    cluster.append(i)
                    curr = finish
                    notServed.remove(i)
                    st = i
                    break
            else:   # for ... else: if didn't use break during for, run else
                break
        cgLv = max(self.job[i][2] for i in cluster)
        return [cluster, int(cgLv)]

    # get coordinates for patients
    def getPoints(self):
        if self.pts is None:
            chDist = (self.dist + self.dist.T) / 2
            mds = MDS(n_components = 2, dissimilarity = "precomputed", random_state = 42)
            self.pts = mds.fit_transform(chDist)
        return self.pts
        
    # use CH to get visiting routes of cgs
    def convexHull(self):
        # apply MDS to construct CH (need symm matrix)
        pts = self.getPoints()
        notServed = list(range(len(self.job)))
        clusters = []

        while len(notServed) >= 3:
            ptsNS = pts[notServed]
            cH = ConvexHull(ptsNS)
            # map hull vertices back to original indices
            cHidx = [notServed[i] for i in cH.vertices]

            st = cHidx[0]
            cluster = self.findCluster(notServed, st)
            clusters.append(cluster)

        # maybe 1 or 2 patients left
        while notServed:
            st = max(notServed, key = lambda i: self.job[i][2])
            cluster = self.findCluster(notServed, st)
            clusters.append(cluster)

        return clusters

    def getCost(self, clusters):
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
                cost += self.dist[route[i]][route[i + 1]] * TR
            costs.append(cost)

            #time
            serve = 0
            for i in range(len(route) - 1):
                serve += self.job[route[i]][3]
                serve += self.dist[route[i]][route[i + 1]]
            serve += self.job[route[-1]][3]
            times.append(int(serve))
        return [costs, times]

    # plot process
    def plotProcess(self, routes):  
        SIZE = 500
        visited = set()   
        route = []
        pts = self.getPoints()

        for cluster in routes:
            route = cluster[0]

            # Update visited patients from previous routes
            unvisited_idx = [i for i in range(len(pts)) if i not in visited]
            unvisited = pts[unvisited_idx]

            # Draw all patients
            for i, (x, y) in enumerate(pts):
                if i in visited:
                    color = "lightgray"
                else:
                    level = self.job[i][2]
                    color = {1: "lime", 2: "cyan", 3: "red"}[level]

                plt.scatter(x, y, color=color, marker="o", s=SIZE, zorder=2, edgecolors="black")
                plt.text(x, y, str(i + 1), fontsize=10, color="black", ha="center", va="center", zorder=3)

            # Draw convex hull of unvisited patients
            if len(unvisited) >= 3:
                hull = ConvexHull(unvisited)
                for simplex in hull.simplices:
                    plt.plot(unvisited[simplex, 0], unvisited[simplex, 1], dashes=[3, 6], color="black", zorder=1, linewidth=1)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
            plt.show()

            # Draw the route
            route_x = [pts[i][0] for i in route]
            route_y = [pts[i][1] for i in route]

            plt.figure(figsize=(8, 8))

            for i, (x, y) in enumerate(pts):
                if i in visited:
                    color = "lightgray"
                else:
                    level = self.job[i][2]
                    color = {1: "lime", 2: "cyan", 3: "red"}[level]

                plt.scatter(x, y, color=color, marker="o", s=SIZE, zorder=2, edgecolors="black")
                plt.text(x, y, str(i + 1), fontsize=10, color="black", ha="center", va="center", zorder=3)

            # Draw convex hull
            if len(unvisited) >= 3:
                for simplex in hull.simplices:
                    plt.plot(unvisited[simplex, 0], unvisited[simplex, 1], dashes=[3, 6], color="black", zorder=1, linewidth=1)

            plt.plot(route_x, route_y, color="black", zorder=1)
            plt.text(route_x[0] - 1, route_y[0] + 2.1, "start", fontsize=10, color="black", va="center", ha="center", zorder=3)
            plt.text(route_x[-1], route_y[-1] + 2.1, "end", fontsize=10, color="black", va="center", ha="center", zorder=3)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
            plt.show()

            # Mark this route's patients as visited
            visited.update(route)

# def plotWhole(pts, job, routes):
#     SIZE = 500 # 200
#     EPSIZE = 200
#     visited = []    
#     route = []
    
#     plt.figure(figsize=(8, 8))

#     for i, pt in enumerate(pts):
#         x, y = pt[0], pt[1]
#         level = job[i][2]
#         if level == 1:
#             color = "lime" # "black"
#         elif level == 2:
#             color = "cyan" # "black"
#         elif level == 3:
#             color = "red" # "black"
#         plt.scatter(x, y, color = color, marker = "o", s = SIZE, zorder = 2, edgecolors = "black")
#         plt.text(x, y, str(i + 1), fontsize = 10, color = "black", ha = "center", va = "center", zorder = 3)
    
#     for route in routes:
#         # route
#         print(route)
#         route_x = [pts[i][0] for i in route[0]]
#         route_y = [pts[i][1] for i in route[0]]
#         plt.text(route_x[0], route_y[0] + 1.2, str("start"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
#         plt.text(route_x[-1], route_y[-1] + 1.2, str("end"), fontsize = 10, color = "black", va = "center", ha = "center", zorder = 3)
#         plt.plot(route_x, route_y, zorder = 1, color = "black")

#     plt.xticks([])
#     plt.yticks([])
#     plt.axis = False
#     plt.show()