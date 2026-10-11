import random as rd
import pygad
import numpy as np

from const import *
from dhich import DHICH

class HHC_GA(pygad.GA):
    def __init__(self, dist, job, config) -> None:

        self.dist = dist
        self.job = job
        n = len(self.job)

        solver = DHICH(self.dist, self.job)
        clusters = solver.convexHull()
        init_seq = [pat for cluster in clusters for pat in cluster[0]]
        # print(init_seq)
        # *** need to put in best sol !!!

        # init_pop
        init_pop = [init_seq]
        for _ in range(config["population"] - 1):
            i = rd.random()
            if i <= 0.25:
                init_pop.append(self.swap_perturb(init_seq))
            elif 0.25 < i <= 0.5:
                init_pop.append(self.insert_perturb(init_seq))
            elif 0.5 < i <= 0.75:
                init_pop.append(self.shuffle_perturb(init_seq))
            elif 0.75 < i <= 1:
                init_pop.append(np.random.permutation(n))

        self.mut_prob = config["mut_prob"]

        super().__init__(
            num_generations = config["generation"],
            num_parents_mating = config["num_par_mat"],
            fitness_func = self._fitness_func,
            crossover_type = self._crossover_func,
            mutation_type = self._mutation_func,
            mutation_probability = config["mut_prob"],
            sol_per_pop = config["population"],
            num_genes = n,
            initial_population = init_pop,
            gene_type = int,
            on_generation = self.callback_generation,
            K_tournament = config["K_tour"] if config["parent_selection_type"] == "tournament" else None,
            parent_selection_type = config["parent_selection_type"],
            keep_elitism = config["keep_elitism"],
        )

        self.records = {"best_fit_val": [],
                        "best_sol": None,
                        "lv3": [],
                        "lv2": [],
                        "lv1": [],
                        }
        
    def swap_perturb(self, gene, swaps = 5):
        new_gene = gene.copy()
        n = len(new_gene)
        for _ in range(swaps):
            i, j = rd.sample(range(n), 2)
            new_gene[i], new_gene[j] = new_gene[j], new_gene[i]
        return new_gene
    
    def insert_perturb(self, gene, insertions = 5):
        new_gene = gene.copy()
        n = len(new_gene)
        for _ in range(insertions):
            i = rd.randrange(n)
            element = new_gene.pop(i)
            j = rd.randrange(n)
            new_gene.insert(j, element)
        return new_gene
    
    def shuffle_perturb(self, gene, deck = 5):
        new_gene = gene.copy()
        n = len(new_gene)
        start = rd.randrange(n - deck + 1)
        end = start + deck
        subseq = new_gene[start: end]
        rd.shuffle(subseq)
        new_gene[start: end] = subseq
        return new_gene

    def _fitness_func(self, ga_instance, sol, sol_idx):
        fitness = 0
        routes = self.make_schedule(sol)
        # print(schedules)
        for route in routes:
            if route[1] == 1:
                fitness += LV1CG
            elif route[1] == 2:
                fitness += LV2CG
            elif route[1] == 3:
                fitness += LV3CG
            for i in range(len(route[0]) - 1):
                fitness += self.dist[route[0][i]][route[0][i + 1]] * TR
        # print("FIT:", objective)
        return -fitness

    # make schedules for an individual
    def make_schedule(self, sol):
        routes = []
        # curr: current patient
        # route: [route, lv, curr time, service time]
        curr = self.job[sol[0]]
        route = [[sol[0]], -1, curr[0] + curr[3], curr[3]]

        for pat in sol[1:]:
            last_pat = route[0][-1]
            last_job = self.job[last_pat]
            d = self.dist[pat][last_pat]
            if self.check(route, pat):
                route[0].append(pat)
                route[2] += (d + self.job[pat][3])
                route[3] += (d + self.job[pat][3])
            else:
                route[1] = max(self.job[i][2] for i in route[0])
                routes.append(route)
                job = self.job[pat]
                route = [[pat], -1, job[0] + job[3], job[3]]

        # add last cg's route
        route[1] = max(self.job[i][2] for i in route[0])
        routes.append(route)

        return routes

    # check if pat can be added to route
    def check(self, route, pat):
        last_pat = route[0][-1]
        curr = self.job[pat]
        d = self.dist[last_pat][pat]
        serve = self.job[pat][3]
        if (route[3] + d + serve <= WLUB and    # workload
			curr[0] <= route[2] + d and          # starting
			route[2] + d + curr[3] <= curr[1]):   # ending
            return True
        return False

    def order_crossover(self, p1, p2):
        size = len(p1)
        start, end = sorted(np.random.choice(size, 2, replace = False))
        child = [-1] * size
        child[start: end] = p1[start: end]

        pos = 0
        for gene in p2:
            if gene not in child:
                while child[pos] != -1:
                    pos += 1
                child[pos] = gene
                pos += 1
        return child

    def _crossover_func(self, parents, offspring_size, ga_instance):
        offspring = []
        for _ in range(offspring_size[0]):
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            child = self.order_crossover(p1, p2)
            offspring.append(child)
        return np.array(offspring)
    
    def _mutation_func(self, offspring, ga_instance):
        for child in offspring:
            if np.random.rand() < self.mut_prob:
                i, j = np.random.choice(len(child), 2, replace = False)
                temp = child[i]
                child[i] = child[j]
                child[j] = temp
        return offspring


    def callback_generation(self, ga_instance):
        sol, best_fit, _ = ga_instance.best_solution()
        self.records["best_fit_val"].append(best_fit)
        # print("SOL:", sol)
        best_schedule = self.make_schedule(sol)
        lv1, lv2, lv3 = 0, 0, 0
        for schedule in best_schedule:
            if schedule[1] == 1:
                lv1 += 1
            elif schedule[1] == 2:
                lv2 += 1
            elif schedule[1] == 3:
                lv3 += 1
        self.records["lv3"].append(lv3)
        self.records["lv2"].append(lv2)
        self.records["lv1"].append(lv1)
        
        # on finish
        if self.generations_completed == self.num_generations:
            self.records["best_sol"] = sol
            