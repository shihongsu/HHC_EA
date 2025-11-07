import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import pygad
import numpy as np
from const import *
from dhich import getInit

class HHC_GA(pygad.GA):
    def __init__(self, dist, job, config) -> None:

        self.distances = dist
        self.patients = job
        n = len(self.patients)

        init_seq = getInit(dist, job)

        # *** need to put in best sol !!!

        # init_pop
        init_pop = [init_seq]
        for _ in range(config["population"]-1):
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
            keep_parents = config["keep_parents"],
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
        objective = 0
        schedules = self.make_schedule(sol)
        for cg in schedules:
            if cg[1] == 1:
                objective += LV1CG
            elif cg[1] == 2:
                objective += LV2CG
            elif cg[1] == 3:
                objective += LV3CG
            for i in range(len(cg[0]) - 1):
                dist = self.distances[cg[0][i]][cg[0][i + 1]]
                objective += dist * TR
        # print("FIT:", objective)
        return -objective

    def make_schedule(self, sol):
        caregivers = []

        job = self.patients[sol[0]]
        cg = [[sol[0]], job[2], job[0] + job[3], job[3]]
        # seq, lv, curr, serv
        for pat in sol[1:]:
            last_pat = cg[0][-1]
            last_job = self.patients[cg[0][-1]]
            dist = self.distances[pat][last_pat]
            if self.check(cg, pat):
                cg[0].append(pat)
                cg[2] += (dist + last_job[3])
                cg[3] += (dist + last_job[3])
            else:
                caregivers.append(cg)
                job = self.patients[pat]
                cg = [[pat], job[2], job[0] + job[3], job[3]]
        caregivers.append(cg)
        return caregivers

    def check(self, cg, pat):
        last_pat = cg[0][-1]
        job = self.patients[pat]
        dist = self.distances[pat][last_pat]
        serve = self.patients[pat][3]
        if (cg[1] >= job[2] and 
			cg[3] + dist + serve <= WLUB and
			job[0] <= cg[2] + dist and
			cg[2] + dist + job[3] <= job[1]):
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

    def plot_best_sol(self):
        sol, fitness, _ = self.get_best_sol()
        print("Is tree", nx.is_tree(sol))
        print("fitness", fitness)

        # calculate total SPT
        start = next(node for node, data in sol.nodes(data=True) if data.get("start", False))
        dijkstra = nx.single_source_dijkstra_path_length(sol, start, weight = "weight")
        result = {}
        for node in sol.nodes():
            if node in dijkstra:  # If the node is reachable, use its Dijkstra value
                result[node] = dijkstra[node]
            else:  # If the node is not reachable, assign a penalty
                result[node] = self.graph_total_weight
        print(sum(result.values()))

        self.plot_embedded_graph(sol)

    def get_best_sol(self, pop_fitness=None):
        gene_sol, solution_fitness, solution_idx = self.best_solution(pop_fitness)
        
        sol = self.geno2pheno(gene_sol)

        return sol, solution_fitness, solution_idx

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
            