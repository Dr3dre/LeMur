from inspyred import ec
from random import Random
from intervaltree import IntervalTree, Interval
import sys

seed = 123
prng = Random()
prng.seed(seed)

def overlaps(intervals):
    es = []
    for a, b in intervals:
        es.append((a, -1))
        es.append((b, 1))
    es.sort()
    result = 0
    n = 0
    for a, s in es:
        if s == -1: result += n
        n -= s
    return result


# individual = iterable of cycles c such as : [ pos, load_beg[num_levate[c]], load_end[num_levate[c]], unload_beg[num_levate[c]], unload_end[num_levate[c]]]

def ga_refinement (products, num_machines, base_op_cost, levata_cost, velocity_step_size, time_units_in_a_day, end_shift, gap_at_day) :
    # copy class variables
    penalty_weight = 7
    penalty_weight *= time_units_in_a_day
    machine = 1

    seed = [[] for _ in range(num_machines)]
    for p, prod in products:
        for c in prod.setup_beg.keys():
            cycle = {
                "pos": -1,
                "product": p,
                "velocity": prod.velocity[c],
                "load_beg": [],
                "load_end": [],
                "unload_beg": [],
                "unload_end": []
            }

            for l in range(prod.num_levate[c]):
                cycle["load_beg"].append(prod.load_beg[c,l])
                cycle["load_end"].append(prod.load_end[c,l])
                cycle["unload_beg"].append(prod.unload_beg[c,l])
                cycle["unload_end"].append(prod.unload_end[c,l])

            # append individual to seed of machine "m"
            seed[prod.machine[c]].append(cycle)

    for m in range(num_machines):
        seed[m] = sorted(seed[m], key=lambda x: x["load_beg"][0])
        for cycle in seed[m]:
            cycle['pos'] = seed[m].index(cycle)

    def generator (random, args) :
        return seed[machine]

    def evaluator (candidates, args) :

        fitness = []
        for individual in candidates :
            # fitness is basically the machine makespan
            this_fitness = individual[-1]["unload_end"][-1]
            # penalties are assign to avoid unfeasible spaces
            penalty = 0
            # Penalty for non-respecting levata_cost[p]
            for cycle in individual :
                effective_levata_cost = levata_cost[cycle["product"]] - cycle["velocity"]*velocity_step_size[cycle["product"]]
                for l in range(len(cycle["load_beg"])) :
                    levata_width = cycle["unload_beg"][l]-cycle["load_end"][l]

                    # penalize on amount of times the levata cost is not respected
                    if levata_width < effective_levata_cost:
                        penalty += 1

            # penalty for overlapping cycles
            intervals = [(cycle["load_beg"][0],cycle["unload_end"][-1]) for cycle in individual]
            # penalize on amount of overlapping intervals
            penalty += overlaps(intervals)

            # total fitness accounts for makespan and weighted penalty
            fitness.append(this_fitness + penalty_weight * penalty)

        return fitness

    def adjust_operation (beg, end, cost) :
        # consider in which day it ends up
        beg_day = beg % time_units_in_a_day
        end = beg + cost
        # add proper gap if necessary
        if end > (end_shift + time_units_in_a_day*beg_day) :
            end += gap_at_day[beg_day]

    def adjust_cycle (cycle) :
        effective_levata_cost = levata_cost[cycle["product"]] - cycle["velocity"]*velocity_step_size[cycle["product"]]
        for l in range(len(cycle["load_beg"])) :
            adjust_operation(cycle["load_beg"][l], cycle["load_end"][l], base_op_cost[cycle["product"], machine])
            adjust_operation(cycle["load_end"][l], cycle["unload_beg"][l], effective_levata_cost)
            adjust_operation(cycle["unload_beg"][l], cycle["unload_end"][l], base_op_cost[cycle["product"], machine]) 

    def mutation (random, candidates, args) :
        # 50% chance mutating levate :
        for individual in candidates :
            if random.choice([0,1]) == 0 : 
                for cycle in individual :
                    for l in range(len(cycle["load_beg"])) :
                        if random.choice([0,1]) == 0 : # 50% chance of mutating Load of levata "l"
                            cycle["load_beg"][l] += random.choice([-1, 1])
                            adjust_operation(cycle["load_beg"][l], cycle["load_end"][l], base_op_cost[cycle["product"], machine])
                        if random.choice([0,1]) == 0 : # 50% chance of mutating Load of levata "l"
                            cycle["unload_beg"][l] += random.choice([-1, 1])
                            adjust_operation(cycle["unload_beg"][l], cycle["unload_end"][l], base_op_cost[cycle["product"], machine])
            else :
                # mutate cycle position
                cycle1 = random.choice(individual)
                cycle2 = random.choice(individual)
                if cycle1["pos"] != cycle2["pos"] :
                    cycle1["pos"], cycle2["pos"] = cycle2["pos"], cycle1["pos"]
                    cycle1["load_beg"][0], cycle2["load_beg"][0] = cycle2["load_beg"][0], cycle1["load_beg"][0]
                    adjust_cycle(cycle1)
                    adjust_cycle(cycle2)
        
        return candidates

    def crossover (random, candidate, args) :
        pass

    algorithm = ec.EvolutionaryComputation(prng)
    algorithm.terminator = ec.terminators.default_termination
    algorithm.replacer = ec.replacers.steady_state_replacement    
    algorithm.variator = [ec.variators.default_variation, mutation]
    algorithm.selector = ec.selectors.tournament_selection

    final_pop = algorithm.evolve(
        generator=generator,
        evaluator=evaluator,
        pop_size=50,
        maximize=False,
        mutation_rate=1,
        max_generations=500,
    )

    best_idx = 0
    best_fitness = sys.maxsize
    for idx, solution in enumerate(final_pop) :
        if solution.fitness < best_fitness :
            best_fitness = solution.fitness
            best_idx = idx
    
    print(final_pop[best_idx].fitness)

    return final_pop[best_idx]


