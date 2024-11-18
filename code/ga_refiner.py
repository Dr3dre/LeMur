from inspyred import ec
from random import Random
from intervaltree import IntervalTree
from data_init import RunningProduct
import matplotlib.pyplot as plt
import numpy

from ga_utils import *

class GA_Refiner:
    def __init__(self, seed=140):
        self.prng = Random()
        self.prng.seed(seed)

    def refine_solution(self, products, args, plot_metrics=True):
        self.penalty_weight = 1 * args["time_units_in_a_day"]
        self.seed = self._initialize_seed(products, args)
        self.products = products
        self.best_candidate = None
        args["minimize"] = True
        args["plot_metrics"] = plot_metrics
        # Populate overlap Tree for domain gaps
        args["domain_interval_tree"] = IntervalTree.from_tuples(args["prohibited_intervals"])
        args["curr_generation"] = 0
        # EA configuration and execution
        ga = ec.GA(self.prng)
        ga.variator = [ec.variators.uniform_crossover, self.mutation]
        ga.observer = self.observer

        ga.terminator = ec.terminators.generation_termination
        ga.selector = ec.selectors.rank_selection
        #ga.selector = ec.selectors.uniform_selection
        #ga.selector = ec.selectors.tournament_selection
        ga.replacer = ec.replacers.comma_replacement
        #ga.replacer = ec.replacers.steady_state_replacement   
        #ga.replacer = ec.replacers.random_replacement

        generations = 150
        population = 50

        # Probability of applying _compact_and_shift mutation
        # It degrades over generations and is proportional to the
        # number of machines in the input space
        initial_temp = 1
        goal_temp = 0.1
        saturation_point = int(args["num_machines"] * 2)
        args["temperature"] = [temperature_profile(t, saturation_point, initial_temp, goal_temp) for t in range(generations+1)]

        _ = ga.evolve(
            generator=self.generator,
            evaluator=self.evaluator,
            pop_size=population,
            maximize=(not args["minimize"]),
            num_selected=int(population * 0.5),
            mutation_rate=1,
            crossover_rate=0,
            max_generations=generations,
            #tournament_size=3,
            **args
        )

        if plot_metrics:
            plt.savefig(f"plots/ga_loss.png", bbox_inches='tight')
            plt.close()

        # extract best candidate found over generations
        solution = self.best_candidate.candidate

        # solution go through conformity check
        # to ensure it is feasible
        is_valid = check_solution_conformity(solution, args)
        if not is_valid:
            print("GA refinement returned INVALID SOLUTION, returning original solution")
            return products

        print("\nGA refinement returned VALID SOLUTION !!!")
        print(f"Solution Fitness => {self.best_candidate.fitness}\n")
        # Assign back values to product instances
        for m, machine_queue in enumerate(solution):
            for elem in machine_queue:
                p = elem["product"]
                c = elem["cycle"]
                for _, prod in products:
                    if prod.id == p:
                        prod.machine[c] = m
                        prod.setup_beg[c] = elem["setup_beg"]
                        prod.setup_end[c] = elem["setup_end"]
                        for l in range(len(elem["load_beg"])) :
                            prod.load_beg[c,l] = elem["load_beg"][l]
                            prod.load_end[c,l] = elem["load_end"][l]
                            prod.unload_beg[c,l] = elem["unload_beg"][l]
                            prod.unload_end[c,l] = elem["unload_end"][l]
                        
                        prod.cycle_end[c] = elem["unload_end"][-1]
                        break
        return products
    
    def _initialize_seed(self, products, args):
        seed = [[] for _ in range(args["num_machines"])]

        args["setup_reduced_cost"] = {}
        args["load_reduced_cost"] = {}
        args["unload_reduced_cost"] = {}

        for p, prod in products:
            for c in prod.setup_beg.keys():
                cycle = {
                    "pos": -1,
                    "fixed": isinstance(prod, RunningProduct) and c == 0,
                    "product": p,
                    "cycle": c,
                    "velocity": prod.velocity[c],
                    "setup_operator": prod.setup_operator[c],
                    "setup_beg": prod.setup_beg[c],
                    "setup_end": prod.setup_end[c],
                    "load_operator": [prod.load_operator[c, l] for l in range(prod.num_levate[c])],
                    "load_beg": [prod.load_beg[c, l] for l in range(prod.num_levate[c])],
                    "load_end": [prod.load_end[c, l] for l in range(prod.num_levate[c])],
                    "unload_operator": [prod.unload_operator[c, l] for l in range(prod.num_levate[c])],
                    "unload_beg": [prod.unload_beg[c, l] for l in range(prod.num_levate[c])],
                    "unload_end": [prod.unload_end[c, l] for l in range(prod.num_levate[c])]
                }
                seed[prod.machine[c]].append(cycle)

        for m in range(args["num_machines"]):
            seed[m] = sorted(seed[m], key=lambda x: x["setup_beg"])
            for cycle in seed[m]:
                cycle['pos'] = seed[m].index(cycle)

        return seed
    
    def observer(self, population, num_generations, num_evaluations, args):
        # Keeps track of current generation
        args["curr_generation"] = num_generations + 1
        
        if num_generations == 0 :
            # iniitialize best candidate with first candidate
            # in population only in first generation
            self.best_candidate = population[0]

        # Look for the best candidate in the population & store it
        for candidate in population :
            if args["minimize"]:
                if candidate.fitness < self.best_candidate.fitness :
                    self.best_candidate = candidate
            else :
                if candidate.fitness > self.best_candidate.fitness :
                    self.best_candidate = candidate

        if args["plot_metrics"]:
            if num_generations % 10 == 0:
                print(f"Gen [{num_generations}] Fitness => {self.best_candidate.fitness}")
            generational_stats_plot(population, num_generations, num_evaluations, args)


    def generator(self, random, args):
        return [self.seed[m] for m in range(args["num_machines"])]

    def evaluator(self, candidates, args):
        fitness = []
        for individual in candidates :
            # penalties are assigned to discourage unfeasible spaces
            penalty = 0
            schedule_makespan = 0
            cumulative_schedule_makespan = 0
            for machine_queue in individual :
                for cycle in machine_queue :
                    # Strong penalty for not respecting time window (Start Date & Due Date)
                    lb = args["start_date"][cycle["product"]]
                    if cycle["setup_beg"] < lb :
                        #penalty += lb - cycle["setup_beg"]
                        penalty += 1
                    ub = args["due_date"][cycle["product"]]
                    if cycle["unload_end"][-1] > ub:
                        #penalty += cycle["unload_end"][-1] - ub
                        penalty += 1
                # Schedule makespan is the maximum among all cycle end times
                machine_makespan = 0 if len(machine_queue) == 0 else machine_queue[-1]["unload_end"][-1]
                schedule_makespan = max(schedule_makespan, machine_makespan)
                # Cumulative makespan accounts for all cycle ends instead of the maximum one only
                cumulative_schedule_makespan += sum([cycle["unload_end"][-1] for cycle in machine_queue])
            
            # Fitness accounts both for schedule makespan and compactness of solution
            # This allows tune machines which doesn't represent a bottleneck
            weight_compactness = 1 / (args["horizon"] * sum([args["max_cycles"][p] for p, _ in self.products]))
            schedule_fitness = schedule_makespan + cumulative_schedule_makespan*weight_compactness
            # Final fitness accounts for both schedule makespan and penalty
            individual_fitness = schedule_fitness + self.penalty_weight*penalty
            # Store fitness
            fitness.append(individual_fitness)

        return fitness

    def mutation(self, random, candidates, args):
        for individual in candidates:

            if random.random() < args["temperature"][args["curr_generation"]] :
                individual = self._compact_and_shift(individual, random, args)
            else :
                mutation_choice = random.choice([0, 1])
                if mutation_choice == 0 :
                    individual = self._random_swap_two(individual, random, args)
                elif mutation_choice == 1 :
                    individual = self._pop_and_push(individual, random, args)
                else:
                    pass

        return candidates

    def crossover (self, random, candidate, args) :
        pass

    def _compact_and_shift(self, individual, random, args):
        # choose a random machine queue in the solution
        machine_idx = random.choice([m for m in range(args["num_machines"]) if len(individual[m]) > 0])
        machine_queue = individual[machine_idx]
        # Generate overlap tree for each operator group (excluding selected machines)
        operator_interval_tree = build_operator_inter_tree(individual, args, exclude_machines=[machine_idx])  

        # Anchor "Now" as new starting point for machine
        # the algorithm will assign in a greedy way the first
        # available slot for each cycle operation
        anchor = args["time_units_from_midnight"]
        machine_queue = adjust_machine(machine_queue, machine_idx, anchor, operator_interval_tree, args)
        
        return individual

    def _random_swap_two(self, individual, random, args):
       # choose a random machine queue in the solution
        machine_idx = random.choice([m for m in range(args["num_machines"]) if not (len(individual[m]) == 0 or ( (len(individual[m]) == 1) and individual[m][0]["fixed"])) ])
        machine_queue = individual[machine_idx]
        # Generate overlap tree for each operator group (excluding selected machines)
        operator_interval_tree = build_operator_inter_tree(individual, args, exclude_machines=[machine_idx])

        # change cycle position
        cycle_candidates = [c for c in machine_queue if not c["fixed"]]
        cycle1 = random.choice(cycle_candidates)
        cycle2 = random.choice(cycle_candidates)
        if cycle1["pos"] != cycle2["pos"] :
            # swap cycles
            machine_queue[cycle1["pos"]], machine_queue[cycle2["pos"]] = machine_queue[cycle2["pos"]], machine_queue[cycle1["pos"]]
            cycle1["pos"], cycle2["pos"] = cycle2["pos"], cycle1["pos"]
            # find which cycle starts first between the two
            anchor = min(cycle1["setup_beg"], cycle2["setup_beg"])
            # Adjust machine
            machine_queue = adjust_machine(machine_queue, machine_idx, anchor, operator_interval_tree, args, from_pos=min(cycle1["pos"], cycle2["pos"]))
        
        return individual

    def _pop_and_push(self, individual, random, args):
        # pick a machine randomly (excluding empty machines & machines with only a single fixed cycles)
        # "fixed" basically flags if such cycle is a RunningProduct cycle => it cannot be moved
        source_machine_idx = random.choice([m for m in range(args["num_machines"]) if not (len(individual[m]) == 0 or ( (len(individual[m]) == 1) and individual[m][0]["fixed"])) ])
        source_machine = individual[source_machine_idx]
        # pick compatible machine IN SAME MACHINE GROUP !!!!!!
        target_machine_idx = random.choice([m for m in range(args["num_machines"]) if m != source_machine_idx])
        target_machine = individual[target_machine_idx]

        # discard fixed cycles from source machine
        source_candidates = [c for c in source_machine if not c["fixed"]]
        # skip mutation if no candidates are available
        if len(source_candidates) > 0 :
            # pick a random cycle to move and remove it from source machine
            source_cycle = random.choice(source_candidates) 
            source_machine.remove(source_cycle)
            # gather source info
            source_anchor = source_cycle["setup_beg"]
            source_from_pos = source_cycle["pos"]
            # fix machine positions on source machine
            for i, cycle in enumerate(source_machine):
                cycle["pos"] = i

            # gather target info
            target_anchor = args["time_units_from_midnight"] if len(target_machine) == 0 else target_machine[-1]["unload_end"][-1]
            target_from_pos = len(target_machine)
            # append cycle to target machine
            target_machine.append(source_cycle)
            for i, cycle in enumerate(target_machine):
                cycle["pos"] = i
            
            # cycles to exclude are the ones requiring an adjustment
            #   1) those in source machine starting from source_from_pos
            cycles_to_exclude = [(elem["product"], elem["cycle"]) for elem in source_machine if elem["pos"] >= source_from_pos]
            #   2) the last cycle in target machine which was previously appended
            cycles_to_exclude.append((target_machine[-1]["product"], target_machine[-1]["cycle"]))
            # Generate overlap tree excluding the specified cycles
            operator_interval_tree = build_operator_inter_tree(individual, args, exclude_cycles=cycles_to_exclude)

            # Choose which machine to adjust first
            # (it matters due to operator overlaps)
            if random.choice([0, 1]) == 0 :
                # Adjust first Source machine
                source_machine = adjust_machine(source_machine, source_machine_idx, source_anchor, operator_interval_tree, args, from_pos=source_from_pos)
                # Add source intervals to overlap tree
                operator_interval_tree = add_intervals_to_tree (operator_interval_tree, source_machine, from_pos=source_from_pos)
                # fix target machine
                target_machine = adjust_machine(target_machine, target_machine_idx, target_anchor, operator_interval_tree, args, from_pos=target_from_pos)
            else :
                # Adjust first Target machine
                target_machine = adjust_machine(target_machine, target_machine_idx, target_anchor, operator_interval_tree, args, from_pos=target_from_pos)
                # Add source intervals to overlap tree
                operator_interval_tree = add_intervals_to_tree (operator_interval_tree,target_machine, from_pos=target_from_pos)
                # fix source machine
                source_machine = adjust_machine(source_machine, source_machine_idx, source_anchor, operator_interval_tree, args, from_pos=source_from_pos)
        
        return individual


    
