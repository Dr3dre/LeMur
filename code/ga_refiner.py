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
        self.best_candidate = None
        args["plot_metrics"] = plot_metrics
        # Populate overlap Tree for domain gaps
        args["domain_interval_tree"] = IntervalTree.from_tuples(args["prohibited_intervals"])

        # EA configuration and execution
        ga = ec.GA(self.prng)
        ga.terminator = ec.terminators.generation_termination
        ga.replacer = ec.replacers.steady_state_replacement    
        ga.variator = [ec.variators.uniform_crossover, self.mutation]
        ga.selector = ec.selectors.rank_selection
        ga.observer = self.observer

        population = 10

        _ = ga.evolve(
            generator=self.generator,
            evaluator=self.evaluator,
            pop_size=population,
            maximize=False,
            num_selected=population,
            mutation_rate=1,
            crossover_rate=0,
            max_generations=100,
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
        
        if num_generations == 0 :
            # iniitialize best candidate with first candidate
            # in population only in first generation
            self.best_candidate = population[0]

        # Look for the best candidate in the population & store it
        for candidate in population :
            if candidate.fitness < self.best_candidate.fitness :
                self.best_candidate = candidate
        
        if args["plot_metrics"]:
            stats = ec.analysis.fitness_statistics(population)
            best_fitness = stats['best']
            worst_fitness = stats['worst']
            median_fitness = stats['median']
            average_fitness = stats['mean']
            colors = ['black', 'blue', 'green', 'red']
            labels = ['average', 'median', 'best', 'worst']
            data = []
            if num_generations == 0:
                plt.figure('A')
                plt.ion()
                data = [[num_evaluations], [average_fitness], [median_fitness], [best_fitness], [worst_fitness]]
                lines = []
                for i in range(4):
                    line, = plt.plot(data[0], data[i+1], color=colors[i], label=labels[i])
                    lines.append(line)
                args['plot_data'] = data
                args['plot_lines'] = lines
                plt.xlabel('Evaluations')
                plt.ylabel('Fitness')
            else:
                data = args['plot_data']
                data[0].append(num_evaluations)
                data[1].append(average_fitness)
                data[2].append(median_fitness)
                data[3].append(best_fitness)
                data[4].append(worst_fitness)
                lines = args['plot_lines']
                for i, line in enumerate(lines):
                    line.set_xdata(numpy.array(data[0]))
                    line.set_ydata(numpy.array(data[i+1]))
                args['plot_data'] = data
                args['plot_lines'] = lines
            ymin = min([min(d) for d in data[1:]])
            ymax = max([max(d) for d in data[1:]])
            yrange = ymax - ymin
            plt.xlim((0, num_evaluations))
            plt.ylim((ymin - 0.1*yrange, ymax + 0.1*yrange))
            plt.draw()
            plt.legend()
        
        if num_generations % 25 == 0:
            print(f"Gen [{num_generations}] Fitness => {self.best_candidate.fitness}")




    def generator(self, random, args):
        return [self.seed[m] for m in range(args["num_machines"])]

    def evaluator(self, candidates, args):
        fitness = []
        for individual in candidates :
            # fitness is basically the machine makespan
            machine_fitness = []
            # penalties are assign to avoid unfeasible spaces
            penalty = 0

            for machine_queue in individual :
                # Strong penalty for not respecting time window
                for cycle in machine_queue :
                    # lower bound is either "Now"
                    lb = args["time_units_from_midnight"]
                    if cycle["setup_beg"] < lb :
                        penalty += lb - cycle["setup_beg"]
                    ub = args["horizon"]
                    if cycle["unload_end"][-1] > ub:
                        penalty += cycle["unload_end"][-1] - ub

                # total fitness accounts for machine makespan and weighted penalty
                machine_makespan = 0 if len(machine_queue) == 0 else sum([cycle["unload_end"][-1] for cycle in machine_queue])
                machine_fitness.append(machine_makespan + self.penalty_weight*penalty)
        
            # Solution fitness is sum of all machine fitness
            fitness.append(sum(machine_fitness))
        return fitness

    def mutation(self, random, candidates, args):
        for individual in candidates:
            mutation_type = self.prng.choice([0, 1, 2])

            if mutation_type == 0:
                individual =self._compact_and_shift(individual, random, args)
            elif mutation_type == 1:
                individual = self._random_2_swap(individual, random, args)
            elif mutation_type == 2:
                individual = self._pop_and_push(individual, random, args)
        
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

    def _random_2_swap(self, individual, random, args):
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


    
