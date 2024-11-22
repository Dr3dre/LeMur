from inspyred import ec
from random import Random
from intervaltree import IntervalTree
from data_init import RunningProduct
import matplotlib.pyplot as plt
from ga_utils import *
from datetime import datetime

# Customizable parameters
POPULATION_SIZE = 100
GENERATIONS = 200
PENALTY_COEFFICIENT = 3

# enable temperature profiling to adjust mutation probabilities over generations
TEMPERATURE_PROFILING = True
INITIAL_TEMP = 0.8
GOAL_TEMP = 0.333
MIDPOINT = int(GENERATIONS / 2.5) # midpoint of the sigmoid function expressed in generations
STEEPNESS = 0.125 # steepness of the sigmoid function

RAND_SEED = int(datetime.now().timestamp()) # set fixed seed for reproducibility

# Output parameters
OUTPUT_LOSS_PLOT = "plots/loss_plot.png"
OUTPUT_TEMP_PROFILE_PLOT = "plots/temperature_profile.png"
PLOT_METRICS = True

class GA_Refiner:
    def __init__(self, seed=RAND_SEED):
        self.prng = Random()
        self.prng.seed(seed)

    def refine_solution(self, products, args):
        # Customizable parameters
        self.penalty_weight = PENALTY_COEFFICIENT * args["time_units_in_a_day"]
        args["minimize"] = True
        # Variables initialization
        self.products = products
        self.best_candidate = None
        args["curr_generation"] = 0
        
        # Correct some aspects input arguments, for instance :
        #  - 1-based to 0-based conversion for input referring to machines, as
        #    machine indexes are 0-based in genetic optimization while they're 1-based in input
        #  - Modify prohibited_intervals for easier control in the genetic optimization
        convert_args(args)
        
        # Initialize as seed the given schedule
        self.seed = self._initialize_seed(products, args)
        # Populate overlap Tree for domain gaps
        args["domain_interval_tree"] = IntervalTree.from_tuples(args["prohibited_intervals"])

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

        # Probability of applying _compact_and_shift mutation
        # It degrades over generations and is proportional to the
        # number of machines in the input space
        args["temperature"] = [temperature_profile(t, MIDPOINT, STEEPNESS, INITIAL_TEMP, GOAL_TEMP) for t in range(GENERATIONS+1)]
        if PLOT_METRICS and TEMPERATURE_PROFILING :
            visualize_temperature_profile(temperature_profile, (0, GENERATIONS+1), OUTPUT_TEMP_PROFILE_PLOT,
                midpoint=MIDPOINT,
                steepness=STEEPNESS,
                initial_temp=INITIAL_TEMP,
                goal_temp=GOAL_TEMP
            )

        _ = ga.evolve(
            generator=self.generator,
            evaluator=self.evaluator,
            pop_size=POPULATION_SIZE,
            maximize=(not args["minimize"]),
            num_selected=int(POPULATION_SIZE * 0.5),
            mutation_rate=1,
            crossover_rate=0,
            max_generations=GENERATIONS,
            #tournament_size=3,
            **args
        )

        if PLOT_METRICS:
            plt.savefig(OUTPUT_LOSS_PLOT, bbox_inches='tight')
            plt.close()

        # extract best candidate found over generations
        solution = self.best_candidate.candidate
        # solution go through conformity check
        # to ensure it is feasible
        is_valid = check_solution_conformity(solution, args)
        solution_fitness = self.best_candidate.fitness if is_valid else -1

        if is_valid :
            # Assign back values to product instances
            for m, machine_queue in enumerate(solution):
                for elem in machine_queue:
                    p = elem["product"]
                    c = elem["cycle"]
                    for _, prod in products:
                        if prod.id == p:
                            # 0-based to 1-based conversion upon returning values as 
                            # they're corrected to 0-based for the whole genetic optimization process
                            prod.machine[c] = (m + 1)
                            prod.setup_beg[c] = elem["setup_beg"]
                            prod.setup_end[c] = elem["setup_end"]
                            for l in range(len(elem["load_beg"])) :
                                prod.load_beg[c,l] = elem["load_beg"][l]
                                prod.load_end[c,l] = elem["load_end"][l]
                                prod.unload_beg[c,l] = elem["unload_beg"][l]
                                prod.unload_end[c,l] = elem["unload_end"][l]
                            
                            prod.cycle_end[c] = elem["unload_end"][-1]
                            break
        
        return products, is_valid, solution_fitness
    
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
                # Apply 1-based to 0-based conversion to prod.machine[c]
                seed[prod.machine[c]-1].append(cycle)

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

        if PLOT_METRICS:
            generational_stats_plot(population, num_generations, num_evaluations, args)


    def generator(self, random, args):
        return [self.seed[m] for m in range(args["num_machines"])]

    def evaluator(self, candidates, args):
        fitness = []
        for individual in candidates :
            # penalties are assigned to discourage unfeasible spaces
            penalty = 0
            individual_fitness = 0
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
                # Fitness score is the cumulative makespan, which accounts for all
                # cycle ends instead of the maximum one only (as in traditional makespan)
                individual_fitness += sum([cycle["unload_end"][-1] for cycle in machine_queue])
            # Store fitness
            fitness.append(individual_fitness)

        return fitness

    def mutation(self, random, candidates, args):
        for individual in candidates:
            # Enabling temperature profiling assign much stronger probability to _pop_and_push mutation
            #   => That's because it's the most effective mutation in terms of potential fitness improvement
            # After a certain number of generations, the probability of applying _pop_and_push mutation
            # decreases with a sigmoid function, allowing other minor mutations (_compact_and_shift + _random_swap_two) to be performed
            if TEMPERATURE_PROFILING :
                if random.random() < args["temperature"][args["curr_generation"]] :
                    individual = self._pop_and_push(individual, random, args)
                else :
                    mutation_choice = random.choice([0, 1])
                    if mutation_choice == 0 :
                        individual = self._random_swap_two(individual, random, args)
                    elif mutation_choice == 1 :
                        individual = self._compact_and_shift(individual, random, args)
                    else :
                        pass
            # Otherwise, apply mutations uniformly
            else :
                mutation_choice = random.choice([0, 1, 2])
                if mutation_choice == 0 :
                    individual = self._compact_and_shift(individual, random, args)
                elif mutation_choice == 1 :
                    individual = self._random_swap_two(individual, random, args)
                elif mutation_choice == 2 :
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

        # Anchor "Now" as new starting point for machine
        # the algorithm will assign in a greedy way the first
        # available slot for each cycle operation
        machine_queues = {machine_idx: machine_queue}
        anchors = {machine_idx: args["time_units_from_midnight"]}
        adjustment_start = {machine_idx: 0}
        # Adjust machine
        res = adjust_machines(random, individual, machine_queues, anchors, adjustment_start, args)
        machine_queue = res[machine_idx]

        return individual

    def _random_swap_two(self, individual, random, args):
       # choose a random machine queue in the solution
        machine_idx = random.choice([m for m in range(args["num_machines"]) if not (len(individual[m]) == 0 or ( (len(individual[m]) == 1) and individual[m][0]["fixed"])) ])
        machine_queue = individual[machine_idx]

        # change cycle position
        cycle_candidates = [c for c in machine_queue if not c["fixed"]]
        cycle1 = random.choice(cycle_candidates)
        cycle2 = random.choice(cycle_candidates)
        if cycle1["pos"] != cycle2["pos"] :
            # Swap cycles
            machine_queue[cycle1["pos"]], machine_queue[cycle2["pos"]] = machine_queue[cycle2["pos"]], machine_queue[cycle1["pos"]]
            cycle1["pos"], cycle2["pos"] = cycle2["pos"], cycle1["pos"]
            # Setup parameters for machine adjustment
            machine_queues = {machine_idx: machine_queue}
            anchors = {machine_idx: min(cycle1["setup_beg"], cycle2["setup_beg"])}
            adjustment_start = {machine_idx: min(cycle1["pos"], cycle2["pos"])}
            # Adjust machine
            res = adjust_machines(random, individual, machine_queues, anchors, adjustment_start, args)
            machine_queue = res[machine_idx]
        
        return individual

    def _pop_and_push(self, individual, random, args):
        # pick a machine randomly (excluding empty machines & machines with only a single fixed cycles)
        # "fixed" basically flags if such cycle is a RunningProduct cycle => it cannot be moved
        source_machine_idx = random.choice([m for m in range(args["num_machines"]) if not (len(individual[m]) == 0 or ( (len(individual[m]) == 1) and individual[m][0]["fixed"])) ])
        source_machine = individual[source_machine_idx]
        # discard fixed cycles from source machine
        source_candidates = [c for c in source_machine if not c["fixed"]]
        if len(source_candidates) == 0 : return individual # skip mutation if source has no cycles available
        # pick a random cycle to move from source machine
        source_cycle = random.choice(source_candidates)
        source_pos = source_cycle["pos"]
        # remove cycle from source machine
        source_machine.remove(source_cycle)
        source_anchor = source_cycle["setup_beg"] # set anchor point for source machine
        # fix machine positions on source machine
        for i, cycle in enumerate(source_machine):
            cycle["pos"] = i

        # pick compatible machine according to the selected cycle on source machine
        target_machine_candidates = [m for m in args["prod_to_machine_comp"][source_cycle["product"]] if m != source_machine_idx and m not in args["broken_machines"]]
        if len(target_machine_candidates) == 0 : return individual # skip mutation if no target has no machines available
        # pick a random target machine
        target_machine_idx = random.choice(target_machine_candidates)
        target_machine = individual[target_machine_idx]
        # look for a position to insert the cycle in the target machine
        target_candidates_pos = [c["pos"] for c in individual[target_machine_idx] if not c["fixed"]]
        target_candidates_pos.append(len(target_machine)) # dummy position to append cycle at the end
        target_pos = random.choice(target_candidates_pos)
        
        # set anchor point for target machine
        if target_pos == 0 :
            # if first position is chosen then use source cycle setup_beg as anchor
            target_anchor = args["time_units_from_midnight"]
        elif 0 < target_pos < len(target_machine) :
            # if a middle position is chosen then use setup_beg as anchor
            target_anchor = target_machine[target_pos]["setup_beg"]
        else :
            # if dummy position is chosen then use last cycle unload_end as anchor
            target_anchor = target_machine[-1]["unload_end"][-1]
        
        # insert source cycle to target machine at specified position
        target_machine.insert(target_pos, source_cycle)
        # fix machine positions on target machine
        for i, cycle in enumerate(target_machine):
            cycle["pos"] = i
        
        # Setup parameters for machine adjustment
        machine_queue = {source_machine_idx: source_machine, target_machine_idx: target_machine}
        anchor = {source_machine_idx: source_anchor, target_machine_idx: target_anchor}
        adjustment_start = {source_machine_idx: source_pos, target_machine_idx: target_pos}
        # Adjust machines
        res = adjust_machines(random, individual, machine_queue, anchor, adjustment_start, args)
        source_machine = res[source_machine_idx]
        target_machine = res[target_machine_idx]

        return individual
