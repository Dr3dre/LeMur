from inspyred import ec
from random import Random
from intervaltree import IntervalTree
from data_init import RunningProduct
import matplotlib.pyplot as plt
from ga_utils import *
from datetime import datetime

# Customizable parameters
POPULATION_SIZE = 100
GENERATIONS = 250
# penalty coefficient for unfeasible solutions, if GA returns too often an unfeasible solution, try to increase this value
PENALTY_COEFFICIENT = 3

# enable temperature profiling to adjust mutation probabilities over generations
TEMPERATURE_PROFILING = True
INITIAL_TEMP = 0.95
GOAL_TEMP = 0.5
MIDPOINT = int(GENERATIONS / 2) # midpoint of the sigmoid function expressed in generations
STEEPNESS = 0.1 # steepness of the sigmoid function

RAND_SEED = int(datetime.now().timestamp()) # set fixed seed for reproducibility

# Output parameters
OUTPUT_LOSS_PLOT = "plots/loss_plot.png"
OUTPUT_TEMP_PROFILE_PLOT = "plots/temperature_profile.png"
PLOT_METRICS = False

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
        args = adapt_args(products, args)
        
        # Initialize as seed the given schedule
        self.seed = self._initialize_seed(products, args)
        # Populate overlap Tree for domain gaps
        args["domain_interval_tree"] = IntervalTree.from_tuples(args["prohibited_intervals"])

        # EA configuration and execution
        ga = ec.GA(self.prng)
        ga.variator = [self.crossover, self.mutation]
        ga.observer = self.observer
        ga.terminator = ec.terminators.generation_termination
        ga.selector = ec.selectors.rank_selection
        ga.replacer = ec.replacers.comma_replacement

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
                    p = elem["p"]
                    c = elem["c"]
                    for _, prod in products:
                        if prod.id == p:
                            # 0-based to 1-based conversion upon returning values as 
                            # they're corrected to 0-based for the whole genetic optimization process
                            prod.machine[c] = (m + 1)
                            # Setup
                            prod.setup_beg[c] = elem["setup_beg"]
                            prod.setup_end[c] = elem["setup_end"]
                            if len(args["domain_interval_tree"].overlap(prod.setup_beg[c], prod.setup_end[c])) == 0 :
                                prod.setup_gap[c] = 0
                            else :
                                prod.setup_gap[c] = args["gap_at_day"][prod.setup_beg[c] // args["time_units_in_a_day"]]
                            
                            for l in range(len(elem["load_beg"])) :
                                # Load
                                prod.load_beg[c,l] = elem["load_beg"][l]
                                prod.load_end[c,l] = elem["load_end"][l]
                                if len(args["domain_interval_tree"].overlap(prod.load_beg[c,l], prod.load_end[c,l])) == 0 :
                                    prod.load_gap[c,l] = 0
                                else :
                                    prod.load_gap[c,l] = args["gap_at_day"][prod.load_beg[c,l] // args["time_units_in_a_day"]]
                                # Unload
                                prod.unload_beg[c,l] = elem["unload_beg"][l]
                                prod.unload_end[c,l] = elem["unload_end"][l]
                                if len(args["domain_interval_tree"].overlap(prod.unload_beg[c,l], prod.unload_end[c,l])) == 0 :
                                    prod.unload_gap[c,l] = 0
                                else :
                                    prod.unload_gap[c,l] = args["gap_at_day"][prod.unload_beg[c,l] // args["time_units_in_a_day"]]
                            # Cycle end
                            prod.cycle_end[c] = elem["unload_end"][-1]
                            break

        return products, is_valid, solution_fitness
    
    def _initialize_seed(self, products, args):
        seed = [[] for _ in range(args["num_machines"])]
        for p, prod in products:
            for c in prod.setup_beg.keys():
                cycle = {
                    # Static information (necessary to keep track of cycle information)
                    "p": p,
                    "c": c,
                    "fixed": isinstance(prod, RunningProduct),  # "fixed" basically flags if such cycle is a RunningProduct instance => it cannot be moved
                    # Dynamic information (will be updated during optimization)
                    "setup_beg": prod.setup_beg[c],
                    "setup_end": prod.setup_end[c],
                    "load_beg": [prod.load_beg[c, l] for l in range(prod.num_levate[c])],
                    "load_end": [prod.load_end[c, l] for l in range(prod.num_levate[c])],
                    "unload_beg": [prod.unload_beg[c, l] for l in range(prod.num_levate[c])],
                    "unload_end": [prod.unload_end[c, l] for l in range(prod.num_levate[c])]
                }
                # Apply 1-based to 0-based conversion to prod.machine[c]
                seed[prod.machine[c]-1].append(cycle)

        for m in range(args["num_machines"]):
            seed[m] = sorted(seed[m], key=lambda x: x["setup_beg"])

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
                    lb = args["start_date"][cycle["p"]]
                    if cycle["setup_beg"] < lb :
                        penalty += (lb - cycle["setup_beg"])
                    ub = args["due_date"][cycle["p"]]
                    if cycle["unload_end"][-1] > ub:
                        penalty += (cycle["unload_end"][-1] - ub)
                    # apply penalty also for exceding the makespan found by CP-SAT in first stage optimization
                    if cycle["unload_end"][-1] > args["makespan_limit"] :
                        penalty += (cycle["unload_end"][-1] - args["makespan_limit"]) // 2

                # Fitness score is the cumulative makespan, which accounts for all
                # cycle ends instead of the maximum one only (as in traditional makespan)
                individual_fitness += sum([cycle["unload_end"][-1] for cycle in machine_queue])
            # Store fitness
            fitness.append(individual_fitness + self.penalty_weight*penalty)
            # for each time unit which violates start date / due date
            # the solutions adds PENALTY_COEFFICIENT days to the fitness
        return fitness

    def mutation(self, random, candidates, args):
        for individual in candidates:
            # Enabling temperature profiling assign much stronger probability to some mutations which have
            # been empirically proven to be more effective at the beginning of the optimization process
            #   - After a certain number of generations, the probability of applying such mutations
            #     decreases with a sigmoid function, allowing other minor mutations to be performed more often
            
            if TEMPERATURE_PROFILING :
                if random.random() < args["temperature"][args["curr_generation"]] :
                    mutation_choice = random.choice([0, 1])
                    if mutation_choice == 0 :
                        individual = self._pop_and_push(individual, random, args)
                    elif mutation_choice == 1 :
                        individual = self._between_machines_swap(individual, random, args)
                    else :
                        pass
                else :
                    mutation_choice = random.choice([0, 1])
                    if mutation_choice == 0 :
                        individual = self._within_machine_swap(individual, random, args)
                    elif mutation_choice == 1 :
                        individual = self._compact_and_shift(individual, random, args)
                    else :
                        pass
            # Otherwise, apply mutations uniformly
            else :
                mutation_choice = random.choice([0, 1, 2, 3])
                if mutation_choice == 0 :
                    individual = self._compact_and_shift(individual, random, args)
                elif mutation_choice == 1 :
                    individual = self._within_machine_swap(individual, random, args)
                elif mutation_choice == 2 :
                    individual = self._pop_and_push(individual, random, args)
                elif mutation_choice == 3 :
                    individual = self._between_machines_swap(individual, random, args)
                else:
                    pass
        
        return candidates

    def crossover (self, random, candidates, args) :
        # Crossover is not implemented in this version as we believe
        # with such representation of the solution space it doesn't
        # bring any significant improvement which mutations by themselves already provide
        return candidates

    def _compact_and_shift(self, individual, random, args):
        # choose a random machine queue in the solution
        machine_candidates = [m for m in range(args["num_machines"]) if len(individual[m]) > 0]
        if len(machine_candidates) == 0 : return individual # skip mutation if no machines are available
        machine_idx = random.choice(machine_candidates)
        machine_queue = individual[machine_idx]

        # Anchor "Now" as new starting point for machine
        # the algorithm will assign in a greedy way the first
        # available slot for each cycle operation
        queues = {machine_idx: machine_queue}
        anchors = {machine_idx: args["start_schedule"]}
        adjustment_start = {machine_idx: 0}
        # Adjust machine
        res = adjust_machines(random, individual, queues, anchors, adjustment_start, args)
        machine_queue = res[machine_idx]

        return individual

    def _within_machine_swap(self, individual, random, args):
        # choose a random machine queue in the solution
        machine_candidates = [m for m in range(args["num_machines"]) if not (len(individual[m]) == 0 or ( (len(individual[m]) == 1) and individual[m][0]["fixed"])) ]
        if len(machine_candidates) == 0 : return individual # skip mutation if no machines are available
        machine_idx = random.choice(machine_candidates)
        machine_queue = individual[machine_idx]

        # pick two random cycles to swap
        candidate_cycles = list(range(len(machine_queue)) if not machine_queue[0]["fixed"] else range(1, len(machine_queue))) # discard fixed cycles from available choices
        if len(candidate_cycles) <= 1 : return individual # skip mutation if there aren't at least 2 cycles to swap
        cycle_A_pos = random.choice(candidate_cycles)
        cycle_A = machine_queue[cycle_A_pos]
        candidate_cycles.remove(cycle_A_pos)
        cycle_B_pos = random.choice(candidate_cycles)
        cycle_B = machine_queue[cycle_B_pos]

        # set appropriate anchor points for source and target machines
        start_point = min(cycle_A_pos, cycle_B_pos)
        anchor_point = min(cycle_A["setup_beg"], cycle_B["setup_beg"]) if start_point > 0 else args["start_schedule"]
        
        # Swap cycles
        machine_queue[cycle_A_pos], machine_queue[cycle_B_pos] = machine_queue[cycle_B_pos], machine_queue[cycle_A_pos]
        # Setup parameters for machine adjustment
        machine_queues = {machine_idx: machine_queue}
        anchors = {machine_idx: anchor_point}
        adjustment_start = {machine_idx: start_point}
        # Adjust machine
        res = adjust_machines(random, individual, machine_queues, anchors, adjustment_start, args)
        machine_queue = res[machine_idx]

        return individual

    def _pop_and_push(self, individual, random, args):
        # pick a machine randomly (excluding empty machines & machines with only a single fixed cycles)
        source_machine_candidates = [m for m in range(args["num_machines"]) if not (len(individual[m]) == 0 or ( (len(individual[m]) == 1) and individual[m][0]["fixed"])) ]
        if len(source_machine_candidates) == 0 : return individual # skip mutation if no machines are available
        source_machine_idx = random.choice(source_machine_candidates)
        source_machine = individual[source_machine_idx]
        # discard fixed cycles from available choices
        source_candidate_pos = list(range(len(source_machine)) if not source_machine[0]["fixed"] else range(1, len(source_machine)))
        # pick a random cycle to move from source machine
        source_pos = random.choice(source_candidate_pos)
        source_cycle = source_machine[source_pos]

        # pick compatible machine according to the selected cycle on source machine
        target_machine_candidates = [m for m in args["prod_to_machine_comp"][source_cycle["p"]] if m != source_machine_idx and m not in args["broken_machines"]]
        if len(target_machine_candidates) == 0 : return individual # skip mutation if no target has no machines available
        # pick a random target machine
        target_machine_idx = random.choice(target_machine_candidates)
        target_machine = individual[target_machine_idx]
        # look for a position to insert the cycle in the target machine
        target_candidates_pos = list(range(len(target_machine) + 1) if len(target_machine) == 0 or not target_machine[0]["fixed"] else range(1, len(target_machine) + 1))
        target_pos = random.choice(target_candidates_pos)
        
        # set anchor point for source machine
        source_anchor = source_cycle["setup_beg"] if source_pos > 0 else args["start_schedule"]
        # set anchor point for target machine
        if target_pos == 0 :
            # if first position is chosen then use source cycle setup_beg as anchor
            target_anchor = args["start_schedule"]
        elif 0 < target_pos < len(target_machine) :
            # if a middle position is chosen then use setup_beg as anchor
            target_anchor = target_machine[target_pos]["setup_beg"]
        else :
            # if dummy position is chosen then use last cycle unload_end as anchor
            target_anchor = target_machine[-1]["unload_end"][-1]
        
        # pop cycle from source and push it to target
        source_machine.remove(source_cycle)
        target_machine.insert(target_pos, source_cycle)

        # Setup parameters for machine adjustment
        machine_queue = {source_machine_idx: source_machine, target_machine_idx: target_machine}
        anchor = {source_machine_idx: source_anchor, target_machine_idx: target_anchor}
        adjustment_start = {source_machine_idx: source_pos, target_machine_idx: target_pos}
        # Adjust machines
        res = adjust_machines(random, individual, machine_queue, anchor, adjustment_start, args)
        source_machine = res[source_machine_idx]
        target_machine = res[target_machine_idx]

        return individual

    def _between_machines_swap(self, individual, random, args):
        # pick a machine randomly (excluding empty machines & machines with only a single fixed cycles)
        source_machine_candidates = [m for m in range(args["num_machines"]) if not (len(individual[m]) == 0 or ( (len(individual[m]) == 1) and individual[m][0]["fixed"])) ]
        if len(source_machine_candidates) == 0 : return individual # skip mutation if no machines are available
        source_machine_idx = random.choice(source_machine_candidates)
        source_machine = individual[source_machine_idx]
        # discard fixed cycles from available choices
        source_candidate_pos = list(range(len(source_machine)) if not source_machine[0]["fixed"] else range(1, len(source_machine)))
        # pick a random cycle to move from source machine
        source_pos = random.choice(source_candidate_pos)
        source_cycle = source_machine[source_pos]

        # pick compatible machine according to the selected cycle on source machine
        target_machine_candidates = [m for m in args["prod_to_machine_comp"][source_cycle["p"]] if (m != source_machine_idx) and (m not in args["broken_machines"]) and not (len(individual[m]) == 0 or ((len(individual[m]) == 1) and individual[m][0]["fixed"]))]
        if len(target_machine_candidates) == 0 : return individual # skip mutation if no target has no machines available
        # pick a random target machine
        target_machine_idx = random.choice(target_machine_candidates)
        target_machine = individual[target_machine_idx]
        # look for a position to insert the cycle in the target machine
        # this time we also need to consider target cycle compatibility with source cycle
        target_candidates_pos = [pos for pos, cycle in enumerate(target_machine) if source_machine_idx in args["prod_to_machine_comp"][cycle["p"]] and not target_machine[pos]["fixed"]]
        if len(target_candidates_pos) == 0 : return individual # skip mutation if target machine has no cycles compatible with source machine
        target_pos = random.choice(target_candidates_pos)
        target_cycle = target_machine[target_pos]
        
        # set appropriate anchor points for source and target machines
        source_anchor = source_cycle["setup_beg"] if source_pos > 0 else args["start_schedule"] 
        target_anchor = target_cycle["setup_beg"] if target_pos > 0 else args["start_schedule"]

        # Swap cycles
        source_machine[source_pos], target_machine[target_pos] = target_machine[target_pos], source_machine[source_pos]
        
        # Setup parameters for machine adjustment
        machine_queue = {source_machine_idx: source_machine, target_machine_idx: target_machine}
        anchor = {source_machine_idx: source_anchor, target_machine_idx: target_anchor}
        adjustment_start = {source_machine_idx: source_pos, target_machine_idx: target_pos}
        # Adjust machine
        res = adjust_machines(random, individual, machine_queue, anchor, adjustment_start, args)
        source_machine = res[source_machine_idx]
        target_machine = res[target_machine_idx]

        return individual