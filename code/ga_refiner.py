from inspyred import ec
from random import Random
from intervaltree import IntervalTree, Interval
import sys
import copy

import matplotlib.pyplot as plt
import numpy

seed = 140
prng = Random()
prng.seed(seed)

def print_solution (candidate) :
    for cycle in candidate :
        print(f"Cycle {cycle['cycle']}, Pos [{cycle['pos']}] :")
        print(f"    Setup : ({cycle['setup_beg']}, {cycle['setup_end']})")
        for l in range(len(cycle["load_beg"])) :
            print(f"        [{l}] : ({cycle['load_beg'][l]}, {cycle['load_end'][l]}) => ({cycle['unload_beg'][l]}, {cycle['unload_end'][l]})")


# individual = iterable of cycles c such as : [ pos, load_beg[num_levate[c]], load_end[num_levate[c]], unload_beg[num_levate[c]], unload_end[num_levate[c]]]

def ga_refinement (products, input) :
    # copy class variables
    penalty_weight = 7
    penalty_weight *= input["time_units_in_a_day"]

    seed = [[] for _ in range(input["num_machines"])]
    for p, prod in products:
        for c in prod.setup_beg.keys():
            cycle = {
                "pos": -1,
                "product": p,
                "cycle": c,
                "velocity": prod.velocity[c],
                
                "setup_operator": prod.setup_operator[c],
                "setup_beg": prod.setup_beg[c],
                "setup_end": prod.setup_end[c],

                "load_operator" : [prod.load_operator[c,l] for l in range(prod.num_levate[c])],
                "load_beg": [prod.load_beg[c,l] for l in range(prod.num_levate[c])],
                "load_end": [prod.load_end[c,l] for l in range(prod.num_levate[c])],

                "unload_operator" : [prod.unload_operator[c,l] for l in range(prod.num_levate[c])],
                "unload_beg": [prod.unload_beg[c,l] for l in range(prod.num_levate[c])],
                "unload_end": [prod.unload_end[c,l] for l in range(prod.num_levate[c])]
            }
            # append individual to seed of machine "m"
            seed[prod.machine[c]].append(cycle)

    for m in range(input["num_machines"]):
        seed[m] = sorted(seed[m], key=lambda x: x["setup_beg"])
        for cycle in seed[m]:
            cycle['pos'] = seed[m].index(cycle)

    # Populate overlap Tree for domain gaps
    domain_overlap_tree = IntervalTree.from_tuples(input["prohibited_intervals"])

    def generator (random, args) :
        return [seed[m] for m in range(input["num_machines"])]

    def evaluator (candidates, args) :

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
                    lb = input["time_units_from_midnight"]
                    if cycle["setup_beg"] < lb :
                        penalty += lb - cycle["setup_beg"]
                    ub = input["horizon"]
                    if cycle["unload_end"][-1] > ub:
                        penalty += cycle["unload_end"][-1] - ub

                # total fitness accounts for machine makespan and weighted penalty
                machine_makespan = 0 if len(machine_queue) == 0 else sum([cycle["unload_end"][-1] for cycle in machine_queue])
                machine_fitness.append(machine_makespan + penalty_weight*penalty)
            
            # Solution fitness is sum of all machine fitness
            fitness.append(sum(machine_fitness))
    
        return fitness


    def set_beg_in_domain (beg) :
        # atomic check for start of feasibility (beg+1 is excluded)
        beg_overlap_set = domain_overlap_tree.overlap(beg, beg+1)
        if len(beg_overlap_set) > 0 :
            # sfhift beginning towards first free slot
            beg = next(iter(beg_overlap_set)).end
        return beg

    def set_end_in_domain (beg, base_cost) :
        day = beg // input["time_units_in_a_day"]
        ub_day = input["end_shift"] + day*input["time_units_in_a_day"]
        # end is contained withing beg day
        end = beg + base_cost
        if end > ub_day and 0 <= day < len(input["gap_at_day"]):
            end += input["gap_at_day"][day]
            
        return end

    def adjust_operation (point, min_distance, operation_base_cost, overlap_tree) :
        """
        returns beginning & end of operation (Load / Unload) such that
        it starts at least 'min_distance' units from point and doesn't
        produce any overlap within associated operator group
        """
        beg = set_beg_in_domain(point+min_distance)  
        # Check if beg is feasible
        beg_overlap_set = overlap_tree.overlap(beg, beg+1) # atomic check for start of feasibility (beg+1 is excluded)
        if len(beg_overlap_set) > 0 :
            # sfhift beginning towards first free slot
            beg = next(iter(beg_overlap_set)).end
            
        # Find appropriate setting
        end = beg + operation_base_cost
        while True :
            # Assign appropriate end value to the operation
            end = set_end_in_domain(beg, operation_base_cost)            
            # Check for whole interval feasibility
            end_overlap_set = overlap_tree.overlap(beg, end)
            # If conflicts are found :
            if len(end_overlap_set) > 0 :
                conflicts = sorted(end_overlap_set)
                beg = conflicts[-1].end
                continue
            
            # If no conflicts are found break the loop
            break
        
        return beg, end

    def adjust_cycle (cycle, cycle_shift, machine_id, overlap_tree) :
        cycle["setup_beg"], cycle["setup_end"] = adjust_operation(cycle["setup_beg"], cycle_shift, input["base_setup_cost"][cycle["product"], machine_id], overlap_tree[cycle["setup_operator"]])
        # scale levata cost according to velocity
        effective_levata_cost = input["base_levata_cost"][cycle["product"]] - cycle["velocity"]*input["velocity_step_size"][cycle["product"]]
        # first anchor point is end of setup
        anchor = cycle["setup_end"]
        for l in range(len(cycle["load_beg"])) :
            cycle["load_beg"][l], cycle["load_end"][l] = adjust_operation(anchor, 0, input["base_op_cost"][cycle["product"], machine_id], overlap_tree[cycle["load_operator"][l]])
            cycle["unload_beg"][l], cycle["unload_end"][l] = adjust_operation(cycle["load_end"][l], effective_levata_cost, input["base_op_cost"][cycle["product"], machine_id], overlap_tree[cycle["unload_operator"][l]])
            # set next anchor point
            anchor = cycle["unload_end"][l]
        
        return cycle

    def adjust_machine(machine_queue, machine_idx, anchor, overlap_tree, from_pos=0) :
        for cycle in machine_queue :
            if cycle["pos"] >= from_pos :
                cycle["setup_beg"] = anchor
                cycle = adjust_cycle(cycle, 0, machine_idx, overlap_tree)
                anchor = cycle["unload_end"][-1]
        
        return machine_queue


    def individual_overlap_tree (individual, exclude_machines=[]) :
        # Generate overlap tree for each operator group
        overlap_tree = {}
        for o in range(input["num_operator_groups"]) :
            operator_intervals = []
            for m in range(input["num_machines"]) :
                if m not in exclude_machines :
                    for cycle in individual[m] :
                        if cycle["setup_operator"] == o and (cycle["setup_end"] - cycle["setup_beg"]) > 0 :
                            operator_intervals.append((cycle["setup_beg"], cycle["setup_end"]))
                        for l in range(len(cycle["load_operator"])) :
                            if cycle["load_operator"][l] == o and (cycle["load_end"][l] - cycle["load_beg"][l]) > 0 :
                                operator_intervals.append((cycle["load_beg"][l], cycle["load_end"][l]))
                            if cycle["unload_operator"][l] == o and (cycle["unload_end"][l] - cycle["unload_beg"][l]) > 0 :
                                operator_intervals.append((cycle["unload_beg"][l], cycle["unload_end"][l]))
            
            overlap_tree[o] = IntervalTree.from_tuples(operator_intervals)
        
        return overlap_tree

    def add_intervals_to_tree (tree, machine_queue) :
        for cycle in machine_queue :
            # Add setup interval
            if (cycle["setup_end"] - cycle["setup_beg"]) > 0 :
                tree[cycle["setup_operator"]].add(Interval(cycle["setup_beg"], cycle["setup_end"]))
            # Add Load & Unload intervals
            for l in range(len(cycle["load_beg"])) :
                if (cycle["load_end"][l] - cycle["load_beg"][l]) > 0 :
                    tree[cycle["load_operator"][l]].add(Interval(cycle["load_beg"][l], cycle["load_end"][l]))
                if (cycle["unload_end"][l] - cycle["unload_beg"][l]) > 0 :
                    tree[cycle["unload_operator"][l]].add(Interval(cycle["unload_beg"][l], cycle["unload_end"][l]))
                
        return tree

    def mutation (random, candidates, args) :
        
        for individual in candidates :

            # Pick a mutation randomly
            mutation_type = random.choice([0, 1, 2])
            
            # Compact & Shift 
            if mutation_type == 0 :
                # choose a random machine queue in the solution
                machine_idx = random.choice([m for m in range(input["num_machines"]) if len(individual[m]) > 0])
                machine_queue = individual[machine_idx]
                # Generate overlap tree for each operator group (excluding selected machines)
                overlap_tree = individual_overlap_tree(individual, exclude_machines=[machine_idx])  

                # Anchor "Now" as new starting point for machine
                # the algorithm will assign in a greedy way the first
                # available slot for each cycle operation
                anchor = input["time_units_from_midnight"]
                adjust_machine(machine_queue, machine_idx, anchor, overlap_tree, from_pos=0) 

            # Cycles swapping
            if mutation_type == 1 :
                # choose a random machine queue in the solution
                machine_idx = random.choice([m for m in range(input["num_machines"]) if len(individual[m]) > 0])
                machine_queue = individual[machine_idx]
                # Generate overlap tree for each operator group (excluding selected machines)
                overlap_tree = individual_overlap_tree(individual, exclude_machines=[machine_idx])

                # change cycle position
                cycle1 = random.choice(machine_queue)
                cycle2 = random.choice(machine_queue)
                if cycle1["pos"] != cycle2["pos"] :
                    # swap positions and beginnings and actual order in the list
                    cycle1["pos"], cycle2["pos"] = cycle2["pos"], cycle1["pos"]
                    cycle1["setup_beg"], cycle2["setup_beg"] = cycle2["setup_beg"], cycle1["setup_beg"]
                    # phisically swap their positions in the list
                    machine_queue[cycle1["pos"]], machine_queue[cycle2["pos"]] = machine_queue[cycle2["pos"]], machine_queue[cycle1["pos"]]
                    # find which cycle starts first between the two
                    first_cycle = cycle1 if cycle1["pos"] < cycle2["pos"] else cycle2
                    anchor = first_cycle["setup_beg"]
                    # Adjust machine
                    machine_queue = adjust_machine(machine_queue, machine_idx, anchor, overlap_tree, from_pos=min(cycle1["pos"], cycle2["pos"])) 
            
            
            if mutation_type == 2 :
                # pick a machine randomly
                source_machine_idx = random.choice([m for m in range(input["num_machines"]) if len(individual[m]) > 0])
                source_machine = individual[source_machine_idx]
                # pick compatible machine IN SAME MACHINE GROUP !!!!!!
                target_machine_idx = random.choice([m for m in range(input["num_machines"]) if m != source_machine_idx])
                target_machine = individual[target_machine_idx]
                # Generate overlap tree for each operator group (excluding selected machines)
                overlap_tree = individual_overlap_tree(individual, exclude_machines=[source_machine_idx, target_machine_idx])

                # pop element from source machine randomly
                source_cycle = random.choice(source_machine) 
                source_machine.remove(source_cycle)
                # gather source info
                source_anchor = source_cycle["setup_beg"]
                source_from_pos = source_cycle["pos"]
                # fix machine positions on source machine
                for i, cycle in enumerate(source_machine):
                    cycle["pos"] = i

                # gather target info
                target_anchor = input["time_units_from_midnight"] if len(target_machine) == 0 else target_machine[-1]["unload_end"][-1]
                target_from_pos = len(target_machine)
                # append cycle to target machine
                target_machine.append(source_cycle)
                for i, cycle in enumerate(target_machine):
                    cycle["pos"] = i
                
                # Choose which machine to adjust first
                # (it matters due to operator overlaps)
                if random.choice([0, 1]) == 0 :
                    # Adjust first Source machine
                    source_machine = adjust_machine(source_machine, source_machine_idx, source_anchor, overlap_tree, from_pos=source_from_pos)
                    # Add source intervals to overlap tree
                    overlap_tree = add_intervals_to_tree (overlap_tree, source_machine)
                    # fix target machine
                    target_machine = adjust_machine(target_machine, target_machine_idx, target_anchor, overlap_tree, from_pos=target_from_pos)
                else :
                    # Adjust first Target machine
                    target_machine = adjust_machine(target_machine, target_machine_idx, target_anchor, overlap_tree, from_pos=target_from_pos)
                    # Add source intervals to overlap tree
                    overlap_tree = add_intervals_to_tree (overlap_tree, target_machine)
                    # fix source machine
                    source_machine = adjust_machine(source_machine, source_machine_idx, source_anchor, overlap_tree, from_pos=source_from_pos)
                    
        return candidates

    def crossover (random, candidate, args) :
        pass
    
    best_candidate = []
    def observer(population, num_generations, num_evaluations, args):
        # set default best candidate
        if num_generations == 0 :
            best_candidate.append(population[0])
        
        for candidate in population :
            if candidate.fitness < best_candidate[-1].fitness :
                best_candidate[-1] = candidate
        print(f"[{num_generations}] => {best_candidate[-1].fitness}")

        if False :
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
    
    algorithm = ec.GA(prng)
    algorithm.terminator = ec.terminators.generation_termination
    algorithm.replacer = ec.replacers.steady_state_replacement    
    algorithm.variator = [ec.variators.uniform_crossover, mutation]
    algorithm.selector = ec.selectors.rank_selection
    algorithm.observer = observer

    population = 10

    final_pop = algorithm.evolve(
        generator=generator,
        evaluator=evaluator,
        pop_size=population,
        maximize=False,
        num_selected=population,
        mutation_rate=1,
        crossover_rate=0,
        max_generations=100
    )

    #plt.show()
    #plt.waitforbuttonpress()

    # Make sure no overlaps are present
    solution = best_candidate[-1].candidate
    #overlap_tree = individual_overlap_tree(solution)

    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["product"]
            c = elem["cycle"]
            for _, prod in products:
                if prod.id == p:
                    prod.machine[c] = m
                    prod.setup_beg[c] = elem["setup_beg"]
                    prod.setup_end[c] = elem["setup_end"]
                    #print(overlap_tree[elem["setup_operator"]].overlap(prod.setup_beg[c], prod.setup_end[c]))
                    for l in range(len(elem["load_beg"])) :
                        prod.load_beg[c,l] = elem["load_beg"][l]
                        prod.load_end[c,l] = elem["load_end"][l]
                        #print(overlap_tree[elem["load_operator"][l]].overlap(prod.load_beg[c,l], prod.load_end[c,l]))
                        prod.unload_beg[c,l] = elem["unload_beg"][l]
                        prod.unload_end[c,l] = elem["unload_end"][l]
                        #print(overlap_tree[elem["unload_operator"][l]].overlap(prod.unload_beg[c,l], prod.unload_end[c,l]))
                    
                    prod.cycle_end[c] = elem["unload_end"][-1]
                    break

    return products


