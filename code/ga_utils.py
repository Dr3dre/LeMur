from inspyred import ec
from intervaltree import IntervalTree, Interval
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

"""
Data Visualization Functions
"""

def generational_stats_plot(population, num_generations, num_evaluations, args):
    """
    Generate a plot of the best, worst, median, and average fitnesses in
    the population given a certain generation of the evolutionary algorithm
    """
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
            line, = plt.plot(data[0], data[i + 1], color=colors[i], label=labels[i])
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
            line.set_xdata(np.array(data[0]))
            line.set_ydata(np.array(data[i + 1]))
        args['plot_data'] = data
        args['plot_lines'] = lines

    # Calculate ymin, ymax, and adjust yrange to prevent warnings
    ymin = min([min(d) for d in data[1:]])
    ymax = max([max(d) for d in data[1:]])

    if ymin == ymax:  # Handle the case where all y-values are identical
        ymin -= 1
        ymax += 1

    yrange = ymax - ymin
    plt.xlim((0, num_evaluations))
    plt.ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))  # Adjusted ylim to avoid singular transformation
    plt.draw()
    plt.legend()


def print_solution (candidate) :
    """
    format output for a candidate solution
    """
    for cycle in candidate :
        print(f"Job {cycle['product']}, Cycle {cycle['cycle']}, Pos [{cycle['pos']}] :")
        print(f"    Setup : ({cycle['setup_beg']}, {cycle['setup_end']})")
        for l in range(len(cycle["load_beg"])) :
            print(f"        [{l}] : ({cycle['load_beg'][l]}, {cycle['load_end'][l]}) => ({cycle['unload_beg'][l]}, {cycle['unload_end'][l]})")


def temperature_profile(t, midpoint, steepness, initial_temp, goal_temp):
    """
    Generates a sigmoid curve bounded between initial_temp and goal_temp.
    
    Parameters:
    - t : current time step
    - midpoint : the t-value where the temperature reaches the midpoint between initial_temp and goal_temp
    - steepness : controls the steepness of the sigmoid transition
    - initial_temp : lower bound of the temperature
    - goal_temp : upper bound of the temperature
    """
    # Calculate the range of temperatures
    temp_range = goal_temp - initial_temp
    # Compute the sigmoid value
    sigmoid = 1 / (1 + math.exp(-steepness * (t - midpoint)))
    # Scale and shift the sigmoid to the desired range
    return initial_temp + temp_range * sigmoid

def visualize_temperature_profile(temperature_function, time_range, out_filepath, **kwargs):
    """
    Visualizes the temperature profile over a given time range.
    """
    # Unpack the time range
    start_time, end_time = time_range
    # Generate time steps
    times = range(start_time, end_time + 1)
    # Compute temperature values
    temperatures = [temperature_function(t, **kwargs) for t in times]
    # Plot the profile
    plt.figure(figsize=(8, 6))
    plt.plot(times, temperatures, label="Temperature Profile", color="blue")
    plt.scatter(times, temperatures, color="red", s=10, label="Data Points")
    plt.title("Temperature Profile Visualization")
    plt.xlabel("Time Steps")
    plt.ylabel("Temperature")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.savefig(out_filepath)
    plt.close()

def convert_args(args) :
    """
    Apply corrections to the arguments dictionary, such as
    - converting 1-indexed to 0-indexed (keys referring to machines)
        - This is necessary because the code input data is 1-indexed while the genetic algorithm works with 0-indexed data
    - modifying scheduled maintenances content
    - modifying prod-to-machine compatibility content
    """
    # FIXING INDEXES (1-indexed to 0-indexed) for all machine-related data
    # correct broken machines list
    broken_machines = copy.copy(args["broken_machines"])
    args["broken_machines"] = [m-1 for m in broken_machines]
    # correct scheduled maintenances
    scheduled_maintenances = copy.copy(args["scheduled_maintenances"])
    args["scheduled_maintenances"] = {(m-1) : [(beg, beg+length) for beg, length in scheduled_maintenances[m]] for m in scheduled_maintenances.keys()}
    # correct prod-to-machine compatibility
    prod_to_machine_comp = copy.copy(args["prod_to_machine_comp"])
    args["prod_to_machine_comp"] = {p: [m-1 for m in prod_to_machine_comp[p]] for p in prod_to_machine_comp.keys()}
    # correct setup base costs
    base_setup_cost = copy.copy(args["base_setup_cost"])
    args["base_setup_cost"] = {(p,m-1) : base_setup_cost[(p,m)] for (p,m) in base_setup_cost.keys()}
    # correct load base costs
    base_load_cost = copy.copy(args["base_load_cost"])
    args["base_load_cost"] = {(p,m-1) : base_load_cost[(p,m)] for (p,m) in base_load_cost.keys()}
    # correct unload base costs
    base_unload_cost = copy.copy(args["base_unload_cost"])
    args["base_unload_cost"] = {(p,m-1) : base_unload_cost[(p,m)] for (p,m) in base_unload_cost.keys()}

    # FIXING PROHIBITED INTERVALS
    # shift by 1 unit beginning to allow allocation on such position
    prohibited_intervals = copy.copy(args["prohibited_intervals"])
    args["prohibited_intervals"] = [(beg+1, end) for beg, end in prohibited_intervals]

"""
DOMAIN ADJUSTMENT FUNCTIONS
"""

def build_operator_inter_tree (individual, args, exclude_cycles=[]) :
    """
    Generate overlap tree for each operator group
    excluding specified machines and cycles
    """
    # Generate overlap tree for each operator group
    overlap_tree = {}
    operator_intervals = {o: [] for o in range(args["num_operator_groups"])}
    for m in range(args["num_machines"]) :
        for elem in individual[m] :
            # tuples (p,c) in "exclude_cycles" list are also not inserted in the tree
            if (elem["product"], elem["cycle"]) not in exclude_cycles :
                # assign intervals to relative groups (always excluding width zero tuples)
                if (elem["setup_end"] - elem["setup_beg"]) > 0 : operator_intervals[elem["setup_operator"]].append((elem["setup_beg"], elem["setup_end"]))
                for l in range(len(elem["load_operator"])) :
                    if (elem["load_end"][l] - elem["load_beg"][l]) > 0 : operator_intervals[elem["load_operator"][l]].append((elem["load_beg"][l], elem["load_end"][l]))
                    if (elem["unload_end"][l] - elem["unload_beg"][l]) > 0 : operator_intervals[elem["unload_operator"][l]].append((elem["unload_beg"][l], elem["unload_end"][l]))
    
    # Generate overlap tree for each operator group from extracted tuples
    for o in range(args["num_operator_groups"]) :
        overlap_tree[o] = IntervalTree.from_tuples(operator_intervals[o])
    
    return overlap_tree


def greedy_set_value_in_domain (value, overlap_tree) :
    """
    given a value, it shifts it to the first feasible value
    non-overlapping with the given interval tree
    """
    # atomic check for start of feasibility (value+1 is excluded)
    overlap_set = overlap_tree.overlap(value, value+1)
    if len(overlap_set) > 0 :
        # sfhift value towards first free slot
        value = next(iter(overlap_set)).end
    return value

def set_end_in_domain (beg, base_cost, args) :
    """
    given the beginning of an operation and its base cost,
    it returns the end of the operation, taking into account
    """
    day = beg // args["time_units_in_a_day"]
    ub_day = args["end_shift"] + day*args["time_units_in_a_day"]
    # end is contained withing beg day
    end = beg + base_cost
    if end > ub_day and 0 <= day < len(args["gap_at_day"]):
        end += args["gap_at_day"][day]
        
    return end

def adjust_operation (point, min_distance, operation_base_cost, operator_overlap_tree, args) :
    """
    returns beginning & end of operation (Load / Unload) such that
    it starts at least 'min_distance' units from point and doesn't
    produce any overlap within associated operator group
    """
    # set point + min_distance to first feasible value according to worktime domain  
    beg = greedy_set_value_in_domain(point+min_distance, args["domain_interval_tree"])  
    # set beg to first feasible value according to operator intervals domain
    beg = greedy_set_value_in_domain(beg, operator_overlap_tree)
        
    # Find appropriate setting
    end = beg + operation_base_cost
    while True :
        # Assign appropriate end value to the operation
        end = set_end_in_domain(beg, operation_base_cost, args)            
        # Check for whole interval feasibility
        end_overlap_set = operator_overlap_tree.overlap(beg, end)
        # If conflicts are found :
        if len(end_overlap_set) > 0 :
            conflicts = sorted(end_overlap_set)
            beg = greedy_set_value_in_domain(conflicts[-1].end, operator_overlap_tree)
            #beg = conflicts[-1].end
            continue
        
        # If no conflicts are found break the loop
        break
    
    return beg, end

def adjust_cycle (cycle, machine_id, operator_overlap_tree, args) :
    """
    Adjust a cycle by setting its Setup / Load / Unload operations
    in a greedy way, such that no overlaps are produced
    """
    cycle["setup_beg"], cycle["setup_end"] = adjust_operation(cycle["setup_beg"], 0, args["base_setup_cost"][cycle["product"], machine_id], operator_overlap_tree[cycle["setup_operator"]], args)
    # scale levata cost according to velocity
    effective_levata_cost = args["base_levata_cost"][cycle["product"]] - cycle["velocity"]*args["velocity_step_size"][cycle["product"]]
    # first anchor point is end of setup
    anchor = cycle["setup_end"]
    for l in range(len(cycle["load_beg"])) :
        cycle["load_beg"][l], cycle["load_end"][l] = adjust_operation(anchor, 0, args["base_load_cost"][cycle["product"], machine_id], operator_overlap_tree[cycle["load_operator"][l]], args)
        cycle["unload_beg"][l], cycle["unload_end"][l] = adjust_operation(cycle["load_end"][l], effective_levata_cost, args["base_unload_cost"][cycle["product"], machine_id], operator_overlap_tree[cycle["unload_operator"][l]], args)
        # set next anchor point
        anchor = cycle["unload_end"][l]
    
    return cycle


def adjust_machines (random, individual, machine_queue, anchor, adjustment_start, args) :
    # exclude cycles starting from specified positions untill the end of machines
    cycles_to_exclude = []
    for m in machine_queue.keys() :
        cycles_to_exclude += [(elem["product"], elem["cycle"]) for elem in machine_queue[m] if elem["pos"] >= adjustment_start[m]]
    # Generate overlap tree for each operator group excluding the specified cycles
    operator_overlap_tree = build_operator_inter_tree(individual, args, exclude_cycles=cycles_to_exclude)

    # Extract scheduled maintenances for the machine as overlap tree (if any)
    scheduled_maintenances_tree = {}
    for m in machine_queue.keys() :
        scheduled_maintenances_tree[m] = None if m not in args["scheduled_maintenances"].keys() else IntervalTree.from_tuples(args["scheduled_maintenances"][m])

    # keep track of advancement status for each machine
    cycle_to_adjust = {m: adjustment_start[m] for m in machine_queue.keys()}
    # Keep advancing until all cycles are adjusted
    while len(cycle_to_adjust.keys()) > 0 :
        # Select a random machine to advance
        m = random.choice(list(cycle_to_adjust.keys()))
        # If the machine has no more cycles to adjust, remove it from the dictionary
        if cycle_to_adjust[m] >= len(machine_queue[m]):
            cycle_to_adjust.pop(m, None)
            continue
        # pick the cycle
        cycle = machine_queue[m][cycle_to_adjust[m]]
        # Adjust the machine queue
        cycle["setup_beg"] = anchor[m] if anchor[m] >= args["start_date"][cycle["product"]] else args["start_date"][cycle["product"]]
        cycle = adjust_cycle(cycle, m, operator_overlap_tree, args)
        
        while scheduled_maintenances_tree[m] is not None :
            # Check for if cycle assignment is feasible according to scheduled maintenances
            maintenance_overlap_set = scheduled_maintenances_tree[m].overlap(cycle["setup_beg"], cycle["unload_end"][-1])
            # if set is not empty, find the first maintenance end point
            if len(maintenance_overlap_set) > 0 :
                conflicts = sorted(maintenance_overlap_set)
                cycle["setup_beg"] = conflicts[0].end
                cycle = adjust_cycle(cycle, m, operator_overlap_tree, args)
                continue
            # Go ahead as there are no conflicts
            break
        
        # Update operator overlap tree with new cycle intervals
        if (cycle["setup_end"] - cycle["setup_beg"]) > 0 :
            # Add Setup interval
            operator_overlap_tree[cycle["setup_operator"]].add(Interval(cycle["setup_beg"], cycle["setup_end"]))
        for l in range(len(cycle["load_beg"])) :
            # Add Load & Unload intervals
            if (cycle["load_end"][l] - cycle["load_beg"][l]) > 0 :
                operator_overlap_tree[cycle["load_operator"][l]].add(Interval(cycle["load_beg"][l], cycle["load_end"][l]))
            if (cycle["unload_end"][l] - cycle["unload_beg"][l]) > 0 :
                operator_overlap_tree[cycle["unload_operator"][l]].add(Interval(cycle["unload_beg"][l], cycle["unload_end"][l]))

        # next cycle will have as anchor point the end of this cycle
        anchor[m] = cycle["unload_end"][-1]
        # Update the advancement dictionary
        cycle_to_adjust[m] += 1
    
    return machine_queue

    
"""
Test Cases Functions
"""

def check_solution_conformity(solution, args):
    """
    Check if a solution conforms to the problem constraints.
    Prints conflicts if found any and returns False. Otherwise, returns True.
    """
    is_valid_solution = True

    # LOOK FOR DOMAIN VIOLATIONS
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["product"]
            c = elem["cycle"]
            # Check for setup start/end points
            if len(args["domain_interval_tree"].overlap(elem["setup_beg"], elem["setup_beg"]+1)) > 0:
                print(
                    f"Domain Violation : Job {p}, Cycle {c}, on machine {m+1}\n"
                    f"      Setup beginning is outside worktime domain"
                )
                is_valid_solution = False
            if len(args["domain_interval_tree"].overlap(elem["setup_end"], elem["setup_end"]+1)) > 0:
                print(
                    f"Domain Violation : Job {p}, Cycle {c}, on machine {m+1}\n"
                    f"      Setup end is outside worktime domain"
                )
                is_valid_solution = False
            # Check for load/unload start/end points
            for l in range(len(elem["load_beg"])):
                if len(args["domain_interval_tree"].overlap(elem["load_beg"][l], elem["load_beg"][l]+1)) > 0:
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n"
                        f"      Load beginning is outside worktime domain"
                    )
                    is_valid_solution = False
                if len(args["domain_interval_tree"].overlap(elem["load_end"][l], elem["load_end"][l]+1)) > 0:
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n"
                        f"      Load end is outside worktime domain"
                    )
                    is_valid_solution = False
                if len(args["domain_interval_tree"].overlap(elem["unload_beg"][l], elem["unload_beg"][l]+1)) > 0:
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n"
                        f"      Unload beginning is outside worktime domain"
                    )
                    is_valid_solution = False
                if len(args["domain_interval_tree"].overlap(elem["unload_end"][l], elem["unload_end"][l]+1)) > 0:
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n"
                        f"      Unload end is outside worktime domain"
                    )
                    is_valid_solution = False
    

    # LOOK FOR WITHIN MACHINE OVERLAPS
    for m, machine_queue in enumerate(solution):
        # Generate intervals for each machine
        intervals = []
        for cycle in machine_queue:
            if (cycle["setup_end"] - cycle["setup_beg"]) > 0:
                intervals.append((cycle["setup_beg"], cycle["setup_end"]))
            for l in range(len(cycle["load_beg"])):
                if (cycle["load_end"][l] - cycle["load_beg"][l]) > 0:
                    intervals.append((cycle["load_beg"][l], cycle["load_end"][l]))
                if (cycle["unload_end"][l] - cycle["unload_beg"][l]) > 0:
                    intervals.append((cycle["unload_beg"][l], cycle["unload_end"][l]))
        # Check for overlaps within the machine
        machine_interval_tree = IntervalTree.from_tuples(intervals)
        for beg, end in intervals:
            conflict = machine_interval_tree.overlap(beg, end)
            if len(conflict) > 1:
                print(
                    f"Within-Machine Overlap :\n"
                    f"  Machine [{m+1}] : some cycles are scheduled at the same time"
                )
                is_valid_solution = False

    # LOOK FOR OPERATOR OVERLAPS
    operator_interval_tree = build_operator_inter_tree(solution, args)
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["product"]
            c = elem["cycle"]

            setup_overlaps = operator_interval_tree[elem["setup_operator"]].overlap(elem["setup_beg"], elem["setup_end"])
            if len(setup_overlaps) > 1:
                print(
                    f"Operator overlap (SETUP) : An operator group is assigned to more than one operation at the same time\n"
                    f"      Product [{p+65}], Cycle [{c}] has conflicts"
                )
                is_valid_solution = False

            for l in range(len(elem["load_beg"])):
                load_overlaps = operator_interval_tree[elem["load_operator"][l]].overlap(elem["load_beg"][l], elem["load_end"][l])
                if len(load_overlaps) > 1:
                    print(
                        f"Operator overlap (LOAD) : An operator group is assigned to more than one operation at the same time\n"
                        f"      Product [{p+65}], Cycle [{c}], Levata [{l}], has conflicts"
                    )
                    is_valid_solution = False

                unload_overlaps = operator_interval_tree[elem["unload_operator"][l]].overlap(elem["unload_beg"][l], elem["unload_end"][l])
                if len(unload_overlaps) > 1:
                    print(
                        f"Operator overlap (UNLOAD) : An operator group is assigned to more than one operation at the same time\n"
                        f"      Product [{p+65}], Cycle [{c}], Levata [{l}], has conflicts"
                    )
                    is_valid_solution = False

    # LOOK FOR TIME WINDOW VIOLATIONS
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["product"]
            c = elem["cycle"]

            if elem["setup_beg"] < args["start_date"][p]:
                print(
                    f"Start Date Violation : Job {p}, Cycle {c}, on machine {m+1}\n"
                    f"      Start date => {args['start_date'][p]}\n"
                    f"      But cycle starts at => {elem['setup_beg']}"
                )
                is_valid_solution = False
            
            if elem["unload_end"][-1] > args["due_date"][p]:
                print(
                    f"Due Date Violation : Job {p}, Cycle {c}, on machine {m+1}\n"
                    f"      Due date => {args['due_date'][p]}\n"
                    f"      But cycle ends at => {(elem['unload_beg'][-1])}"
                )
                is_valid_solution = False

    # LOOK FOR MACHINE ASSIGNMENT CORRECTNESS
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["product"]
            c = elem["cycle"]

            if m not in args["prod_to_machine_comp"][p] :
                print(f"Machine Assignment Violation :\n"
                      f"    Job {p}, Cycle {c}, on invalid machine {m+1}"
                )
                is_valid_solution = False

    # LOOK FOR BROKEN MACHINE VIOLATIONS
    for m, machine_queue in enumerate(solution):
        if m in args["broken_machines"] and len(machine_queue) > 0:
            print(f"Machine {m+1} is Broken, still some cycles have been allocated")
            is_valid_solution = False
    
    # LOOK FOR SCHEDULED MANTENANCE VIOLATION
    for m, machine_queue in enumerate(solution):
        machine_scheduled_maintenances_tree = None if m not in args["scheduled_maintenances"].keys() else IntervalTree.from_tuples(args["scheduled_maintenances"][m])
        if machine_scheduled_maintenances_tree is not None :
            for elem in machine_queue:
                p = elem["product"]
                c = elem["cycle"]
                maintenance_overlap_set = machine_scheduled_maintenances_tree.overlap(elem["setup_beg"], elem["unload_end"][-1])
                if len(maintenance_overlap_set) > 0 :
                    print(
                        f"Scheduled Maintenance Violation :\n"
                        f"    Job {p}, Cycle {c}, on machine {m+1} violates scheduled maintenance"
                    )
                    is_valid_solution = False

    return is_valid_solution


