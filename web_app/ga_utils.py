from inspyred import ec
from intervaltree import IntervalTree, Interval
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
from data_init import RunningProduct

"""
Data Visualization Functions
"""


def generational_stats_plot(population, num_generations, num_evaluations, args):
    """
    Generate a plot of the best, worst, median, and average fitnesses in
    the population given a certain generation of the evolutionary algorithm
    """
    stats = ec.analysis.fitness_statistics(population)
    best_fitness = stats["best"]
    worst_fitness = stats["worst"]
    median_fitness = stats["median"]
    average_fitness = stats["mean"]
    colors = ["black", "blue", "green", "red"]
    labels = ["average", "median", "best", "worst"]
    data = []

    if num_generations == 0:
        plt.figure("A")
        plt.ion()
        data = [
            [num_evaluations],
            [average_fitness],
            [median_fitness],
            [best_fitness],
            [worst_fitness],
        ]
        lines = []
        for i in range(4):
            (line,) = plt.plot(data[0], data[i + 1], color=colors[i], label=labels[i])
            lines.append(line)
        args["plot_data"] = data
        args["plot_lines"] = lines
        plt.xlabel("Evaluations")
        plt.ylabel("Fitness")
    else:
        data = args["plot_data"]
        data[0].append(num_evaluations)
        data[1].append(average_fitness)
        data[2].append(median_fitness)
        data[3].append(best_fitness)
        data[4].append(worst_fitness)
        lines = args["plot_lines"]
        for i, line in enumerate(lines):
            line.set_xdata(np.array(data[0]))
            line.set_ydata(np.array(data[i + 1]))
        args["plot_data"] = data
        args["plot_lines"] = lines

    # Calculate ymin, ymax, and adjust yrange to prevent warnings
    ymin = min([min(d) for d in data[1:]])
    ymax = max([max(d) for d in data[1:]])

    if ymin == ymax:  # Handle the case where all y-values are identical
        ymin -= 1
        ymax += 1

    yrange = ymax - ymin
    plt.xlim((0, num_evaluations))
    plt.ylim(
        (ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    )  # Adjusted ylim to avoid singular transformation
    plt.draw()
    plt.legend()


def print_machine_queue(machine_queue, machine_id, args):
    """
    format output of a machine queue
    """
    print(f"Machine {machine_id+1} :")
    for pos, cycle in enumerate(machine_queue):
        print(
            f"    Job {cycle['p']}, min. start_date {args['start_date'][cycle['p']]} :"
        )
        print(f"    Cycle {cycle['c']}, Pos [{pos}] :")
        print(f"        Setup : ({cycle['setup_beg']}, {cycle['setup_end']})")
        for l in range(len(cycle["load_beg"])):
            print(
                f"            [{l}] : ({cycle['load_beg'][l]}, {cycle['load_end'][l]}) => ({cycle['unload_beg'][l]}, {cycle['unload_end'][l]})"
            )


def visualize_temperature_profile(
    temperature_function, time_range, out_filepath, **kwargs
):
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


"""
UTILITY FUNCTIONS (to make code more readable)
"""


def adapt_args(products, args):
    """
    Apply corrections / Gather info and set to the arguments dictionary
    - add start_date and due_date for each product
    - converting 1-indexed to 0-indexed all keys referring to machines
    - modifying scheduled maintenances content
    - modifying prod-to-machine compatibility content
    - modified prohibited intervals content
    """
    # Update the args with the products data
    args["start_date"] = {p: prod.start_date for p, prod in products}
    args["due_date"] = {p: prod.due_date for p, prod in products}
    args["velocity"] = {
        (p, c): prod.velocity[c] for p, prod in products for c in prod.velocity.keys()
    }
    # Info specific to RunningProduct instances
    args["current_op_type"] = {
        p: prod.current_op_type
        for p, prod, in products
        if isinstance(prod, RunningProduct)
    }
    args["remaining_time"] = {
        p: prod.remaining_time
        for p, prod in products
        if isinstance(prod, RunningProduct)
    }
    args["machine_fixation"] = {
        p: (prod.machine[0] - 1)
        for p, prod in products
        if isinstance(prod, RunningProduct)
    }
    args["setup_operator"] = {
        (p, c): prod.setup_operator[c]
        for p, prod in products
        for c in prod.setup_operator.keys()
    }
    args["load_operator"] = {
        (p, c, l): prod.load_operator[c, l]
        for p, prod in products
        for (c, l) in prod.load_operator.keys()
    }
    args["unload_operator"] = {
        (p, c, l): prod.unload_operator[c, l]
        for p, prod in products
        for (c, l) in prod.unload_operator.keys()
    }

    # FIXING INDEXES (1-indexed to 0-indexed) for all machine-related data
    # correct broken machines list
    broken_machines = copy.copy(args["broken_machines"])
    args["broken_machines"] = [m - 1 for m in broken_machines]
    # correct scheduled maintenances
    scheduled_maintenances = copy.copy(args["scheduled_maintenances"])
    args["scheduled_maintenances"] = {
        (m - 1): [(beg, beg + length) for beg, length in scheduled_maintenances[m]]
        for m in scheduled_maintenances.keys()
    }
    # correct prod-to-machine compatibility
    prod_to_machine_comp = copy.copy(args["prod_to_machine_comp"])
    args["prod_to_machine_comp"] = {
        p: [m - 1 for m in prod_to_machine_comp[p]] for p in prod_to_machine_comp.keys()
    }
    # correct setup base costs
    base_setup_cost = copy.copy(args["base_setup_cost"])
    args["base_setup_cost"] = {
        (p, m - 1): base_setup_cost[(p, m)] for (p, m) in base_setup_cost.keys()
    }
    # correct load base costs
    base_load_cost = copy.copy(args["base_load_cost"])
    args["base_load_cost"] = {
        (p, m - 1): base_load_cost[(p, m)] for (p, m) in base_load_cost.keys()
    }
    # correct unload base costs
    base_unload_cost = copy.copy(args["base_unload_cost"])
    args["base_unload_cost"] = {
        (p, m - 1): base_unload_cost[(p, m)] for (p, m) in base_unload_cost.keys()
    }

    # FIXING PROHIBITED INTERVALS
    # shift by 1 unit beginning to allow allocation on such position
    prohibited_intervals = copy.copy(args["prohibited_intervals"])
    args["prohibited_intervals"] = [(beg + 1, end) for beg, end in prohibited_intervals]
    # Populate overlap Tree for domain gaps
    args["domain_interval_tree"] = IntervalTree.from_tuples(
        args["prohibited_intervals"]
    )
    args["domain_interval_tree"].merge_overlaps()

    return args


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


def build_operator_inter_tree(individual, args, exclude_cycles=[]):
    """
    Generate overlap tree for each operator group
    excluding specified machines and cycles
    """

    # Generate overlap tree for each operator group
    overlap_tree = {}
    operator_intervals = {o: [] for o in range(args["num_operator_groups"])}
    for m in range(args["num_machines"]):
        for elem in individual[m]:
            p, c = elem["p"], elem["c"]
            # tuples (p,c) in "exclude_cycles" list are also not inserted in the tree
            if (p, c) not in exclude_cycles:
                # assign intervals to relative groups (always excluding width zero tuples)
                if (elem["setup_end"] - elem["setup_beg"]) > 0:
                    operator_intervals[args["setup_operator"][p, c]].append(
                        (elem["setup_beg"], elem["setup_end"])
                    )
                for l in range(len(elem["load_beg"])):
                    if (elem["load_end"][l] - elem["load_beg"][l]) > 0:
                        operator_intervals[args["load_operator"][p, c, l]].append(
                            (elem["load_beg"][l], elem["load_end"][l])
                        )
                    if (elem["unload_end"][l] - elem["unload_beg"][l]) > 0:
                        operator_intervals[args["unload_operator"][p, c, l]].append(
                            (elem["unload_beg"][l], elem["unload_end"][l])
                        )

    # Generate overlap tree for each operator group from extracted tuples
    for o in range(args["num_operator_groups"]):
        overlap_tree[o] = IntervalTree.from_tuples(operator_intervals[o])

    return overlap_tree


def gather_operation_info(cycle, machine_id, levata_id, operation_to_adjust, args):
    """
    Extracts operation information from the cycle and the machine
    Adjusting values in the case of fixed operations
    (Function for readability purposes)
    """
    p, c, l, m = cycle["p"], cycle["c"], levata_id, machine_id
    # Pick the proper operation according to the operation advancement
    operation = None
    if operation_to_adjust[machine_id] == 0:
        operation = {
            "type": "setup",
            "base_cost": args["base_setup_cost"][p, m],
            "operator_group": args["setup_operator"][p, c],
            "min_distance": 0,
        }
    elif (operation_to_adjust[m] - 1) % 2 == 0:
        operation = {
            "type": "load",
            "base_cost": args["base_load_cost"][p, m],
            "operator_group": args["load_operator"][p, c, l],
            "min_distance": 0,
        }
    else:
        operation = {
            "type": "unload",
            "base_cost": args["base_unload_cost"][p, m],
            "operator_group": args["unload_operator"][p, c, l],
            "min_distance": args["base_levata_cost"][p]
            - args["velocity"][p, c] * args["velocity_step_size"][p],
        }

    # if cycle is fixed and we're at first levata we need to apply some corrections
    if l <= 0 and cycle["fixed"]:
        if (
            operation["type"] == "setup"
            and args["current_op_type"][p] == 0
            or operation["type"] == "load"
            and args["current_op_type"][p] == 1
        ):
            # if evaluating a Setup / Load & current operation is one of them (0: Setup. 1: Load), correct the base_cost
            operation["base_cost"] = args["remaining_time"][p]

        elif operation["type"] == "unload":
            # if evaluating an Unload there are 2 cases to take care of
            if args["current_op_type"][p] == 2:
                # current operation is (2 : running), adjust only min_distance
                operation["min_distance"] = args["remaining_time"][p]

            elif args["current_op_type"][p] == 3:
                # current operation is (3 : Unload), adjust base_cost and zero the min_distance
                operation["base_cost"] = args["remaining_time"][p]
                operation["min_distance"] = 0
        else:
            return False, operation

    return True, operation


"""
DOMAIN ADJUSTMENT FUNCTIONS
"""


def greedy_set_value_in_domain(value, overlap_tree):
    """
    given a value, it shifts it to the first feasible value
    non-overlapping with the given interval tree
    """
    # atomic check for start of feasibility (value+1 is excluded)
    overlap_set = overlap_tree.overlap(value, value + 1)
    if len(overlap_set) > 0:
        # sfhift value towards first free slot
        value = next(iter(overlap_set)).end
    return value


def set_end_in_domain(beg, base_cost, args):
    """
    given the beginning of an operation and its base cost,
    it returns the end of the operation, taking into account
    """
    day = beg // args["time_units_in_a_day"]
    ub_day = args["end_shift"] + day * args["time_units_in_a_day"]
    # end is contained withing beg day
    end = beg + base_cost
    if end > ub_day and 0 <= day < len(args["gap_at_day"]):
        end += args["gap_at_day"][day]

    return end


def adjust_operation(
    point, min_distance, operation_base_cost, operator_overlap_tree, args
):
    """
    returns beginning & end of operation (Setup / Load / Unload) such that
    it starts at least 'min_distance' units from point and doesn't
    produce any overlap within associated operator group
    """
    anchor = point + min_distance
    # set point + min_distance to first feasible value according to worktime domain
    beg = greedy_set_value_in_domain(anchor, args["domain_interval_tree"])
    # set beg to first feasible value according to operator intervals domain
    beg = greedy_set_value_in_domain(beg, operator_overlap_tree)

    # Find appropriate setting
    end = beg + operation_base_cost
    while True:
        # Assign appropriate end value to the operation
        end = set_end_in_domain(beg, operation_base_cost, args)
        # Check for whole interval feasibility
        end_overlap_set = operator_overlap_tree.overlap(beg, end)
        # If conflicts are found :
        if len(end_overlap_set) > 0:
            conflicts = sorted(end_overlap_set)
            # beg = greedy_set_value_in_domain(conflicts[-1].end, operator_overlap_tree)
            beg = conflicts[-1].end
            continue
        # If no conflicts are found break the loop
        break

    return beg, end


def adjust_machines(random, individual, machine_queue, anchor, adjustment_start, args):
    # exclude cycles starting from specified positions untill the end of machines
    cycles_to_exclude = []
    for m in machine_queue.keys():
        cycles_to_exclude += [
            (elem["p"], elem["c"])
            for pos, elem in enumerate(machine_queue[m])
            if pos >= adjustment_start[m]
        ]
    # Generate overlap tree for each operator group excluding the specified cycles
    operator_overlap_tree = build_operator_inter_tree(
        individual, args, exclude_cycles=cycles_to_exclude
    )

    # Extract scheduled maintenances for the machine as overlap tree (if any)
    scheduled_maintenances_tree = {}
    for m in machine_queue.keys():
        scheduled_maintenances_tree[m] = (
            None
            if m not in args["scheduled_maintenances"].keys()
            else IntervalTree.from_tuples(args["scheduled_maintenances"][m])
        )
    # keep track of advancement status for each machine
    cycle_to_adjust = {m: adjustment_start[m] for m in machine_queue.keys()}
    operation_to_adjust = {m: 0 for m in machine_queue.keys()}

    ####
    ## Operations adjustment loop
    ####

    # Keep advancing until all cycles are adjusted
    while len(cycle_to_adjust.keys()) > 0:
        # Select a random machine to advance
        m = random.choice(list(cycle_to_adjust.keys()))
        # If the machine has no more cycles to adjust, remove it from the dictionary
        if cycle_to_adjust[m] >= len(machine_queue[m]):
            cycle_to_adjust.pop(m, None)
            continue
        # pick the cycle
        cycle = machine_queue[m][cycle_to_adjust[m]]
        p, c = cycle["p"], cycle["c"]

        # If the cycle has no more operations to adjust reset
        # the operation advancement and increment the cycle advancement
        l = (operation_to_adjust[m] - 1) // 2  # levata counter
        if l >= len(cycle["load_beg"]):
            cycle_to_adjust[m] += 1
            operation_to_adjust[m] = 0
            continue

        # Gather operation information
        needs_adjustment, operation = gather_operation_info(
            cycle, m, l, operation_to_adjust, args
        )
        # if cycle is fixed and operation has been done in the past...
        if not needs_adjustment:
            # skip it incrementing advancement
            operation_to_adjust[m] += 1
            continue

        # modify anchor to be compliant with start date on setup
        # (on operations following setup it will be implicit to be satisfied)
        anchor[m] = (
            max(anchor[m], args["start_date"][p])
            if operation["type"] == "setup"
            else anchor[m]
        )
        # Adjust the operation
        operation_beg, operation_end = adjust_operation(
            anchor[m],
            operation["min_distance"],
            operation["base_cost"],
            operator_overlap_tree[operation["operator_group"]],
            args,
        )
        # Store back the adjusted operation
        if operation["type"] == "setup":
            cycle["setup_beg"], cycle["setup_end"] = operation_beg, operation_end
        elif operation["type"] == "load":
            cycle["load_beg"][l], cycle["load_end"][l] = operation_beg, operation_end
        else:
            cycle["unload_beg"][l], cycle["unload_end"][l] = (
                operation_beg,
                operation_end,
            )

        ####
        ## Update parameters
        ####

        # update operator overlap tree with operation interval
        operator_overlap_tree[operation["operator_group"]].add(
            Interval(operation_beg, operation_end)
        )

        # if adjustment is on a machine with scheduled maintenances
        # check for conflicts. if any are found, repeat the cycle assignment process
        # imposing as new anchor the end of the first maintenance conflict
        if scheduled_maintenances_tree[m] is not None:
            # Check for if cycle assignment is feasible according to scheduled maintenances
            maintenance_overlap_set = scheduled_maintenances_tree[m].overlap(
                cycle["setup_beg"], operation_end
            )
            # if set is not empty it means there are conflicts
            if len(maintenance_overlap_set) > 0:
                # all intervals allocated by the cycle are removed from the operator interval tree
                operator_overlap_tree[args["setup_operator"][p, c]].discard(
                    Interval(cycle["setup_beg"], cycle["setup_end"])
                )

                for l in range(((operation_to_adjust[m] - 1) // 2) + 1):
                    operator_overlap_tree[args["load_operator"][p, c, l]].discard(
                        Interval(cycle["load_beg"][l], cycle["load_end"][l])
                    )
                    operator_overlap_tree[args["unload_operator"][p, c, l]].discard(
                        Interval(cycle["unload_beg"][l], cycle["unload_end"][l])
                    )
                # reset progress on the cycle
                conflicts = sorted(maintenance_overlap_set)
                anchor[m] = conflicts[0].end
                operation_to_adjust[m] = 0
                # go ahead with operation adjustment loop
                continue

        # next operation will have as anchor point the end of current operation
        anchor[m] = operation_end
        # Update the advancement dictionary
        operation_to_adjust[m] += 1

    return machine_queue


"""
TEST CASES FUNCTIONS
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
            p = elem["p"]
            c = elem["c"]
            # Check for setup start/end points
            if (
                len(
                    args["domain_interval_tree"].overlap(
                        elem["setup_beg"], elem["setup_beg"] + 1
                    )
                )
                > 0
            ):
                print(
                    f"Domain Violation : Job {p}, Cycle {c}, on machine {m+1}\n",
                    f"      Setup beginning is outside worktime domain",
                    f"{args['domain_interval_tree'].overlap(elem['setup_beg'], elem['setup_beg']+1)}",
                    f"{elem['setup_beg'], elem['setup_beg']+1}",
                )
                is_valid_solution = False
            if (
                len(
                    args["domain_interval_tree"].overlap(
                        elem["setup_end"], elem["setup_end"] + 1
                    )
                )
                > 0
            ):
                print(
                    f"Domain Violation : Job {p}, Cycle {c}, on machine {m+1}\n",
                    f"      Setup end is outside worktime domain",
                    f"{args['domain_interval_tree'].overlap(elem['setup_end'], elem['setup_end']+1)}",
                    f"{elem['setup_end'], elem['setup_end']+1}",
                )
                is_valid_solution = False
            # Check for load/unload start/end points
            for l in range(len(elem["load_beg"])):
                if (
                    len(
                        args["domain_interval_tree"].overlap(
                            elem["load_beg"][l], elem["load_beg"][l] + 1
                        )
                    )
                    > 0
                ):
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n",
                        f"      Load beginning is outside worktime domain",
                        f"{args['domain_interval_tree'].overlap(elem['load_beg'][l], elem['load_beg'][l]+1)}",
                        f"{elem['load_beg'][l], elem['load_beg'][l]+1}",
                    )
                    is_valid_solution = False
                if (
                    len(
                        args["domain_interval_tree"].overlap(
                            elem["load_end"][l], elem["load_end"][l] + 1
                        )
                    )
                    > 0
                ):
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n",
                        f"      Load end is outside worktime domain",
                        f"{args['domain_interval_tree'].overlap(elem['load_end'][l], elem['load_end'][l]+1)}",
                        f"{elem['load_end'][l], elem['load_end'][l]+1}",
                    )
                    is_valid_solution = False
                if (
                    len(
                        args["domain_interval_tree"].overlap(
                            elem["unload_beg"][l], elem["unload_beg"][l] + 1
                        )
                    )
                    > 0
                ):
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n",
                        f"      Unload beginning is outside worktime domain",
                        f"{args['domain_interval_tree'].overlap(elem['unload_beg'][l], elem['unload_beg'][l]+1)}",
                        f"{elem['unload_beg'][l], elem['unload_beg'][l]+1}",
                    )
                    is_valid_solution = False
                if (
                    len(
                        args["domain_interval_tree"].overlap(
                            elem["unload_end"][l], elem["unload_end"][l] + 1
                        )
                    )
                    > 0
                ):
                    print(
                        f"Domain Violation : Job {p}, Cycle {c}, Levata {l}, on machine {m+1}\n",
                        f"      Unload end is outside worktime domain",
                        f"{args['domain_interval_tree'].overlap(elem['unload_end'][l], elem['unload_end'][l]+1)}",
                        f"{elem['unload_end'][l], elem['unload_end'][l]+1}",
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
            p = elem["p"]
            c = elem["c"]

            setup_overlaps = operator_interval_tree[
                args["setup_operator"][p, c]
            ].overlap(elem["setup_beg"], elem["setup_end"])
            if len(setup_overlaps) > 1:
                print(
                    f"Operator overlap (SETUP) : An operator group is assigned to more than one operation at the same time\n"
                    f"      Product [{p+65}], Cycle [{c}] has conflicts"
                )
                is_valid_solution = False

            for l in range(len(elem["load_beg"])):
                load_overlaps = operator_interval_tree[
                    args["load_operator"][p, c, l]
                ].overlap(elem["load_beg"][l], elem["load_end"][l])
                if len(load_overlaps) > 1:
                    print(
                        f"Operator overlap (LOAD) : An operator group is assigned to more than one operation at the same time\n"
                        f"      Product [{p+65}], Cycle [{c}], Levata [{l}], has conflicts"
                    )
                    is_valid_solution = False

                unload_overlaps = operator_interval_tree[
                    args["unload_operator"][p, c, l]
                ].overlap(elem["unload_beg"][l], elem["unload_end"][l])
                if len(unload_overlaps) > 1:
                    print(
                        f"Operator overlap (UNLOAD) : An operator group is assigned to more than one operation at the same time\n"
                        f"      Product [{p+65}], Cycle [{c}], Levata [{l}], has conflicts"
                    )
                    is_valid_solution = False

    # LOOK FOR TIME WINDOW VIOLATIONS
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["p"]
            c = elem["c"]

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
            p = elem["p"]
            c = elem["c"]

            if m not in args["prod_to_machine_comp"][p]:
                print(
                    f"Machine Assignment Violation :\n"
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
        machine_scheduled_maintenances_tree = (
            None
            if m not in args["scheduled_maintenances"].keys()
            else IntervalTree.from_tuples(args["scheduled_maintenances"][m])
        )
        if machine_scheduled_maintenances_tree is not None:
            for elem in machine_queue:
                p = elem["p"]
                c = elem["c"]
                maintenance_overlap_set = machine_scheduled_maintenances_tree.overlap(
                    elem["setup_beg"], elem["unload_end"][-1]
                )
                if len(maintenance_overlap_set) > 0:
                    print(
                        f"Scheduled Maintenance Violation :\n"
                        f"    Job {p}, Cycle {c}, on machine {m+1} violates scheduled maintenance"
                    )
                    is_valid_solution = False

    # LOOK FOR FIXED CYCLES VIOLATIONS
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["p"]
            c = elem["c"]
            if elem["fixed"]:
                if m != args["machine_fixation"][p]:
                    print(
                        f"Fixed Cycle Violation :\n"
                        f"    Job {p}, Cycle {c}, on machine {m+1} is not fixed"
                    )
                    is_valid_solution = False

    # CHECK IF PAST OPERATIONS STILL HAVE ZERO COST
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["p"]
            c = elem["c"]
            if elem["fixed"]:
                if (
                    args["current_op_type"][p] > 0
                    and elem["setup_end"] - elem["setup_beg"] > 0
                ):
                    print(
                        f"Fixed Cycle Violation :\n"
                        f"    Job {p}, Cycle {c}, on machine {m+1} has non-zero setup cost"
                    )
                    is_valid_solution = False
                if (
                    args["current_op_type"][p] > 1
                    and elem["load_end"][0] - elem["load_beg"][0] > 0
                ):
                    print(
                        f"Fixed Cycle Violation :\n"
                        f"    Job {p}, Cycle {c}, on machine {m+1} has non-zero load cost"
                    )
                    is_valid_solution = False

    return is_valid_solution
