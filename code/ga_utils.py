from inspyred import ec
from random import Random
from intervaltree import IntervalTree, Interval
from data_init import RunningProduct
import matplotlib.pyplot as plt
import numpy
import copy

def build_operator_inter_tree (individual, args, exclude_machines=[], exclude_cycles=[]) :
    """
    Generate overlap tree for each operator group
    excluding specified machines and cycles
    """
    # Generate overlap tree for each operator group
    overlap_tree = {}
    operator_intervals = {o: [] for o in range(args["num_operator_groups"])}
    for m in range(args["num_machines"]) :
        # machines in "exclude_machines" list are not included in the tree
        if m not in exclude_machines :
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

def add_intervals_to_tree (tree, machine_queue, from_pos=0) :
    """
    Add intervals from a machine queue to an overlap tree
    excluding the ones before "from_pos"
    """
    for cycle in machine_queue :
        if cycle["pos"] >= from_pos :
            # Add setup interval
            if (cycle["setup_end"] - cycle["setup_beg"]) > 0 : tree[cycle["setup_operator"]].add(Interval(cycle["setup_beg"], cycle["setup_end"]))
            # Add Load & Unload intervals
            for l in range(len(cycle["load_beg"])) :
                if (cycle["load_end"][l] - cycle["load_beg"][l]) > 0 : tree[cycle["load_operator"][l]].add(Interval(cycle["load_beg"][l], cycle["load_end"][l]))
                if (cycle["unload_end"][l] - cycle["unload_beg"][l]) > 0 : tree[cycle["unload_operator"][l]].add(Interval(cycle["unload_beg"][l], cycle["unload_end"][l]))
    return tree


"""
DOMAIN ADJUSTMENT FUNCTIONS
"""

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
            #beg = greedy_set_value_in_domain(conflicts[-1].end, operator_overlap_tree)
            beg = conflicts[-1].end
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
        cycle["load_beg"][l], cycle["load_end"][l] = adjust_operation(anchor, 0, args["base_op_cost"][cycle["product"], machine_id], operator_overlap_tree[cycle["load_operator"][l]], args)
        cycle["unload_beg"][l], cycle["unload_end"][l] = adjust_operation(cycle["load_end"][l], effective_levata_cost, args["base_op_cost"][cycle["product"], machine_id], operator_overlap_tree[cycle["unload_operator"][l]], args)
        # set next anchor point
        anchor = cycle["unload_end"][l]
    
    return cycle

def adjust_machine(machine_queue, machine_id, anchor, operator_overlap_tree, args, from_pos=0) :
    """
    Adjust all cycles ina machine queue in a greedy way,
    such that no overlaps are produced
    """
    for cycle in machine_queue :
        if cycle["pos"] >= from_pos :
            cycle["setup_beg"] = anchor
            cycle = adjust_cycle(cycle, machine_id, operator_overlap_tree, args)
            anchor = cycle["unload_end"][-1]
    
    return machine_queue


def print_solution (candidate) :
    """
    format output for a candidate solution
    """
    for cycle in candidate :
        print(f"Product {chr(cycle['product']+65)}, Cycle {cycle['cycle']}, Pos [{cycle['pos']}] :")
        print(f"    Setup : ({cycle['setup_beg']}, {cycle['setup_end']})")
        for l in range(len(cycle["load_beg"])) :
            print(f"        [{l}] : ({cycle['load_beg'][l]}, {cycle['load_end'][l]}) => ({cycle['unload_beg'][l]}, {cycle['unload_end'][l]})")


def check_solution_conformity(solution, args):
    """
    Check if a solution conforms to the problem constraints.
    Prints conflicts if found and returns False. Otherwise, returns True.
    """
    has_conflicts = False

    # LOOK FOR WITHIN MACHINE OVERLAPS
    for m, machine_queue in enumerate(solution):
        intervals = []
        for cycle in machine_queue:
            if (cycle["setup_end"] - cycle["setup_beg"]) > 0:
                intervals.append((cycle["setup_beg"], cycle["setup_end"]))
            for l in range(len(cycle["load_beg"])):
                if (cycle["load_end"][l] - cycle["load_beg"][l]) > 0:
                    intervals.append((cycle["load_beg"][l], cycle["load_end"][l]))
                if (cycle["unload_end"][l] - cycle["unload_beg"][l]) > 0:
                    intervals.append((cycle["unload_beg"][l], cycle["unload_end"][l]))

        machine_interval_tree = IntervalTree.from_tuples(intervals)
        for beg, end in intervals:
            conflict = machine_interval_tree.overlap(beg, end)
            if len(conflict) > 1:
                print(
                    f"Within-Machine Overlap Found on Machine {m}\n"
                    f"{print_solution(machine_queue)}"
                )
                has_conflicts = True

    # LOOK FOR OPERATOR OVERLAPS
    operator_interval_tree = build_operator_inter_tree(solution, args)
    for m, machine_queue in enumerate(solution):
        for elem in machine_queue:
            p = elem["product"]
            c = elem["cycle"]

            setup_overlaps = operator_interval_tree[elem["setup_operator"]].overlap(
                elem["setup_beg"], elem["setup_end"]
            )
            if len(setup_overlaps) > 1:
                print(
                    f"operatorSETUP OVERLAP: {(chr(p + 65), c)} - "
                    f"{(elem['setup_beg'], elem['setup_end'])} => "
                    f"{setup_overlaps}"
                )
                has_conflicts = True

            for l in range(len(elem["load_beg"])):
                load_overlaps = operator_interval_tree[
                    elem["load_operator"][l]
                ].overlap(elem["load_beg"][l], elem["load_end"][l])
                if len(load_overlaps) > 1:
                    print(
                        f"LOAD OVERLAP: {(chr(p + 65), c)} - "
                        f"{(elem['load_beg'][l], elem['load_end'][l])} => "
                        f"{load_overlaps}"
                    )
                    has_conflicts = True

                unload_overlaps = operator_interval_tree[
                    elem["unload_operator"][l]
                ].overlap(elem["unload_beg"][l], elem["unload_end"][l])
                if len(unload_overlaps) > 1:
                    print(
                        f"UNLOAD OVERLAP: {(chr(p + 65), c)} - "
                        f"{(elem['unload_beg'][l], elem['unload_end'][l])} => "
                        f"{unload_overlaps}"
                    )
                    has_conflicts = True

    if has_conflicts:
        print("CONFLICTS FOUND\n")
        for m, machine_queue in enumerate(solution):
            print(f"Machine {m}\n{print_solution(machine_queue)}")
        print("\n--------------------")
        return False

    return True
