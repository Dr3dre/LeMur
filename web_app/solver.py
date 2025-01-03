from ortools.sat.python import cp_model
import math

from data_init import Product, RunningProduct, Schedule, init_csv_data
import os
from ga_refiner import GA_Refiner

from datetime import datetime

"""
INPUT DATA
"""

# timouts for solver
MAKESPAN_SOLVER_TIMEOUT = 600
CYCLE_OPTIM_SOLVER_TIMEOUT = 60
GENERATIONS = 250

LOGGING = False

USE_ADD_ELEMENT = True
PRESOLVE_SECOND_SEARCH = True

STOP_AT_FIRST_SOLUTION = False

# output txt files
OUTPUT_SCHEDULE = "output/schedule.txt"
OUTPUT_REFINED_SCHEDULE = "output/refined_schedule.txt"

# constraints = [] # list of constraints for debugging

machine_velocities = 1


def get_time_intervals(
    horizon_days,
    time_units_in_a_day,
    start_shift,
    end_shift,
    festive_days=[],
    now=datetime.now(),
):
    # Compute horizon
    horizon = horizon_days * time_units_in_a_day

    # Current time data
    days_in_week = 7
    minutes_from_midnight = (
        now.hour * 60 + now.minute
    )  # Current time in minutes from midnight
    current_day_of_week = now.weekday()  # 0: Monday, 1: Tuesday, ..., 6: Sunday

    # Time steps convertion
    time_units_from_midnight = minutes_from_midnight
    if time_units_in_a_day == 24:
        time_units_from_midnight = minutes_from_midnight // 60
    elif time_units_in_a_day == 48:
        time_units_from_midnight = minutes_from_midnight // 30
    elif time_units_in_a_day == 96:
        time_units_from_midnight = minutes_from_midnight // 15

    # Define worktime intervals according to current daytime
    worktime_intervals = []
    # first interval (scheduling startin in or out workday)
    if (
        (time_units_from_midnight < end_shift)
        and (time_units_from_midnight > start_shift)
        and (current_day_of_week % days_in_week not in [6])
        and 0 not in festive_days
    ):
        worktime_intervals.append((time_units_from_midnight, end_shift))
    # Handle remaining cases
    for day in range(1, horizon_days):
        if (day + current_day_of_week) % days_in_week not in [
            6
        ] and day not in festive_days:
            workday_start = day * time_units_in_a_day + start_shift
            workday_end = day * time_units_in_a_day + end_shift
            worktime_intervals.append((workday_start, workday_end))

    # Define prohibited intervals as complementar set of worktime
    prohibited_intervals = []
    # first interval (scheduling startin in or out workday)
    if time_units_from_midnight < worktime_intervals[0][0]:
        prohibited_intervals.append(
            (time_units_from_midnight, worktime_intervals[0][0])
        )
    # handle remaining cases
    for i in range(len(worktime_intervals) - 1):
        _, gap_start = worktime_intervals[i]
        gap_end, _ = worktime_intervals[i + 1]
        prohibited_intervals.append((gap_start, gap_end))
    # Append last interval (from last worktime end to horizon)
    prohibited_intervals.append((worktime_intervals[-1][1], horizon))

    # List of gap size relative to day index (0 being today)
    gap_at_day = []
    gap_idx = 0 if prohibited_intervals[0][0] > 0 else 1
    for g in range(horizon_days):
        time_step = g * time_units_in_a_day + start_shift
        # Check if need to advance in the prohibited intervals
        while (
            gap_idx < len(prohibited_intervals)
            and time_step >= prohibited_intervals[gap_idx][1]
        ):
            gap_idx += 1

        gap_start, gap_end = prohibited_intervals[gap_idx]
        if gap_start <= time_step <= gap_end:
            gap_at_day.append(1000000000)  # Day inside a prohibited interval
        else:
            gap_at_day.append(gap_end - gap_start)

    return (
        worktime_intervals,
        prohibited_intervals,
        gap_at_day,
        time_units_from_midnight,
        worktime_intervals[0][0],
    )


def solve(
    common_products,
    running_products,
    article_to_machine_comp,
    fuses_machines_associations,
    base_setup_art_cost,
    base_load_art_cost,
    base_unload_art_cost,
    base_levata_art_cost,
    standard_levate_art,
    kg_per_levata_art,
    broken_machines,
    scheduled_maintenances,
    num_operator_groups,
    festivities,
    horizon_days,
    time_units_in_a_day,
    start_shift,
    end_shift,
    timeout=MAKESPAN_SOLVER_TIMEOUT,
    timeout_cycle=CYCLE_OPTIM_SOLVER_TIMEOUT,
    generations=GENERATIONS,
    now=datetime.now(),
):
    """
    SOLVER

    Args:
        common_products (list): list of common products
        running_products (list): list of running products
        article_to_machine_comp (dict): article to machine compatibility {article: [machines]}
        base_setup_art_cost (dict): base setup cost for each article
        base_load_art_cost (dict): base load cost for each article
        base_unload_art_cost (dict): base unload cost for each article
        base_levata_art_cost (dict): base levata cost for each article
        standard_levate_art (dict): standard levate for each article
        kg_per_levata_art (dict): kg per levata for each article
        broken_machines (list): list of broken machines
        scheduled_maintenances (list): list of scheduled maintenances
        num_operator_groups (int): number of operator groups
        festivities (list): list of festivities (days)
        horizon_days (int): horizon days
        time_units_in_a_day (int): time units in a day
        start_shift (int): start shift
        end_shift (int): end shift
        timeout (int): timeout for makespan solver
        timeout_cycle (int): timeout for cycle optimization solver
        generations (int): number of generations for genetic algorithm

    Returns:
        Schedule: the schedule
    """
    horizon = horizon_days * time_units_in_a_day
    num_machines = max([m for a, ms in article_to_machine_comp.items() for m in ms])

    # Make joint tuples (for readability purp.)
    common_products = [(prod.id, prod) for prod in common_products]
    running_products = [(prod.id, prod) for prod in running_products]
    all_products = common_products + running_products

    # convert_standard levate and kg to be indexed on product id
    standard_levate = {}
    kg_per_levata = {}
    base_levata_cost = {}
    base_setup_cost = {}
    gaps_setup = {}
    base_load_cost = {}
    gaps_load = {}
    base_unload_cost = {}
    gaps_unload = {}
    for p, prod in all_products:
        try:
            standard_levate[p] = standard_levate_art[prod.article]
            base_levata_cost[p] = base_levata_art_cost[prod.article]
            for m in article_to_machine_comp[prod.article]:
                kg_per_levata[m, p] = kg_per_levata_art[m, prod.article]
                base_setup_cost[p, m] = base_setup_art_cost[prod.article, m]
                base_load_cost[p, m] = base_load_art_cost[prod.article, m]
                base_unload_cost[p, m] = base_unload_art_cost[prod.article, m]
        except:
            ValueError(
                f"ERROR : Inconsistency in assigning standard levate and kg to articles for product {prod.article}\nBreaking execution..."
            )

    # convert machine and article compatibility to be indexed on product id
    prod_to_machine_comp = {}
    machine_to_prod_comp = {}
    for m in range(1, num_machines + 1):
        machine_to_prod_comp[m] = []
    for p, prod in all_products:
        try:
            prod_to_machine_comp[p] = article_to_machine_comp[prod.article]
            for m in article_to_machine_comp[prod.article]:
                machine_to_prod_comp[m].append(p)
        except:
            ValueError(
                f"ERROR : Inconsistency in assigning machines to articles compatibility for product {prod.article}\nBreaking execution..."
            )

    """
        DERIVED CONSTANTS
    """
    worktime_intervals, prohibited_intervals, gap_at_day, time_units_from_midnight, start_schedule = get_time_intervals(horizon_days, time_units_in_a_day, start_shift, end_shift, festivities, now=now)


    # Velocity gears
    velocity_levels = list(
        range(-(machine_velocities // 2), (machine_velocities // 2) + 1)
    )  # machine velocities = 3 => [-1, 0, 1]
    velocity_step_size = {}
    for p, prod in all_products:
        velocity_step_size[p] = int(
            (0.05 / max(1, max(velocity_levels))) * base_levata_cost[p]
        )

    # Generate custom domain excluding out of work hours, weekends, etc.
    #   make it personalized for each product, excluding values < start_date[p] and > due_date[p]
    worktime_domain = {}
    for p, prod in all_products:
        adjusted_intervals = []
        current_start = prod.start_date
        for start, end in worktime_intervals:
            if end <= prod.start_date:
                continue
            interval_start = max(start, current_start)
            interval_end = min(end, prod.due_date)
            if interval_start >= prod.due_date:
                break
            adjusted_intervals.append((interval_start, interval_end))
            current_start = interval_end
            if interval_end == prod.due_date:
                break
        if adjusted_intervals and adjusted_intervals[-1][1] < prod.due_date:
            adjusted_intervals[-1] = (
                adjusted_intervals[-1][0],
                min(adjusted_intervals[-1][1], prod.due_date),
            )

        # check empty intervals
        valid = False
        for interval in adjusted_intervals:
            if interval[0] < interval[1]:
                valid = True
        if not valid:
            print("EMPTY DOMAIN FOR ARTICLE", prod.article)
            raise ValueError(
                f"EMPTY DOMAIN FOR ARTICLE {prod.article}. This might mean that the horizon is too short for the dates provided for the product."
            )

        # generate worktime domain for p
        worktime_domain[p] = cp_model.Domain.FromIntervals(adjusted_intervals)
    # domain for velocity gear steps
    velocity_domain = cp_model.Domain.FromValues(velocity_levels)

    # Max amount of cycles a product might go through (it's assigned to the slowest machine)
    max_cycles = {}
    best_kg_cycle = {}
    for p, prod in all_products:
        if isinstance(prod, RunningProduct):
            max_cycles[p] = 1
        else:
            max_cycles[p] = max(
                [
                    math.ceil(
                        prod.kg_request / (kg_per_levata[m, p] * standard_levate[p])
                    )
                    for m in prod_to_machine_comp[p]
                ]
            )
        best_kg_cycle[p] = max(
            [
                math.ceil(kg_per_levata[m, p] * standard_levate[p])
                for m in prod_to_machine_comp[p]
            ]
        )

    # Adjust remaining time for running products
    # That's a corner case for when scheduling starts outside working hours
    # And a running product is already in the middle of 'running'operation
    for p, prod in all_products:
        if isinstance(prod, RunningProduct) and prod.current_op_type == 2 and start_schedule > time_units_from_midnight:
            prod.remaining_time = max(0, prod.remaining_time - (start_schedule-time_units_from_midnight))

    """
    SETS
    """
    # sets of running products requiring at hoc operations according to at which stage they're
    SETUP_EXCL = [
        (p, c)
        for p, prod in all_products
        for c in range(max_cycles[p])
        if (isinstance(prod, (RunningProduct)) and prod.current_op_type >= 0 and c == 0)
    ]
    LOAD_EXCL = [
        (p, c, l)
        for p, prod in all_products
        for c in range(max_cycles[p])
        for l in range(standard_levate[p])
        if (
            isinstance(prod, (RunningProduct))
            and prod.current_op_type >= 1
            and c == 0
            and l == 0
        )
    ]
    LEVATA_EXCL = [
        (p, c, l)
        for p, prod in all_products
        for c in range(max_cycles[p])
        for l in range(standard_levate[p])
        if (
            isinstance(prod, (RunningProduct))
            and prod.current_op_type >= 2
            and c == 0
            and l == 0
        )
    ]
    UNLOAD_EXCL = [
        (p, c, l)
        for p, prod in all_products
        for c in range(max_cycles[p])
        for l in range(standard_levate[p])
        if (
            isinstance(prod, (RunningProduct))
            and prod.current_op_type == 3
            and c == 0
            and l == 0
        )
    ]

    # Create the model
    model = cp_model.CpModel()
    print("Initializing model...")

    """
        DECISION VARIBLES
        """

    # Assignment variable (product, cycle, machine)
    A = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for m in prod_to_machine_comp[p]:
                if m not in broken_machines:
                    A[p, c, m] = model.NewBoolVar(f"A[{p},{c},{m}]")
                else:
                    A[p, c, m] = model.NewConstant(0)

    # States if the cycle is a completion cycle
    COMPLETE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            COMPLETE[p, c] = model.NewBoolVar(f"COMPLETE[{p},{c}]")
    # Number of levate operations in a cycle
    NUM_LEVATE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            NUM_LEVATE[p, c] = model.NewIntVar(
                0, standard_levate[p], f"NUM_LEVATE[{p},{c}]"
            )

    # beginning of setup operation
    SETUP_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            if (p, c) not in SETUP_EXCL:
                SETUP_BEG[p, c] = model.NewIntVarFromDomain(
                    worktime_domain[p], f"SETUP_BEG[{p},{c}]"
                )
            else:
                SETUP_BEG[p, c] = model.NewConstant(start_schedule)

    # beginning of load operation
    LOAD_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p, c, l) not in LOAD_EXCL:
                    LOAD_BEG[p, c, l] = model.NewIntVarFromDomain(
                        worktime_domain[p], f"LOAD_BEG[{p},{c},{l}]"
                    )
                else:
                    LOAD_BEG[p, c, l] = model.NewConstant(start_schedule)

    # beginning of unload operation
    UNLOAD_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p, c, l) not in UNLOAD_EXCL:
                    UNLOAD_BEG[p, c, l] = model.NewIntVarFromDomain(
                        worktime_domain[p], f"UNLOAD_BEG[{p},{c},{l}]"
                    )
                else:
                    UNLOAD_BEG[p, c, l] = model.NewConstant(start_schedule)

    # Velocity gear at which the cycle is performed
    VELOCITY = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            VELOCITY[p, c] = model.NewIntVarFromDomain(
                velocity_domain, f"VELOCITY[{p},{c}]"
            )

    # Operator group assignment to setup operations
    A_OP_SETUP = {}
    for o in range(num_operator_groups):
        for p, prod in all_products:
            for c in range(max_cycles[p]):
                A_OP_SETUP[o, p, c] = model.NewBoolVar(f"A_OP_SETUP[{o},{p},{c}]")

    # Operator group assignment to load / unload operations of some levata
    A_OP = {}
    for o in range(num_operator_groups):
        for p, prod in all_products:
            for c in range(max_cycles[p]):
                for l in range(standard_levate[p]):
                    for t in [0, 1]:
                        A_OP[o, p, c, l, t] = model.NewBoolVar(
                            f"A_OP[{o},{p},{c},{l},{t}]"
                        )

    """
        OTHER VARIABLES (no search needed, easily calculated)
        """
    # states if a levata is active (it exists) or not
    ACTIVE_LEVATA = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                ACTIVE_LEVATA[p, c, l] = model.NewBoolVar(f"ACTIVE_LEVATA[{p},{c},{l}]")
                # behaviour
                model.Add(l < NUM_LEVATE[p, c]).OnlyEnforceIf(ACTIVE_LEVATA[p, c, l])
                # constraints.append(f"states if a levata is active (it exists) or not 1 {l} < {NUM_LEVATE[p,c]}")
                model.Add(l >= NUM_LEVATE[p, c]).OnlyEnforceIf(
                    ACTIVE_LEVATA[p, c, l].Not()
                )
                # constraints.append(f"states if a levata is active (it exists) or not 2 {l} >= {NUM_LEVATE[p,c]}")

    # states if a cycle is active (it exists) or not
    ACTIVE_CYCLE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # This isn't an acutal variable, it's done for readability
            ACTIVE_CYCLE[p, c] = ACTIVE_LEVATA[p, c, 0]

    # states if a cycle is partial (not all levate are done)
    PARTIAL_CYCLE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            PARTIAL_CYCLE[p, c] = model.NewBoolVar(f"PARTIAL_CYCLE[{p},{c}]")

    ##################
    # COST VARIABLES #
    ##################

    # Base cost of setup operation (accounts for machine specific setup)
    BASE_SETUP_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            BASE_SETUP_COST[p, c] = model.NewIntVar(
                0, prod.due_date, f"BASE_SETUP_COST[{p},{c}]"
            )
            # behaviour
            if (p, c) not in SETUP_EXCL:
                for m in prod_to_machine_comp[p]:
                    model.Add(
                        BASE_SETUP_COST[p, c] == base_setup_cost[p, m]
                    ).OnlyEnforceIf(A[p, c, m], ACTIVE_CYCLE[p, c])
                    # constraints.append(f"Base cost of setup operation (accounts for machine specific setup) 1 {BASE_SETUP_COST[p,c]} == {base_setup_cost[p,m]}")
                model.Add(BASE_SETUP_COST[p, c] == 0).OnlyEnforceIf(
                    ACTIVE_CYCLE[p, c].Not()
                )
                # constraints.append(f"Base cost of setup operation (accounts for machine specific setup) 2 {BASE_SETUP_COST[p,c]} == 0")
            elif prod.current_op_type == 0:
                # set as base cost the remaining for running products (if p is in setup)
                model.Add(BASE_SETUP_COST[p, c] == prod.remaining_time)
                # constraints.append(f"Base cost of setup operation (accounts for machine specific setup) 3 {BASE_SETUP_COST[p,c]} == {prod.remaining_time}")

    # Base cost load operation (accounts for machine specific setup)
    BASE_LOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                BASE_LOAD_COST[p, c, l] = model.NewIntVar(
                    0, prod.due_date, f"BASE_LOAD_COST[{p},{c},{l}]"
                )
                # behaviour
                if (p, c, l) not in LOAD_EXCL:
                    for m in prod_to_machine_comp[p]:
                        model.Add(
                            BASE_LOAD_COST[p, c, l] == base_load_cost[p, m]
                        ).OnlyEnforceIf(A[p, c, m], ACTIVE_LEVATA[p, c, l])
                        # constraints.append(f"Base cost load operation (accounts for machine specific setup) 1 {BASE_LOAD_COST[p,c,l]} == {base_load_cost[p,m]}")
                    model.Add(BASE_LOAD_COST[p, c, l] == 0).OnlyEnforceIf(
                        ACTIVE_LEVATA[p, c, l].Not()
                    )
                    # constraints.append(f"Base cost load operation (accounts for machine specific setup) 2 {BASE_LOAD_COST[p,c,l]} == 0")
                elif prod.current_op_type == 1:
                    # set as base cost the remaining for running products (if p is in load)
                    model.Add(BASE_LOAD_COST[p, c, l] == prod.remaining_time)
                    # constraints.append(f"Base cost load operation (accounts for machine specific setup) 3 {BASE_LOAD_COST[p,c,l]} == {prod.remaining_time}")

    # Base cost unload operation (accounts for machine specific setup)
    BASE_UNLOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                BASE_UNLOAD_COST[p, c, l] = model.NewIntVar(
                    0, prod.due_date, f"BASE_UNLOAD_COST[{p},{c},{l}]"
                )
                # behaviour
                if (p, c, l) not in UNLOAD_EXCL:
                    for m in prod_to_machine_comp[p]:
                        model.Add(
                            BASE_UNLOAD_COST[p, c, l] == base_unload_cost[p, m]
                        ).OnlyEnforceIf(A[p, c, m], ACTIVE_LEVATA[p, c, l])
                        # constraints.append(f"Base cost unload operation (accounts for machine specific setup) 1 {BASE_UNLOAD_COST[p,c,l]} == {base_unload_cost[p,m]}")
                    model.Add(BASE_UNLOAD_COST[p, c, l] == 0).OnlyEnforceIf(
                        ACTIVE_LEVATA[p, c, l].Not()
                    )
                    # constraints.append(f"Base cost unload operation (accounts for machine specific setup) 2 {BASE_UNLOAD_COST[p,c,l]} == 0")
                elif prod.current_op_type == 3:
                    # set as base cost the remaining for running products (if p is in unload)
                    model.Add(BASE_UNLOAD_COST[p, c, l] == prod.remaining_time)
                    # constraints.append(f"Base cost unload operation (accounts for machine specific setup) 3 {BASE_UNLOAD_COST[p,c,l]} == {prod.remaining_time}")

    # cost (time) of levata operation
    #   Pay attention :
    #   there's no BASE_LEVATA_COST as LEVATA is an unsupervised
    #   operation, moreover it's independent to machine assignment
    LEVATA_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LEVATA_COST[p, c, l] = model.NewIntVar(
                    0, prod.due_date, f"LEVATA_COST[{p},{c},{l}]"
                )
                if (p, c, l) not in LEVATA_EXCL:
                    # behaviour
                    model.Add(
                        LEVATA_COST[p, c, l]
                        == base_levata_cost[p] - VELOCITY[p, c] * velocity_step_size[p]
                    ).OnlyEnforceIf(ACTIVE_LEVATA[p, c, l])
                    # constraints.append(f"cost (time) of levata operation 1 {LEVATA_COST[p,c,l]} == {base_levata_cost[p] - VELOCITY[p,c] * velocity_step_size[p]}")
                    model.Add(LEVATA_COST[p, c, l] == 0).OnlyEnforceIf(
                        ACTIVE_LEVATA[p, c, l].Not()
                    )
                    # constraints.append(f"cost (time) of levata operation 2 {LEVATA_COST[p,c,l]} == 0")
                elif prod.current_op_type == 2:
                    # set as base cost the remaining for running products (if theh're in levata)
                    #   N.B. => for .remaining_time relative to a setting .current_op_type = 2 (product is running)
                    #           we assume .remaining_time already takes account for the velocity & time in between
                    #           now and start_schedule in the case scheduling starts outside working hours
                    model.Add(LEVATA_COST[p, c, l] == prod.remaining_time)
                    # constraints.append(f"cost (time) of levata operation 3 {LEVATA_COST[p,c,l]} == {prod.remaining_time}")

    def make_gap_var(BEGIN, BASE_COST, IS_ACTIVE, enabled=True):
        """
        Returns a GAP variable to allow setup / load / unload operations
        to be performed over multiple days (if needed)
        """
        if not enabled:
            return 0  # for testing purposes

        # Associate day index to relative gap size
        G = model.NewIntVar(0, horizon_days, f"G_DAY")
        gap_values = list(set(gap_at_day))
        GAP_SIZE = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(gap_values), f"GAP_SIZE"
        )
        model.AddElement(G, gap_at_day, GAP_SIZE)
        # constraints.append(f"Associate day index to relative gap size {G} == {gap_at_day} {GAP_SIZE}")
        # Get current day
        model.AddDivisionEquality(G, BEGIN, time_units_in_a_day)
        # constraints.append(f"Get current day {G} == {BEGIN} / {time_units_in_a_day}")
        UB = end_shift + time_units_in_a_day * G
        # Understand if such operation goes beyond the worktime
        NEEDS_GAP = model.NewBoolVar("NEEDS_GAP")
        model.Add((BEGIN + BASE_COST) > UB).OnlyEnforceIf(NEEDS_GAP)
        # constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} > {UB}")
        model.Add((BEGIN + BASE_COST) <= UB).OnlyEnforceIf(NEEDS_GAP.Not())
        # constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} <= {UB}")
        # Associate to GAP if needed, otherwise it's zero
        GAP = model.NewIntVar(0, max(gap_at_day), f"GAP")
        model.Add(GAP == GAP_SIZE).OnlyEnforceIf(NEEDS_GAP)
        # constraints.append(f"Associate to GAP if needed, otherwise it's zero {GAP} == {GAP_SIZE}")
        model.Add(GAP == 0).OnlyEnforceIf(NEEDS_GAP.Not())
        # constraints.append(f"Associate to GAP if needed, otherwise it's zero {GAP} == 0")
        # If not active, GAP is also zero
        model.AddImplication(IS_ACTIVE.Not(), NEEDS_GAP.Not())
        # constraints.append(f"If not active, GAP is also zero {GAP} == 0")

        return GAP

    def make_gap_var_linear(
        model: cp_model.CpModel,
        BEGIN,
        BASE_COST,
        IS_ACTIVE,
        gap_at_day,
        time_units_in_a_day,
        end_shift,
        horizon_days,
        enabled=True,
    ):
        """
        Returns a GAP variable to allow setup / load / unload operations
        to be performed over multiple days (if needed) without using AddElement.
        """

        if not enabled:
            return 0  # for testing purposes

        # Associate day index to relative gap size
        G = model.NewIntVar(0, horizon_days, f"G_DAY")
        # Get current day
        model.AddDivisionEquality(G, BEGIN, time_units_in_a_day)

        values_gap = list(set(gap_at_day))
        GAP_SIZE = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(values_gap), f"GAP_SIZE"
        )
        for i, gap in enumerate(gap_at_day):
            G_i = model.NewBoolVar(f"G_{i}")
            model.Add(G == i).OnlyEnforceIf(G_i)
            model.Add(G != i).OnlyEnforceIf(G_i.Not())
            model.Add(GAP_SIZE == gap).OnlyEnforceIf(G_i)
            # constraints.append(f"Associate day index to relative gap size {GAP_SIZE} == {gap} {G} == {i}")

        # constraints.append(f"Get current day {G} == {BEGIN} / {time_units_in_a_day}")
        UB = end_shift + time_units_in_a_day * (G)

        # Understand if such operation goes beyond the worktime
        NEEDS_GAP = model.NewBoolVar(f"NEEDS_GAP")
        model.Add((BEGIN + BASE_COST) >= UB).OnlyEnforceIf(NEEDS_GAP)
        # constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} > {UB}")
        model.Add((BEGIN + BASE_COST) < UB).OnlyEnforceIf(NEEDS_GAP.Not())
        # constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} <= {UB}")

        # Associate to GAP if needed, otherwise it's zero
        GAP = model.NewIntVar(0, max(gap_at_day), f"GAP")
        model.Add(GAP == GAP_SIZE).OnlyEnforceIf(NEEDS_GAP)
        # constraints.append(f"Associate to GAP if needed, otherwise it's zero {GAP} == {GAP_SIZE}")
        model.Add(GAP == 0).OnlyEnforceIf(NEEDS_GAP.Not())
        model.AddImplication(IS_ACTIVE.Not(), NEEDS_GAP.Not())
        # constraints.append(f"If not active, GAP is also zero {GAP} == 0")

        return GAP

    # cost (time) of machine setup operation
    SETUP_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            SETUP_COST[p, c] = model.NewIntVar(0, prod.due_date, f"SETUP_COST[{p},{c}]")
            # behaviour
            if USE_ADD_ELEMENT:
                SETUP_GAP = make_gap_var(
                    SETUP_BEG[p, c], BASE_SETUP_COST[p, c], ACTIVE_CYCLE[p, c]
                )
            else:
                SETUP_GAP = make_gap_var_linear(
                    model,
                    SETUP_BEG[p, c],
                    BASE_SETUP_COST[p, c],
                    ACTIVE_CYCLE[p, c],
                    gap_at_day,
                    time_units_in_a_day,
                    end_shift,
                    horizon_days,
                )
            gaps_setup[p, c] = SETUP_GAP
            model.Add(SETUP_COST[p, c] == (BASE_SETUP_COST[p, c] + SETUP_GAP))
            # constraints.append(f"cost (time) of machine setup operation {SETUP_COST[p,c]} == {BASE_SETUP_COST[p,c]} + {SETUP_GAP}")

    # cost (time) of machine load operation
    LOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LOAD_COST[p, c, l] = model.NewIntVar(
                    0, prod.due_date, f"LOAD_COST[{p},{c},{l}]"
                )
                # behaviour
                if USE_ADD_ELEMENT:
                    LOAD_GAP = make_gap_var(
                        LOAD_BEG[p, c, l],
                        BASE_LOAD_COST[p, c, l],
                        ACTIVE_LEVATA[p, c, l],
                    )
                else:
                    LOAD_GAP = make_gap_var_linear(
                        model,
                        LOAD_BEG[p, c, l],
                        BASE_LOAD_COST[p, c, l],
                        ACTIVE_LEVATA[p, c, l],
                        gap_at_day,
                        time_units_in_a_day,
                        end_shift,
                        horizon_days,
                    )
                gaps_load[p, c, l] = LOAD_GAP
                model.Add(LOAD_COST[p, c, l] == (BASE_LOAD_COST[p, c, l] + LOAD_GAP))
                # constraints.append(f"cost (time) of machine load operation {LOAD_COST[p,c,l]} == {BASE_LOAD_COST[p,c,l]} + {LOAD_GAP}")
    # cost (time) of machine unload operation
    UNLOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                UNLOAD_COST[p, c, l] = model.NewIntVar(
                    0, prod.due_date, f"UNLOAD_COST[{p},{c},{l}]"
                )
                # behaviour
                if USE_ADD_ELEMENT:
                    UNLOAD_GAP = make_gap_var(
                        UNLOAD_BEG[p, c, l],
                        BASE_UNLOAD_COST[p, c, l],
                        ACTIVE_LEVATA[p, c, l],
                    )
                else:
                    UNLOAD_GAP = make_gap_var_linear(
                        model,
                        UNLOAD_BEG[p, c, l],
                        BASE_UNLOAD_COST[p, c, l],
                        ACTIVE_LEVATA[p, c, l],
                        gap_at_day,
                        time_units_in_a_day,
                        end_shift,
                        horizon_days,
                    )
                gaps_unload[p, c, l] = UNLOAD_GAP
                model.Add(
                    UNLOAD_COST[p, c, l] == (BASE_UNLOAD_COST[p, c, l] + UNLOAD_GAP)
                )
                # constraints.append(f"cost (time) of machine unload operation {UNLOAD_COST[p,c,l]} == {BASE_UNLOAD_COST[p,c,l]} + {UNLOAD_GAP}")

    # end times for setup
    SETUP_END = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            SETUP_END[p, c] = model.NewIntVarFromDomain(
                worktime_domain[p], f"SETUP_END[{p},{c}]"
            )
            # behaviour
            model.Add(SETUP_END[p, c] == SETUP_BEG[p, c] + SETUP_COST[p, c])
            # constraints.append(f"end times for setup {SETUP_END[p,c]} == {SETUP_BEG[p,c]} + {SETUP_COST[p,c]}")
    # end time for load
    LOAD_END = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LOAD_END[p, c, l] = model.NewIntVarFromDomain(
                    worktime_domain[p], f"LOAD_END[{p},{c},{l}]"
                )
                # behaviour
                model.Add(LOAD_END[p, c, l] == LOAD_BEG[p, c, l] + LOAD_COST[p, c, l])
                # constraints.append(f"end time for load {LOAD_END[p,c,l]} == {LOAD_BEG[p,c,l]} + {LOAD_COST[p,c,l]}")
    # end times for unload
    UNLOAD_END = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                UNLOAD_END[p, c, l] = model.NewIntVarFromDomain(
                    worktime_domain[p], f"UNLOAD_END[{p},{c},{l}]"
                )
                # behaviour
                model.Add(
                    UNLOAD_END[p, c, l] == UNLOAD_BEG[p, c, l] + UNLOAD_COST[p, c, l]
                )
                # constraints.append(f"end times for unload {UNLOAD_END[p,c,l]} == {UNLOAD_BEG[p,c,l]} + {UNLOAD_COST[p,c,l]}")

    # Aliases for cycle Beginning and End (for readability)
    CYCLE_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            CYCLE_BEG[p, c] = SETUP_BEG[p, c]
    CYCLE_END = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            CYCLE_END[p, c] = UNLOAD_END[p, c, standard_levate[p] - 1]
    # Cycle cost
    CYCLE_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            CYCLE_COST[p, c] = model.NewIntVar(0, prod.due_date, f"CYCLE_COST[{p},{c}]")
            # behaviour
            model.Add(
                CYCLE_COST[p, c] == CYCLE_END[p, c] - CYCLE_BEG[p, c]
            ).OnlyEnforceIf(ACTIVE_CYCLE[p, c])
            # constraints.append(f"Cycle cost {CYCLE_COST[p,c]} == {CYCLE_END[p,c]} - {CYCLE_BEG[p,c]}")

    # number of kg produced by a cycle
    KG_CYCLE = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            KG_CYCLE[p, c] = model.NewIntVar(
                0, best_kg_cycle[p] * standard_levate[p], f"KG_CYCLE[{p},{c}]"
            )
            # behaviour
            ACTUAL_KG_PER_LEVATA = model.NewIntVar(
                0,
                int(best_kg_cycle[p] / standard_levate[p]),
                f"ACTUAL_KG_PER_LEVATA[{p},{c}]",
            )
            model.Add(
                ACTUAL_KG_PER_LEVATA
                == sum(
                    [A[p, c, m] * kg_per_levata[m, p] for m in prod_to_machine_comp[p]]
                )
            )
            model.Add(ACTUAL_KG_PER_LEVATA == 0).OnlyEnforceIf(ACTIVE_CYCLE[p, c].Not())
            # constraints.append(f"number of kg produced by a cycle 1")
            model.AddMultiplicationEquality(
                KG_CYCLE[p, c], NUM_LEVATE[p, c], ACTUAL_KG_PER_LEVATA
            )
            model.Add(KG_CYCLE[p, c] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p, c].Not())
            # constraints.append(f"number of kg produced by a cycle 2")

    """
        CONSTRAINTS (search space reduction)
        """
    # Left tightness to search space
    for p, _ in all_products:
        if max_cycles[p] > 1:
            for c in range(max_cycles[p] - 1):
                model.AddImplication(COMPLETE[p, c + 1], COMPLETE[p, c])
                # constraints.append(f"Left tightness to search space {COMPLETE[p,c]} >= {COMPLETE[p,c+1]}")
                model.AddImplication(ACTIVE_CYCLE[p, c + 1], ACTIVE_CYCLE[p, c])
                # constraints.append(f"Left tightness to search space {ACTIVE_CYCLE[p,c]} >= {ACTIVE_CYCLE[p,c+1]}")

    """
        CONSTRAINTS (LeMur specific)
        """
    # 1 : Cycle machine assignment
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 1.1 An active cycle must have one and only one machine assigned
            #   !!! note !!! : for some reason, BoolXor and ExactlyOne don't support the OnlyEnforceIf (documentation says that) only BoolAnd / BoolOr does
            model.AddBoolXOr(
                [A[p, c, m] for m in prod_to_machine_comp[p]]
                + [ACTIVE_CYCLE[p, c].Not()]
            )
            # constraints.append(f"A non active cycle must have 0 machines assigned")

    # 2 : At most one partial cycle per product
    for p, prod in all_products:
        model.AddAtMostOne([PARTIAL_CYCLE[p, c] for c in range(max_cycles[p])])
        # constraints.append(f"At most one partial cycle per product")

    # 3 : Connect cycle specific variables
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 3.1 The complete cycles must be active (only implication to allow for partials)
            model.AddImplication(COMPLETE[p, c], ACTIVE_CYCLE[p, c])
            model.AddImplication(PARTIAL_CYCLE[p, c], ACTIVE_CYCLE[p, c])
            model.AddImplication(ACTIVE_CYCLE[p, c].Not(), PARTIAL_CYCLE[p, c].Not())
            model.AddImplication(ACTIVE_CYCLE[p, c].Not(), COMPLETE[p, c].Not())
            model.AddImplication(COMPLETE[p, c], PARTIAL_CYCLE[p, c].Not())
            # constraints.append(f"The complete cycles must be active {COMPLETE[p,c]} => {ACTIVE_CYCLE[p,c]}")
            # 3.2 The partial cycle is the active but not complete
            # (this carries the atmost one from partial to active so it needs to be a iff)
            model.AddBoolAnd(ACTIVE_CYCLE[p, c], COMPLETE[p, c].Not()).OnlyEnforceIf(
                PARTIAL_CYCLE[p, c]
            )
            # constraints.append(f"The partial cycle is the active but not complete {ACTIVE_CYCLE[p,c]} && {COMPLETE[p,c].Not()} => {PARTIAL_CYCLE[p,c]}")
            model.AddBoolOr(ACTIVE_CYCLE[p, c].Not(), COMPLETE[p, c]).OnlyEnforceIf(
                PARTIAL_CYCLE[p, c].Not()
            )
            # constraints.append(f"The partial cycle is the active but not complete {ACTIVE_CYCLE[p,c].Not()} || {COMPLETE[p,c]} => {PARTIAL_CYCLE[p,c].Not()}")

    # 4 : Tie number of levate to cycles
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 4.1 : If the cycle is complete, then the number of levate is the maximum one
            model.Add(NUM_LEVATE[p, c] == standard_levate[p]).OnlyEnforceIf(
                COMPLETE[p, c]
            )
            # constraints.append(f"If the cycle is complete, then the number of levate is the maximum one {NUM_LEVATE[p,c]} == {standard_levate[p]}")
            # 4.2 : If the cycle is not active the number of levate is 0
            model.Add(NUM_LEVATE[p, c] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p, c].Not())
            # constraints.append(f"If the cycle is not active the number of levate is 0 {NUM_LEVATE[p,c]} == 0")
            # 4.3 : If partial, then we search for the number of levate
            model.AddLinearConstraint(
                NUM_LEVATE[p, c], lb=1, ub=(standard_levate[p] - 1)
            ).OnlyEnforceIf(PARTIAL_CYCLE[p, c])
            # constraints.append(f"If partial, then we search for the number of levate 1 <= {NUM_LEVATE[p,c]} <= {standard_levate[p]-1}")

    # 5 : Start date / Due date - (Defined at domain level)
    #

    # 6. Objective : all products must reach the requested production
    for p, prod in all_products:
        total_production = model.NewIntVar(
            prod.kg_request,
            prod.kg_request + best_kg_cycle[p],
            f"TOTAL_PRODUCTION[{p}]",
        )
        model.Add(
            total_production == sum([KG_CYCLE[p, c] for c in range(max_cycles[p])])
        )
        # constraints.append(f"Objective : all products must reach the requested production {total_production} == {prod.kg_request} + {best_kg_cycle[p]}")

    # 7. Define ordering between time variables
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p, c, l) not in LOAD_EXCL:
                    # 7.1 : Load (Common case)
                    if l == 0:
                        model.Add(LOAD_BEG[p, c, l] >= SETUP_END[p, c]).OnlyEnforceIf(
                            ACTIVE_LEVATA[p, c, l]
                        )
                        # constraints.append(f"Load (Common case) {LOAD_BEG[p,c,l]} >= {SETUP_END[p,c]}")
                    else:
                        model.Add(
                            LOAD_BEG[p, c, l] == UNLOAD_END[p, c, l - 1]
                        ).OnlyEnforceIf(ACTIVE_LEVATA[p, c, l])
                        # constraints.append(f"Load (Common case) {LOAD_BEG[p,c,l]} == {UNLOAD_END[p,c,l-1]}")

                if (p, c, l) not in UNLOAD_EXCL:
                    # 7.2 : Unload (Common case)
                    model.Add(
                        UNLOAD_BEG[p, c, l] >= LOAD_END[p, c, l] + LEVATA_COST[p, c, l]
                    ).OnlyEnforceIf(ACTIVE_LEVATA[p, c, l])
                    # constraints.append(f"Unload (Common case) {UNLOAD_BEG[p,c,l]} >= {LOAD_END[p,c,l]} + {LEVATA_COST[p,c,l]}")

    # 7.3 : Partial Loads / Unloads
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if l > 0:
                    if (p, c, l) not in LOAD_EXCL:
                        # Copy previous Load values
                        model.Add(
                            LOAD_BEG[p, c, l] == LOAD_END[p, c, l - 1]
                        ).OnlyEnforceIf(
                            ACTIVE_CYCLE[p, c], ACTIVE_LEVATA[p, c, l].Not()
                        )
                        # constraints.append(f"Partial Loads / Unloads {LOAD_BEG[p,c,l]} == {LOAD_END[p,c,l-1]}")
                        model.Add(
                            LOAD_END[p, c, l] == LOAD_END[p, c, l - 1]
                        ).OnlyEnforceIf(
                            ACTIVE_CYCLE[p, c], ACTIVE_LEVATA[p, c, l].Not()
                        )
                        # constraints.append(f"Partial Loads / Unloads {LOAD_END[p,c,l]} == {LOAD_END[p,c,l-1]}")
                    if (p, c, l) not in UNLOAD_EXCL:
                        # Copy previous Unload values
                        model.Add(
                            UNLOAD_BEG[p, c, l] == UNLOAD_END[p, c, l - 1]
                        ).OnlyEnforceIf(
                            ACTIVE_CYCLE[p, c], ACTIVE_LEVATA[p, c, l].Not()
                        )
                        # constraints.append(f"Partial Loads / Unloads {UNLOAD_BEG[p,c,l]} == {UNLOAD_END[p,c,l-1]}")
                        model.Add(
                            UNLOAD_END[p, c, l] == UNLOAD_END[p, c, l - 1]
                        ).OnlyEnforceIf(
                            ACTIVE_CYCLE[p, c], ACTIVE_LEVATA[p, c, l].Not()
                        )
                        # constraints.append(f"Partial Loads / Unloads {UNLOAD_END[p,c,l]} == {UNLOAD_END[p,c,l-1]}")

    # 7.4 : Inactive cycles
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                model.Add(LOAD_BEG[p, c, l] == start_schedule).OnlyEnforceIf(
                    ACTIVE_CYCLE[p, c].Not()
                )
                # constraints.append(f"Inactive cycles {LOAD_BEG[p,c,l]} == {start_schedule}")
                model.Add(UNLOAD_BEG[p, c, l] == start_schedule).OnlyEnforceIf(
                    ACTIVE_CYCLE[p, c].Not()
                )
                # constraints.append(f"Inactive cycles {UNLOAD_BEG[p,c,l]} == {start_schedule}")

    # 8. No overlap between product cycles on same machine :
    for m in range(1, num_machines + 1):
        machine_intervals = [
            model.NewOptionalIntervalVar(
                CYCLE_BEG[p, c],
                CYCLE_COST[p, c],
                CYCLE_END[p, c],
                is_present=A[p, c, m],
                name=f"machine_{m}_interval[{p},{c}]",
            )
            for p in machine_to_prod_comp[m]
            for c in range(max_cycles[p])
            if m in prod_to_machine_comp[p]
        ]

        # 8.1 Add maintanance intervals
        if m in scheduled_maintenances:
            for maintanance in scheduled_maintenances[m]:
                print(f"Adding maintanance interval for machine {m} : {maintanance}")
                machine_intervals.append(
                    model.NewFixedSizeIntervalVar(
                        maintanance[0],
                        maintanance[1],
                        f"machine_{m}_maintanance[{maintanance[0]},{maintanance[1]}]",
                    )
                )

        model.AddNoOverlap(machine_intervals)
        # constraints.append(f"No overlap between product cycles on same machine {machine_intervals}")

    # 9. Operators constraints
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 9.1 The active cycles' setups must be assigned to one operator
            model.AddBoolXOr(
                [A_OP_SETUP[o, p, c] for o in range(num_operator_groups)]
                + [ACTIVE_CYCLE[p, c].Not()]
            )
            # constraints.append(f"The active cycles' setups must be assigned to one operator {sum([A_OP_SETUP[o,p,c] for o in range(num_operator_groups)])} == 1")

    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                for t in [0, 1]:
                    # 9.3 The levate must have an operator assigned for the load and the unload operation:
                    model.AddBoolXOr(
                        [A_OP[o, p, c, l, t] for o in range(num_operator_groups)]
                        + [ACTIVE_LEVATA[p, c, l].Not()]
                    )
                    # constraints.append(f"The levate must have an operator assigned for the load and the unload operation {sum([A_OP[o,p,c,l,t] for o in range(num_operator_groups)])} == 1")

    for o in range(num_operator_groups):
        # 9.5 create intervals for each operation operators have to handle:
        setup_intervals = [
            model.NewOptionalIntervalVar(
                SETUP_BEG[p, c],
                SETUP_COST[p, c],
                SETUP_END[p, c],
                A_OP_SETUP[o, p, c],
                f"op[{o}]_setup_int[{p},{c}]",
            )
            for p, _ in all_products
            for c in range(max_cycles[p])
        ]
        load_intervals = [
            model.NewOptionalIntervalVar(
                LOAD_BEG[p, c, l],
                LOAD_COST[p, c, l],
                LOAD_END[p, c, l],
                A_OP[o, p, c, l, t],
                f"op[{o}]_load_int[{p},{c},{l},{t}]",
            )
            for p, _ in all_products
            for c in range(max_cycles[p])
            for l in range(standard_levate[p])
            for t in [0]
        ]
        unload_intervals = [
            model.NewOptionalIntervalVar(
                UNLOAD_BEG[p, c, l],
                UNLOAD_COST[p, c, l],
                UNLOAD_END[p, c, l],
                A_OP[o, p, c, l, t],
                f"op[{o}]_unload_int[{p},{c},{l},{t}]",
            )
            for p, _ in all_products
            for c in range(max_cycles[p])
            for l in range(standard_levate[p])
            for t in [1]
        ]
        model.AddNoOverlap(setup_intervals + load_intervals + unload_intervals)
        # constraints.append(f"create intervals for each operation operators have to handle {setup_intervals + load_intervals + unload_intervals}")

    # 10. Handle initialization of running products.
    for p, prod in running_products:
        # Only first cycles / first levate needs adjustments
        cycle = lev = 0
        # Fix Machine assignment
        model.Add(A[p, cycle, prod.machine[cycle]] == 1)
        # constraints.append(f"Fix Machine assignment {A[p,cycle,prod.machine[cycle]]} == 1")
        # Fix Velocity
        model.Add(VELOCITY[p, cycle] == prod.velocity[cycle])
        # constraints.append(f"Fix Velocity {VELOCITY[p,cycle]} == {prod.velocity[cycle]}")
        # Fix Cycle
        model.Add(ACTIVE_CYCLE[p, cycle] == 1)
        # constraints.append(f"Fix Cycle {ACTIVE_CYCLE[p,cycle]} == 1")
        # Fix Levate
        print(
            f"Remaining Levate for {p} : {prod.remaining_levate} vs {standard_levate[p]}"
        )
        if prod.remaining_levate < standard_levate[p]:
            model.Add(PARTIAL_CYCLE[p, cycle] == 1)
            # constraints.append(f"Fix Levate {PARTIAL_CYCLE[p,cycle]} == 1")
            model.Add(COMPLETE[p, cycle] == 0)
            # constraints.append(f"Fix Levate {COMPLETE[p,cycle]} == 0")
            # model.Add(NUM_LEVATE[p,cycle] == prod.remaining_levate)
            continue
        else:
            model.Add(PARTIAL_CYCLE[p, cycle] == 0)
            # constraints.append(f"Fix Levate {PARTIAL_CYCLE[p,cycle]} == 0")
            model.Add(COMPLETE[p, cycle] == 1)
            # constraints.append(f"Fix Levate {COMPLETE[p,cycle]} == 1")

        # operator assignments
        if prod.current_op_type == 0:
            model.Add(A_OP_SETUP[prod.operator, p, cycle] == 1)
            # constraints.append(f"operator assignments {A_OP_SETUP[prod.operator,p,cycle]} == 1")
        elif prod.current_op_type == 1:
            model.Add(A_OP[prod.operator, p, cycle, lev, 0] == 1)
            # constraints.append(f"operator assignments {A_OP[prod.operator,p,cycle,lev,0]} == 1")
        elif prod.current_op_type == 3:
            model.Add(A_OP[prod.operator, p, cycle, lev, 1] == 1)
            # constraints.append(f"operator assignments {A_OP[prod.operator,p,cycle,lev,1]} == 1")

        # Load needs to be done or has been done prev.
        if prod.current_op_type >= 1:
            model.Add(BASE_SETUP_COST[p, cycle] == 0)  # zero previous cost
            # constraints.append(f"Load needs to be done or has been done prev. {BASE_SETUP_COST[p,cycle]} == 0")

        # Levata needs to be done or has been done prev.
        if prod.current_op_type >= 2:
            model.Add(BASE_LOAD_COST[p, cycle, lev] == 0)  # zero previous cost
            # constraints.append(f"Levata needs to be done or has been done prev. {BASE_LOAD_COST[p,cycle,lev]} == 0")

        # Unload needs to be done or has been done prev.
        if prod.current_op_type == 3:
            model.Add(LEVATA_COST[p, cycle, lev] == 0)  # zero previous cost
            # constraints.append(f"Unload needs to be done or has been done prev. {LEVATA_COST[p,cycle,lev]} == 0")

    """
        OBJECTIVE
        """
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan,
        [CYCLE_END[p, c] for p, _ in all_products for c in range(max_cycles[p])],
    )

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.log_search_progress = LOGGING
    solver.parameters.num_search_workers = os.cpu_count()
    # solver.parameters.add_lp_constraints_lazily = True
    solver.parameters.stop_after_first_solution = STOP_AT_FIRST_SOLUTION

    model.Minimize(makespan)
    print("-----------------------------------------------------")
    print("Searching...")
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("1° stage optimization (Makespan minimization) over.")
        print(f"    Status: {solver.StatusName(status)}")
        print(f"    WallTime: {solver.WallTime()}s")
        print(f"    Objective value: {solver.ObjectiveValue()}")
        # fix as much as possible
        for p, prod in all_products:
            for c in range(max_cycles[p]):
                model.Add(NUM_LEVATE[p, c] == solver.Value(NUM_LEVATE[p, c]))
                model.Add(PARTIAL_CYCLE[p, c] == solver.Value(PARTIAL_CYCLE[p, c]))
                model.Add(COMPLETE[p, c] == solver.Value(COMPLETE[p, c]))
                model.Add(KG_CYCLE[p, c] == solver.Value(KG_CYCLE[p, c]))
                model.Add(ACTIVE_CYCLE[p, c] == solver.Value(ACTIVE_CYCLE[p, c]))
                model.Add(SETUP_END[p, c] <= solver.Value(SETUP_END[p, c]))
                model.Add(SETUP_BEG[p,c] == solver.Value(SETUP_BEG[p,c]))
                for l in range(standard_levate[p]):
                    model.Add(LOAD_END[p, c, l] <= solver.Value(LOAD_END[p, c, l]))
                    model.Add(UNLOAD_END[p, c, l] <= solver.Value(UNLOAD_END[p, c, l]))
                for m in prod_to_machine_comp[p]:
                    model.Add(A[p, c, m] == solver.Value(A[p, c, m]))
        stage_1_makespan = (
            solver.ObjectiveValue()
        )  # store 1° stage makespan apart for GA refinement

        solver_new = cp_model.CpSolver()
        # minimize makespan on each machine need to check A[p,c,m] for each product
        model.Minimize(
            sum(
                [CYCLE_END[p, c] for p, _ in all_products for c in range(max_cycles[p])]
            )
        )

        solver_new.parameters.max_time_in_seconds = timeout_cycle
        # avoid presolve
        solver_new.parameters.cp_model_presolve = PRESOLVE_SECOND_SEARCH
        solver_new.parameters.log_search_progress = LOGGING
        solver_new.parameters.num_search_workers = os.cpu_count()
        print("-----------------------------------------------------")
        print("Searching...")
        stat = solver_new.Solve(model)

        if stat == cp_model.OPTIMAL or stat == cp_model.FEASIBLE:
            status = stat
            solver = solver_new

    if (
        status == cp_model.UNKNOWN
        or status == cp_model.MODEL_INVALID
        or status == cp_model.INFEASIBLE
    ):
        print("ERROR : Solver status is unknown or model is invalid.")
        model_proto = model.Proto()
        variables = model_proto.variables
        constraints = model_proto.constraints
        raise ValueError(
            "Solver status is unknown or model is invalid.\n Hints => \n  Horizon days might be too short, try increasing it.\n  Input might be invalid.\nBreaking execution"
        )

    print("2° stage optimization (Compactness maximization) over.")
    print(f"    Status: {solver.StatusName(status)}")
    print(f"    WallTime: {solver.WallTime()}s")
    print(f"    Objective value: {solver.ObjectiveValue()}")
    
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            if solver.Value(ACTIVE_CYCLE[p, c]):
                # Store mahcine assignments
                for m in prod_to_machine_comp[p]:
                    if solver.Value(A[p, c, m]):
                        prod.machine[c] = m
                # Setup / Cycle
                for o in range(num_operator_groups):
                    if solver.Value(A_OP_SETUP[o, p, c]):
                        prod.setup_operator[c] = o
                        break
                prod.setup_beg[c] = solver.Value(SETUP_BEG[p, c])
                prod.setup_end[c] = solver.Value(SETUP_END[p, c])
                prod.setup_base_cost[c] = solver.Value(BASE_SETUP_COST[p, c])
                prod.setup_gap[c] = solver.Value(gaps_setup[p, c])
                prod.cycle_end[c] = solver.Value(CYCLE_END[p, c])
                # Velocity
                prod.velocity[c] = solver.Value(VELOCITY[p, c])
                # Num Levate
                prod.num_levate[c] = solver.Value(NUM_LEVATE[p, c])
                # Loads / Unloads
                for l in range(standard_levate[p]):
                    if solver.Value(ACTIVE_LEVATA[p, c, l]):
                        # load
                        for o in range(num_operator_groups):
                            if solver.Value(A_OP[o, p, c, l, 0]):
                                prod.load_operator[c, l] = o
                                break
                        # Time
                        prod.load_beg[c, l] = solver.Value(LOAD_BEG[p, c, l])
                        prod.load_end[c, l] = solver.Value(LOAD_END[p, c, l])
                        prod.load_base_cost[c, l] = solver.Value(
                            BASE_LOAD_COST[p, c, l]
                        )
                        prod.load_gap[c, l] = solver.Value(gaps_load[p, c, l])
                        # unload
                        for o in range(num_operator_groups):
                            if solver.Value(A_OP[o, p, c, l, 1]):
                                prod.unload_operator[c, l] = o
                                break
                        # Time
                        prod.unload_beg[c, l] = solver.Value(UNLOAD_BEG[p, c, l])
                        prod.unload_end[c, l] = solver.Value(UNLOAD_END[p, c, l])
                        prod.unload_base_cost[c, l] = solver.Value(
                            BASE_UNLOAD_COST[p, c, l]
                        )
                        prod.unload_gap[c, l] = solver.Value(gaps_unload[p, c, l])

    production_schedule = Schedule(all_products, invalid_intervals=prohibited_intervals)
    # save production schedule string form to file
    with open(OUTPUT_SCHEDULE, "w") as f:
        f.write(str(production_schedule))

    cp_sat_solution_cost = sum(
        [prod.cycle_end[c] for _, prod in all_products for c in prod.cycle_end.keys()]
    )

    """
        GENETIC REFINEMENT
    """
    if generations > 0:
        # Dictionary of variables necessary to GA computation
        vars = {
            # machine info
            "num_machines": num_machines,
            "fuses_machines_associations": fuses_machines_associations,
            "broken_machines": broken_machines,
            "scheduled_maintenances": scheduled_maintenances,
            "velocity_step_size": velocity_step_size,
            # costs
            "base_setup_cost": base_setup_cost,
            "base_load_cost": base_load_cost,
            "base_levata_cost": base_levata_cost,
            "base_unload_cost": base_unload_cost,
            # time related
            "makespan_limit": stage_1_makespan,
            "horizon": horizon,
            "start_shift": start_shift,
            "end_shift": end_shift,
            "time_units_in_a_day": time_units_in_a_day,
            "gap_at_day": gap_at_day,
            "prohibited_intervals": prohibited_intervals,
            "start_schedule": start_schedule,
            # products and operators
            "num_operator_groups": num_operator_groups,
            "prod_to_machine_comp": prod_to_machine_comp,
            "generations": generations,
        }
        # Refine it
        ga_refiner = GA_Refiner()
        print("-----------------------------------------------------")
        print("Searching...")
        refinement, validity, fitness = ga_refiner.refine_solution(all_products, vars)
        genetic_refinement_solution_cost = sum(
            [
                prod.cycle_end[c]
                for _, prod in all_products
                for c in prod.cycle_end.keys()
            ]
        )
        print("3° stage optimization (Genetic refinement) over.")
        print(f"    Valid solution: {validity}")
        print(f"    Objective value: {fitness}")
        if not validity:
            print("    (Invalid => CP-SAT solution will be used instead)")
        print("-----------------------------------------------------")
        print(
            f"    CP-SAT solution cost    :: {(cp_sat_solution_cost/time_units_in_a_day):.2f} days"
        )
        print(
            f"    Genetic refinement cost :: {(genetic_refinement_solution_cost/time_units_in_a_day):.2f} days"
        )
        print(
            f"                Improvement => {((cp_sat_solution_cost-genetic_refinement_solution_cost)/time_units_in_a_day):.2f} days"
        )

        # Plot refinement
        production_schedule = Schedule(
            refinement, invalid_intervals=prohibited_intervals
        )
        # save production schedule string form to file
        with open(OUTPUT_REFINED_SCHEDULE, "w") as f:
            f.write(str(production_schedule))

    print("-+-+- Scheduling completed -+-+-")

    return production_schedule


if __name__ == "__main__":
    COMMON_P_PATH = "input/new_orders.csv"
    RUNNING_P_PATH = "input/running_products.csv"
    J_COMPATIBILITY_PATH = "input/articoli_macchine.json"
    M_INFO_PATH = "input/macchine_info.json"
    ARTICLE_LIST_PATH = "input/lista_articoli.csv"

    broken_machines = []  # put here the number of the broken machines
    scheduled_maintenances = {
        # machine : [(start, duration), ...]
        # 1 : [(0, 10), (100, 24), ...],
    }
    festivities = [
        # day from scheduling start (0 is the current day, 1 is tomorrow, etc.)
        # 0,1,2,...
    ]

    hour_resolution = 1  # 1: hours, 2: half-hours, 4: quarter-hours, ..., 60: minutes
    horizon_days = 200
    time_units_in_a_day = (
        24 * hour_resolution
    )  # 24 : hours, 48 : half-hours, 96 : quarter-hours, ..., 1440 : minutes
    horizon = horizon_days * time_units_in_a_day

    start_shift = int(
        8 * hour_resolution
    )  # 8:00 MUST BE COMPATIBLE WITH time_units_in_a_day
    end_shift = int(
        16 * hour_resolution
    )  # 16:00 MUST BE COMPATIBLE WITH time_units_in_a_day
    num_operator_groups = 2
    num_operators_per_group = 4

    setup_cost = 4 * hour_resolution
    load_hours_fuso_person = 6 / 256 * hour_resolution
    unload_hours_fuso_person = 2 / 256 * hour_resolution

    costs = (
        setup_cost,  # Setup time
        load_hours_fuso_person / num_operators_per_group,  # Load time for 1 "fuso"
        unload_hours_fuso_person / num_operators_per_group,  # Unload time for 1 "fuso"
    )

    if machine_velocities % 2 == 0:
        raise ValueError("Machine velocities must be odd numbers.")

    (
        common_products,
        running_products,
        article_to_machine_comp,
        fuses_machines_associations,
        base_setup_art_cost,
        base_load_art_cost,
        base_unload_art_cost,
        base_levata_art_cost,
        standard_levate_art,
        kg_per_levata_art,
    ) = init_csv_data(
        COMMON_P_PATH,
        RUNNING_P_PATH,
        J_COMPATIBILITY_PATH,
        M_INFO_PATH,
        ARTICLE_LIST_PATH,
        costs=costs,
    )

    schedule = solve(
        common_products,
        running_products,
        article_to_machine_comp,
        fuses_machines_associations,
        base_setup_art_cost,
        base_load_art_cost,
        base_unload_art_cost,
        base_levata_art_cost,
        standard_levate_art,
        kg_per_levata_art,
        broken_machines,
        scheduled_maintenances,
        num_operator_groups,
        festivities,
        horizon_days,
        time_units_in_a_day,
        start_shift,
        end_shift,
    )
