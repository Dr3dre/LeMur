from ortools.sat.python import cp_model
import argparse
import math

from data_init import *
from data_plotting import *
from utils import *

# Parsing args.
parser = argparse.ArgumentParser(description='Job Scheduling Optimization')
parser.add_argument('--minimize', type=str, default='makespan', help="The optimization criterion: ['makespan'].")
args = parser.parse_args()
 
'''
INPUT DATA
'''

num_common_jobs = 8
num_running_jobs = 2
num_machines = 4
machine_velocities = 3

horizon_days = 90
time_units_in_a_day = 24   # 24 : hours, 48 : half-hours, 96 : quarter-hours, ..., 1440 : minutes
horizon = horizon_days * time_units_in_a_day

start_shift = 8       # 8:00
end_shift = 18        # 16:00
num_operator_groups = 2
num_operators_per_group = 2

if num_running_jobs > num_machines:
    raise ValueError("Number of jobs must be less than or equal to the number of machines.")
if machine_velocities % 2 == 0 :
    raise ValueError("Machine velocities must be odd numbers.")

# Data initialization (at random for the moment)
common_products, running_products, prod_to_machine_comp, machine_to_prod_comp, base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata = init_data(num_common_jobs, num_running_jobs, num_machines, num_operator_groups, horizon)

# Make joint tuples (for readability purp.)
common_products = [(prod.id, prod) for prod in common_products]
running_products = [(prod.id, prod) for prod in running_products]
all_products = common_products + running_products

'''
DERIVED CONSTANTS
'''
worktime_intervals, prohibited_intervals, gap_at_day, time_units_from_midnight = get_time_intervals(horizon_days, time_units_in_a_day, start_shift, end_shift)

# Velocity gears
velocity_levels = list(range(-(machine_velocities//2), (machine_velocities//2) + 1)) # machine velocities = 3 => [-1, 0, 1]
velocity_step_size = {}
for p, prod in all_products:
    velocity_step_size[p] = int((0.05 / max(velocity_levels)) * base_levata_cost[p])

# Generate custom domain excluding out of work hours, weekends, etc.
#   make it personalized for each product, excluding values < start_date[p] and > due_date[p]
worktime_domain = {}
for p, prod in all_products :
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
        adjusted_intervals[-1] = (adjusted_intervals[-1][0], min(adjusted_intervals[-1][1],prod.due_date))
    
    # generate worktime domain for p
    worktime_domain[p] = cp_model.Domain.FromIntervals(adjusted_intervals)
# domain for velocity gear steps
velocity_domain = cp_model.Domain.FromValues(velocity_levels)

# Max amount of cycles a product might go through (it's assigned to the slowest machine)
max_cycles = {}
best_kg_cycle = {}
for p, prod in all_products:
    max_cycles[p] =  max([math.ceil(prod.kg_request / (kg_per_levata[m,p] * standard_levate[p])) for m in prod_to_machine_comp[p]])
    best_kg_cycle[p] = max([math.ceil(kg_per_levata[m,p] * standard_levate[p]) for m in prod_to_machine_comp[p]])

'''
SETS
'''
# sets of running products requiring at hoc operations according to at which stage they're
SETUP_EXCL = [(p,c) for p, prod in all_products for c in range(max_cycles[p]) if (isinstance(prod, (RunningProduct)) and prod.current_op_type >= 0 and c == 0)]
LOAD_EXCL = [(p,c,l) for p, prod in all_products for c in range(max_cycles[p]) for l in range(standard_levate[p]) if (isinstance(prod, (RunningProduct)) and prod.current_op_type >= 1 and c == 0 and l == 0)]
LEVATA_EXCL = [(p,c,l) for p, prod in all_products for c in range(max_cycles[p]) for l in range(standard_levate[p]) if (isinstance(prod, (RunningProduct)) and prod.current_op_type >= 2 and c == 0 and l == 0)]
UNLOAD_EXCL = [(p,c,l) for p, prod in all_products for c in range(max_cycles[p]) for l in range(standard_levate[p]) if (isinstance(prod, (RunningProduct)) and prod.current_op_type == 3 and c == 0 and l == 0)]

if __name__ == '__main__':
    # Create the model
    model = cp_model.CpModel()
    print("Initializing model...")
    
    '''
    DECISION VARIBLES
    '''
    
    # Assignment variable (product, cycle, machine)
    A = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for m in prod_to_machine_comp[p]:
                A[p,c,m] = model.NewBoolVar(f'A[{p},{c},{m}]')
    # States if the cycle is a completion cycle
    COMPLETE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            COMPLETE[p,c] = model.NewBoolVar(f'COMPLETE[{p},{c}]')
    # Number of levate operations in a cycle
    NUM_LEVATE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            NUM_LEVATE[p,c] = model.NewIntVar(0, standard_levate[p], f'NUM_LEVATE[{p},{c}]')


    # beginning of setup operation
    SETUP_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            if (p,c) not in SETUP_EXCL :
                SETUP_BEG[p,c] = model.NewIntVarFromDomain(worktime_domain[p], f'SETUP_BEG[{p},{c}]')
            else :
                SETUP_BEG[p,c] = model.NewConstant(time_units_from_midnight)
    
    # beginning of load operation
    LOAD_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p,c,l) not in LOAD_EXCL :
                    LOAD_BEG[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'LOAD_BEG[{p},{c},{l}]')
                else :
                    LOAD_BEG[p,c,l] = model.NewConstant(time_units_from_midnight)
    
    # beginning of unload operation
    UNLOAD_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p,c,l) not in UNLOAD_EXCL :
                    UNLOAD_BEG[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'UNLOAD_BEG[{p},{c},{l}]')
                else :
                    UNLOAD_BEG[p,c,l] = model.NewConstant(time_units_from_midnight)

    # Velocity gear at which the cycle is performed
    VELOCITY = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            VELOCITY[p,c] = model.NewIntVarFromDomain(velocity_domain, f'VELOCITY[{p},{c}]')

    # Operator group assignment to setup operations
    A_OP_SETUP = {}
    for o in range(num_operator_groups):
        for p, prod in all_products:
            for c in range(max_cycles[p]):
                A_OP_SETUP[o,p,c] = model.NewBoolVar(f'A_OP_SETUP[{o},{p},{c}]')
                

    # Operator group assignment to load / unload operations of some levata
    A_OP = {}
    for o in range(num_operator_groups):
        for p, prod in all_products:
            for c in range(max_cycles[p]):
                for l in range(standard_levate[p]):
                    for t in [0,1]:
                        A_OP[o,p,c,l,t] = model.NewBoolVar(f'A_OP[{o},{p},{c},{l},{t}]')

    '''
    OTHER VARIABLES (no search needed, easily calculated)
    '''
    # states if a levata is active (it exists) or not
    ACTIVE_LEVATA = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                ACTIVE_LEVATA[p,c,l] = model.NewBoolVar(f'ACTIVE_LEVATA[{p},{c},{l}]')
                # behaviour
                model.Add(l < NUM_LEVATE[p,c]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                model.Add(l >= NUM_LEVATE[p,c]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())

    # states if a cycle is active (it exists) or not
    ACTIVE_CYCLE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # This isn't an acutal variable, it's done for readability
            ACTIVE_CYCLE[p,c] = ACTIVE_LEVATA[p,c,0]

    # states if a cycle is partial (not all levate are done)
    PARTIAL_CYCLE = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            PARTIAL_CYCLE[p,c] = model.NewBoolVar(f'PARTIAL_CYCLE[{p},{c}]')
            # behaviour
            # ... (defined in constraints LeMur specific) ...


    ##################
    # COST VARIABLES #
    ##################

    # Base cost of setup operation (accounts for machine specific setup)
    BASE_SETUP_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            BASE_SETUP_COST[p,c] = model.NewIntVar(0, prod.due_date, f"BASE_SETUP_COST[{p},{c}]")
            # behaviour
            if (p,c) not in SETUP_EXCL :
                for m in prod_to_machine_comp[p] :
                    model.Add(BASE_SETUP_COST[p,c] == base_setup_cost[p,m]).OnlyEnforceIf(A[p,c,m])
                model.Add(BASE_SETUP_COST[p,c] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())
            elif prod.current_op_type == 0 :
                # set as base cost the remaining for running products (if p is in setup)
                model.Add(BASE_SETUP_COST[p,c] == prod.remaining_time)

    # Base cost load operation (accounts for machine specific setup)
    BASE_LOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                BASE_LOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f"BASE_LOAD_COST[{p},{c},{l}]")
                # behaviour
                if (p,c,l) not in LOAD_EXCL :
                    for m in prod_to_machine_comp[p] :
                        model.Add(BASE_LOAD_COST[p,c,l] == base_load_cost[p,m]).OnlyEnforceIf(A[p,c,m])
                    model.Add(BASE_LOAD_COST[p,c,l] == 0).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
                elif prod.current_op_type == 1 :
                    # set as base cost the remaining for running products (if p is in load)
                    model.Add(BASE_LOAD_COST[p,c,l] == prod.remaining_time)
    
    # Base cost unload operation (accounts for machine specific setup)
    BASE_UNLOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                BASE_UNLOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f"BASE_UNLOAD_COST[{p},{c},{l}]")
                # behaviour
                if (p,c,l) not in UNLOAD_EXCL :
                    for m in prod_to_machine_comp[p] :
                        model.Add(BASE_UNLOAD_COST[p,c,l] == base_unload_cost[p,m]).OnlyEnforceIf(A[p,c,m])
                    model.Add(BASE_UNLOAD_COST[p,c,l] == 0).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
                elif prod.current_op_type == 3 :
                    # set as base cost the remaining for running products (if p is in unload)
                    model.Add(BASE_UNLOAD_COST[p,c,l] == prod.remaining_time)

    # cost (time) of levata operation
    #   Pay attention :
    #   there's no BASE_LEVATA_COST as LEVATA is an unsupervised
    #   operation, moreover it's independent to machine assignment
    LEVATA_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LEVATA_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f'LEVATA_COST[{p},{c},{l}]')
                if (p,c,l) not in LEVATA_EXCL :
                    # behaviour 
                    model.Add(LEVATA_COST[p,c,l] == base_levata_cost[p] - VELOCITY[p,c] * velocity_step_size[p]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                    model.Add(LEVATA_COST[p,c,l] == 0).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
                elif prod.current_op_type == 2 :
                    # set as base cost the remaining for running products (if theh're in levata)
                    model.Add(LEVATA_COST[p,c,l] == prod.remaining_time - VELOCITY[p,c]*velocity_step_size[p])

    def make_gap_var(BEGIN, BASE_COST, IS_ACTIVE, enabled=True):
        '''
        Returns a GAP variable to allow setup / load / unload operations
        to be performed over multiple days (if needed)
        '''
        if not enabled: return 0 # for testing purposes
        
        # Associate day index to relative gap size
        G = model.NewIntVar(0, horizon_days, 'G')
        GAP_SIZE = model.NewIntVar(0, max(gap_at_day), 'GAP_SIZE')
        model.AddElement(G, gap_at_day, GAP_SIZE)#.OnlyEnforceIf(IS_ACTIVE)
        # Get current day
        model.AddDivisionEquality(G, BEGIN, time_units_in_a_day)
        UB = end_shift + time_units_in_a_day*G
        # Understand if such operation goes beyond the worktime
        NEEDS_GAP = model.NewBoolVar('NEEDS_GAP')
        model.Add(BEGIN + BASE_COST > UB).OnlyEnforceIf(NEEDS_GAP, IS_ACTIVE)
        model.Add(BEGIN + BASE_COST <= UB).OnlyEnforceIf(NEEDS_GAP.Not(), IS_ACTIVE)
        # Associate to GAP if needed, otherwise it's zero
        GAP = model.NewIntVar(0, max(gap_at_day), 'GAP')
        model.Add(GAP == GAP_SIZE).OnlyEnforceIf(NEEDS_GAP, IS_ACTIVE)
        model.Add(GAP == 0).OnlyEnforceIf(NEEDS_GAP.Not(), IS_ACTIVE)
        # If not active, GAP is also zero
        model.Add(GAP == 0).OnlyEnforceIf(IS_ACTIVE.Not())
        
        return GAP

    # cost (time) of machine setup operation
    SETUP_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            SETUP_COST[p,c] = model.NewIntVar(0, prod.due_date, f'SETUP_COST[{p},{c}]')
            # behaviour
            SETUP_GAP = make_gap_var(SETUP_BEG[p,c], BASE_SETUP_COST[p,c], ACTIVE_CYCLE[p,c])
            model.Add(SETUP_COST[p,c] == BASE_SETUP_COST[p,c] + SETUP_GAP)
    
    # cost (time) of machine load operation
    LOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f'LOAD_COST[{p},{c},{l}]')
                # behaviour
                LOAD_GAP = make_gap_var(LOAD_BEG[p,c,l], BASE_LOAD_COST[p,c,l], ACTIVE_LEVATA[p,c,l])
                model.Add(LOAD_COST[p,c,l] == BASE_LOAD_COST[p,c,l] + LOAD_GAP)
    # cost (time) of machine unload operation
    UNLOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                UNLOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f'UNLOAD_COST[{p},{c},{l}]')
                # behaviour
                UNLOAD_GAP = make_gap_var(UNLOAD_BEG[p,c,l], BASE_UNLOAD_COST[p,c,l], ACTIVE_LEVATA[p,c,l])
                model.Add(UNLOAD_COST[p,c,l] == BASE_UNLOAD_COST[p,c,l] + UNLOAD_GAP)


    # end times for setup
    SETUP_END = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            SETUP_END[p,c] = model.NewIntVarFromDomain(worktime_domain[p], f'SETUP_END[{p},{c}]')
            # behaviour
            model.Add(SETUP_END[p,c] == SETUP_BEG[p,c] + SETUP_COST[p,c])
    # end time for load
    LOAD_END = {}
    for p, _ in all_products :
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LOAD_END[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'LOAD_END[{p},{c},{l}]')
                # behaviour
                model.Add(LOAD_END[p,c,l] == LOAD_BEG[p,c,l] + LOAD_COST[p,c,l])
    # end times for unload
    UNLOAD_END = {}
    for p, _ in all_products :
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                UNLOAD_END[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'UNLOAD_END[{p},{c},{l}]')
                # behaviour
                model.Add(UNLOAD_END[p,c,l] == UNLOAD_BEG[p,c,l] + UNLOAD_COST[p,c,l])


    # Aliases for cycle Beginning and End (for readability)
    CYCLE_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            CYCLE_BEG[p,c] = SETUP_BEG[p,c]
    CYCLE_END = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            CYCLE_END[p,c] = UNLOAD_END[p, c, standard_levate[p]-1]
    # Cycle cost
    CYCLE_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            CYCLE_COST[p,c] = model.NewIntVar(0, prod.due_date, f'CYCLE_COST[{p},{c}]')
            # behaviour
            model.Add(CYCLE_COST[p,c] == CYCLE_END[p,c] - CYCLE_BEG[p,c]).OnlyEnforceIf(ACTIVE_CYCLE[p,c])

    # number of operators for each group
    OPERATORS_PER_GROUP = {}
    for o in range(num_operator_groups):
        OPERATORS_PER_GROUP[o] = model.NewIntVar(0, 5, f'OPERATORS_PER_GROUP[{o}]')
        model.Add(OPERATORS_PER_GROUP[o] == num_operators_per_group)
    
    # number of kg produced by a cycle
    KG_CYCLE = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            KG_CYCLE[p,c] = model.NewIntVar(0, prod.kg_request+best_kg_cycle[p], f'KG_CYCLE[{p},{c}]')
            # behaviour
            ACTUAL_KG_PER_LEVATA = model.NewIntVar(0, prod.kg_request, f'ACTUAL_KG_PER_LEVATA[{p},{c}]')
            model.Add(ACTUAL_KG_PER_LEVATA == sum([A[p,c,m]*kg_per_levata[m,p] for m in prod_to_machine_comp[p]]))
            model.AddMultiplicationEquality(KG_CYCLE[p,c], NUM_LEVATE[p,c], ACTUAL_KG_PER_LEVATA)


    '''
    CONSTRAINTS (search space reduction)
    '''
    # Left tightness to search space
    for p, _ in all_products:
        for c in range(max_cycles[p]-1):
            model.Add(COMPLETE[p,c] >= COMPLETE[p,c+1])
            model.Add(ACTIVE_CYCLE[p,c] >= ACTIVE_CYCLE[p,c+1])


    '''
    CONSTRAINTS (LeMur specific)
    '''
    # 1 : Cycle machne assignment
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 1.1 An active cycle must have one and only one machine assigned
            #   !!! note !!! : for some reason, BoolXor and ExactlyOne don't support the OnlyEnforceIf (documentation says that) only BoolAnd / BoolOr does
            model.Add(sum([A[p,c,m] for m in prod_to_machine_comp[p]]) == 1).OnlyEnforceIf(ACTIVE_CYCLE[p,c])
            # 1.2 A non active cycle must have 0 machines assigned
            model.AddBoolAnd([A[p,c,m].Not() for m in prod_to_machine_comp[p]]).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())    

    # 2 : At most one partial cycle per product
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            if not (isinstance(prod, RunningProduct) and c == 0):
                model.AddAtMostOne([PARTIAL_CYCLE[p,c] for c in range(max_cycles[p])])
            else :
                # in the case of running products, we allow for 2 partial cycles
                model.Add(sum([PARTIAL_CYCLE[p,c] for c in range(max_cycles[p])]) <= 2)

    # 3 : Connect cycle specific variables
    for p, _ in all_products:
        for c in range(max_cycles[p]):            
            # 3.1 The complete cycles must be active (only implication to allow for partials)
            model.AddImplication(COMPLETE[p,c], ACTIVE_CYCLE[p,c])
            # 3.2 The partial cycle is the active but not complete
            # (this carries the atmost one from partial to active so it needs to be a iff)
            model.AddBoolAnd(ACTIVE_CYCLE[p,c], COMPLETE[p,c].Not()).OnlyEnforceIf(PARTIAL_CYCLE[p,c])
            model.AddBoolOr(ACTIVE_CYCLE[p,c].Not(), COMPLETE[p,c]).OnlyEnforceIf(PARTIAL_CYCLE[p,c].Not())


    # 4 : Tie number of levate to cycles 
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 4.1 : If the cycle is complete, then the number of levate is the maximum one
            model.Add(NUM_LEVATE[p,c] == standard_levate[p]).OnlyEnforceIf(COMPLETE[p,c])
            # 4.2 : If the cycle is not active the number of levate is 0
            model.Add(NUM_LEVATE[p,c] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())
            # 4.3 : If partial, then we search for the number of levate
            model.AddLinearConstraint(NUM_LEVATE[p,c], lb=1, ub=(standard_levate[p]-1)).OnlyEnforceIf(PARTIAL_CYCLE[p,c])
    
    # 5 : Start date / Due date - (Defined at domain level)
    #

    # 6. Objective : all products must reach the requested production
    for p, prod in all_products:
        total_production = sum([KG_CYCLE[p,c] for c in range(max_cycles[p])])
        model.AddLinearConstraint(total_production, lb=prod.kg_request, ub=(prod.kg_request + best_kg_cycle[p]))

    # 7. Define ordering between time variables
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p,c,l) not in LOAD_EXCL :
                    # 7.1 : Load (Common case)
                    if l == 0 :
                        model.Add(LOAD_BEG[p,c,l] >= SETUP_END[p,c]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                    else :
                        model.Add(LOAD_BEG[p,c,l] == UNLOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                
                if (p,c,l) not in UNLOAD_EXCL :
                    # 7.2 : Unload (Common case)
                    model.Add(UNLOAD_BEG[p,c,l] >= LOAD_END[p,c,l] + LEVATA_COST[p,c,l]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
    
    # 7.3 : Partial Loads / Unloads
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if l > 0 :
                    if (p,c,l) not in LOAD_EXCL :
                        # Copy previous Load values
                        model.Add(LOAD_BEG[p,c,l] == LOAD_BEG[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())
                        model.Add(LOAD_END[p,c,l] == LOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())
                    if (p,c,l) not in UNLOAD_EXCL :
                        # Copy previous Unload values
                        model.Add(UNLOAD_BEG[p,c,l] == UNLOAD_BEG[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())
                        model.Add(UNLOAD_END[p,c,l] == UNLOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())

    # 7.4 : Inactive cycles
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                model.Add(LOAD_BEG[p,c,l] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())
                model.Add(UNLOAD_BEG[p,c,l] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())

    # 8. No overlap between product cycles on same machine :
    for m in range(num_machines):
        machine_intervals = [model.NewOptionalIntervalVar(CYCLE_BEG[p,c], CYCLE_COST[p,c], CYCLE_END[p,c], A[p,c,m], f'machine_{m}_interval[{p},{c}]') for p in machine_to_prod_comp[m] for c in range(max_cycles[p])]
        model.AddNoOverlap(machine_intervals)
    
    # 9. Operators constraints
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 9.1 The active cycles' setups must be assigned to one operator
            model.Add(sum([A_OP_SETUP[o,p,c] for o in range(num_operator_groups)]) == 1).OnlyEnforceIf(ACTIVE_CYCLE[p,c])
            # 9.2 The non active cycles' setups must have no operator assigned
            model.AddBoolAnd([A_OP_SETUP[o,p,c].Not() for o in range(num_operator_groups)]).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())

    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                for t in [0,1]:
                    # 9.3 The levate must have an operator assigned for the load and the unload operation:
                    model.Add(sum([A_OP[o,p,c,l,t] for o in range(num_operator_groups)]) == 1).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                    # 9.4 The non active levate must have no operator assigned for the load and the unload operation:
                    model.AddBoolAnd([A_OP[o,p,c,l,t].Not() for o in range(num_operator_groups)]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
    
    for o in range(num_operator_groups) :
        # 9.5 create intervals for each operation operators have to handle:
        setup_intervals = [model.NewOptionalIntervalVar(SETUP_BEG[p,c], SETUP_COST[p,c], SETUP_END[p,c], A_OP_SETUP[o,p,c], f"op[{o}]_setup_int[{p},{c}]") for p, _ in all_products for c in range(max_cycles[p])]
        load_intervals = [model.NewOptionalIntervalVar(LOAD_BEG[p,c,l], LOAD_COST[p,c,l], LOAD_END[p,c,l], A_OP[o,p,c,l,t], f"op[{o}]_load_int[{p},{c},{l},{t}]") for p, _ in all_products for c in range(max_cycles[p]) for l in range(standard_levate[p]) for t in [0]]
        unload_intervals = [model.NewOptionalIntervalVar(UNLOAD_BEG[p,c,l], UNLOAD_COST[p,c,l], UNLOAD_END[p,c,l], A_OP[o,p,c,l,t], f"op[{o}]_unload_int[{p},{c},{l},{t}]") for p, _ in all_products for c in range(max_cycles[p]) for l in range(standard_levate[p]) for t in [1]]
        model.AddNoOverlap(setup_intervals + load_intervals + unload_intervals)            

    # 10. Handle initialization of running products.
    for p, prod in all_products:
        if isinstance(prod, RunningProduct):
            # Only first cycles / first levate needs adjustments
            c = l = 0
            # Fix Machine assignment
            model.Add(A[p,c,prod.machine[c]] == 1)
            # Fix Velocity
            model.Add(VELOCITY[p,c] == prod.velocity[c])
            
            # need to not count to the partial cycles, otherwise it can't work
            #model.Add(NUM_LEVATE[p,c] == prod.remaining_levate)
            # Also missing operator assignments
            
            # Load needs to be done or has been done prev.
            if prod.current_op_type >= 1 :    
                model.Add(BASE_SETUP_COST[p,c] == 0) # zero previous cost
            
            # Levata needs to be done or has been done prev.
            if prod.current_op_type >= 2 : 
                model.Add(BASE_LOAD_COST[p,c,l] == 0) # zero previous cost
            
            # Unload needs to be done or has been done prev.
            if prod.current_op_type == 3 :
                model.Add(LEVATA_COST[p,c,l] == 0) # zero previous cost

    '''
    OBJECTIVE
    '''

    # Objective based on argument
    if args.minimize == 'makespan':
        print("Criterion: Makespan")
        print("Searching for a solution...\n")
        makespan = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(makespan, [CYCLE_END[p,c] for p, _ in all_products for c in range(max_cycles[p])])
        model.Minimize(makespan)
    else :
        raise ValueError("Unsupported optimization criterion. Use 'tardiness' or 'makespan'.")

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    '''
    PLOTTING
    '''

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"    Status: {solver.StatusName(status)}")
        print(f"    WallTime: {solver.WallTime()}s")
        print(f"    Objective value: {solver.ObjectiveValue()}")
        print("---")        
        for p, prod in all_products:
            for c in range(max_cycles[p]):
                if solver.Value(ACTIVE_CYCLE[p,c]):
                    # Store mahcine assignments
                    for m in prod_to_machine_comp[p]:
                        if solver.Value(A[p,c,m]):
                            prod.machine[c] = m
                    # Setup / Cycle
                    prod.setup_beg[c] = solver.Value(SETUP_BEG[p,c])
                    prod.setup_end[c] = solver.Value(SETUP_END[p,c])
                    prod.cycle_end[c] = solver.Value(CYCLE_END[p,c])
                    # Velocity
                    prod.velocity[c] = solver.Value(VELOCITY[p,c])
                    # Num Levate
                    prod.num_levate[c] = solver.Value(NUM_LEVATE[p,c])
                    # Loads / Unloads
                    for l in range(standard_levate[p]):
                        if solver.Value(ACTIVE_LEVATA[p,c,l]):
                            prod.load_beg[c,l] = solver.Value(LOAD_BEG[p,c,l])
                            prod.load_end[c,l] = solver.Value(LOAD_END[p,c,l])
                            prod.unload_beg[c,l] = solver.Value(UNLOAD_BEG[p,c,l])
                            prod.unload_end[c,l] = solver.Value(UNLOAD_END[p,c,l])

        production_schedule = Schedule(all_products)
        print(production_schedule)
        
        # Plot schedule
        plot_gantt_chart(production_schedule, max_cycles, num_machines, horizon, prohibited_intervals, time_units_from_midnight)
    else:
        print("No solution found. Try increasing the horizon days.")