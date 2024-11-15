from ortools.sat.python import cp_model
from google.protobuf import text_format
import argparse
import math

from data_init import *
from data_plotting import *
from utils import *
import os

# Parsing args.
parser = argparse.ArgumentParser(description='Job Scheduling Optimization')
parser.add_argument('--minimize', type=str, default='makespan', help="The optimization criterion: ['makespan'].")
args = parser.parse_args()
 
'''
INPUT DATA
'''

# timouts for solver
MAKESPAN_SOLVER_TIMEOUT = 600
CYCLE_OPTIM_SOLVER_TIMEOUT = 60

USE_ADD_ELEMENT = False

COMMON_P_PATH = 'data/new_orders.csv'
J_COMPATIBILITY_PATH = 'data/utils/articoli_macchine.json'
M_COMPATIBILITY_PATH = 'data/utils/macchine_articoli.json'
M_INFO_PATH = 'data/utils/macchine_info.json'
ARTICLE_LIST_PATH = 'data/valid/lista_articoli.csv'

constraints = []
broken_machines = [] # put here the number of the broken machines
scheduled_maintenances = {
    # machine : [(start, duration), ...]
    # 0 : [(50, 150)],
}

num_machines = 72
machine_velocities = 3

horizon_days = 60
time_units_in_a_day = 24   # 24 : hours, 48 : half-hours, 96 : quarter-hours, ..., 1440 : minutes
horizon = horizon_days * time_units_in_a_day

start_shift = 8       # 8:00 MUST BE COMPATIBLE WITH time_units_in_a_day
end_shift = 18        # 16:00 MUST BE COMPATIBLE WITH time_units_in_a_day
num_operator_groups = 2
num_operators_per_group = 2

if machine_velocities % 2 == 0 :
    raise ValueError("Machine velocities must be odd numbers.")

common_products, running_products, article_to_machine_comp, machine_to_article_comp, base_setup_art_cost, base_load_art_cost, base_unload_art_cost, base_levata_art_cost, standard_levate_art, kg_per_levata_art = init_csv_data(COMMON_P_PATH, J_COMPATIBILITY_PATH, M_COMPATIBILITY_PATH, M_INFO_PATH, ARTICLE_LIST_PATH, costs=(1, 1/256, 1/256))

for prod in common_products:
    for m in machine_to_article_comp:
        if kg_per_levata_art[m,prod.article] <= 0:
            print('no kg_per_levata_art found')
            print(f'm: {m} --- article: {prod.article}')
            exit(0)

# Make joint tuples (for readability purp.)
common_products = [(prod.id, prod) for prod in common_products]
running_products = [(prod.id, prod) for prod in running_products]
all_products = common_products + running_products

# convert_standard levate and kg to be indexed on product id
standard_levate={}
kg_per_levata={}
base_levata_cost={}
base_setup_cost={}
base_load_cost={}
base_unload_cost={}
for p, prod in all_products:
    standard_levate[p] = standard_levate_art[prod.article]
    base_levata_cost[p] = base_levata_art_cost[prod.article]
    for m in article_to_machine_comp[prod.article]:
        try:
            kg_per_levata[m,p] = kg_per_levata_art[m,prod.article]
            base_setup_cost[p,m] = base_setup_art_cost[prod.article,m]
            base_load_cost[p,m] = base_load_art_cost[prod.article,m]
            base_unload_cost[p,m] = base_unload_art_cost[prod.article,m]
        except:
            breakpoint()

# convert machine and article compatibility to be indexed on product id
prod_to_machine_comp = {}
machine_to_prod_comp = {}
for m in range(1,num_machines+1):
    machine_to_prod_comp[m] = []
for p, prod in all_products:
    prod_to_machine_comp[p] = article_to_machine_comp[prod.article]
    for m in article_to_machine_comp[prod.article]:
        machine_to_prod_comp[m].append(p)
# breakpoint()
'''
DERIVED CONSTANTS
'''
worktime_intervals, prohibited_intervals, gap_at_day, time_units_from_midnight, start_schedule = get_time_intervals(horizon_days, time_units_in_a_day, start_shift, end_shift)

# Velocity gears
velocity_levels = list(range(-(machine_velocities//2), (machine_velocities//2) + 1)) # machine velocities = 3 => [-1, 0, 1]
velocity_step_size = {}
for p, prod in all_products:
    velocity_step_size[p] = int((0.05 / max(1,max(velocity_levels))) * base_levata_cost[p])

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
    
    # check empty intervals
    valid = False
    for interval in adjusted_intervals:
        if interval[0] < interval[1]:
            valid = True
    if not valid:
        print("EMPTY DOMAIN FOR ARTICLE",prod.article)
        raise ValueError
    
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
                if m not in broken_machines:
                    A[p,c,m] = model.NewBoolVar(f'A[{p},{c},{m}]')
                else:
                    A[p,c,m] = model.NewConstant(0)

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
                SETUP_BEG[p,c] = model.NewConstant(start_schedule)
    
    # beginning of load operation
    LOAD_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p,c,l) not in LOAD_EXCL :
                    LOAD_BEG[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'LOAD_BEG[{p},{c},{l}]')
                else :
                    LOAD_BEG[p,c,l] = model.NewConstant(start_schedule)
    
    # beginning of unload operation
    UNLOAD_BEG = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p,c,l) not in UNLOAD_EXCL :
                    UNLOAD_BEG[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'UNLOAD_BEG[{p},{c},{l}]')
                else :
                    UNLOAD_BEG[p,c,l] = model.NewConstant(start_schedule)

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
                constraints.append(f"states if a levata is active (it exists) or not 1 {l} < {NUM_LEVATE[p,c]}")
                model.Add(l >= NUM_LEVATE[p,c]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
                constraints.append(f"states if a levata is active (it exists) or not 2 {l} >= {NUM_LEVATE[p,c]}")

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
                    model.Add(BASE_SETUP_COST[p,c] == base_setup_cost[p,m]).OnlyEnforceIf(A[p,c,m], ACTIVE_CYCLE[p,c])
                    constraints.append(f"Base cost of setup operation (accounts for machine specific setup) 1 {BASE_SETUP_COST[p,c]} == {base_setup_cost[p,m]}")
                model.Add(BASE_SETUP_COST[p,c] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())
                constraints.append(f"Base cost of setup operation (accounts for machine specific setup) 2 {BASE_SETUP_COST[p,c]} == 0")
            elif prod.current_op_type == 0 :
                # set as base cost the remaining for running products (if p is in setup)
                model.Add(BASE_SETUP_COST[p,c] == prod.remaining_time)
                constraints.append(f"Base cost of setup operation (accounts for machine specific setup) 3 {BASE_SETUP_COST[p,c]} == {prod.remaining_time}")

    # Base cost load operation (accounts for machine specific setup)
    BASE_LOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                BASE_LOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f"BASE_LOAD_COST[{p},{c},{l}]")
                # behaviour
                if (p,c,l) not in LOAD_EXCL :
                    for m in prod_to_machine_comp[p] :
                        model.Add(BASE_LOAD_COST[p,c,l] == base_load_cost[p,m]).OnlyEnforceIf(A[p,c,m], ACTIVE_LEVATA[p,c,l])
                        constraints.append(f"Base cost load operation (accounts for machine specific setup) 1 {BASE_LOAD_COST[p,c,l]} == {base_load_cost[p,m]}")
                    model.Add(BASE_LOAD_COST[p,c,l] == 0).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
                    constraints.append(f"Base cost load operation (accounts for machine specific setup) 2 {BASE_LOAD_COST[p,c,l]} == 0")
                elif prod.current_op_type == 1 :
                    # set as base cost the remaining for running products (if p is in load)
                    model.Add(BASE_LOAD_COST[p,c,l] == prod.remaining_time)
                    constraints.append(f"Base cost load operation (accounts for machine specific setup) 3 {BASE_LOAD_COST[p,c,l]} == {prod.remaining_time}")
    
    # Base cost unload operation (accounts for machine specific setup)
    BASE_UNLOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                BASE_UNLOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f"BASE_UNLOAD_COST[{p},{c},{l}]")
                # behaviour
                if (p,c,l) not in UNLOAD_EXCL :
                    for m in prod_to_machine_comp[p] :
                        model.Add(BASE_UNLOAD_COST[p,c,l] == base_unload_cost[p,m]).OnlyEnforceIf(A[p,c,m], ACTIVE_LEVATA[p,c,l])
                        constraints.append(f"Base cost unload operation (accounts for machine specific setup) 1 {BASE_UNLOAD_COST[p,c,l]} == {base_unload_cost[p,m]}")
                    model.Add(BASE_UNLOAD_COST[p,c,l] == 0).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
                    constraints.append(f"Base cost unload operation (accounts for machine specific setup) 2 {BASE_UNLOAD_COST[p,c,l]} == 0")
                elif prod.current_op_type == 3 :
                    # set as base cost the remaining for running products (if p is in unload)
                    model.Add(BASE_UNLOAD_COST[p,c,l] == prod.remaining_time)
                    constraints.append(f"Base cost unload operation (accounts for machine specific setup) 3 {BASE_UNLOAD_COST[p,c,l]} == {prod.remaining_time}")

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
                    constraints.append(f"cost (time) of levata operation 1 {LEVATA_COST[p,c,l]} == {base_levata_cost[p] - VELOCITY[p,c] * velocity_step_size[p]}")
                    model.Add(LEVATA_COST[p,c,l] == 0).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l].Not())
                    constraints.append(f"cost (time) of levata operation 2 {LEVATA_COST[p,c,l]} == 0")
                elif prod.current_op_type == 2 :
                    # set as base cost the remaining for running products (if theh're in levata)
                    model.Add(LEVATA_COST[p,c,l] == prod.remaining_time - VELOCITY[p,c]*velocity_step_size[p])
                    constraints.append(f"cost (time) of levata operation 3 {LEVATA_COST[p,c,l]} == {prod.remaining_time - VELOCITY[p,c]*velocity_step_size[p]}")

    def make_gap_var(BEGIN, BASE_COST, IS_ACTIVE, enabled=True):
        '''
        Returns a GAP variable to allow setup / load / unload operations
        to be performed over multiple days (if needed)
        '''
        if not enabled: return 0 # for testing purposes
        
        # Associate day index to relative gap size
        G = model.NewIntVar(0, horizon_days, 'G')
        GAP_SIZE = model.NewIntVar(0, max(gap_at_day), 'GAP_SIZE')
        model.AddElement(G, gap_at_day, GAP_SIZE)
        constraints.append(f"Associate day index to relative gap size {G} == {gap_at_day} {GAP_SIZE}")
        # Get current day
        model.AddDivisionEquality(G, BEGIN, time_units_in_a_day)
        constraints.append(f"Get current day {G} == {BEGIN} / {time_units_in_a_day}")
        UB = end_shift + time_units_in_a_day*G
        # Understand if such operation goes beyond the worktime
        NEEDS_GAP = model.NewBoolVar('NEEDS_GAP')
        model.Add((BEGIN + BASE_COST) > UB).OnlyEnforceIf(NEEDS_GAP, IS_ACTIVE)
        constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} > {UB}")
        model.Add((BEGIN + BASE_COST) <= UB).OnlyEnforceIf(NEEDS_GAP.Not(), IS_ACTIVE)
        constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} <= {UB}")
        # Associate to GAP if needed, otherwise it's zero
        GAP = model.NewIntVar(0, max(gap_at_day), 'GAP')
        model.Add(GAP == GAP_SIZE).OnlyEnforceIf(NEEDS_GAP, IS_ACTIVE)
        constraints.append(f"Associate to GAP if needed, otherwise it's zero {GAP} == {GAP_SIZE}")
        model.Add(GAP == 0).OnlyEnforceIf(NEEDS_GAP.Not(), IS_ACTIVE)
        constraints.append(f"Associate to GAP if needed, otherwise it's zero {GAP} == 0")
        # If not active, GAP is also zero
        model.Add(GAP == 0).OnlyEnforceIf(IS_ACTIVE.Not())
        constraints.append(f"If not active, GAP is also zero {GAP} == 0")
        
        return GAP

    def make_gap_var_linear(model, BEGIN, BASE_COST, IS_ACTIVE, gap_at_day, time_units_in_a_day, end_shift, horizon_days, constraints, enabled=True):
        '''
        Returns a GAP variable to allow setup / load / unload operations
        to be performed over multiple days (if needed) without using AddElement.
        '''
        if not enabled: return 0 # for testing purposes
        
        # Associate day index to relative gap size
        G = model.NewIntVar(0, horizon_days, 'G')
        GAP_SIZE = model.NewIntVar(0, max(gap_at_day), 'GAP_SIZE')
        for i, gap in enumerate(gap_at_day):
            G_i = model.NewBoolVar(f'G_{i}')
            model.Add(G == i).OnlyEnforceIf(G_i)
            model.Add(GAP_SIZE == gap).OnlyEnforceIf(G_i)
            constraints.append(f"Associate day index to relative gap size {GAP_SIZE} == {gap} {G} == {i}")

        # Get current day
        model.AddDivisionEquality(G, BEGIN, time_units_in_a_day)
        constraints.append(f"Get current day {G} == {BEGIN} / {time_units_in_a_day}")
        UB = end_shift + time_units_in_a_day*G

        # Understand if such operation goes beyond the worktime
        NEEDS_GAP = model.NewBoolVar('NEEDS_GAP')
        model.Add((BEGIN + BASE_COST) > UB).OnlyEnforceIf(NEEDS_GAP, IS_ACTIVE)
        constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} > {UB}")
        model.Add((BEGIN + BASE_COST) <= UB).OnlyEnforceIf(NEEDS_GAP.Not(), IS_ACTIVE)
        constraints.append(f"Understand if such operation goes beyond the worktime {BEGIN} + {BASE_COST} <= {UB}")
        
        # Associate to GAP if needed, otherwise it's zero
        GAP = model.NewIntVar(0, max(gap_at_day), 'GAP')
        model.Add(GAP == GAP_SIZE).OnlyEnforceIf(NEEDS_GAP, IS_ACTIVE)
        constraints.append(f"Associate to GAP if needed, otherwise it's zero {GAP} == {GAP_SIZE}")
        model.Add(GAP == 0).OnlyEnforceIf(NEEDS_GAP.Not(), IS_ACTIVE)
        constraints.append(f"Associate to GAP if needed, otherwise it's zero {GAP} == 0")
        # If not active, GAP is also zero
        model.Add(GAP == 0).OnlyEnforceIf(IS_ACTIVE.Not())
        constraints.append(f"If not active, GAP is also zero {GAP} == 0")

        return GAP


    # cost (time) of machine setup operation
    SETUP_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            SETUP_COST[p,c] = model.NewIntVar(0, prod.due_date, f'SETUP_COST[{p},{c}]')
            # behaviour
            if USE_ADD_ELEMENT:
                SETUP_GAP = make_gap_var(SETUP_BEG[p,c], BASE_SETUP_COST[p,c], ACTIVE_CYCLE[p,c])
            else:
                SETUP_GAP = make_gap_var_linear(model, SETUP_BEG[p,c], BASE_SETUP_COST[p,c], ACTIVE_CYCLE[p,c], gap_at_day, time_units_in_a_day, end_shift, horizon_days, constraints)
            model.Add(SETUP_COST[p,c] == BASE_SETUP_COST[p,c] + SETUP_GAP)
            constraints.append(f"cost (time) of machine setup operation {SETUP_COST[p,c]} == {BASE_SETUP_COST[p,c]} + {SETUP_GAP}")
    
    # cost (time) of machine load operation
    LOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f'LOAD_COST[{p},{c},{l}]')
                # behaviour
                if USE_ADD_ELEMENT:
                    LOAD_GAP = make_gap_var(LOAD_BEG[p,c,l], BASE_LOAD_COST[p,c,l], ACTIVE_LEVATA[p,c,l])
                else:
                    LOAD_GAP = make_gap_var_linear(model, LOAD_BEG[p,c,l], BASE_LOAD_COST[p,c,l], ACTIVE_LEVATA[p,c,l], gap_at_day, time_units_in_a_day, end_shift, horizon_days, constraints)
                model.Add(LOAD_COST[p,c,l] == BASE_LOAD_COST[p,c,l] + LOAD_GAP)
                constraints.append(f"cost (time) of machine load operation {LOAD_COST[p,c,l]} == {BASE_LOAD_COST[p,c,l]} + {LOAD_GAP}")
    # cost (time) of machine unload operation
    UNLOAD_COST = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                UNLOAD_COST[p,c,l] = model.NewIntVar(0, prod.due_date, f'UNLOAD_COST[{p},{c},{l}]')
                # behaviour
                if USE_ADD_ELEMENT:
                    UNLOAD_GAP = make_gap_var(UNLOAD_BEG[p,c,l], BASE_UNLOAD_COST[p,c,l], ACTIVE_LEVATA[p,c,l])
                else:
                    UNLOAD_GAP = make_gap_var_linear(model, UNLOAD_BEG[p,c,l], BASE_UNLOAD_COST[p,c,l], ACTIVE_LEVATA[p,c,l], gap_at_day, time_units_in_a_day, end_shift, horizon_days, constraints)
                model.Add(UNLOAD_COST[p,c,l] == BASE_UNLOAD_COST[p,c,l] + UNLOAD_GAP)
                constraints.append(f"cost (time) of machine unload operation {UNLOAD_COST[p,c,l]} == {BASE_UNLOAD_COST[p,c,l]} + {UNLOAD_GAP}")


    # end times for setup
    SETUP_END = {}
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            SETUP_END[p,c] = model.NewIntVarFromDomain(worktime_domain[p], f'SETUP_END[{p},{c}]')
            # behaviour
            model.Add(SETUP_END[p,c] == SETUP_BEG[p,c] + SETUP_COST[p,c])
            constraints.append(f"end times for setup {SETUP_END[p,c]} == {SETUP_BEG[p,c]} + {SETUP_COST[p,c]}")
    # end time for load
    LOAD_END = {}
    for p, _ in all_products :
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                LOAD_END[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'LOAD_END[{p},{c},{l}]')
                # behaviour
                model.Add(LOAD_END[p,c,l] == LOAD_BEG[p,c,l] + LOAD_COST[p,c,l])
                constraints.append(f"end time for load {LOAD_END[p,c,l]} == {LOAD_BEG[p,c,l]} + {LOAD_COST[p,c,l]}")
    # end times for unload
    UNLOAD_END = {}
    for p, _ in all_products :
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                UNLOAD_END[p,c,l] = model.NewIntVarFromDomain(worktime_domain[p], f'UNLOAD_END[{p},{c},{l}]')
                # behaviour
                model.Add(UNLOAD_END[p,c,l] == UNLOAD_BEG[p,c,l] + UNLOAD_COST[p,c,l])
                constraints.append(f"end times for unload {UNLOAD_END[p,c,l]} == {UNLOAD_BEG[p,c,l]} + {UNLOAD_COST[p,c,l]}")


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
            constraints.append(f"Cycle cost {CYCLE_COST[p,c]} == {CYCLE_END[p,c]} - {CYCLE_BEG[p,c]}")

    # number of operators for each group
    OPERATORS_PER_GROUP = {}
    for o in range(num_operator_groups):
        OPERATORS_PER_GROUP[o] = model.NewIntVar(0, 5, f'OPERATORS_PER_GROUP[{o}]')
        model.Add(OPERATORS_PER_GROUP[o] == num_operators_per_group)
        constraints.append(f"number of operators for each group {OPERATORS_PER_GROUP[o]} == {num_operators_per_group}")
    
    # number of kg produced by a cycle
    KG_CYCLE = {}
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            KG_CYCLE[p,c] = model.NewIntVar(0, prod.kg_request+best_kg_cycle[p], f'KG_CYCLE[{p},{c}]')
            # behaviour
            ACTUAL_KG_PER_LEVATA = model.NewIntVar(0, prod.kg_request, f'ACTUAL_KG_PER_LEVATA[{p},{c}]')
            model.Add(ACTUAL_KG_PER_LEVATA == sum([A[p,c,m]*kg_per_levata[m,p] for m in prod_to_machine_comp[p]]))
            constraints.append(f"number of kg produced by a cycle 1")
            model.AddMultiplicationEquality(KG_CYCLE[p,c], NUM_LEVATE[p,c], ACTUAL_KG_PER_LEVATA)
            constraints.append(f"number of kg produced by a cycle 2")


    '''
    CONSTRAINTS (search space reduction)
    '''
    # Left tightness to search space
    for p, _ in all_products:
        if max_cycles[p]>1:
            for c in range(max_cycles[p]-2):
                model.Add(COMPLETE[p,c] >= COMPLETE[p,c+1])
                constraints.append(f"Left tightness to search space {COMPLETE[p,c]} >= {COMPLETE[p,c+1]}")
                model.Add(ACTIVE_CYCLE[p,c] >= ACTIVE_CYCLE[p,c+1])
                constraints.append(f"Left tightness to search space {ACTIVE_CYCLE[p,c]} >= {ACTIVE_CYCLE[p,c+1]}")


    '''
    CONSTRAINTS (LeMur specific)
    '''
    # 1 : Cycle machne assignment
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 1.1 An active cycle must have one and only one machine assigned
            #   !!! note !!! : for some reason, BoolXor and ExactlyOne don't support the OnlyEnforceIf (documentation says that) only BoolAnd / BoolOr does
            model.AddBoolXOr([A[p,c,m] for m in prod_to_machine_comp[p]]+[ACTIVE_CYCLE[p,c].Not()])
            constraints.append(f"A non active cycle must have 0 machines assigned sum([A[{p},{c},{m}] for m in prod_to_machine_comp[{p}]]) == 1")

    # 2 : At most one partial cycle per product
    for p, prod in all_products:
        model.AddAtMostOne([PARTIAL_CYCLE[p,c] for c in range(max_cycles[p])])
        constraints.append(f"At most one partial cycle per product")

    # 3 : Connect cycle specific variables
    for p, _ in all_products:
        for c in range(max_cycles[p]):            
            # 3.1 The complete cycles must be active (only implication to allow for partials)
            model.AddImplication(COMPLETE[p,c], ACTIVE_CYCLE[p,c])
            constraints.append(f"The complete cycles must be active {COMPLETE[p,c]} => {ACTIVE_CYCLE[p,c]}")
            # 3.2 The partial cycle is the active but not complete
            # (this carries the atmost one from partial to active so it needs to be a iff)
            model.AddBoolAnd(ACTIVE_CYCLE[p,c], COMPLETE[p,c].Not()).OnlyEnforceIf(PARTIAL_CYCLE[p,c])
            constraints.append(f"The partial cycle is the active but not complete {ACTIVE_CYCLE[p,c]} && {COMPLETE[p,c].Not()} => {PARTIAL_CYCLE[p,c]}")
            model.AddBoolOr(ACTIVE_CYCLE[p,c].Not(), COMPLETE[p,c]).OnlyEnforceIf(PARTIAL_CYCLE[p,c].Not())
            constraints.append(f"The partial cycle is the active but not complete {ACTIVE_CYCLE[p,c].Not()} || {COMPLETE[p,c]} => {PARTIAL_CYCLE[p,c].Not()}")


    # 4 : Tie number of levate to cycles
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 4.1 : If the cycle is complete, then the number of levate is the maximum one
            model.Add(NUM_LEVATE[p,c] == standard_levate[p]).OnlyEnforceIf(COMPLETE[p,c])
            constraints.append(f"If the cycle is complete, then the number of levate is the maximum one {NUM_LEVATE[p,c]} == {standard_levate[p]}")
            # 4.2 : If the cycle is not active the number of levate is 0
            model.Add(NUM_LEVATE[p,c] == 0).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())
            constraints.append(f"If the cycle is not active the number of levate is 0 {NUM_LEVATE[p,c]} == 0")
            # 4.3 : If partial, then we search for the number of levate
            model.AddLinearConstraint(NUM_LEVATE[p,c], lb=1, ub=(standard_levate[p]-1)).OnlyEnforceIf(PARTIAL_CYCLE[p,c])
            constraints.append(f"If partial, then we search for the number of levate 1 <= {NUM_LEVATE[p,c]} <= {standard_levate[p]-1}")
    
    # 5 : Start date / Due date - (Defined at domain level)
    #

    # 6. Objective : all products must reach the requested production
    for p, prod in all_products:
        total_production = sum([KG_CYCLE[p,c] for c in range(max_cycles[p])])
        model.AddLinearConstraint(total_production, lb=prod.kg_request, ub=(prod.kg_request + best_kg_cycle[p]))
        constraints.append(f"Objective : all products must reach the requested production {total_production} == {prod.kg_request} + {best_kg_cycle[p]}")

    # 7. Define ordering between time variables
    for p, prod in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if (p,c,l) not in LOAD_EXCL :
                    # 7.1 : Load (Common case)
                    if l == 0 :
                        model.Add(LOAD_BEG[p,c,l] >= SETUP_END[p,c]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                        constraints.append(f"Load (Common case) {LOAD_BEG[p,c,l]} >= {SETUP_END[p,c]}")
                    else :
                        model.Add(LOAD_BEG[p,c,l] == UNLOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                        constraints.append(f"Load (Common case) {LOAD_BEG[p,c,l]} == {UNLOAD_END[p,c,l-1]}")
                
                if (p,c,l) not in UNLOAD_EXCL :
                    # 7.2 : Unload (Common case)
                    model.Add(UNLOAD_BEG[p,c,l] >= LOAD_END[p,c,l] + LEVATA_COST[p,c,l]).OnlyEnforceIf(ACTIVE_LEVATA[p,c,l])
                    constraints.append(f"Unload (Common case) {UNLOAD_BEG[p,c,l]} >= {LOAD_END[p,c,l]} + {LEVATA_COST[p,c,l]}")
    
    # 7.3 : Partial Loads / Unloads
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                if l > 0 :
                    if (p,c,l) not in LOAD_EXCL :
                        # Copy previous Load values
                        model.Add(LOAD_BEG[p,c,l] == LOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())
                        constraints.append(f"Partial Loads / Unloads {LOAD_BEG[p,c,l]} == {LOAD_END[p,c,l-1]}")
                        model.Add(LOAD_END[p,c,l] == LOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())
                        constraints.append(f"Partial Loads / Unloads {LOAD_END[p,c,l]} == {LOAD_END[p,c,l-1]}")
                    if (p,c,l) not in UNLOAD_EXCL :
                        # Copy previous Unload values
                        model.Add(UNLOAD_BEG[p,c,l] == UNLOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())
                        constraints.append(f"Partial Loads / Unloads {UNLOAD_BEG[p,c,l]} == {UNLOAD_END[p,c,l-1]}")
                        model.Add(UNLOAD_END[p,c,l] == UNLOAD_END[p,c,l-1]).OnlyEnforceIf(ACTIVE_CYCLE[p,c], ACTIVE_LEVATA[p,c,l].Not())
                        constraints.append(f"Partial Loads / Unloads {UNLOAD_END[p,c,l]} == {UNLOAD_END[p,c,l-1]}")

    # 7.4 : Inactive cycles
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                model.Add(LOAD_BEG[p,c,l] == start_schedule).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())
                constraints.append(f"Inactive cycles {LOAD_BEG[p,c,l]} == {start_schedule}")
                model.Add(UNLOAD_BEG[p,c,l] == start_schedule).OnlyEnforceIf(ACTIVE_CYCLE[p,c].Not())
                constraints.append(f"Inactive cycles {UNLOAD_BEG[p,c,l]} == {start_schedule}")

    # 8. No overlap between product cycles on same machine :
    for m in range(1,num_machines + 1):
        machine_intervals = [
            model.NewOptionalIntervalVar(
                CYCLE_BEG[p,c], 
                CYCLE_COST[p,c], 
                CYCLE_END[p,c],
                is_present=A[p,c,m], 
                name=f'machine_{m}_interval[{p},{c}]') 
            for p in machine_to_prod_comp[m]
                for c in range(max_cycles[p]) 
            if m in prod_to_machine_comp[p]]
        
        # 8.1 Add maintanance intervals
        if m in scheduled_maintenances :
            for maintanance in scheduled_maintenances[m]:
                print(f"Adding maintanance interval for machine {m} : {maintanance}")
                machine_intervals.append(model.NewFixedSizeIntervalVar(maintanance[0], maintanance[1], f'machine_{m}_maintanance[{maintanance[0]},{maintanance[1]}]'))
        
        model.AddNoOverlap(machine_intervals)
        constraints.append(f"No overlap between product cycles on same machine {machine_intervals}")
        

    
    # 9. Operators constraints
    for p, _ in all_products:
        for c in range(max_cycles[p]):
            # 9.1 The active cycles' setups must be assigned to one operator
            model.AddBoolXOr([A_OP_SETUP[o,p,c] for o in range(num_operator_groups)] + [ACTIVE_CYCLE[p,c].Not()])
            constraints.append(f"The active cycles' setups must be assigned to one operator {sum([A_OP_SETUP[o,p,c] for o in range(num_operator_groups)])} == 1")

    for p, _ in all_products:
        for c in range(max_cycles[p]):
            for l in range(standard_levate[p]):
                for t in [0,1]:
                    # 9.3 The levate must have an operator assigned for the load and the unload operation:
                    model.AddBoolXOr([A_OP[o,p,c,l,t] for o in range(num_operator_groups)] + [ACTIVE_LEVATA[p,c,l].Not()])
                    constraints.append(f"The levate must have an operator assigned for the load and the unload operation {sum([A_OP[o,p,c,l,t] for o in range(num_operator_groups)])} == 1")
    
    for o in range(num_operator_groups) :
        # 9.5 create intervals for each operation operators have to handle:
        setup_intervals = [model.NewOptionalIntervalVar(SETUP_BEG[p,c], SETUP_COST[p,c], SETUP_END[p,c], A_OP_SETUP[o,p,c], f"op[{o}]_setup_int[{p},{c}]") for p, _ in all_products for c in range(max_cycles[p])]
        load_intervals = [model.NewOptionalIntervalVar(LOAD_BEG[p,c,l], LOAD_COST[p,c,l], LOAD_END[p,c,l], A_OP[o,p,c,l,t], f"op[{o}]_load_int[{p},{c},{l},{t}]") for p, _ in all_products for c in range(max_cycles[p]) for l in range(standard_levate[p]) for t in [0]]
        unload_intervals = [model.NewOptionalIntervalVar(UNLOAD_BEG[p,c,l], UNLOAD_COST[p,c,l], UNLOAD_END[p,c,l], A_OP[o,p,c,l,t], f"op[{o}]_unload_int[{p},{c},{l},{t}]") for p, _ in all_products for c in range(max_cycles[p]) for l in range(standard_levate[p]) for t in [1]]
        model.AddNoOverlap(setup_intervals + load_intervals + unload_intervals)            
        constraints.append(f"create intervals for each operation operators have to handle {setup_intervals + load_intervals + unload_intervals}")

    # 10. Handle initialization of running products.
    for p, prod in running_products:
        # Only first cycles / first levate needs adjustments
        cycle = lev = 0
        # Fix Machine assignment
        model.Add(A[p,cycle,prod.machine[cycle]] == 1)
        constraints.append(f"Fix Machine assignment {A[p,cycle,prod.machine[cycle]]} == 1")
        # Fix Velocity
        model.Add(VELOCITY[p,cycle] == prod.velocity[cycle])
        constraints.append(f"Fix Velocity {VELOCITY[p,cycle]} == {prod.velocity[cycle]}")
        # Fix Cycle
        model.Add(ACTIVE_CYCLE[p,cycle] == 1)
        constraints.append(f"Fix Cycle {ACTIVE_CYCLE[p,cycle]} == 1")
        # Fix Levate
        print(f"Remaining Levate for {p} : {prod.remaining_levate} vs {standard_levate[p]}")
        if prod.remaining_levate < standard_levate[p] :
            model.Add(PARTIAL_CYCLE[p,cycle] == 1)
            constraints.append(f"Fix Levate {PARTIAL_CYCLE[p,cycle]} == 1")
            model.Add(COMPLETE[p,cycle] == 0)
            constraints.append(f"Fix Levate {COMPLETE[p,cycle]} == 0")
            # model.Add(NUM_LEVATE[p,cycle] == prod.remaining_levate)
            continue
        else :
            model.Add(PARTIAL_CYCLE[p,cycle] == 0)
            constraints.append(f"Fix Levate {PARTIAL_CYCLE[p,cycle]} == 0")
            model.Add(COMPLETE[p,cycle] == 1)
            constraints.append(f"Fix Levate {COMPLETE[p,cycle]} == 1")

        # operator assignments
        if prod.current_op_type == 0 :
            model.Add(A_OP_SETUP[prod.operator,p,cycle] == 1)
            constraints.append(f"operator assignments {A_OP_SETUP[prod.operator,p,cycle]} == 1")
        elif prod.current_op_type == 1 :
            model.Add(A_OP[prod.operator,p,cycle,lev,0] == 1)
            constraints.append(f"operator assignments {A_OP[prod.operator,p,cycle,lev,0]} == 1")
        elif prod.current_op_type == 3 :
            model.Add(A_OP[prod.operator,p,cycle,lev,1] == 1)
            constraints.append(f"operator assignments {A_OP[prod.operator,p,cycle,lev,1]} == 1")

        # Load needs to be done or has been done prev.
        if prod.current_op_type >= 1 :    
            model.Add(BASE_SETUP_COST[p,cycle] == 0) # zero previous cost
            constraints.append(f"Load needs to be done or has been done prev. {BASE_SETUP_COST[p,cycle]} == 0")
            
        # Levata needs to be done or has been done prev.
        if prod.current_op_type >= 2 : 
            model.Add(BASE_LOAD_COST[p,cycle,lev] == 0) # zero previous cost
            constraints.append(f"Levata needs to be done or has been done prev. {BASE_LOAD_COST[p,cycle,lev]} == 0")
            
        # Unload needs to be done or has been done prev.
        if prod.current_op_type == 3 :
            model.Add(LEVATA_COST[p,cycle,lev] == 0) # zero previous cost
            constraints.append(f"Unload needs to be done or has been done prev. {LEVATA_COST[p,cycle,lev]} == 0")

    '''
    OBJECTIVE
    '''

    # Objective based on argument
    if args.minimize == 'makespan':
        print("Criterion: Makespan")
        print("Searching for a solution...\n")
        makespan = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(makespan, [CYCLE_END[p,c] for p, _ in all_products for c in range(max_cycles[p])])
        # product end variables
    else :
        raise ValueError("Unsupported optimization criterion. Use 'tardiness' or 'makespan'.")

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAKESPAN_SOLVER_TIMEOUT
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = os.cpu_count()
    solver.parameters.add_lp_constraints_lazily = True
        
    model.Minimize(makespan)


    status = solver.Solve(model)


    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

        
        # fix as much as possible
        for p, prod in all_products:
            for c in range(max_cycles[p]):
                model.Add(NUM_LEVATE[p,c] == solver.Value(NUM_LEVATE[p,c]))
                model.Add(PARTIAL_CYCLE[p,c] == solver.Value(PARTIAL_CYCLE[p,c]))
                model.Add(COMPLETE[p,c] == solver.Value(COMPLETE[p,c]))
                model.Add(KG_CYCLE[p,c] == solver.Value(KG_CYCLE[p,c]))
                model.Add(ACTIVE_CYCLE[p,c] == solver.Value(ACTIVE_CYCLE[p,c]))
                for m in prod_to_machine_comp[p]:
                    model.Add(A[p,c,m] == solver.Value(A[p,c,m]))

        
        # minimize makespan on each machine need to check A[p,c,m] for each product
        model.Minimize(sum([CYCLE_END[p,c] for p, _ in all_products for c in range(max_cycles[p])]))
        
        solver.parameters.max_time_in_seconds = CYCLE_OPTIM_SOLVER_TIMEOUT
        # avoid presolve
        solver.parameters.cp_model_presolve = False

        stat = solver.Solve(model)

        if stat == cp_model.OPTIMAL or stat == cp_model.FEASIBLE:
            status = stat

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

        num_partials = sum([solver.Value(PARTIAL_CYCLE[p,c]) for p, _ in all_products for c in range(max_cycles[p])])
        production_schedule = Schedule(all_products)
        print(production_schedule)
        
        # Plot schedule
        plot_gantt_chart(production_schedule, max_cycles, num_machines, horizon, prohibited_intervals, time_units_from_midnight)
    else:
        print("No solution found. Try increasing the horizon days.")