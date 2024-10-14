import copy
from itertools import combinations
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ortools.sat.python import cp_model


# ============================
# 1. Data Definitions
# ============================

SOLVER_TIME_LIMIT = 60  # Time limit in seconds for the solver

# Define Machines
MACHINES = ['M1', 'M2', 'M3']  # List of machine names

PRIORITY_LEVELS = 3  # Number of priority levels
# Define Orders
ORDERS = {
    'Order1': {
        'due_date': 300,  # in hours
        'products': {
            'ProductA': 200,
            'ProductB': 250
        },
        'priority': 0
    },
    'Order2': {
        'due_date': 300,  # in hours
        'products': {
            'ProductC': 450
        },
        'priority': 2
    },
    'Order3': {
        'due_date': 300,  # in hours
        'products': {
            'ProductA': 200,
            'ProductB': 250,
            'ProductC': 450
        },
        'priority': 1
    },
}

# Define Products with Cycles and Operations
PRODUCTS = {
    'ProductA': {
        'Kg_prod_per_levata': 100,
        'levate': [
            ['PrepareMachine', 'Op1'], #this is a "levata"
            ['RechargeMachine', 'Op1'],
        ]
    },
    'ProductB': {
        'Kg_prod_per_levata': 250,
        'levate': [
            ['PrepareMachine', 'Op2'],
            ['RechargeMachine', 'Op2'],
        ]
    },
    'ProductC': {
        'Kg_prod_per_levata': 150,
        'levate': [
            ['PrepareMachine', 'Op3'],
            ['RechargeMachine', 'Op3'],
            ['RechargeMachine', 'Op3'],
        ]
    }
}

# Define Operations with Machines, Base Times, and Supervision Requirement
OPERATION_SPECIFICATIONS = {
    'PrepareMachine': {'machines': ['M1', 'M2', 'M3'], 'base_time': {'M1': 5, 'M2': 3, 'M3': 5}, 'requires_supervision': True},
    'RechargeMachine': {'machines': ['M1', 'M2', 'M3'], 'base_time': {'M1': 4, 'M2': 2, 'M3': 3}, 'requires_supervision': True},
    'Op1': {'machines': ['M1', 'M3'], 'base_time': {'M1': 35, 'M3': 35}, 'requires_supervision': False},
    'Op2': {'machines': ['M2'], 'base_time': {'M2': 35}, 'requires_supervision': False},
    'Op3': {'machines': ['M1', 'M2'], 'base_time': {'M1': 35, 'M2': 35}, 'requires_supervision': False},
}

# Speed Adjustment Factors
SPEED_LEVELS = [0.95, 1.0, 1.05]

WORKING_HOURS = (8, 16)  # Start and end times for the day shift
START_DAY = 0  # Day of the week to start scheduling (0 = Monday)
FESTIVE_DAYS = []  # Saturday and Sunday are already non-working days, add more if needed

# Define Shift Parameters for Each Weekday
OPERATOR_GROUPS_SHIFT = 2  # Number of operators available per shift (define the number of parallel operations allowed)
OPERATORS_PER_GROUP = 1  # Number of operators in each group

ALREADY_EXECUTING = [
    {
        'Product': 'ProductA',
        'EndLevata': 1,
        'CurrentLevata': 0,
        'Operation': 1,
        'Machine': 'M1',
        'Speed': 1.0,
        'Completion': 5,
        'DueDate': 500
    }
]

# ============================
# 2. Preprocess Data
# ============================

# Scheduling Horizon
scheduling_horizon = max([order_info['due_date'] for order_info in ORDERS.values()])
scheduling_horizon_in_days = int(scheduling_horizon / 24) + 1

# add saturday and sunday as non-working days
for i in range(scheduling_horizon_in_days):
    day_of_week = (START_DAY + i) % 7
    if day_of_week == 5 or day_of_week == 6:
        FESTIVE_DAYS.append(i)

# Precompute shift start and end times for each day
working_shifts = []
for d in range(scheduling_horizon_in_days):
    if d not in FESTIVE_DAYS:
        working_shifts.extend([d * 24 + h for h in range(WORKING_HOURS[0], WORKING_HOURS[1])])

# Create unique products and operations per order
product_instances = {} # Dictionary to hold all products of all orders

order_product_keys = {}  # Dictionary to hold all product instances of all orders
operations_product = {} # Dictionary to hold all operations of all orders
operation_instances = []  # List to hold all operation instances

operation_max_time = {}  # Dictionary to hold the maximum time for each operation
operation_time_shift = {}  # Dictionary to hold operation start times for each shift

for order_id, order_info in ORDERS.items():
    order_product_keys[order_id] = {}
    for product, requested_quantity in order_info['products'].items():
        full_cycles = requested_quantity // (PRODUCTS[product]['Kg_prod_per_levata'] * len(PRODUCTS[product]['levate']))
        remaining_quantity = requested_quantity % (PRODUCTS[product]['Kg_prod_per_levata'] * len(PRODUCTS[product]['levate']))
        num_partial_cycle_levate = 0
        if remaining_quantity > 0.8 * (PRODUCTS[product]['Kg_prod_per_levata'] * len(PRODUCTS[product]['levate'])):
            full_cycles += 1
        elif remaining_quantity > 0:
            num_partial_cycle_levate = math.ceil(remaining_quantity / PRODUCTS[product]['Kg_prod_per_levata'])

        order_product_keys[order_id][product] = []

        for cycle_num in range(full_cycles):
            time = 0
            product_key = f"{product}#{cycle_num}_{order_id}"
            order_product_keys[order_id][product].append(product_key)

            product_levate = copy.deepcopy(PRODUCTS[product]['levate'])
            operations_product[product_key] = []

            # Adjust operation names to be unique per product per order
            prev = None
            for levata_idx, levata_product in enumerate(product_levate):
                new_operations = []

                for op_idx, op in enumerate(levata_product):
                    op_instance = f"{op}_{product_key}_Cycle{levata_idx + 1}_Op{op_idx + 1}"
                    new_operations.append(op_instance)
                    operations_product[product_key].append(op_instance)
                    operation_instances.append(op_instance)
                    operation_time_shift[op_instance] = time
                    operation_max_time[op_instance] = order_info['due_date']
                    prev = op_instance
                    # Update time based on operation duration
                    time += min([OPERATION_SPECIFICATIONS[op]['base_time'][m] / SPEED_LEVELS[0] for m in OPERATION_SPECIFICATIONS[op]['machines']])

                product_levate[levata_idx] = new_operations

            product_instances[product_key] = product_levate

        if num_partial_cycle_levate > 0:
            product_key = f"{product}#{full_cycles}_partial#{num_partial_cycle_levate}_{order_id}"
            order_product_keys[order_id][product].append(product_key)
            product_levate = copy.deepcopy(PRODUCTS[product]['levate'])[0:num_partial_cycle_levate]
            operations_product[product_key] = []

            for levate in range(num_partial_cycle_levate):
                new_operations = []
                for op_idx, op in enumerate(product_levate[levate]):
                    op_instance = f"{op}_{product_key}_Cycle{levate + 1}_Op{op_idx + 1}"
                    new_operations.append(op_instance)
                    operations_product[product_key].append(op_instance)
                    operation_instances.append(op_instance)
                    operation_time_shift[op_instance] = time
                    operation_max_time[op_instance] = order_info['due_date']
                    # Update time based on operation duration
                    time += min([OPERATION_SPECIFICATIONS[op]['base_time'][m] / SPEED_LEVELS[0] for m in OPERATION_SPECIFICATIONS[op]['machines']])
                product_levate[levate] = new_operations
            product_instances[product_key] = product_levate

# Create operations that are already executing
started_operations = []
for idx, op_info in enumerate(ALREADY_EXECUTING):
    product = op_info['Product']
    current_levata = op_info['CurrentLevata']
    end_levata = op_info['EndLevata']
    operation = op_info['Operation']

    order_id = f"OrderStarted{idx+1}"

    order_info = {'due_date': op_info['DueDate'], 'priority': 0, 'products': {product: 0}}
    ORDERS[order_id] = order_info

    product_key = f"{product}#Started{idx}_{order_id}"

    order_product_keys[order_id] = {product: [product_key]}
    product_instances[product_key] = []
    operations_product[product_key] = []
    
    # Set current operation to the last operation of the current levata
    levata_operations = []
    op_type = PRODUCTS[product]['levate'][current_levata][operation]
    operation_key = f"{op_type}_{product_key}_Cycle{current_levata}_Op{0}"
    started_operations.append(operation_key)
    operation_instances.append(operation_key)
    operation_time_shift[operation_key] = 0
    operation_max_time[operation_key] = op_info['DueDate']
    operations_product[product_key].append(operation_key)
    levata_operations.append(operation_key)

    # Add the remaining operations of the current levata
    current_levata_op = PRODUCTS[product]['levate'][current_levata][(operation+1):]
    for idx_op, op in enumerate(current_levata_op):
        operation_key = f"{op}_{product_key}_Cycle{current_levata}_Op{idx_op+1}"
        operation_instances.append(operation_key)
        operation_time_shift[operation_key] = 0
        operation_max_time[operation_key] = op_info['DueDate']
        operations_product[product_key].append(operation_key)
        levata_operations.append(operation_key)

    product_instances[product_key].append(levata_operations)

    # Add the remaining levate of the product
    for levata_idx, levata_product in enumerate(PRODUCTS[product]['levate'][(current_levata+1):(end_levata+1)]):
        new_operations = []
        for op_idx, op in enumerate(levata_product):
            op_instance = f"{op}_{product_key}_Cycle{levata_idx + 1}_Op{op_idx + 1}"
            new_operations.append(op_instance)
            operations_product[product_key].append(op_instance)
            operation_instances.append(op_instance)
            operation_time_shift[op_instance] = 0
            operation_max_time[op_instance] = op_info['DueDate']
        product_instances[product_key].append(new_operations)


# Mapping Operations to Their Operation Type
op_instance_to_type = {}
for product_key, product_info in product_instances.items():
    for levata_product in product_info:
        for op_instance in levata_product:
            # Extract the base operation name (e.g., Op1 from Op1_ProductA_Order1)
            base_op = op_instance.split('_')[0]
            op_instance_to_type[op_instance] = base_op

supervised_ops = [op for op in operation_instances if OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['requires_supervision']]
unsupervised_ops = [op for op in operation_instances if not OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['requires_supervision']]

# Mapping Products to Orders
product_to_order = {}
for product_key in product_instances:
    # Extract order_id from product name
    order_id = product_key.split('_')[-1]
    product_to_order[product_key] = order_id


# ============================
# 3. Initialize the Model
# ============================

model = cp_model.CpModel()


# ============================
# 4. Decision Variables
# ============================

# 4.1. Machine and Speed Assignment Variables
# x[(op, m, s)] = 1 if operation op is assigned to machine m with speed s
operation_machine_speed_assignment = {}
for op in operation_instances:
    op_type = op_instance_to_type[op]
    for m in OPERATION_SPECIFICATIONS[op_type]['machines']:
        for s in SPEED_LEVELS:
            operation_machine_speed_assignment[(op, m, s)] = model.NewBoolVar(f"Op_machine_speed_assignment_{op}_{m}_{int(s*100)}")

# 4.2. Start Time, Processing Time, and Completion Time Variables for Operations
start_times = {}
for op in operation_instances:
    min_time = int(operation_time_shift[op])
    max_time_op = operation_max_time[op]
    start_times[op] = model.NewIntVarFromDomain(cp_model.Domain.FromValues([t for t in working_shifts if t >= min_time and t <= max_time_op]), f"start_{op}")

processing_time = {}
for op in operation_instances:
    processing_time[op] = model.NewIntVar(0, scheduling_horizon, f"processing_time_{op}")

completion_time = {}
for op in operation_instances:
    completion_time[op] = model.NewIntVar(0, scheduling_horizon, f"completion_{op}")

# 4.3. Start Time and Completion Time for Products
start_time_product = {}
for product in product_instances:
    start_time_product[product] = model.NewIntVar(0, scheduling_horizon, f"start_product_{product}")

completion_time_product = {}
for product in product_instances:
    completion_time_product[product] = model.NewIntVar(0, scheduling_horizon, f"completion_product_{product}")

# 4.4. Completion Time for Orders
completion_time_order = {}
for order_id in ORDERS:
    completion_time_order[order_id] = model.NewIntVar(0, scheduling_horizon, f"completion_order_{order_id}")

# 4.5. Product to Machine Assignment Variables
product_machine_assignment = {}
for p in product_instances:
    for m in MACHINES:
        product_machine_assignment[(p, m)] = model.NewBoolVar(f"y_{p}_{m}")

# 4.6. Operator Assignment Variables
operator_operation_assignments = {}
for op in supervised_ops:
    for operator_id in range(OPERATOR_GROUPS_SHIFT):
        operator_operation_assignments[(op, operator_id)] = model.NewBoolVar(f"operator_{op}_id{operator_id}")

# 4.7. Next shift days for operations that cannot finish in a single shift
next_shift_days = []
for d in range(scheduling_horizon_in_days):
    next_shift_days.append(model.NewIntVar(0, scheduling_horizon_in_days, f"next_shift_day_{d}"))
    next_shift = 1
    while next_shift+d in FESTIVE_DAYS: next_shift += 1
    model.Add(next_shift_days[d] == next_shift)


# ============================
# 5. Constraints
# ============================

# 5.1. Each Operation Assigned to Exactly One (Machine, Speed)
for op in operation_instances:
    op_type = op_instance_to_type[op]
    possible_assignments = []
    for m in OPERATION_SPECIFICATIONS[op_type]['machines']:
        for s in SPEED_LEVELS:
            possible_assignments.append(operation_machine_speed_assignment[(op, m, s)])
    model.AddExactlyOne(possible_assignments)

# 5.2. Already Executing Operations assign the machine and speed
for idx, op_id in enumerate(started_operations):
    model.AddBoolAnd(operation_machine_speed_assignment[(op_id, ALREADY_EXECUTING[idx]['Machine'], ALREADY_EXECUTING[idx]['Speed'])])

# 5.3. Define Processing Time Based on Machine and Speed
for op in operation_instances:
    op_type = op_instance_to_type[op]

    base_times = []
    for m in OPERATION_SPECIFICATIONS[op_type]['machines']:
        for s in SPEED_LEVELS:
            if op in unsupervised_ops:
                time = int(math.ceil(OPERATION_SPECIFICATIONS[op_type]['base_time'][m] / s))
            else:
                time = int(math.ceil(OPERATION_SPECIFICATIONS[op_type]['base_time'][m] / OPERATORS_PER_GROUP))
            base_times.append((operation_machine_speed_assignment[(op, m, s)], time))

    if op in started_operations:
        model.Add(
            processing_time[op] == ALREADY_EXECUTING[idx]['Completion']
        )
    elif op in unsupervised_ops:
        model.Add(
            processing_time[op] == sum(x_var * time for (x_var, time) in base_times)
        )
    else:
        # When the operation cannot finish in a single shift, we need to add the time for the next shift
        base_processing_time = model.NewIntVar(0, scheduling_horizon, f"base_processing_time_{op}")
        model.Add(base_processing_time == sum(x_var * time for (x_var, time) in base_times))
        estimated_completion_time = model.NewIntVar(0, scheduling_horizon, f"estimated_completion_time_{op}")
        model.Add(estimated_completion_time == start_times[op] + base_processing_time)

        day_start_op = model.NewIntVar(0, scheduling_horizon_in_days, f"day_{op}")
        model.AddDivisionEquality(day_start_op, start_times[op], 24)
        day_end_op = model.NewIntVar(0, scheduling_horizon_in_days, f"day_{op}")
        model.AddDivisionEquality(day_end_op, estimated_completion_time, 24)
        time_end_op = model.NewIntVar(0, 24, f"day_time_{op}")
        model.AddModuloEquality(time_end_op, estimated_completion_time, 24)
        
        # Define Boolean variables for the conditions
        same_day = model.NewBoolVar(f"same_day_{op}")
        model.Add(day_start_op == day_end_op).OnlyEnforceIf(same_day)
        model.Add(day_start_op != day_end_op).OnlyEnforceIf(same_day.Not())

        within_shift_hours = model.NewBoolVar(f"within_shift_hours_{op}")
        model.Add(time_end_op <= WORKING_HOURS[1]).OnlyEnforceIf(within_shift_hours).OnlyEnforceIf(same_day)
        model.Add(time_end_op > WORKING_HOURS[1]).OnlyEnforceIf(within_shift_hours.Not()).OnlyEnforceIf(same_day)

        # Define 'in_shift' as the conjunction of 'same_day' and 'within_shift_hours'
        in_shift = model.NewBoolVar(f"in_shift_{op}")
        model.AddBoolAnd([same_day, within_shift_hours]).OnlyEnforceIf(in_shift)
        model.AddBoolOr([same_day.Not(), within_shift_hours.Not()]).OnlyEnforceIf(in_shift.Not())

        # If the operation is in the shift, the processing time is the base processing time
        model.Add( processing_time[op] == base_processing_time).OnlyEnforceIf(in_shift)

        # If estimated completion time is beyond the shift end time, add the next shift time considering also the festive days
        next_shift = model.NewIntVar(0, scheduling_horizon_in_days, f"next_shift_{op}")
        model.AddElement(day_start_op, next_shift_days, next_shift)
        
        # base time + time to get to the next day + time to get to the next shift day + time to get to the next shift
        model.Add(processing_time[op] == base_processing_time + (24-WORKING_HOURS[1]) + (next_shift-1) * 24 + WORKING_HOURS[0]).OnlyEnforceIf(in_shift.Not())

# 5.4. Completion Time Constraints
for op in operation_instances:
    # Define completion_time = start_time + processing_time
    model.Add(completion_time[op] == start_times[op] + processing_time[op])

# 5.5. Start Time for already executing operations
# for idx, op_id in enumerate(started_operations):
#     model.Add(start_times[op_id] == ALREADY_EXECUTING[idx]['Start'])

# 5.6. Precedence Constraints
for product in product_instances:
    ops = operations_product[product]
    for i in range(len(ops) - 1):
        op_prev = ops[i]
        op_next = ops[i + 1]
        # Ensure that op_next starts after op_prev completes
        if op_next in unsupervised_ops:
            # Ensure that unsupervised operations start right after supervised operations
            if op_prev in supervised_ops:
                model.Add(start_times[op_next] == completion_time[op_prev])
            else:
                model.Add(start_times[op_next] >= completion_time[op_prev])
        else:
            model.Add(start_times[op_next] >= completion_time[op_prev])

# 5.7. Completion and Start Time for Products
for product in product_instances:
    last_op = operations_product[product][-1]
    model.Add(completion_time_product[product] == completion_time[last_op])
    first_op = operations_product[product][0]
    model.Add(start_time_product[product] == start_times[first_op])

# 5.8. Product Sequencing Constraints on Machines
# Each product assigned to exactly one machine
for p in product_instances:
    model.Add(sum([product_machine_assignment[(p, m)] for m in MACHINES]) == 1)

# Link operation machine assignment with product machine assignment
for p in product_instances:
    ops_p = operations_product[p]
    for op in ops_p:
        op_type = op_instance_to_type[op]
        for m in OPERATION_SPECIFICATIONS[op_type]['machines']:
            model.AddBoolOr([operation_machine_speed_assignment[(op, m, s)] for s in SPEED_LEVELS]).OnlyEnforceIf(product_machine_assignment[(p, m)])
            model.AddBoolAnd([operation_machine_speed_assignment[(op, m, s)].Not() for s in SPEED_LEVELS]).OnlyEnforceIf(product_machine_assignment[(p, m)].Not())

# Before starting a new product, the previous product scheduled on the same machine must be completed
for m in MACHINES:
    for p1, p2 in combinations(product_instances.keys(), 2):
        or_var = model.NewBoolVar(f"precedence_product_{p1}_{p2}_{m}")
        model.Add(completion_time_product[p1] <= start_time_product[p2]).OnlyEnforceIf(product_machine_assignment[(p1, m)]).OnlyEnforceIf(product_machine_assignment[(p2, m)]).OnlyEnforceIf(or_var)
        model.Add(completion_time_product[p2] <= start_time_product[p1]).OnlyEnforceIf(product_machine_assignment[(p1, m)]).OnlyEnforceIf(product_machine_assignment[(p2, m)]).OnlyEnforceIf(or_var.Not())

# 5.9. Completion Time for Orders
for order_id, order_info in ORDERS.items():
    products_order = []
    for product in order_info['products'].keys():
        for product_key in order_product_keys[order_id][product]:
            products_order.append(completion_time_product[product_key])
        
    model.AddMaxEquality(completion_time_order[order_id], products_order)
    # The order should be completed before the due date
    model.Add(completion_time_order[order_id] <= order_info['due_date'])

# 5.10. Machine Availability Constraints
# Prevent overlapping operations on the same machine
intervals_machine = {}  # Dictionary to hold interval variables for each operation on each machine
for m in MACHINES: intervals_machine[m] = []
for op in operation_instances:
    for m in OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['machines']:
        assigned = model.NewBoolVar(f"assigned_{op}_{m}")
        # If operation is assigned to machine m, assigned = 1
        model.AddBoolOr([operation_machine_speed_assignment[(op, m, s)] for s in SPEED_LEVELS]).OnlyEnforceIf(assigned)
        model.AddBoolAnd([operation_machine_speed_assignment[(op, m, s)].Not() for s in SPEED_LEVELS]).OnlyEnforceIf(assigned.Not())

        interval = model.NewOptionalIntervalVar(
            start=start_times[op],
            size=processing_time[op],
            end=completion_time[op],
            is_present=assigned,
            name=f"interval_{op}_{m}"
        )
        intervals_machine[m].append(interval)
    
for m in MACHINES:
    model.AddNoOverlap(intervals_machine[m])

# 5.11 Operator Assignment Constraints
# Each operation that requires supervision must be assigned to exactly one operator group
intervals_operator = [[] for _ in range(OPERATOR_GROUPS_SHIFT)]
for op in supervised_ops:
    # If operation requires supervision, assign exactly one operator
    model.AddExactlyOne([operator_operation_assignments[(op, operator_id)] for operator_id in range(OPERATOR_GROUPS_SHIFT)])

    # Create intervals for each operator assignment
    for operator_id in range(OPERATOR_GROUPS_SHIFT):
        intervals_operator[operator_id].append(model.NewOptionalIntervalVar(
            start=start_times[op],
            size=processing_time[op],  # Ensure this is correctly defined based on speed 's'
            end=completion_time[op],
            is_present=operator_operation_assignments[(op, operator_id)],
            name=f"interval_{op}_operator_{operator_id}"
        ))

for operator_id in range(OPERATOR_GROUPS_SHIFT):
    model.AddNoOverlap(intervals_operator[operator_id])


# ============================
# 6. Objective Function
# ============================

# Define Makespan Minimization
makespan = model.NewIntVar(0, scheduling_horizon, "makespan")
model.AddMaxEquality(makespan, [completion_time_order[order_id] for order_id in ORDERS])

# Penalize Speed-Up Usage to Minimize It
speed_up_penalty = model.NewIntVar(0, scheduling_horizon, "speed_up_penalty")
model.Add(speed_up_penalty == sum([(operation_machine_speed_assignment[(op, m, s)])*int((s*100)**2-10000) for op in operation_instances for m in OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['machines'] for s in SPEED_LEVELS]))
speed_weight = 0.01

# Define Total Sum of Completion Times for tighter optimization and some level of control on which orders are completed first
num_orders = len(ORDERS)
total_sum = model.NewIntVar(0, scheduling_horizon*num_orders*PRIORITY_LEVELS, "total_sum")
model.Add(total_sum == sum([completion_time_order[order_id] * (PRIORITY_LEVELS-ORDERS[order_id]['priority']) for order_id in ORDERS]))
order_priority_weight = 0.5 / sum([PRIORITY_LEVELS-ORDERS[order_id]['priority'] for order_id in ORDERS])

# product tightness variable
num_products = len(product_instances)
product_tightness = model.NewIntVar(0, scheduling_horizon*num_products, "product_tightness")
model.Add(product_tightness == sum([completion_time_product[p] for p in product_instances]))
weight_product_tightness = 0.1 / num_products 

# Objective:
model.Minimize(makespan + order_priority_weight * total_sum + speed_weight * speed_up_penalty + weight_product_tightness * product_tightness)


# ============================
# 7. Solve the Problem
# ============================

# Create a solver and minimize the makespan
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True
solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT

status = solver.Solve(model)


# ============================
# 8. Output the Results
# ============================

print("Status:", solver.StatusName(status))

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    schedule = []
    for order_id, order_info in ORDERS.items():
        print(f"\nOrder: {order_id}")
        print(f"Due Date: {order_info['due_date']} hours")
        for product in order_info['products'].keys():
            for product_key in order_product_keys[order_id][product]:

                print(f"  Product: {product_key}")
                ops = operations_product[product_key]
                for op in ops:
                    # Find the machine and speed assignment
                    assigned_m = None
                    assigned_s = None
                    for m in OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['machines']:
                        for s in SPEED_LEVELS:
                            if solver.Value(operation_machine_speed_assignment[(op, m, s)]) == 1:
                                assigned_m = m
                                assigned_s = s
                                break
                        if assigned_m is not None:
                            break
                    # Get start time and day
                    st = solver.Value(start_times[op])
                    day = st // 24
                    s_op = st % 24
                    day_name = weekdays[(day + START_DAY) % 7]
                    # Get completion time
                    ct = solver.Value(completion_time[op])
                    # Determine shift
                    if OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['requires_supervision']:
                        op_shift = 'Day'
                    else:
                        op_shift = 'Independent'
                    schedule.append({
                        'Operation': op,
                        'Product': product,
                        'Order': order_id,
                        'Machine': assigned_m,
                        'Speed': assigned_s,
                        'Start': st,
                        'End': ct,
                        'Day': day + 1,  # Days are 1-indexed for display
                        'Shift': op_shift
                    })
                    print(f"    Operation: {op}")
                    print(f"      Assigned Machine: {assigned_m}")
                    print(f"      Speed Factor: {assigned_s}")
                    print(f"      Start Time: {st} hours (Day {day + 1} - {day_name} at {s_op} hours)")
                    print(f"      Completion Time: {ct} hours")
                    print(f"      Requires Supervision: {OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['requires_supervision']}")
                # Get product completion time
                ct_product = solver.Value(completion_time_product[product_key])
                print(f"  Completion Time for {product_key}: {ct_product} hours")
        # Get order completion time
        for order_id_inner in ORDERS:
            if order_id_inner == order_id:
                ct_order = solver.Value(completion_time_order[order_id_inner])
                print(f"Completion Time for {order_id_inner}: {ct_order} hours")
    print("\nMakespan =", solver.Value(makespan))
    print("Total Speed-Up Penalty =", solver.Value(speed_up_penalty))
    
    # ============================
    # 9. Graphical Visualization
    # ============================

    # Assign a unique color to each operation type using a color palette
    products_types = list(PRODUCTS.keys())
    num_product_types = len(products_types)

    # Use a colormap with enough distinct colors
    cmap = plt.get_cmap('tab20')  # Supports up to 20 distinct colors
    if num_product_types > 20:
        cmap = plt.get_cmap('hsv')  # Fallback to hsv if more colors are needed
    product_type_colors = {op_type: cmap(i % cmap.N) for i, op_type in enumerate(products_types)}

    # Define patterns for operations based on supervision requirement
    hatch_patterns = {
        True: '',      # No hatch for supervised operations
        False: '//',   # Hatched pattern for independent operations
    }

    fig, ax = plt.subplots(figsize=(20, 10))  # Increased figure size for clarity

    # To avoid duplicate legend entries, keep track of added operation types and supervision types
    added_operation_labels = set()
    added_supervision_labels = set()

    # Sort the schedule by machine and start time for better visualization
    schedule_sorted = sorted(schedule, key=lambda x: (x['Machine'], x['Start']))

    # Plot each operation as a bar
    for op_entry in schedule_sorted:
        operation_name = op_entry['Operation']
        op_type = operation_name.split('_')[0]
        product_key = op_entry['Product']
        machine_name = op_entry['Machine']
        start_time = op_entry['Start']
        end_time = op_entry['End']
        requires_supervision = OPERATION_SPECIFICATIONS[op_instance_to_type[operation_name]]['requires_supervision']

        # Check for valid start and end times
        if start_time is None or end_time is None:
            print(f"Invalid Start or End time for operation {operation_name}")
            continue

        color = product_type_colors.get(product_key.split('_')[0], 'gray')  # Default to gray if not found
        hatch = hatch_patterns.get(requires_supervision, '')

        # Add a unique label for each operation type to the legend
        if product_key.split('_')[0] not in added_operation_labels:
            label_operation = product_key.split('_')[0]
            added_operation_labels.add(product_key.split('_')[0])
        else:
            label_operation = ""

        # Plot the bar with operation-specific color and hatch
        ax.barh(
            y=machine_name,
            width=end_time - start_time,
            left=start_time,
            height=0.4,
            color=color,
            edgecolor='black',
            hatch=hatch,
            label=label_operation if label_operation else ""
        )

        # Only add labels for bars that are wide enough to avoid overlap
        if end_time - start_time > 0:  # Adjust this threshold as needed
            ax.text(
                x=start_time + (end_time - start_time) / 2,
                y=machine_name,
                s=f"{op_type} {product_key} ({op_entry['Order']})",
                va='center',
                ha='center',
                color='black',
                fontsize=8, 
                rotation=90
            )

    # Add day and night shift backgrounds
    for d in range(scheduling_horizon_in_days):
        # Define the absolute start and end times for shifts
        day_shift_start = d * 24 + WORKING_HOURS[0]
        day_shift_end = d * 24 + WORKING_HOURS[1]

        if d not in FESTIVE_DAYS:
            # Day shift background (yellow)
            ax.axvspan(day_shift_start, day_shift_end, color='yellow', alpha=0.3, zorder=0)

            # Night shift backgrounds (darker gray)
            night_start_time = d * 24
            night_end_time = day_shift_start
            if night_start_time < day_shift_start:
                ax.axvspan(night_start_time, day_shift_start, color='black', alpha=0.1, zorder=0)

            night_start_time = day_shift_end
            night_end_time = (d + 1) * 24
            if day_shift_end < night_end_time:
                ax.axvspan(day_shift_end, night_end_time, color='black', alpha=0.1, zorder=0)

    # Labels and title
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title('Job Shop Scheduling Gantt Chart with Order Information', fontsize=14)

    # Create custom legends
    # Operation Type Legend
    operation_patches = [mpatches.Patch(color=product_type_colors[op_type], label=op_type) for op_type in products_types]
    first_legend = ax.legend(handles=operation_patches, loc='upper right', title='Operation Types', bbox_to_anchor=(1.15, 1))
    ax.add_artist(first_legend)

    # Supervision Legend
    supervision_patches = [
        mpatches.Patch(facecolor='white', edgecolor='black', label='Requires Supervision', hatch=''),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Runs Independently', hatch='//')
    ]
    second_legend = ax.legend(handles=supervision_patches, loc='lower right', title='Operation Type', bbox_to_anchor=(1.15, 0))
    ax.add_artist(second_legend)

    plt.tight_layout()
    plt.show()

else:
    print("No optimal solution found.")

    # ============================
    # 10. Debugging Output
    # ============================

    # Display the CP model for debugging
    with open('cp_model.txt', 'w') as file:
        file.write(str(model))
