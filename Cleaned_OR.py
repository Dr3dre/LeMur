import copy
import math
from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

# ============================
# 1. Data Definitions
# ============================

# 1.1. Constants
SOLVER_TIME_LIMIT = 60  # Time limit in seconds for the solver
PRIORITY_LEVELS = 3  # Number of priority levels
SPEED_LEVELS = [0.95, 1.0, 1.05]  # Speed adjustment factors
WORKING_HOURS = (8, 16)  # Start and end times for the day shift (8 AM to 4 PM)
START_HOUR = 0  # Hour to start scheduling
START_DAY = 0  # Day of the week to start scheduling (0 = Monday)
FESTIVE_DAYS = []  # Additional non-working days beyond weekends
OPERATOR_GROUPS_SHIFT = 2  # Number of operator groups available per shift
OPERATORS_PER_GROUP = 1  # Number of operators in each group

# 1.2. Machines
MACHINES = ['M1', 'M2', 'M3']  # List of machine names

# 1.3. Orders
ORDERS = {
    'Order1': {
        'due_date': 750,  # in hours
        'start_date': 100,  # in hours
        'products': {
            'ProductA': 600,  # in kg
            'ProductB': 250,  # in kg
        },
        'priority': 0,  # Priority level (0 is highest)
    },
    'Order2': {
        'due_date': 500,
        'start_date': 0,
        'products': {
            'ProductC': 450,
        },
        'priority': 2,
    },
    'Order3': {
        'due_date': 500,
        'start_date': 0,
        'products': {
            'ProductA': 200,
            'ProductB': 250,
            'ProductC': 450,
        },
        'priority': 1,
    },
}

# 1.4. Products
PRODUCTS = {
    'ProductA': {
        'Kg_prod_per_levata': 100,  # Production per levata in kg
        'levate': [
            ['PrepareMachine', 'Op1'],  # Sequence of operations for each levata
            ['RechargeMachine', 'Op1', 'UnloadMachine'],
        ],
    },
    'ProductB': {
        'Kg_prod_per_levata': 250,
        'levate': [
            ['PrepareMachine', 'Op2'],
            ['RechargeMachine', 'Op2', 'UnloadMachine'],
        ],
    },
    'ProductC': {
        'Kg_prod_per_levata': 150,
        'levate': [
            ['PrepareMachine', 'Op3'],
            ['RechargeMachine', 'Op3'],
            ['RechargeMachine', 'Op3', 'UnloadMachine'],
        ],
    },
}

# 1.5. Operation Specifications
OPERATION_SPECIFICATIONS = {
    'PrepareMachine': {
        'machines': ['M1', 'M2', 'M3'],  # Machines that can perform this operation
        'base_time': {'M1': 5, 'M2': 3, 'M3': 5},  # Base processing time per machine
        'requires_supervision': True,  # Whether supervision is required
    },
    'RechargeMachine': {
        'machines': ['M1', 'M2', 'M3'],
        'base_time': {'M1': 4, 'M2': 2, 'M3': 3},
        'requires_supervision': True,
    },
    'UnloadMachine': {
        'machines': ['M1', 'M2', 'M3'],
        'base_time': {'M1': 5, 'M2': 3, 'M3': 5},
        'requires_supervision': True,
    },
    'Op1': {
        'machines': ['M1', 'M3'],
        'base_time': {'M1': 35, 'M3': 35},
        'requires_supervision': False,
    },
    'Op2': {
        'machines': ['M2'],
        'base_time': {'M2': 35},
        'requires_supervision': False,
    },
    'Op3': {
        'machines': ['M1', 'M2'],
        'base_time': {'M1': 35, 'M2': 35},
        'requires_supervision': False,
    },
}

# 1.6. Already Executing Operations
ALREADY_EXECUTING = [
    {
        'Product': 'ProductA',
        'CurrentLevata': 0,  # Current levata index
        'Remaining': 1,  # Number of remaining levate excluding the current one
        'Operation': 1,  # Index of the current operation within the levata
        'Machine': 'M1',  # Machine on which the operation is running
        'Speed': 1.0,  # Speed factor
        'Completion': 15,  # Remaining processing time in hours
        'DueDate': 500,  # Due date for the operation
    },
]

# ============================
# 2. Preprocess Data
# ============================

# 2.1. Compute Scheduling Horizon
scheduling_horizon = max(order_info['due_date'] for order_info in ORDERS.values())
scheduling_horizon_in_days = int(scheduling_horizon / 24) + 1  # Convert hours to days

# 2.2. Add Saturday and Sunday as Non-Working Days
for i in range(scheduling_horizon_in_days):
    day_of_week = (START_DAY + i) % 7
    if day_of_week in (5, 6):  # 5 = Saturday, 6 = Sunday
        FESTIVE_DAYS.append(i)

# 2.3. Precompute Shift Start and End Times for Each Day
working_shifts = [0]  # Initialize with start time for already executing operations
for day in range(scheduling_horizon_in_days):
    if day not in FESTIVE_DAYS:
        # Determine start hour for the first day
        start_hour = WORKING_HOURS[0] if day != 0 else max(START_HOUR, WORKING_HOURS[0])
        # Extend working shifts for the current day
        working_shifts.extend(day * 24 + hour for hour in range(start_hour, WORKING_HOURS[1]))

# 2.4. Initialize Data Structures
product_instances = {}  # Holds all products of all orders
order_product_keys = {}  # Holds product instances per order
operations_product = {}  # Holds operations per product
operation_instances = []  # List of all operation instances
operation_max_time = {}  # Max time for each operation
operation_time_shift = {}  # Operation start times for each shift

# 2.5. Create Unique Products and Operations per Order
for order_id, order_info in ORDERS.items():
    order_product_keys[order_id] = {}
    for product, requested_quantity in order_info['products'].items():
        # Compute the number of full cycles and any partial cycles needed
        levate_per_cycle = PRODUCTS[product]['Kg_prod_per_levata'] * len(PRODUCTS[product]['levate'])
        full_cycles = requested_quantity // levate_per_cycle
        remaining_quantity = requested_quantity % levate_per_cycle
        num_partial_cycle_levate = 0

        # Determine if a partial cycle is needed
        if remaining_quantity > 0.8 * levate_per_cycle:
            full_cycles += 1
        elif remaining_quantity > 0:
            num_partial_cycle_levate = math.ceil(remaining_quantity / PRODUCTS[product]['Kg_prod_per_levata'])

        order_product_keys[order_id][product] = []

        # Handle full cycles
        for cycle_num in range(full_cycles):
            time = 0
            product_key = f"{product}#{cycle_num}_{order_id}"
            order_product_keys[order_id][product].append(product_key)

            product_levate = copy.deepcopy(PRODUCTS[product]['levate'])
            operations_product[product_key] = []

            # Adjust operation names to be unique per product per order
            for levata_idx, levata_product in enumerate(product_levate):
                new_operations = []
                for op_idx, op in enumerate(levata_product):
                    op_instance = f"{op}_{product_key}_Cycle{levata_idx + 1}_Op{op_idx + 1}"
                    new_operations.append(op_instance)
                    operations_product[product_key].append(op_instance)
                    operation_instances.append(op_instance)
                    operation_time_shift[op_instance] = time + order_info['start_date']
                    operation_max_time[op_instance] = order_info['due_date']

                    # Update time based on the minimum possible processing time
                    time += min(
                        OPERATION_SPECIFICATIONS[op]['base_time'][m] / SPEED_LEVELS[0]
                        for m in OPERATION_SPECIFICATIONS[op]['machines']
                    )
                product_levate[levata_idx] = new_operations
            product_instances[product_key] = product_levate

        # Handle partial cycles
        if num_partial_cycle_levate > 0:
            product_key = f"{product}#{full_cycles}_partial#{num_partial_cycle_levate}_{order_id}"
            order_product_keys[order_id][product].append(product_key)
            product_levate = copy.deepcopy(PRODUCTS[product]['levate'])
            operations_product[product_key] = []

            # Select the necessary levate for the partial cycle
            levate_product = list(range(max(0, num_partial_cycle_levate - 1))) + [len(PRODUCTS[product]['levate']) - 1]
            time = 0

            for levata in levate_product:
                new_operations = []
                for op_idx, op in enumerate(product_levate[levata]):
                    op_instance = f"{op}_{product_key}_Cycle{levata + 1}_Op{op_idx + 1}"
                    new_operations.append(op_instance)
                    operations_product[product_key].append(op_instance)
                    operation_instances.append(op_instance)
                    operation_time_shift[op_instance] = time + order_info['start_date']
                    operation_max_time[op_instance] = order_info['due_date']

                    # Update time based on the minimum possible processing time
                    time += min(
                        OPERATION_SPECIFICATIONS[op]['base_time'][m] / SPEED_LEVELS[0]
                        for m in OPERATION_SPECIFICATIONS[op]['machines']
                    )
                product_levate[levata] = new_operations
            product_instances[product_key] = product_levate

# 2.6. Create Operations That Are Already Executing
started_operations = []
for idx, op_info in enumerate(ALREADY_EXECUTING):
    product = op_info['Product']
    current_levata = op_info['CurrentLevata']
    remaining_num_levate = op_info['Remaining']
    operation = op_info['Operation']
    order_id = f"OrderStarted{idx + 1}"

    # Create a new order for the already executing operation
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
    current_levata_op = PRODUCTS[product]['levate'][current_levata][(operation + 1):]
    for idx_op, op in enumerate(current_levata_op):
        operation_key = f"{op}_{product_key}_Cycle{current_levata}_Op{idx_op + 1}"
        operation_instances.append(operation_key)
        operation_time_shift[operation_key] = 0
        operation_max_time[operation_key] = op_info['DueDate']
        operations_product[product_key].append(operation_key)
        levata_operations.append(operation_key)
    product_instances[product_key].append(levata_operations)

    # Add the remaining levate of the product
    remaining_levate = PRODUCTS[product]['levate'][(current_levata + 1):(current_levata + 1 + remaining_num_levate + 1)]
    if current_levata < len(PRODUCTS[product]['levate']) - 1 and remaining_num_levate > 0:
        remaining_levate[-1] = PRODUCTS[product]['levate'][-1]

    for levata_idx, levata_product in enumerate(remaining_levate):
        new_operations = []
        for op_idx, op in enumerate(levata_product):
            op_instance = f"{op}_{product_key}_Cycle{levata_idx + 1}_Op{op_idx + 1}"
            new_operations.append(op_instance)
            operations_product[product_key].append(op_instance)
            operation_instances.append(op_instance)
            operation_time_shift[op_instance] = 0
            operation_max_time[op_instance] = op_info['DueDate']
        product_instances[product_key].append(new_operations)

# 2.7. Map Operations to Their Operation Type
op_instance_to_type = {}
for product_key, product_info in product_instances.items():
    for levata_product in product_info:
        for op_instance in levata_product:
            # Extract the base operation name
            base_op = op_instance.split('_')[0]
            op_instance_to_type[op_instance] = base_op

# 2.8. Separate Supervised and Unsupervised Operations
supervised_ops = [
    op for op in operation_instances if OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['requires_supervision']
]
unsupervised_ops = [
    op for op in operation_instances if not OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['requires_supervision']
]

# 2.9. Map Products to Orders
product_to_order = {}
for product_key in product_instances:
    # Extract order_id from product name
    order_id = product_key.split('_')[-1]
    product_to_order[product_key] = order_id

# ============================
# 3. Initialize the Model
# ============================

# Create a new constraint programming model
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
            key = (op, m, s)
            operation_machine_speed_assignment[key] = model.NewBoolVar(f"Op_machine_speed_assignment_{op}_{m}_{int(s * 100)}")

# 4.2. Start Time, Processing Time, and Completion Time Variables for Operations
start_times = {}
processing_time = {}
completion_time = {}
for op in operation_instances:
    min_time = int(operation_time_shift[op])  # Earliest possible start time
    max_time_op = operation_max_time[op]  # Latest possible completion time
    # Start times are constrained to working shifts
    start_times[op] = model.NewIntVarFromDomain(
        cp_model.Domain.FromValues([t for t in working_shifts if min_time <= t <= max_time_op]),
        f"start_{op}"
    )
    processing_time[op] = model.NewIntVar(0, scheduling_horizon, f"processing_time_{op}")
    completion_time[op] = model.NewIntVar(0, scheduling_horizon, f"completion_{op}")

# 4.3. Start Time and Completion Time for Products
start_time_product = {}
completion_time_product = {}
for product in product_instances:
    start_time_product[product] = model.NewIntVar(0, scheduling_horizon, f"start_product_{product}")
    completion_time_product[product] = model.NewIntVar(0, scheduling_horizon, f"completion_product_{product}")

# 4.4. Completion Time for Orders
completion_time_order = {}
for order_id in ORDERS:
    completion_time_order[order_id] = model.NewIntVar(0, scheduling_horizon, f"completion_order_{order_id}")

# 4.5. Product to Machine Assignment Variables
# y[(p, m)] = 1 if product p is assigned to machine m
product_machine_assignment = {}
for p in product_instances:
    for m in MACHINES:
        product_machine_assignment[(p, m)] = model.NewBoolVar(f"y_{p}_{m}")

# 4.6. Operator Assignment Variables
operator_operation_assignments = {}
for op in supervised_ops:
    for operator_id in range(OPERATOR_GROUPS_SHIFT):
        key = (op, operator_id)
        operator_operation_assignments[key] = model.NewBoolVar(f"operator_{op}_id{operator_id}")

# 4.7. Next Shift Days for Operations That Cannot Finish in a Single Shift
next_shift_days = []
for d in range(scheduling_horizon_in_days):
    next_shift_day = model.NewIntVar(0, scheduling_horizon_in_days, f"next_shift_day_{d}")
    next_shift = 1
    # Skip festive days
    while next_shift + d in FESTIVE_DAYS:
        next_shift += 1
    model.Add(next_shift_day == next_shift)
    next_shift_days.append(next_shift_day)

# ============================
# 5. Constraints
# ============================

# 5.1. Each Operation Assigned to Exactly One (Machine, Speed)
for op in operation_instances:
    op_type = op_instance_to_type[op]
    possible_assignments = [
        operation_machine_speed_assignment[(op, m, s)]
        for m in OPERATION_SPECIFICATIONS[op_type]['machines']
        for s in SPEED_LEVELS
    ]
    model.AddExactlyOne(possible_assignments)

# 5.2. Already Executing Operations Assign the Machine and Speed
for idx, op_id in enumerate(started_operations):
    key = (op_id, ALREADY_EXECUTING[idx]['Machine'], ALREADY_EXECUTING[idx]['Speed'])
    model.AddBoolAnd([operation_machine_speed_assignment[key]])

# 5.3. Define Processing Time Based on Machine and Speed
for op in operation_instances:
    op_type = op_instance_to_type[op]
    base_times = []
    for m in OPERATION_SPECIFICATIONS[op_type]['machines']:
        for s in SPEED_LEVELS:
            if op in unsupervised_ops:
                # Adjust processing time based on speed
                time = int(math.ceil(OPERATION_SPECIFICATIONS[op_type]['base_time'][m] / s))
            else:
                # For supervised operations, processing time depends on the number of operators
                time = int(math.ceil(OPERATION_SPECIFICATIONS[op_type]['base_time'][m] / OPERATORS_PER_GROUP))
            base_times.append((operation_machine_speed_assignment[(op, m, s)], time))

    if op in started_operations:
        idx = started_operations.index(op)
        model.Add(processing_time[op] == ALREADY_EXECUTING[idx]['Completion'])
    elif op in unsupervised_ops:
        # Processing time is the weighted sum of possible times based on machine and speed
        model.Add(processing_time[op] == sum(x_var * time for (x_var, time) in base_times))
    else:
        # For operations that may not finish in a single shift
        base_processing_time = model.NewIntVar(0, scheduling_horizon, f"base_processing_time_{op}")
        model.Add(base_processing_time == sum(x_var * time for (x_var, time) in base_times))
        estimated_completion_time = model.NewIntVar(0, scheduling_horizon, f"estimated_completion_time_{op}")
        model.Add(estimated_completion_time == start_times[op] + base_processing_time)

        # Determine if the operation finishes within the same day and shift hours
        day_start_op = model.NewIntVar(0, scheduling_horizon_in_days, f"day_start_{op}")
        model.AddDivisionEquality(day_start_op, start_times[op], 24)
        day_end_op = model.NewIntVar(0, scheduling_horizon_in_days, f"day_end_{op}")
        model.AddDivisionEquality(day_end_op, estimated_completion_time, 24)
        time_end_op = model.NewIntVar(0, 24, f"time_end_{op}")
        model.AddModuloEquality(time_end_op, estimated_completion_time, 24)

        # Boolean variables for conditions
        same_day = model.NewBoolVar(f"same_day_{op}")
        model.Add(day_start_op == day_end_op).OnlyEnforceIf(same_day)
        model.Add(day_start_op != day_end_op).OnlyEnforceIf(same_day.Not())

        within_shift_hours = model.NewBoolVar(f"within_shift_hours_{op}")
        model.Add(time_end_op <= WORKING_HOURS[1]).OnlyEnforceIf(within_shift_hours).OnlyEnforceIf(same_day)
        model.Add(time_end_op > WORKING_HOURS[1]).OnlyEnforceIf(within_shift_hours.Not()).OnlyEnforceIf(same_day)

        # Determine if the operation is within the shift
        in_shift = model.NewBoolVar(f"in_shift_{op}")
        model.AddBoolAnd([same_day, within_shift_hours]).OnlyEnforceIf(in_shift)
        model.AddBoolOr([same_day.Not(), within_shift_hours.Not()]).OnlyEnforceIf(in_shift.Not())

        # Adjust processing time based on whether it fits in the shift
        model.Add(processing_time[op] == base_processing_time).OnlyEnforceIf(in_shift)

        # If not, add time to wait for the next shift
        next_shift = model.NewIntVar(0, scheduling_horizon_in_days, f"next_shift_{op}")
        model.AddElement(day_start_op, next_shift_days, next_shift)
        model.Add(
            processing_time[op]
            == base_processing_time + (24 - WORKING_HOURS[1]) + (next_shift - 1) * 24 + WORKING_HOURS[0]
        ).OnlyEnforceIf(in_shift.Not())

# 5.4. Completion Time Constraints
for op in operation_instances:
    # Completion time equals start time plus processing time
    model.Add(completion_time[op] == start_times[op] + processing_time[op])

# 5.5. Start Time for Already Executing Operations
for idx, op_id in enumerate(started_operations):
    model.Add(start_times[op_id] == 0)  # Already started at time 0

# 5.6. Precedence Constraints
for product in product_instances:
    ops = operations_product[product]
    for i in range(len(ops) - 1):
        op_prev = ops[i]
        op_next = ops[i + 1]
        # Ensure operations are performed in sequence
        if op_next in unsupervised_ops and op_prev in supervised_ops:
            # Unsupervised operation starts immediately after supervised operation
            model.Add(start_times[op_next] == completion_time[op_prev])
        else:
            model.Add(start_times[op_next] >= completion_time[op_prev])

        op_type_prev = op_instance_to_type[op_prev]
        op_type_next = op_instance_to_type[op_next]
        # Ensure operations are performed on the same machine and speed
        for m in OPERATION_SPECIFICATIONS[op_type_prev]['machines']:
            if m in OPERATION_SPECIFICATIONS[op_type_next]['machines']:
                for s in SPEED_LEVELS:
                    key_prev = (op_prev, m, s)
                    key_next = (op_next, m, s)
                    model.Add(operation_machine_speed_assignment[key_prev] == operation_machine_speed_assignment[key_next])

# 5.7. Completion and Start Time for Products
for product in product_instances:
    last_op = operations_product[product][-1]
    first_op = operations_product[product][0]
    model.Add(completion_time_product[product] == completion_time[last_op])
    model.Add(start_time_product[product] == start_times[first_op])

# 5.8. Product Sequencing Constraints on Machines
for p in product_instances:
    # Each product is assigned to exactly one machine 
    model.Add(sum(product_machine_assignment[(p, m)] for m in MACHINES) == 1)

# Link operation machine assignment with product machine assignment
for p in product_instances:
    ops_p = operations_product[p]
    for op in ops_p:
        op_type = op_instance_to_type[op]
        for m in OPERATION_SPECIFICATIONS[op_type]['machines']:
            # If product is assigned to machine m, then operations can be assigned to m
            model.AddBoolOr(
                [operation_machine_speed_assignment[(op, m, s)] for s in SPEED_LEVELS]
            ).OnlyEnforceIf(product_machine_assignment[(p, m)])
            # If not, operations cannot be assigned to m
            model.AddBoolAnd(
                [operation_machine_speed_assignment[(op, m, s)].Not() for s in SPEED_LEVELS]
            ).OnlyEnforceIf(product_machine_assignment[(p, m)].Not())

# Ensure no overlapping products on the same machine
for m in MACHINES:
    for p1, p2 in combinations(product_instances.keys(), 2):
        or_var = model.NewBoolVar(f"precedence_product_{p1}_{p2}_{m}")
        model.Add(completion_time_product[p1] <= start_time_product[p2]).OnlyEnforceIf(
            product_machine_assignment[(p1, m)], product_machine_assignment[(p2, m)], or_var
        )
        model.Add(completion_time_product[p2] <= start_time_product[p1]).OnlyEnforceIf(
            product_machine_assignment[(p1, m)], product_machine_assignment[(p2, m)], or_var.Not()
        )

# 5.9. Completion Time for Orders
for order_id, order_info in ORDERS.items():
    products_order = []
    for product in order_info['products'].keys():
        for product_key in order_product_keys[order_id][product]:
            products_order.append(completion_time_product[product_key])
    # The order's completion time is the maximum completion time of its products
    model.AddMaxEquality(completion_time_order[order_id], products_order)
    # Ensure the order is completed before its due date
    model.Add(completion_time_order[order_id] <= order_info['due_date'])

# 5.10. Machine Availability Constraints
intervals_machine = {m: [] for m in MACHINES}
for op in operation_instances:
    for m in OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['machines']:
        assigned = model.NewBoolVar(f"assigned_{op}_{m}")
        model.AddBoolOr(
            [operation_machine_speed_assignment[(op, m, s)] for s in SPEED_LEVELS]
        ).OnlyEnforceIf(assigned)
        model.AddBoolAnd(
            [operation_machine_speed_assignment[(op, m, s)].Not() for s in SPEED_LEVELS]
        ).OnlyEnforceIf(assigned.Not())
        interval = model.NewOptionalIntervalVar(
            start_times[op],
            processing_time[op],
            completion_time[op],
            assigned,
            f"interval_{op}_{m}"
        )
        intervals_machine[m].append(interval)
# Ensure no overlapping operations on the same machine
for m in MACHINES:
    model.AddNoOverlap(intervals_machine[m])

# 5.11. Operator Assignment Constraints
intervals_operator = [[] for _ in range(OPERATOR_GROUPS_SHIFT)]
for op in supervised_ops:
    # Each supervised operation is assigned to exactly one operator group
    model.AddExactlyOne([operator_operation_assignments[(op, operator_id)] for operator_id in range(OPERATOR_GROUPS_SHIFT)])
    for operator_id in range(OPERATOR_GROUPS_SHIFT):
        interval = model.NewOptionalIntervalVar(
            start_times[op],
            processing_time[op],
            completion_time[op],
            operator_operation_assignments[(op, operator_id)],
            f"interval_{op}_operator_{operator_id}"
        )
        intervals_operator[operator_id].append(interval)
# Ensure operators are not assigned to overlapping operations
for operator_id in range(OPERATOR_GROUPS_SHIFT):
    model.AddNoOverlap(intervals_operator[operator_id])

# ============================
# 6. Objective Function
# ============================

# Define makespan as the maximum completion time across all orders
makespan = model.NewIntVar(0, scheduling_horizon, "makespan")
model.AddMaxEquality(makespan, [completion_time_order[order_id] for order_id in ORDERS])

# Penalize speed-up usage to minimize it
speed_up_penalty = model.NewIntVar(0, scheduling_horizon, "speed_up_penalty")
model.Add(
    speed_up_penalty
    == sum(
        operation_machine_speed_assignment[(op, m, s)] * int((s * 100) ** 2 - 10000)
        for op in operation_instances
        for m in OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['machines']
        for s in SPEED_LEVELS
    )
)
speed_weight = 0.01  # Weight for speed-up penalty in the objective function

# Sum of weighted completion times for orders based on priority
num_orders = len(ORDERS)
total_sum = model.NewIntVar(0, scheduling_horizon * num_orders * PRIORITY_LEVELS, "total_sum")
model.Add(
    total_sum
    == sum(
        completion_time_order[order_id] * (PRIORITY_LEVELS - ORDERS[order_id]['priority'])
        for order_id in ORDERS
    )
)
order_priority_weight = 0.5 / sum(PRIORITY_LEVELS - ORDERS[order_id]['priority'] for order_id in ORDERS)

# Total completion time of all products
num_products = len(product_instances)
product_tightness = model.NewIntVar(0, scheduling_horizon * num_products, "product_tightness")
model.Add(product_tightness == sum(completion_time_product[p] for p in product_instances))
weight_product_tightness = 0.2 / num_products

# Objective Function: Minimize makespan, speed-up penalty, and product tightness
model.Minimize(
    makespan
    + order_priority_weight * total_sum
    + speed_weight * speed_up_penalty
    + weight_product_tightness * product_tightness
)

# ============================
# 7. Solve the Problem
# ============================

solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True  # Enable search progress logging
solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT  # Set time limit
# Set number of threads to use
# solver.parameters.num_search_workers = 1

status = solver.Solve(model)

# ============================
# 8. Output the Results
# ============================

print("Status:", solver.StatusName(status))
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    schedule = []
    for order_id, order_info in ORDERS.items():
        print(f"\nOrder: {order_id}")
        print(f"Due Date: {order_info['due_date']} hours")
        for product in order_info['products'].keys():
            for product_key in order_product_keys[order_id][product]:
                print(f"  Product: {product_key}")
                ops = operations_product[product_key]
                for op in ops:
                    # Find the assigned machine and speed
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
                    st = solver.Value(start_times[op])
                    day = st // 24
                    s_op = st % 24
                    day_name = weekdays[(day + START_DAY) % 7]
                    ct = solver.Value(completion_time[op])
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
                        'Day': day + 1,
                        'Shift': op_shift
                    })
                    print(f"    Operation: {op}")
                    print(f"      Assigned Machine: {assigned_m}")
                    print(f"      Speed Factor: {assigned_s}")
                    print(f"      Start Time: {st} hours (Day {day + 1} - {day_name} at {s_op} hours)")
                    print(f"      Completion Time: {ct} hours")
                    print(f"      Requires Supervision: {OPERATION_SPECIFICATIONS[op_instance_to_type[op]]['requires_supervision']}")
                ct_product = solver.Value(completion_time_product[product_key])
                print(f"  Completion Time for {product_key}: {ct_product} hours")
        ct_order = solver.Value(completion_time_order[order_id])
        print(f"Completion Time for {order_id}: {ct_order} hours")
    print("\nMakespan =", solver.Value(makespan))
    print("Total Speed-Up Penalty =", solver.Value(speed_up_penalty))

    # ============================
    # 9. Graphical Visualization
    # ============================

    # Map product types to colors for plotting
    products_types = list(PRODUCTS.keys())
    num_product_types = len(products_types)
    cmap = plt.get_cmap('tab20')
    if num_product_types > 20:
        cmap = plt.get_cmap('hsv')
    product_type_colors = {op_type: cmap(i % cmap.N) for i, op_type in enumerate(products_types)}
    hatch_patterns = {
        True: '',    # No hatch for supervised operations
        False: '//',  # Hatched pattern for unsupervised operations
    }

    fig, ax = plt.subplots(figsize=(20, 10))
    added_operation_labels = set()

    # Sort the schedule for better visualization
    schedule_sorted = sorted(schedule, key=lambda x: (x['Machine'], x['Start']))

    for op_entry in schedule_sorted:
        operation_name = op_entry['Operation']
        op_type = operation_name.split('_')[0]
        product_key = op_entry['Product']
        machine_name = op_entry['Machine']
        start_time = op_entry['Start']
        end_time = op_entry['End']
        requires_supervision = OPERATION_SPECIFICATIONS[op_instance_to_type[operation_name]]['requires_supervision']

        if start_time is None or end_time is None:
            print(f"Invalid Start or End time for operation {operation_name}")
            continue

        color = product_type_colors.get(product_key.split('_')[0], 'gray')
        hatch = hatch_patterns.get(requires_supervision, '')

        if product_key.split('_')[0] not in added_operation_labels:
            label_operation = product_key.split('_')[0]
            added_operation_labels.add(product_key.split('_')[0])
        else:
            label_operation = ""

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

        if end_time - start_time > 0:
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

    # Add shift backgrounds for visualization
    for d in range(scheduling_horizon_in_days):
        day_shift_start = d * 24 + WORKING_HOURS[0]
        day_shift_end = d * 24 + WORKING_HOURS[1]
        if d not in FESTIVE_DAYS:
            ax.axvspan(day_shift_start, day_shift_end, color='yellow', alpha=0.3, zorder=0)
            night_start_time = d * 24
            night_end_time = day_shift_start
            if night_start_time < day_shift_start:
                ax.axvspan(night_start_time, day_shift_start, color='black', alpha=0.1, zorder=0)
            night_start_time = day_shift_end
            night_end_time = (d + 1) * 24
            if day_shift_end < night_end_time:
                ax.axvspan(day_shift_end, night_end_time, color='black', alpha=0.1, zorder=0)

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title('Job Shop Scheduling Gantt Chart with Order Information', fontsize=14)

    # Create legends for operation types and supervision
    operation_patches = [mpatches.Patch(color=product_type_colors[op_type], label=op_type) for op_type in products_types]
    first_legend = ax.legend(handles=operation_patches, loc='upper right', title='Operation Types', bbox_to_anchor=(1.15, 1))
    ax.add_artist(first_legend)

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

    # Write the CP model to a file for debugging
    with open('cp_model.txt', 'w') as file:
        file.write(str(model))
