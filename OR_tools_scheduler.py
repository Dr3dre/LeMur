import copy
from itertools import combinations
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ortools.sat.python import cp_model

# ============================
# 1. Data Definitions
# ============================

# Define Machines
MACHINES = ['M1', 'M2', 'M3']  # List of machine names

# Define Orders
ORDERS = {
    'Order1': {
        'due_date': 150,  # in hours
        'products': ['ProductA', 'ProductA']
    },
    'Order2': {
        'due_date': 380,  # in hours
        'products': ['ProductC']
    },
    'Order3': {
        'due_date': 380,  # in hours
        'products': ['ProductA', 'ProductC']
    },
    # 'Order4': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductB']
    # },
    # 'Order5': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductA']
    # },
    # 'Order6': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductB', 'ProductC']
    # },
    # 'Order7': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductA', 'ProductB']
    # },
    # 'Order8': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductC']
    # },
    # 'Order9': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductA', 'ProductC']
    # },
    # 'Order10': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductB']
    # },
    # 'Order11': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductA']
    # },
    # 'Order12': {
    #     'due_date': 1000,  # in hours
    #     'products': ['ProductB', 'ProductC']
    # },
}

# Define Products with Cycles and Operations
PRODUCTS = {
    'ProductA': {
        'cycle': [
            {'operations': ['PrepareMachine', 'Op1']},
            {'operations': ['RechargeMachine', 'Op1']},
        ]
    },
    'ProductB': {
        'cycle': [
            {'operations': ['PrepareMachine', 'Op2']},
            {'operations': ['RechargeMachine', 'Op2']},
        ]
    },
    'ProductC': {
        'cycle': [
            {'operations': ['PrepareMachine', 'Op3']},
            {'operations': ['RechargeMachine', 'Op3']},
            {'operations': ['RechargeMachine', 'Op3']},
        ]
    }
}

# Define Operations with Machines, Base Times, and Supervision Requirement
operation_specifications = {
    'PrepareMachine': {'machines': ['M1', 'M2', 'M3'], 'base_time': {'M1': 5, 'M2': 3, 'M3': 5}, 'requires_supervision': True},
    'RechargeMachine': {'machines': ['M1', 'M2', 'M3'], 'base_time': {'M1': 4, 'M2': 2, 'M3': 3}, 'requires_supervision': True},
    'Op1': {'machines': ['M1', 'M3'], 'base_time': {'M1': 35, 'M3': 35}, 'requires_supervision': False},
    'Op2': {'machines': ['M2'], 'base_time': {'M2': 35}, 'requires_supervision': False},
    'Op3': {'machines': ['M1', 'M2'], 'base_time': {'M1': 35, 'M2': 35}, 'requires_supervision': False},
}

# ============================
# 2. Preprocess Data
# ============================

# Scheduling Horizon
scheduling_horizon = max([order_info['due_date'] for order_info in ORDERS.values()])
scheduling_horizon_in_days = int(scheduling_horizon / 24) + 1

# Define Shift Parameters for Each Weekday
operators_groups_per_shift = 2  # Number of operators available per shift (define the number of parallel operations allowed)
number_of_operators = 1  # Number of operators in each group

working_hours = (8, 16)  # Start and end times for the day shift

festive_days = []  # Saturday and Sunday are non-working days
# add saturday and sunday as non-working days
start_day = 0  # Day of the week to start scheduling (0 = Monday)
for i in range(scheduling_horizon_in_days):
    day_of_week = (start_day + i) % 7
    if day_of_week == 5 or day_of_week == 6:
        festive_days.append(i)

# Speed Adjustment Factors
speed_levels = [0.95, 1.0, 1.05]

# Precompute shift start and end times for each day
working_shifts = []
for d in range(scheduling_horizon_in_days):
    if d not in festive_days:
        working_shifts.extend([d * 24 + h for h in range(working_hours[0], working_hours[1])])

# ============================
# 3. Create Unique Products and Operations per Order
# ============================

# Create unique products and operations per order
products_per_order = {}
operations_per_order = {}
operation_instances = []  # List to hold all operation instances
operation_max_time = {}  # Dictionary to hold the maximum time for each operation
operation_time_shift = {}  # Dictionary to hold operation start times for each shift

for order_id, order_info in ORDERS.items():
    for product_id, product in enumerate(order_info['products']):
        time = 0
        product_key = f"{product}#{product_id}_{order_id}"
        product_cycles = copy.deepcopy(PRODUCTS[product]['cycle'])
        operations_per_order[product_key] = []

        # Adjust operation names to be unique per product per order
        prev = None
        for cycle_idx, cycle in enumerate(product_cycles):
            new_operations = []
            for op_idx, op in enumerate(cycle['operations']):
                op_instance = f"{op}_{product_key}_Cycle{cycle_idx + 1}_Op{op_idx + 1}"
                new_operations.append(op_instance)
                operations_per_order[product_key].append(op_instance)
                operation_instances.append(op_instance)
                operation_time_shift[op_instance] = time
                operation_max_time[op_instance] = order_info['due_date']
                prev = op_instance
                # Update time based on operation duration
                time += min([operation_specifications[op]['base_time'][m] / speed_levels[0] for m in operation_specifications[op]['machines']])
            product_cycles[cycle_idx]['operations'] = new_operations
        products_per_order[product_key] = {'cycle': product_cycles}

# Mapping Operations to Their Operation Type
op_instance_to_type = {}
for product_key, product_info in products_per_order.items():
    for cycle in product_info['cycle']:
        for op_instance in cycle['operations']:
            # Extract the base operation name (e.g., Op1 from Op1_ProductA_Order1)
            base_op = op_instance.split('_')[0]
            op_instance_to_type[op_instance] = base_op

supervised_ops = [op for op in operation_instances if operation_specifications[op_instance_to_type[op]]['requires_supervision']]
unsupervised_ops = [op for op in operation_instances if not operation_specifications[op_instance_to_type[op]]['requires_supervision']]

# Mapping Products to Orders
product_to_order = {}
for product_key in products_per_order:
    # Extract order_id from product name
    order_id = product_key.split('_')[-1]
    product_to_order[product_key] = order_id

# ============================
# 4. Initialize the Model
# ============================

model = cp_model.CpModel()

# ============================
# 5. Decision Variables
# ============================

# 5.1. Machine and Speed Assignment Variables
# x[(op, m, s)] = 1 if operation op is assigned to machine m with speed s
operation_machine_speed_assignment = {}
for op in operation_instances:
    op_type = op_instance_to_type[op]
    for m in operation_specifications[op_type]['machines']:
        for s in speed_levels:
            operation_machine_speed_assignment[(op, m, s)] = model.NewBoolVar(f"Op_machine_speed_assignment_{op}_{m}_{int(s*100)}")

# 5.2. Start Time Variables
start_times = {}
for op in operation_instances:
    min_time = int(operation_time_shift[op])
    max_time_op = operation_max_time[op]
    start_times[op] = model.NewIntVarFromDomain(cp_model.Domain.FromValues([t for t in working_shifts if t >= min_time and t <= max_time_op]), f"start_{op}")


# 5.3. Completion Time Variables
completion_time = {}
for op in operation_instances:
    completion_time[op] = model.NewIntVar(min_time, max_time_op, f"completion_{op}")

# 5.5. Completion Time for Products and Orders
completion_time_product = {}
for product in products_per_order:
    completion_time_product[product] = model.NewIntVar(0, scheduling_horizon, f"completion_product_{product}")

start_time_product = {}
for product in products_per_order:
    start_time_product[product] = model.NewIntVar(0, scheduling_horizon, f"start_product_{product}")

completion_time_order = {}
for order_id in ORDERS:
    completion_time_order[order_id] = model.NewIntVar(0, scheduling_horizon, f"completion_order_{order_id}")

# 5.6. Makespan Variable
makespan = model.NewIntVar(0, scheduling_horizon, "makespan")

# 5.7. Product to Machine Assignment Variables
product_machine_assignment = {}
for p in products_per_order:
    for m in MACHINES:
        product_machine_assignment[(p, m)] = model.NewBoolVar(f"y_{p}_{m}")

# 5.9 Operator Assignment Variables
operator_operation_assignments = {}
for op in supervised_ops:
    for operator_id in range(operators_groups_per_shift):
        operator_operation_assignments[(op, operator_id)] = model.NewBoolVar(f"operator_{op}_id{operator_id}")

# 5.10 Next shift days for operations that cannot finish in a single shift
next_shift_days = []
for d in range(scheduling_horizon_in_days):
    next_shift_days.append(model.NewIntVar(0, scheduling_horizon_in_days, f"next_shift_day_{d}"))
    next_shift = 1
    while next_shift+d in festive_days: next_shift += 1
    model.Add(next_shift_days[d] == next_shift)


# ============================
# 6. Constraints
# ============================

# 6.1. Each Operation Assigned to Exactly One (Machine, Speed)
for op in operation_instances:
    op_type = op_instance_to_type[op]
    possible_assignments = []
    for m in operation_specifications[op_type]['machines']:
        for s in speed_levels:
            possible_assignments.append(operation_machine_speed_assignment[(op, m, s)])
    model.AddExactlyOne(possible_assignments)

# 6.2. Define Processing Time Based on Machine and Speed
processing_time = {}
for op in operation_instances:
    op_type = op_instance_to_type[op]
    processing_time[op] = model.NewIntVar(0, scheduling_horizon, f"processing_time_{op}")

    base_times = []
    for m in operation_specifications[op_type]['machines']:
        for s in speed_levels:
            if op in unsupervised_ops:
                time = int(math.ceil(operation_specifications[op_type]['base_time'][m] / s))
            else:
                time = int(math.ceil(operation_specifications[op_type]['base_time'][m] / number_of_operators))
            base_times.append((operation_machine_speed_assignment[(op, m, s)], time))

    if op in unsupervised_ops:
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
        model.Add(time_end_op <= working_hours[1]).OnlyEnforceIf(within_shift_hours)
        model.Add(time_end_op > working_hours[1]).OnlyEnforceIf(within_shift_hours.Not())

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
        model.Add(processing_time[op] == base_processing_time + (24-working_hours[1]) + (next_shift-1) * 24 + working_hours[0]).OnlyEnforceIf(in_shift.Not())


# model.Add(start_times[operation_instances[0]] == 8)
# 6.3. Start Time and Completion Time Constraints
for op in operation_instances:
    # Define completion_time = start_time + processing_time
    model.Add(completion_time[op] == start_times[op] + processing_time[op])

# 6.4. Precedence Constraints
for product in products_per_order:
    ops = operations_per_order[product]
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

# 6.5. Completion and Start Time for Products
for product in products_per_order:
    last_op = operations_per_order[product][-1]
    model.Add(completion_time_product[product] == completion_time[last_op])
    first_op = operations_per_order[product][0]
    model.Add(start_time_product[product] == start_times[first_op])

# 6.6. Product Sequencing Constraints on Machines
# Each product assigned to exactly one machine
for p in products_per_order:
    model.Add(sum([product_machine_assignment[(p, m)] for m in MACHINES]) == 1)

# Link operation machine assignment with product machine assignment
for p in products_per_order:
    ops_p = operations_per_order[p]
    for op in ops_p:
        op_type = op_instance_to_type[op]
        for m in operation_specifications[op_type]['machines']:
            model.AddBoolOr([operation_machine_speed_assignment[(op, m, s)] for s in speed_levels]).OnlyEnforceIf(product_machine_assignment[(p, m)])
            model.AddBoolAnd([operation_machine_speed_assignment[(op, m, s)].Not() for s in speed_levels]).OnlyEnforceIf(product_machine_assignment[(p, m)].Not())

# Before starting a new product, the previous product scheduled on the same machine must be completed
for m in MACHINES:
    for p1, p2 in combinations(products_per_order.keys(), 2):
        or_var = model.NewBoolVar(f"precedence_product_{p1}_{p2}_{m}")
        model.Add(completion_time_product[p1] <= start_time_product[p2]).OnlyEnforceIf(product_machine_assignment[(p1, m)]).OnlyEnforceIf(product_machine_assignment[(p2, m)]).OnlyEnforceIf(or_var)
        model.Add(completion_time_product[p2] <= start_time_product[p1]).OnlyEnforceIf(product_machine_assignment[(p1, m)]).OnlyEnforceIf(product_machine_assignment[(p2, m)]).OnlyEnforceIf(or_var.Not())

# 6.7. Completion Time for Orders
for order_id, order_info in ORDERS.items():
    for product_id, product in enumerate(order_info['products']):
        product_key = f"{product}#{product_id}_{order_id}"
        model.Add(completion_time_order[order_id] >= completion_time_product[product_key])
    # The order should be completed before the due date
    model.Add(completion_time_order[order_id] <= order_info['due_date'])

# 6.8. Makespan Constraints
for order_id in ORDERS:
    model.Add(makespan >= completion_time_order[order_id])

# 6.9. Machine Availability Constraints
# Prevent overlapping operations on the same machine
intervals_machine = {}  # Dictionary to hold interval variables for each operation on each machine
for m in MACHINES: intervals_machine[m] = []
for op in operation_instances:
    for m in operation_specifications[op_instance_to_type[op]]['machines']:
        assigned = model.NewBoolVar(f"assigned_{op}_{m}")
        # If operation is assigned to machine m, assigned = 1
        model.AddBoolOr([operation_machine_speed_assignment[(op, m, s)] for s in speed_levels]).OnlyEnforceIf(assigned)
        model.AddBoolAnd([operation_machine_speed_assignment[(op, m, s)].Not() for s in speed_levels]).OnlyEnforceIf(assigned.Not())

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

# 6.11 Operator Assignment Constraints
intervals_operator = [[] for _ in range(operators_groups_per_shift)]
for op in supervised_ops:
    # If operation requires supervision, assign exactly one operator
    model.AddExactlyOne([operator_operation_assignments[(op, operator_id)] for operator_id in range(operators_groups_per_shift)])

    # Create intervals for each operator assignment
    for operator_id in range(operators_groups_per_shift):
        intervals_operator[operator_id].append(model.NewOptionalIntervalVar(
            start=start_times[op],
            size=processing_time[op],  # Ensure this is correctly defined based on speed 's'
            end=completion_time[op],
            is_present=operator_operation_assignments[(op, operator_id)],
            name=f"interval_{op}_operator_{operator_id}"
        ))

# 6.12. Operator Shift Constraints
for operator_id in range(operators_groups_per_shift):
    model.AddNoOverlap(intervals_operator[operator_id])

# ============================
# 7. Objective Function
# ============================

# Define Makespan Minimization
# Penalize Speed-Up Usage to Minimize It
# The objective is a weighted sum of makespan and total speed-up penalties
speed_up_penalty = model.NewIntVar(0, scheduling_horizon, "speed_up_penalty")
# If there were speed levels >1.0, you'd calculate penalties accordingly
model.Add(speed_up_penalty == sum([(operation_machine_speed_assignment[(op, m, s)])*int(s*100-100) for op in operation_instances for m in operation_specifications[op_instance_to_type[op]]['machines'] for s in speed_levels]))

# Set a small weight for speed-up penalties to ensure they are only used when necessary
small_weight = 0.1  # Adjusted since speed_up_penalty is zero

# Objective: Minimize makespan + small_weight * speed_up_penalty
model.Minimize(makespan + small_weight * speed_up_penalty)

# ============================
# 8. Solve the Problem
# ============================

# Create a solver and minimize the makespan
solver = cp_model.CpSolver()
# solver.parameters.log_search_progress = True
# solver.parameters.log_to_stdout = True
solver.parameters.max_time_in_seconds = 60  # 10 minutes time limit
# make verbose output

status = solver.Solve(model)

# ============================
# 9. Output the Results
# ============================

print("Status:", solver.StatusName(status))

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    schedule = []
    for order_id, order_info in ORDERS.items():
        print(f"\nOrder: {order_id}")
        print(f"Due Date: {order_info['due_date']} hours")
        for product_id, product in enumerate(order_info['products']):
            product_key = f"{product}#{product_id}_{order_id}"
            print(f"  Product: {product_key}")
            ops = operations_per_order[product_key]
            for op in ops:
                # Find the machine and speed assignment
                assigned_m = None
                assigned_s = None
                for m in operation_specifications[op_instance_to_type[op]]['machines']:
                    for s in speed_levels:
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
                day_name = weekdays[(day + start_day) % 7]
                # Get completion time
                ct = solver.Value(completion_time[op])
                # Determine shift
                if operation_specifications[op_instance_to_type[op]]['requires_supervision']:
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
                print(f"      Requires Supervision: {operation_specifications[op_instance_to_type[op]]['requires_supervision']}")
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
    # 10. Graphical Visualization
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
        requires_supervision = operation_specifications[op_instance_to_type[operation_name]]['requires_supervision']

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
        if end_time - start_time > 2:  # Adjust this threshold as needed
            ax.text(
                x=start_time + (end_time - start_time) / 2,
                y=machine_name,
                s=f"{op_type} ({op_entry['Order']})",
                va='center',
                ha='center',
                color='black',
                fontsize=8
            )

    # Add day and night shift backgrounds
    for d in range(scheduling_horizon_in_days):
        # Define the absolute start and end times for shifts
        day_shift_start = d * 24 + working_hours[0]
        day_shift_end = d * 24 + working_hours[1]

        if d not in festive_days:
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
