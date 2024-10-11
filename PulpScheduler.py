import pulp
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpStatus
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy

# ============================
# 1. Data Definitions
# ============================

# Define Machines
machines = ['M1', 'M2','M3']  # List of machine names

# Define Orders
orders = {
    'Order1': {
        'due_date': 500,  # in hours
        'products': ['ProductA', 'ProductB']
    },
    'Order2': {
        'due_date': 500,  # in hours
        'products': ['ProductC']
    },
    'Order3': {
        'due_date': 500,  # in hours
        'products': ['ProductA', 'ProductC']
    },
    'Order4': {
        'due_date': 500,  # in hours
        'products': ['ProductB']
    },
    'Order5': {
        'due_date': 500,  # in hours
        'products': ['ProductA']
    },
    'Order6': {
        'due_date': 500,  # in hours
        'products': ['ProductB', 'ProductC']
    },
}

# Define Products with Cycles and Operations
products = {
    'ProductA': {
        'cycles': [
            {'operations': ['PrepareMachine', 'Op1']},
            {'operations': ['RechargeMachine', 'Op1']},
        ]
    },
    'ProductB': {
        'cycles': [
            {'operations': ['PrepareMachine', 'Op2']},
            {'operations': ['RechargeMachine', 'Op2']},
        ]
    },
    'ProductC': {
        'cycles': [
            {'operations': ['PrepareMachine', 'Op3']},
            {'operations': ['RechargeMachine', 'Op3']},
            {'operations': ['RechargeMachine', 'Op3']},
        ]
    }
}

# Define Operations with Machines, Base Times, and Supervision Requirement
operations_data = {
    'PrepareMachine': {'machines': ['M1', 'M2', 'M3'], 'base_time': {'M1': 5, 'M2': 3, 'M3': 5}, 'requires_supervision': True},
    'RechargeMachine': {'machines': ['M1', 'M2', 'M3'], 'base_time': {'M1': 4, 'M2': 2, 'M3': 3}, 'requires_supervision': True},
    'Op1': {'machines': ['M1','M3'], 'base_time': {'M1': 10, 'M3': 8}, 'requires_supervision': False},
    'Op2': {'machines': ['M2'], 'base_time': {'M2': 12}, 'requires_supervision': False},
    'Op3': {'machines': ['M1', 'M2'], 'base_time': {'M1': 8, 'M2': 6}, 'requires_supervision': False},
}

# Number of Days in Scheduling Horizon
max_time = max([order_info['due_date'] for order_info in orders.values()])
num_days = int(max_time / 24) + 1  # scheduling horizon in days

# Define Shift Parameters for Each Weekday
operators_per_shift = 5  # Number of operators available per shift

working_hours = (8, 16)  # Start and end times for the day shift

festive_days = []  # Saturday and Sunday are non-working days

# add saturday and sunday as non-working days
start_day = 0 # Day of the week to start scheduling (0 = Monday)
for i in range(num_days):
    day_of_week = (start_day + i) % 7
    if day_of_week == 5 or day_of_week == 6:
        festive_days.append(i)

# Speed Adjustment Factors
speed_levels = [1.0]  

# DEEMED NOT REQUIRED
# Maximum number of parallel supervised operations
# N_supervised_operations = 3  # Set your desired maximum here

# Big M and Epsilon Constants
big_M = 100000  # Increased Big M to accommodate finer time resolution
epsilon = 0.001  # Small epsilon relative to time resolution

# ============================
# 2. Precompute Shift Times
# ============================

# Precompute shift start and end times for each day
shift_start_times = {}
shift_end_times = {}
for d in range(num_days):
    day_of_week = d % 7
    if d not in festive_days:
        shift_start_times[d] = working_hours[0]
        shift_end_times[d] = working_hours[1]
    else:
        shift_start_times[d] = 0
        shift_end_times[d] = 0

# ============================
# 3. Create Unique Products and Operations per Order
# ============================

# Create unique products and operations per order
products_per_order = {}
operations_per_order = {}
operation_instances = []  # List to hold all operation instances
operation_precedences = {}  # Dictionary to hold the previous operation for each operation

for order_id, order_info in orders.items():
    for product in order_info['products']:
        product_key = f"{product}_{order_id}"
        product_cycles = copy.deepcopy(products[product]['cycles'])
        operations_per_order[product_key] = []

        # Adjust operation names to be unique per product per order
        for cycle_idx, cycle in enumerate(product_cycles):
            new_operations = []
            prev = None
            for op_idx, op in enumerate(cycle['operations']):
                op_instance = f"{op}_{product_key}_Cycle{cycle_idx + 1}_Op{op_idx + 1}"
                new_operations.append(op_instance)
                operations_per_order[product_key].append(op_instance)
                operation_instances.append(op_instance)
                operation_precedences[op_instance] = prev
                prev = op_instance
            product_cycles[cycle_idx]['operations'] = new_operations
        products_per_order[product_key] = {'cycles': product_cycles}

# Mapping Operations to Their Operation Type
op_instance_to_type = {}
for product_key, product_info in products_per_order.items():
    for cycle in product_info['cycles']:
        for op_instance in cycle['operations']:
            # Extract the base operation name (e.g., Op1 from Op1_ProductA_Order1)
            base_op = op_instance.split('_')[0]
            op_instance_to_type[op_instance] = base_op

supervised_ops = [op for op in operation_instances if operations_data[op_instance_to_type[op]]['requires_supervision']]
unsupervised_ops = [op for op in operation_instances if not operations_data[op_instance_to_type[op]]['requires_supervision']]

# Mapping Products to Orders
product_to_order = {}
for product_key in products_per_order:
    # Extract order_id from product name
    order_id = product_key.split('_')[-1]
    product_to_order[product_key] = order_id


# ============================
# 4. Initialize the Problem
# ============================

prob = LpProblem("JobShopSchedulingWithSupervision", LpMinimize)

# ============================
# 5. Decision Variables
# ============================

# 5.1. Machine and Speed Assignment Variables
x = {}
for op in operation_instances:
    op_type = op_instance_to_type[op]
    for m in operations_data[op_type]['machines']:
        for s in speed_levels:
            var_name = f"x_{op}_{m}_{int(s*100)}"
            x[(op, m, s)] = LpVariable(var_name, cat=LpBinary)

# 5.2. Start Time Variables (Total time since scheduling horizon start)
s_vars = {}
for op in operation_instances:
    s_vars[op] = LpVariable(f"s_{op}", lowBound=0, upBound=num_days * 24, cat='Continuous')  # Start time in total hours

# 5.3. Completion Time Variables
completion_time = {}
for op in operation_instances:
    completion_time[op] = LpVariable(f"completion_{op}", lowBound=0, cat='Continuous')

# 5.4. Binary Variables for Day Assignment
is_scheduled_on_day = {}
for op in operation_instances:
    for d in range(num_days):
        var_name = f"is_scheduled_on_day_{op}_d{d}"
        is_scheduled_on_day[(op, d)] = LpVariable(var_name, cat=LpBinary)

# 5.5. Completion Time for Products and Orders
completion_time_product = {}
for product in products_per_order:
    var_name = f"completion_product_{product}"
    completion_time_product[product] = LpVariable(var_name, lowBound=0, cat='Continuous')

completion_time_order = {}
for order_id in orders:
    var_name = f"completion_order_{order_id}"
    completion_time_order[order_id] = LpVariable(var_name, lowBound=0, cat='Continuous')

# 5.6. Makespan Variable
makespan = LpVariable("makespan", lowBound=0, cat='Continuous')

# 5.7. Product to Machine Assignment Variables
y_p_m = {}
for p in products_per_order:
    for m in machines:
        y_p_m[(p, m)] = LpVariable(f"y_{p}_{m}", cat=LpBinary)

# 5.8. Define earliest start time and latest finish time variables for each product on each machine
est_p_m = {}
lft_p_m = {}
for p in products_per_order:
    for m in machines:
        est_p_m[(p, m)] = LpVariable(f"est_{p}_{m}", lowBound=0, cat='Continuous')
        lft_p_m[(p, m)] = LpVariable(f"lft_{p}_{m}", lowBound=0, cat='Continuous')

# 5.9 Operator Assignment Variables
operator_assignments = {}
for op in supervised_ops:
    for operator_id in range(operators_per_shift):
        var_name = f"operator_{op}_id{operator_id}"
        operator_assignments[(op, operator_id)] = LpVariable(var_name, cat=LpBinary)

# ============================
# 6. Constraints
# ============================

# 6.1. Each Operation Assigned to Exactly One (Machine, Speed)
for op in operation_instances:
    op_type = op_instance_to_type[op]
    prob += lpSum([x[(op, m, s)] for m in operations_data[op_type]['machines'] for s in speed_levels]) == 1, f"OneMachineSpeed_{op}"

# 6.2. Define Processing Time Based on Machine and Speed
processing_time = {}
for op in operation_instances:
    op_type = op_instance_to_type[op]
    processing_time[op] = lpSum([
        x[(op, m, s)] * (operations_data[op_type]['base_time'][m] / s)
        for m in operations_data[op_type]['machines']
        for s in speed_levels
    ])

# 6.3. Start Time and Completion Time Constraints
for op in operation_instances:
    op_type = op_instance_to_type[op]
    # Define completion_time
    prob += completion_time[op] == s_vars[op] + processing_time[op], f"CompletionTime_{op}"

    # Assign operation to exactly one day
    prob += lpSum([is_scheduled_on_day[(op, d)] for d in range(num_days)]) == 1, f"OneDayAssignment_{op}"

    # Shift constraints based on supervision requirement
    if operations_data[op_type]['requires_supervision']:
        for d in range(num_days):
            shift_start = d * 24 + shift_start_times[d]
            shift_end = d * 24 + shift_end_times[d]
            # Enforce constraints only if operation is scheduled on day d
            prob += s_vars[op] >= shift_start - big_M * (1 - is_scheduled_on_day[(op, d)]), f"ShiftStart_{op}_d{d}" 
            prob += completion_time[op] <= shift_end + big_M * (1 - is_scheduled_on_day[(op, d)]), f"ShiftEnd_{op}_d{d}" 
    else:
        # For operations that do not require supervision, they can start anytime during the day shift
        for d in range(num_days):
            shift_start = d * 24 + shift_start_times[d]
            shift_end = d * 24 + shift_end_times[d]
            # If scheduled on day d, start time can be anywhere within the shift
            prob += s_vars[op] >= shift_start - big_M * (1 - is_scheduled_on_day[(op, d)]), f"ShiftStart_Independent_{op}_d{d}" 
            prob += s_vars[op] <= shift_end + big_M * (1 - is_scheduled_on_day[(op, d)]), f"ShiftEnd_Independent_{op}_d{d}"

        # Ensure that unsupervised operations start right after supervised operations
        prev_op = operation_precedences.get(op)
        if prev_op and prev_op in supervised_ops:
            prob += completion_time[prev_op] + epsilon == s_vars[op], f"UnsupervisedStartAfterSupervised_{op}"

    # Link is_scheduled_on_day variables with s_vars[op]
    for d in range(num_days):
        prob += s_vars[op] >= d * 24 - big_M * (1 - is_scheduled_on_day[(op, d)]), f"StartTimeLowerBound_{op}_d{d}"
        prob += s_vars[op] <= (d + 1) * 24 + big_M * (1 - is_scheduled_on_day[(op, d)]), f"StartTimeUpperBound_{op}_d{d}"


# 6.4. Precedence Constraints with Waiting Time
for product in products_per_order:
    ops = operations_per_order[product]
    for i in range(len(ops) - 1):
        op_prev = ops[i]
        op_next = ops[i + 1]
        # Ensure that op_next starts after op_prev completes
        prob += s_vars[op_next] >= completion_time[op_prev] + epsilon, f"Precedence_{op_prev}_to_{op_next}"

# 6.5. Completion Time for Products
for product in products_per_order:
    last_op = operations_per_order[product][-1]
    prob += completion_time_product[product] == completion_time[last_op], f"CompletionTimeProduct_{product}"

# 6.6. Product Sequencing Constraints on Machines
# Link est_p_m and lft_p_m with operation start and completion times
for p in products_per_order:
    ops_p = operations_per_order[p]
    for op in ops_p:
        op_type = op_instance_to_type[op]
        for m in operations_data[op_type]['machines']:
            x_op_m = lpSum([x[(op, m, s)] for s in speed_levels])
            # Only link if operation is assigned to machine m
            prob += s_vars[op] >= est_p_m[(p, m)] - big_M * (1 - x_op_m), f"EstLink_{op}_{m}"
            prob += completion_time[op] <= lft_p_m[(p, m)] + big_M * (1 - x_op_m), f"LftLink_{op}_{m}"

# Define binary variables z_p_q_m to indicate the sequencing of products on machines
z_p_q_m = {}
for m in machines:
    for p, q in combinations(products_per_order.keys(), 2):
        if p != q:
            var_name = f"z_{p}_{q}_m_{m}"
            z_p_q_m[(p, q, m)] = LpVariable(var_name, cat=LpBinary)

# Enforce sequencing constraints
for m in machines:
    for p, q in combinations(products_per_order.keys(), 2):
        if p != q:
            # Constraints only apply if both products are assigned to machine m
            prob += lft_p_m[(p, m)] <= est_p_m[(q, m)] + big_M * (1 - z_p_q_m[(p, q, m)]) + big_M * (2 - y_p_m[(p, m)] - y_p_m[(q, m)]), f"Seq1_{p}_{q}_{m}"
            prob += lft_p_m[(q, m)] <= est_p_m[(p, m)] + big_M * z_p_q_m[(p, q, m)] + big_M * (2 - y_p_m[(p, m)] - y_p_m[(q, m)]), f"Seq2_{p}_{q}_{m}"

# 6.7. Completion Time for Orders
for order_id, order_info in orders.items():
    for product in order_info['products']:
        product_key = f"{product}_{order_id}"
        prob += completion_time_order[order_id] >= completion_time_product[product_key], f"OrderCompletion_{order_id}_Product_{product_key}"
    prob += completion_time_order[order_id] <= order_info['due_date'], f"OrderDueDate_{order_id}"

# 6.8. Makespan Constraints
for order_id in orders:
    prob += makespan >= completion_time_order[order_id], f"MakespanConstraint_{order_id}"

# 6.9. Machine Availability Constraints
for m in machines:
    ops_on_m = [op for op in operation_instances if m in operations_data[op_instance_to_type[op]]['machines']]
    for op1, op2 in combinations(ops_on_m, 2):
        # To prevent overlapping only if both operations are assigned to machine m
        # Introduce y binary variable to indicate ordering
        y = LpVariable(f"y_{op1}_{op2}_m{m}", cat=LpBinary)
        prob += s_vars[op1] + processing_time[op1] <= s_vars[op2] + big_M * (1 - y) + big_M * (2 - lpSum([x[(op1, m, s)] for s in speed_levels]) - lpSum([x[(op2, m, s)] for s in speed_levels])), f"NoOverlap1_{op1}_{op2}_m{m}"
        prob += s_vars[op2] + processing_time[op2] <= s_vars[op1] + big_M * y + big_M * (2 - lpSum([x[(op1, m, s)] for s in speed_levels]) - lpSum([x[(op2, m, s)] for s in speed_levels])), f"NoOverlap2_{op1}_{op2}_m{m}"

# 6.10. Product to Machine Assignment Constraints

# Each product assigned to exactly one machine
for p in products_per_order:
    prob += lpSum([y_p_m[(p, m)] for m in machines]) == 1, f"ProductAssignedToOneMachine_{p}"

# Link operation machine assignment with product machine assignment
for p in products_per_order:
    ops_p = operations_per_order[p]
    for op in ops_p:
        op_type = op_instance_to_type[op]
        for m in operations_data[op_type]['machines']:
            for s in speed_levels:
                prob += x[(op, m, s)] <= y_p_m[(p, m)], f"OpMachineLink_{op}_{m}_{s}"

# 6.11 Operator Availability Constraints
# Each supervised operation must be assigned to exactly one operator
for op in supervised_ops:
    prob += lpSum([operator_assignments[(op, operator_id)] for operator_id in range(operators_per_shift)]) == 1, f"OperatorAssignment_{op}"

# Prevent overlapping operations for each operator
operator_overlap_order = {}
for operator_id in range(operators_per_shift):
    for op1, op2 in combinations(supervised_ops, 2):
        y_var = LpVariable(f"y_{op1}_{op2}_operator_{operator_id}", cat='Binary')
        operator_overlap_order[(op1, op2, operator_id)] = y_var
        # If both operations are assigned to the same operator, they must not overlap
        prob += s_vars[op1] + processing_time[op1] <= s_vars[op2] + big_M * (1 - y_var) + big_M * (2 - operator_assignments[(op1, operator_id)] - operator_assignments[(op2, operator_id)]), f"OperatorOverlap1_{op1}_{op2}_operator_{operator_id}"
        prob += s_vars[op2] + processing_time[op2] <= s_vars[op1] + big_M * y_var + big_M * (2 - operator_assignments[(op1, operator_id)] - operator_assignments[(op2, operator_id)]), f"OperatorOverlap2_{op1}_{op2}_operator_{operator_id}"


# ============================
# 7. Objective Function
# ============================

# Define Makespan Minimization
# Penalize Speed-Up Usage to Minimize It
# The objective is a weighted sum of makespan and total speed-up penalties
speed_up_penalty = lpSum([
    x[(op, m, s)] * (s - 1.0)
    for op in operation_instances
    for m in operations_data[op_instance_to_type[op]]['machines']
    for s in speed_levels if s > 1.0
])

# Set a small weight for speed-up penalties to ensure they are only used when necessary
small_weight = 0.1

prob += makespan + small_weight * speed_up_penalty, "Minimize_Makespan_and_SpeedUp"

# ============================
# 8. Solve the Problem
# ============================

prob.solve(pulp.PULP_CBC_CMD(timeLimit=60))

# ============================
# 9. Output the Results
# ============================

print("Status:", LpStatus[prob.status])

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

if LpStatus[prob.status] == 'Optimal':
    schedule = []
    for order_id, order_info in orders.items():
        print(f"\nOrder: {order_id}")
        print(f"Due Date: {order_info['due_date']} hours")
        for product in order_info['products']:
            product_key = f"{product}_{order_id}"
            print(f"  Product: {product_key}")
            ops = operations_per_order[product_key]
            for op in ops:
                # Find the machine and speed assignment
                assigned_m = None
                assigned_s = None
                for m in operations_data[op_instance_to_type[op]]['machines']:
                    for s_level in speed_levels:
                        if pulp.value(x[(op, m, s_level)]) == 1:
                            assigned_m = m
                            assigned_s = s_level
                            break
                    if assigned_m is not None:
                        break
                # Get start time and day
                st = pulp.value(s_vars[op])
                day = int(st // 24)
                s_op = st % 24
                day_name = weekdays[(day + start_day) % 7]
                # Get completion time
                ct = pulp.value(completion_time[op])
                # Determine shift
                if operations_data[op_instance_to_type[op]]['requires_supervision']:
                    op_shift = 'Day'
                else:
                    op_shift = 'Independent'
                schedule.append({
                    'Operation': op,
                    'Product': product_key,
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
                print(f"      Start Time: {st:.2f} hours (Day {day + 1} - {day_name} at {s_op:.2f} hours)")
                print(f"      Completion Time: {ct:.2f} hours")
                print(f"      Requires Supervision: {operations_data[op_instance_to_type[op]]['requires_supervision']}")
            # Get product completion time
            ct_product = pulp.value(completion_time_product[product_key])
            print(f"  Completion Time for {product_key}: {ct_product:.2f} hours")
        # Get order completion time
        for order_id_inner in orders:
            if order_id_inner == order_id:
                ct_order = pulp.value(completion_time_order[order_id_inner])
                print(f"Completion Time for {order_id_inner}: {ct_order:.2f} hours")
    print("\nMakespan =", pulp.value(makespan))
    print("Total Speed-Up Penalty =", pulp.value(speed_up_penalty))
    
    # ============================
    # 10. Graphical Visualization
    # ============================

    # Assign a unique color to each operation type using a color palette
    products_types = list(products.keys())
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
        product_key = operation_name.split('_')[1]
        machine_name = op_entry['Machine']
        start_time = op_entry['Start']
        end_time = op_entry['End']
        requires_supervision = operations_data[op_instance_to_type[operation_name]]['requires_supervision']

        # Check for valid start and end times
        if start_time is None or end_time is None:
            print(f"Invalid Start or End time for operation {operation_name}")
            continue

        color = product_type_colors.get(product_key, 'gray')  # Default to gray if not found
        hatch = hatch_patterns.get(requires_supervision, '')

        # Add a unique label for each operation type to the legend
        if product_key not in added_operation_labels:
            label_operation = product_key
            added_operation_labels.add(product_key)
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
    for d in range(num_days):
        # Define the absolute start and end times for shifts
        day_shift_start = d * 24 + shift_start_times[d]
        day_shift_end = d * 24 + shift_end_times[d]

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

    if prob.status == pulp.LpStatusInfeasible:
        print("The problem is infeasible. Here are some potential reasons:")
        print("- The combination of constraints might be too restrictive.")
        print("- The scheduling horizon may not be sufficient to accommodate all operations.")
        print("- The number of available operators may be insufficient.")
        print("- Product-to-machine assignments may conflict with operation durations and supervision requirements.")
