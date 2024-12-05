from datetime import datetime
from data_init import init_csv_data
from gantt import plot_gantt_chart
from solver import solve
import os

def run_solver(
    horizon_days,
    broken_machines,
    maintenance,
    time_units_in_a_day,
    start_shift,
    end_shift,
    festive_days,
    num_operators_groups,
    num_operators_per_group,
    setup_cost,
    load_cost,
    unload_cost,
    max_seconds,
    second_run_seconds,
    run_ga,
    now=datetime.now(),
):
    # Paths to the files saved above
    COMMON_P_PATH = "input/new_orders.csv"
    J_COMPATIBILITY_PATH = "input/articoli_macchine.json"
    M_INFO_PATH = "input/macchine_info.json"
    RUNNING_P_PATH = "input/running_products.csv"
    ARTICLE_LIST_PATH = "input/lista_articoli.csv"

    # Parse inputs
    try:
        broken_machines = [int(x) for x in broken_machines.split(",") if x.strip()]
    except ValueError:
        return (
            "❌ Invalid input for Broken Machines. Please enter a comma-separated list of integers.",
            None,
        )

    try:
        festive_days = [int(x) for x in festive_days.split(",") if x.strip()]
    except ValueError:
        return (
            "❌ Invalid input for Festive Days. Please enter a comma-separated list of integers.",
            None,
        )

    try:
        load_cost = (
            load_cost / 256 / num_operators_per_group
        )  # Convert to cost per fuso
        unload_cost = (
            unload_cost / 256 / num_operators_per_group
        )  # Convert to cost per fuso
    except Exception as e:
        return f"❌ Error processing load/unload costs: {str(e)}", None

    try:
        maintenance = [
            tuple(map(int, x.strip().strip("()").split(",")))
            for x in maintenance.split("),")
            if x.strip()
        ]
    except ValueError:
        return (
            "❌ Invalid input for Maintenance. Please enter a comma-separated list of tuples like (1,78,2), (3,10,3).",
            None,
        )

    maintenance_schedule = {}
    for machine, day, duration in maintenance:
        maintenance_schedule[machine] = maintenance_schedule.get(machine, []) + [
            (day, duration)
        ]

    # Check if all required files exist
    required_files = [
        COMMON_P_PATH,
        J_COMPATIBILITY_PATH,
        M_INFO_PATH,
        ARTICLE_LIST_PATH,
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        return f"❌ Missing files: {', '.join(missing_files)}", None

    try:
        # Variable referring to starting of scheduling time (can be defined customly for testing purposes)
        # There's no guarantees for it to work properly if setting now to future dates
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
            costs=(setup_cost, load_cost, unload_cost),
            now=now,
            time_units_in_a_day=time_units_in_a_day,
        )
    except Exception as e:
        return f"❌ Error initializing data: {str(e)}", None

    try:
        # Solve the scheduling problem
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
            maintenance_schedule,
            num_operators_groups,
            festive_days,
            horizon_days,
            time_units_in_a_day,
            start_shift,
            end_shift,
            max_seconds,
            second_run_seconds,
            run_ga,
            now,
        )

        print("Generated valid schedule")
        # Convert the schedule to a string to display
        schedule_str = str(schedule)

        # Prepare data for plotting
        # Build max_cycles
        max_cycles = {}
        for p, prod in schedule.products:
            if prod.setup_beg:
                max_cycle = max(prod.setup_beg.keys())
                max_cycles[prod.id] = max_cycle + 1  # Assuming cycles are zero-indexed
            else:
                max_cycles[prod.id] = 0

        # Get num_machines
        used_machines = set()
        for p, prod in schedule.products:
            for c in prod.setup_beg.keys():
                used_machines.add(prod.machine[c])
        num_machines = len(used_machines)

        # Calculate horizon
        horizon = 0
        for p, prod in schedule.products:
            for c in prod.setup_beg.keys():
                horizon = max(horizon, prod.cycle_end[c])
                # Also consider unload_end times
                for l in range(prod.num_levate[c]):
                    if (c, l) in prod.unload_end.keys():
                        horizon = max(horizon, prod.unload_end[c, l])

        # Define prohibited_intervals (if available), otherwise set to empty list
        prohibited_intervals = (
            schedule.invalid_intervals if hasattr(schedule, "invalid_intervals") else []
        )

        # Define time_units_from_midnight (if available), otherwise set to 0
        time_units_from_midnight = 0  # Update as per your requirements

        print("Generating Gantt chart")

        # Call the plotting function
        fig = plot_gantt_chart(
            production_schedule=schedule,
            max_cycles=max_cycles,
            num_machines=num_machines,
            horizon=horizon,
            prohibited_intervals=prohibited_intervals,
            time_units_from_midnight=time_units_from_midnight,
        )

        print("Generated Gantt chart")
        # Return the schedule and the Gantt chart figure
        return schedule_str, fig

    except Exception as e:
        return f"❌ An error occurred during solving: {str(e)}", None
