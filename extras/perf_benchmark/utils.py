from datetime import datetime
import sys

sys.path.append("..\\..\\web_app")

from data_init import init_csv_data
from solver import solve
import os


def run_solver(
    p_path,
    r_path,
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
    COMMON_P_PATH = p_path
    J_COMPATIBILITY_PATH = "../../web_app/input/articoli_macchine.json"
    M_INFO_PATH = "../../web_app/input/macchine_info.json"
    RUNNING_P_PATH = r_path
    ARTICLE_LIST_PATH = "../../web_app/input/lista_articoli.csv"

    # Parse inputs
    try:
        broken_machines = [int(x) for x in broken_machines.split(",") if x.strip()]
    except ValueError:
        raise "❌ Invalid input for Broken Machines. Please enter a comma-separated list of integers."

    try:
        festive_days = [int(x) for x in festive_days.split(",") if x.strip()]
    except ValueError:
        raise "❌ Invalid input for Festive Days. Please enter a comma-separated list of integers."

    try:
        load_cost = (
            load_cost / 256 / num_operators_per_group
        )  # Convert to cost per fuso
        unload_cost = (
            unload_cost / 256 / num_operators_per_group
        )  # Convert to cost per fuso
    except Exception as e:
        raise f"❌ Error processing load/unload costs: {str(e)}"

    try:
        maintenance = [
            tuple(map(int, x.strip().strip("()").split(",")))
            for x in maintenance.split("),")
            if x.strip()
        ]
    except ValueError:
        raise "❌ Invalid input for Maintenance. Please enter a comma-separated list of tuples like (1,78,2), (3,10,3)."

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
        raise f"❌ Missing files: {', '.join(missing_files)}"

    try:
        # Variable referring to starting of scheduling time (can be defined customly for testing purposes)
        # There's no guarantees for it to work properly if setting now to future dates
        (
            common_products,
            running_products,
            article_to_machine_comp,
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
        raise f"❌ Error initializing data: {str(e)}"

    # Solve the scheduling problem
    schedule = solve(
        common_products,
        running_products,
        article_to_machine_comp,
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

    return schedule_str
