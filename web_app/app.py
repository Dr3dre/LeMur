import functools
import gradio as gr
import pandas as pd
import json
import os
from utils import run_solver

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

def upload_csv_file(file, columns):
    if file is not None:
        try:
            df = pd.read_csv(file.name)

            # verify if the columns are at least the same as the ones in the csv file
            for col in columns:
                if col not in df.columns:
                    return f"❌ Invalid CSV file. Please make sure it contains the following columns: {', '.join(columns)}"

            # filter out columns that are not in the columns list
            df = df[columns]

            return df
        except Exception as e:
            return f"Error reading CSV file: {str(e)}"

    return (pd.DataFrame(),)


def save_csv_file(df, filename):
    try:
        df.to_csv("input/" + filename, index=False)
        return f"✅ `{filename}` saved successfully."
    except Exception as e:
        return f"❌ Error saving `{filename}`: {str(e)}"


def upload_file(file_path):
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            raise f"Error reading file: {str(e)}"
    else:
        raise "No file uploaded."


def upload_article_machine_compatibility(file_path):
    # read the file and verify that it's a valid json file with "article_id" : [machine_id, ...] format
    content = upload_file(file_path)
    try:
        data = json.loads(content)
        if not all(
            isinstance(k, str) and isinstance(v, list) and isinstance(i, int)
            for k, v in data.items()
            for i in v
        ):
            return "❌ Invalid JSON format. Please ensure the file is a dictionary with string keys and list values."
        return content
    except json.JSONDecodeError as e:
        return f"❌ JSON Decode Error: {str(e)}"


def upload_machine_info(file_path):
    # read the file and verify that it's a valid json file with "machine_id" : { "n_fusi": int } format and remove any extra fields
    content = upload_file(file_path)
    try:
        data = json.loads(content)
        if not all(
            isinstance(k, str) and isinstance(v, dict) and "n_fusi" in v
            for k, v in data.items()
        ):
            return "❌ Invalid JSON format. Please ensure the file is a dictionary with integer keys and 'n_fusi' values."

        # remove any extra fields
        for k, v in data.items():
            data[k] = {"n_fusi": v["n_fusi"]}

        content = json.dumps(data, indent=4)

        return content
    except json.JSONDecodeError as e:
        return f"❌ JSON Decode Error: {str(e)}"


def save_common_products(df):
    return save_csv_file(df, "input/new_orders.csv")


def save_article_machine_compatibility(content):
    try:
        data = json.loads(content)
        with open("input/articoli_macchine.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return "✅ `articoli_macchine.json` saved successfully."
    except json.JSONDecodeError as e:
        return f"❌ JSON Decode Error: {str(e)}"
    except Exception as e:
        return f"❌ Error saving `articoli_macchine.json`: {str(e)}"


def save_machine_info(content):
    try:
        data = json.loads(content)
        with open("input/macchine_info.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return "✅ `macchine_info.json` saved successfully."
    except json.JSONDecodeError as e:
        return f"❌ JSON Decode Error: {str(e)}"
    except Exception as e:
        return f"❌ Error saving `macchine_info.json`: {str(e)}"


def save_article_list(df):
    return save_csv_file(df, "input/lista_articoli.csv")


def save_running_products(df):
    return save_csv_file(df, "input/running_products.csv")

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## LeMur Scheduling Web App")
    with gr.Tabs():

        # Tab 1: Article-Machine Compatibility
        with gr.TabItem("Upload Article-Machine Compatibility (JSON)"):
            gr.Markdown("### Upload and Edit `articoli_macchine.json`")
            article_machine_file = gr.File(
                label="Upload `articoli_macchine.json`", type="filepath"
            )
            article_machine_content = gr.Code(
                label="Edit `articoli_macchine.json` content",
                language="json",
                max_lines=20,
            )
            article_machine_file.change(
                fn=upload_article_machine_compatibility,
                inputs=article_machine_file,
                outputs=article_machine_content,
            )
            save_article_machine_button = gr.Button("Save `articoli_macchine.json`")
            save_article_machine_status = gr.Textbox(label="Status", interactive=False)
            save_article_machine_button.click(
                fn=save_article_machine_compatibility,
                inputs=article_machine_content,
                outputs=save_article_machine_status,
            )

        # Tab 2: Machine Info
        with gr.TabItem("Upload Machine Info (JSON)"):
            gr.Markdown("### Upload and Edit `macchine_info.json`")
            machine_info_file = gr.File(
                label="Upload `macchine_info.json`", type="filepath"
            )
            machine_info_content = gr.Code(
                label="Edit `macchine_info.json` content", language="json", max_lines=20
            )
            machine_info_file.change(
                fn=upload_machine_info,
                inputs=machine_info_file,
                outputs=machine_info_content,
            )
            save_machine_info_button = gr.Button("Save `macchine_info.json`")
            save_machine_info_status = gr.Textbox(label="Status", interactive=False)
            save_machine_info_button.click(
                fn=save_machine_info,
                inputs=machine_info_content,
                outputs=save_machine_info_status,
            )

        # Tab 3: Article List
        with gr.TabItem("Upload Article List (CSV)"):
            gr.Markdown("### Upload and Edit `lista_articoli.csv`")
            article_list_file = gr.File(
                label="Upload `lista_articoli.csv`", type="filepath"
            )
            article_list_columns = [
                "codarticolo",
                "kg_ora",
                "no_cicli",
                "ore_levata",
            ]
            article_list_content = gr.Dataframe(
                label="Edit `lista_articoli.csv` content",
                datatype=[
                    "str",
                    "number",
                    "number",
                    "number",
                ],  # Adjust based on your CSV columns
                headers=article_list_columns,
                row_count=10,  # Replaced max_rows with row_count
            )
            article_list_file.change(
                fn=functools.partial(upload_csv_file, columns=article_list_columns),
                inputs=article_list_file,
                outputs=article_list_content,
            )
            save_article_list_button = gr.Button("Save `lista_articoli.csv`")
            save_article_list_status = gr.Textbox(label="Status", interactive=False)
            save_article_list_button.click(
                fn=lambda df: save_csv_file(df, "lista_articoli.csv"),
                inputs=article_list_content,
                outputs=save_article_list_status,
            )

        # Tab 4: Common Products
        with gr.TabItem("Upload New Orders (CSV)"):
            gr.Markdown("### Upload and Edit `new_orders.csv`")
            common_products_file = gr.File(
                label="Upload `new_orders.csv`", type="filepath"
            )
            common_products_columns = [
                "cliente",
                "cod_articolo",
                "quantity",
                "data inserimento",
                "data consegna",
            ]
            common_products_content = gr.Dataframe(
                label="Edit `new_orders.csv` content",
                datatype=[
                    "str",
                    "str",
                    "number",
                    "date",
                    "date",
                ],  # Adjust based on your CSV columns
                headers=common_products_columns,
                row_count=10,  # Replaced max_rows with row_count
            )
            common_products_file.change(
                fn=functools.partial(upload_csv_file, columns=common_products_columns),
                inputs=common_products_file,
                outputs=common_products_content,
            )
            save_common_products_button = gr.Button("Save `new_orders.csv`")
            save_common_products_status = gr.Textbox(label="Status", interactive=False)
            save_common_products_button.click(
                fn=lambda df: save_csv_file(df, "new_orders.csv"),
                inputs=common_products_content,
                outputs=save_common_products_status,
            )

        # Tab 5: Running Products
        with gr.TabItem("Upload Running Products (CSV)"):
            gr.Markdown("### Upload and Edit `running_products.csv`")
            running_products_file = gr.File(
                label="Upload `running_products.csv`", type="filepath"
            )
            running_products_columns = [
                "cliente",
                "macchina",
                "cod_articolo",
                "quantity",
                "fine operazione attuale",
                "data consegna",
                "levate rimanenti ciclo",
                "tipo operazione attuale",
                "operatore",
            ]
            running_products_content = gr.Dataframe(
                label="Edit `running_products.csv` content",
                datatype=[
                    "str",
                    "str",
                    "str",
                    "number",
                    "date",
                    "date",
                    "number",
                    "number",
                    "number",
                ],  # Adjust based on your CSV columns
                headers=running_products_columns,
                row_count=10,  # Replaced max_rows with row_count
            )
            gr.Markdown(
                "The tipo operazione attuale can be 0-Setup, 1-Load, 2-machine running, 3-Unload"
            )
            gr.Markdown(
                "The operatore is the operator number assigned to the operation (NOTE: it is 0-based, so we start from 0)"
            )
            running_products_file.change(
                fn=functools.partial(upload_csv_file, columns=running_products_columns),
                inputs=running_products_file,
                outputs=running_products_content,
            )
            save_running_products_button = gr.Button("Save `running_products.csv`")
            save_running_products_status = gr.Textbox(label="Status", interactive=False)
            save_running_products_button.click(
                fn=lambda df: save_csv_file(df, "running_products.csv"),
                inputs=running_products_content,
                outputs=save_running_products_status,
            )

        # Tab 6: Solver Configuration
        with gr.TabItem("Solver Configuration"):
            gr.Markdown("## General Configuration")
            # max seconds for the first run of the solver
            max_seconds = gr.Slider(
                label="Max Seconds for makespan minimization",
                minimum=30,
                maximum=60 * 100,
                step=1,
                value=60,
            )
            # second run of solve time
            second_run_seconds = gr.Slider(
                label="Max seconds for compactness refinement",
                minimum=0,
                maximum=60 * 100,
                step=1,
                value=60,
            )
            # run GA, num of generations and population size
            run_ga = gr.Slider(
                label="Genetic refinement number of generations",
                minimum=0,
                maximum=500,
                step=1,
                value=250,
            )

            # scheduling horizon in days
            horizon_days = gr.Slider(
                label="Scheduling Horizon (Days)",
                minimum=1,
                maximum=365,
                step=1,
                value=30,
            )
            # time units in a day
            time_units_in_a_day = gr.Radio(
                label="Time Units in a Day", choices=[24, 48, 96], value=24
            )
            gr.Markdown(
                "Ensure all the following values are set according to the time units in a day!\n\n"
            )

            gr.Markdown("## Machine Configurations")
            # broken machines
            gr.Markdown(
                "Enter the broken machines as a comma-separated list of integers (e.g., 1, 3, 5)."
            )
            broken_machines = gr.Textbox(
                label="Broken Machines", placeholder="e.g., 1, 3, 5", value=""
            )
            # maintenance, list of tuples (machine, time_unit_start, duration)
            gr.Markdown(
                "Enter the maintenance schedule as a comma-separated list of tuples (machine, time_unit_start, duration)."
            )
            maintenance = gr.Textbox(
                label="Maintenance",
                placeholder="e.g., (1, 78, 2), (3, 10, 3)",
                value="",
            )

            gr.Markdown("## Operator Configurations")
            # start and end shift times
            gr.Markdown(
                "Configure the working shift timings and the festive days (as a list of days, note that sundays are already accounted for)."
            )
            start_shift = gr.Slider(
                label="Start Shift", minimum=0, maximum=23, step=1, value=8
            )
            end_shift = gr.Slider(
                label="End Shift", minimum=0, maximum=23, step=1, value=16
            )
            # festive days (excluding sundays) as a list of integers [0,1,8,...]
            festive_days = gr.Textbox(
                label="Festive Days", placeholder="e.g., 1, 8, 25", value=""
            )
            # operators
            gr.Markdown("Enter the number of operators available for scheduling.")
            num_operators_groups = gr.Slider(
                label="Number of Operators groups",
                minimum=1,
                maximum=5,
                step=1,
                value=2,
            )
            num_operators_per_group = gr.Slider(
                label="Operators per Group", minimum=1, maximum=5, step=1, value=4
            )
            # cost factors
            gr.Markdown(
                """
Enter the cost factors for setup, load, and unload operations.
Setup cost is fixed for all machines.
Load and unload costs are the time units needed for a single person to load a single machine with 256 fusi."""
            )
            setup_cost = gr.Slider(
                label="Setup Cost", minimum=0, maximum=10, step=0.1, value=4
            )
            load_cost = gr.Slider(
                label="Load Cost", minimum=0, maximum=10, step=0.1, value=6
            )
            unload_cost = gr.Slider(
                label="Unload Cost", minimum=0, maximum=10, step=0.1, value=2
            )

        # Tab 7: Run Solver
        with gr.TabItem("Run Solver"):
            gr.Markdown("### Execute the Scheduler and View Results")
            # Run the solver
            run_solver_button = gr.Button("Run Solver")
            run_solver_status = gr.Textbox(
                label="Solver Output", lines=10, interactive=False
            )
            gantt_output = gr.Plot(
                label="Gantt Chart"
            )  # Replaced gr.Plot with gr.Plotly
            run_solver_button.click(
                fn=run_solver,
                inputs=[
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
                ],
                outputs=[run_solver_status, gantt_output],
            )

    gr.Markdown("### Note:")
    gr.Markdown(
        """
- Ensure all required files are uploaded and saved before running the solver.
- The Gantt chart visualizes the scheduling of operations on machines over time.
- Time units are based on the solver's configuration (e.g., hours).
- Machines are 1-indexed.
    """
    )

# Launch the Gradio app
demo.launch()
