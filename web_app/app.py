import functools
import gradio as gr
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

# Import your solver and data_init functions
from solver import solve
from data_init import init_csv_data, RunningProduct

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
        df.to_csv("input/"+filename, index=False)
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
        if not all(isinstance(k, str) and isinstance(v, list) and isinstance(i,int) for k, v in data.items() for i in v):
            return "❌ Invalid JSON format. Please ensure the file is a dictionary with string keys and list values."
        return content
    except json.JSONDecodeError as e:
        return f"❌ JSON Decode Error: {str(e)}"

def upload_machine_info(file_path):
    # read the file and verify that it's a valid json file with "machine_id" : { "n_fusi": int } format and remove any extra fields
    content = upload_file(file_path)
    try:
        data = json.loads(content)
        if not all(isinstance(k, str) and isinstance(v, dict) and "n_fusi" in v for k, v in data.items()):
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
):
    # Paths to the files saved above
    COMMON_P_PATH = "input/new_orders.csv"
    J_COMPATIBILITY_PATH = "input/articoli_macchine.json"
    M_INFO_PATH = "input/macchine_info.json"
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
        load_cost = load_cost / 256  / num_operators_per_group      # Convert to cost per fuso
        unload_cost = unload_cost / 256  / num_operators_per_group  # Convert to cost per fuso
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
        # Initialize data
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
            J_COMPATIBILITY_PATH,
            M_INFO_PATH,
            ARTICLE_LIST_PATH,
            costs=(setup_cost, load_cost, unload_cost),
        )

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
        fig = plot_gantt_chart_1(
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


def plot_gantt_chart_1(
    production_schedule,
    max_cycles,
    num_machines,
    horizon,
    prohibited_intervals,
    time_units_from_midnight,
):
    """
    Plot an interactive Gantt chart for the given job schedule with enhanced readability and navigation.
    """
    # Define start time (hour 0) for the x-axis
    start_time = 0  # Assuming time is measured in hours

    # Identify used machines
    used_machines = set()
    for p, prod in production_schedule.products:
        for c in range(max_cycles[prod.id]):
            if c in prod.machine.keys():
                used_machines.add(prod.machine[c])
    used_machines = sorted(used_machines)
    machine_id_map = {machine: idx for idx, machine in enumerate(used_machines)}
    updated_num_machines = len(used_machines)

    # Initialize Plotly figure
    fig = go.Figure()

    # Add prohibited intervals (e.g., non-working hours) as shapes
    for interval in prohibited_intervals:
        x_start, x_end = interval
        fig.add_vrect(
            x0=x_start,
            x1=x_end,
            fillcolor="gray",
            opacity=0.4,
            layer="below",
            line_width=0,
            annotation_text="Non-working Hours",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="black",
            annotation_bgcolor="rgba(0,0,0,0)",
        )

    # Mark past time as a red shaded area
    fig.add_vrect(
        x0=start_time,
        x1=time_units_from_midnight,
        fillcolor="red",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="Past",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="black",
        annotation_bgcolor="rgba(0,0,0,0)",
    )

    # Colormap for different products
    cmap = plt.get_cmap("tab20")
    colors = [
        f"rgba{tuple(int(c * 255) for c in cmap(i % 20)[:3]) + (0.8,)}"
        for i in range(len(production_schedule.products))
    ]

    # Dictionaries to group operations by type for performance
    setup_traces = {}
    cycle_traces = {}
    load_traces = {}
    unload_traces = {}

    # Plot each job
    for idx, (p, prod) in enumerate(production_schedule.products):
        color = colors[idx]
        for c in range(max_cycles[prod.id]):
            if c in prod.machine.keys():
                machine = prod.machine[c]
                machine_pos = machine_id_map[machine]

                # Setup times
                setup_start = prod.setup_beg[c]  # Assuming in hours
                setup_end = prod.setup_end[c]
                setup_duration = setup_end - setup_start

                # Cycle times
                cycle_start = setup_end  # Assuming cycle starts after setup
                cycle_end = prod.cycle_end[c]
                cycle_duration = cycle_end - cycle_start

                # Add Production Cycle Bar
                cycle_key = f"Cycle_P{prod.id}_C{c}"
                if cycle_key not in cycle_traces:
                    cycle_traces[cycle_key] = {
                        "x": [],
                        "y": [],
                        "base": [],
                        "hovertext": [],
                        "color": color,
                        "line_color": (
                            "red"
                            if isinstance(prod, RunningProduct) and c == 0
                            else "black"
                        ),
                        "line_width": (
                            2 if isinstance(prod, RunningProduct) and c == 0 else 1
                        ),
                    }
                cycle_traces[cycle_key]["x"].append(cycle_duration)
                cycle_traces[cycle_key]["y"].append(machine_pos)
                cycle_traces[cycle_key]["base"].append(cycle_start)
                cycle_traces[cycle_key]["hovertext"].append(
                    f"Product ID: {prod.id}<br>"
                    f"Article: {prod.article}<br>"
                    f"Velocity: {prod.velocity[c]}<br>"
                    f"Cycle: {c}"
                )

                # Add Setup Bar
                setup_key = f"Setup_P{prod.id}_C{c}"
                if setup_key not in setup_traces:
                    setup_traces[setup_key] = {
                        "x": [],
                        "y": [],
                        "base": [],
                        "hovertext": [],
                        "color": "rgba(255,0,0,0)",  # Transparent fill
                        "line_color": "red",
                        "line_width": 2,
                    }
                setup_traces[setup_key]["x"].append(setup_duration)
                setup_traces[setup_key]["y"].append(machine_pos)
                setup_traces[setup_key]["base"].append(setup_start)
                setup_traces[setup_key]["hovertext"].append(
                    f"Job Setup for Product ID: {prod.id}, Cycle: {c}"
                )

                # Add Load/Unload Bars
                for l in range(prod.num_levate[c]):
                    # Load
                    load_key = f"Load_P{prod.id}_C{c}_L{l}"
                    if load_key not in load_traces:
                        load_traces[load_key] = {
                            "x": [],
                            "y": [],
                            "base": [],
                            "hovertext": [],
                            "color": "rgba(0, 128, 0, 0.8)",  # Green with opacity
                            "line_color": "green",
                            "line_width": 1.5,
                        }
                    load_start = prod.load_beg[c, l]
                    load_end = prod.load_end[c, l]
                    load_duration = load_end - load_start
                    load_traces[load_key]["x"].append(load_duration)
                    load_traces[load_key]["y"].append(machine_pos)
                    load_traces[load_key]["base"].append(load_start)
                    load_traces[load_key]["hovertext"].append(
                        f"Load for Product ID: {prod.id}, Cycle: {c}, Load: {l}"
                    )

                    # Unload
                    unload_key = f"Unload_P{prod.id}_C{c}_U{l}"
                    if unload_key not in unload_traces:
                        unload_traces[unload_key] = {
                            "x": [],
                            "y": [],
                            "base": [],
                            "hovertext": [],
                            "color": "rgba(255, 165, 0, 0.8)",  # Orange with opacity
                            "line_color": "orange",
                            "line_width": 1.5,
                        }
                    unload_start = prod.unload_beg[c, l]
                    unload_end = prod.unload_end[c, l]
                    unload_duration = unload_end - unload_start
                    unload_traces[unload_key]["x"].append(unload_duration)
                    unload_traces[unload_key]["y"].append(machine_pos)
                    unload_traces[unload_key]["base"].append(unload_start)
                    unload_traces[unload_key]["hovertext"].append(
                        f"Unload for Product ID: {prod.id}, Cycle: {c}, Unload: {l}"
                    )

                # Add Text Annotation
                # Using Scatter for annotations
                fig.add_trace(
                    go.Scatter(
                        x=[setup_start + setup_duration + cycle_duration / 2],
                        y=[machine_pos],
                        text=[f"JOB {prod.id} C {c} V[{prod.velocity[c]}]"],
                        mode="text",
                        textposition="middle center",
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

    # Add all Cycle Traces
    for trace_key, trace_data in cycle_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Production Cycle",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                showlegend=False,  # We'll add a custom legend later
            )
        )

    # Add all Setup Traces
    for trace_key, trace_data in setup_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Job Setup",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                width=0.4,
                opacity=0.75,
                showlegend=False,
            )
        )

    # Add all Load Traces
    for trace_key, trace_data in load_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Load",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                width=0.3,
                opacity=1.0,
                showlegend=False,
            )
        )

    # Add all Unload Traces
    for trace_key, trace_data in unload_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Unload",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                width=0.3,
                opacity=1.0,
                showlegend=False,
            )
        )

    # Update layout for better readability and navigation
    fig.update_layout(
        barmode="overlay",  # Overlay to stack bars correctly
        height=400
        + 40 * updated_num_machines,  # Dynamic height based on number of machines
        title="Production Schedule Gantt Chart",
        xaxis=dict(
            title="Time (hours)",
            type="linear",
            tickmode="linear",
            tick0=0,
            dtick=max(1, horizon // 10),  # Dynamic tick interval
            range=[start_time, horizon],
            showgrid=True,
            zeroline=False,
            rangeslider=dict(
                visible=True,
                thickness=0.1,
                range=[start_time, min(horizon, start_time + 50)],  # Initial view range
            ),
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=10, label="10h", step="hour", stepmode="backward"),
                        dict(count=20, label="20h", step="hour", stepmode="backward"),
                        dict(count=50, label="50h", step="hour", stepmode="backward"),
                        dict(step="all"),
                    ]
                ),
                x=0,
                y=-0.2,  # Positioning the selector below the chart
                xanchor="left",
                yanchor="top",
            ),
        ),
        yaxis=dict(
            title="Machine ID",
            tickvals=list(range(updated_num_machines)),
            ticktext=[f"[{m}]" for m in used_machines],
            autorange="reversed",  # To have the first machine at the top
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(
            l=100, r=40, t=80, b=150
        ),  # Increased left margin for machine labels
        plot_bgcolor="white",
    )

    # Add Legend Manually using Scatter (since individual traces have showlegend=False)
    legend_elements = [
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="rgba(255,0,0,0)", line=dict(color="red", width=2)),
            name="Job Setup",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(
                color="rgba(0, 128, 0, 0.8)", line=dict(color="green", width=1.5)
            ),
            name="Load",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(
                color="rgba(255, 165, 0, 0.8)", line=dict(color="orange", width=1.5)
            ),
            name="Unload",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="gray", opacity=0.4),
            name="Non-working Hours",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="red", opacity=0.3),
            name="Past",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="blue"),
            name="Production Cycle",
            hoverinfo="none",
        ),
    ]

    for elem in legend_elements:
        fig.add_trace(elem)

    # Update layout to include the custom legend
    fig.update_layout(
        legend=dict(
            itemsizing="constant",
            title="Legend",
            traceorder="normal",
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1,
        )
    )

    # Final Touches: Ensure legend entries are unique and not duplicated
    fig.update_traces(showlegend=False)

    return fig


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
                label="Edit `articoli_macchine.json` content", language="json", max_lines=20
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
                outputs=machine_info_content
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
                datatype=["str","number","number","number"],  # Adjust based on your CSV columns
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
                datatype=["str","str","number","date","date"],  # Adjust based on your CSV columns
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
                "EMPTY"
            ]
            running_products_content = gr.Dataframe(
                label="Edit `running_products.csv` content",
                datatype=["str"],  # Adjust based on your CSV columns
                headers=running_products_columns,
                row_count=10,  # Replaced max_rows with row_count
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
                maximum=60 * 10,
                step=1,
                value=60,
            )
            # second run of solve time
            second_run_seconds = gr.Slider(
                label="Max seconds for compactness refinement",
                minimum=0,
                maximum=60 * 10,
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
