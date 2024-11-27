import gradio as gr
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import tempfile
import os

# Import your solver and data_init functions
from solver import solve
from data_init import init_csv_data

def upload_file(file):
    if file is not None:
        # If multiple files are uploaded, take the first one
        if isinstance(file, list):
            file = file[0]
        
        # If file is a dictionary with 'data' key
        if isinstance(file, dict) and 'data' in file:
            file_path = file['data']
        elif isinstance(file, str):
            # If file is a path string
            file_path = file
        else:
            return "Unsupported file format."
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    return ""

def save_common_products(content):
    try:
        with open("new_orders.csv", "w", encoding='utf-8') as f:
            f.write(content)
        return "✅ `new_orders.csv` saved successfully."
    except Exception as e:
        return f"❌ Error saving `new_orders.csv`: {str(e)}"

def save_article_machine_compatibility(content):
    try:
        data = json.loads(content)
        with open("articoli_macchine.json", "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return "✅ `articoli_macchine.json` saved successfully."
    except json.JSONDecodeError as e:
        return f"❌ JSON Decode Error: {str(e)}"
    except Exception as e:
        return f"❌ Error saving `articoli_macchine.json`: {str(e)}"

def save_machine_info(content):
    try:
        data = json.loads(content)
        with open("macchine_info.json", "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return "✅ `macchine_info.json` saved successfully."
    except json.JSONDecodeError as e:
        return f"❌ JSON Decode Error: {str(e)}"
    except Exception as e:
        return f"❌ Error saving `macchine_info.json`: {str(e)}"

def save_article_list(content):
    try:
        with open("lista_articoli.csv", "w", encoding='utf-8') as f:
            f.write(content)
        return "✅ `lista_articoli.csv` saved successfully."
    except Exception as e:
        return f"❌ Error saving `lista_articoli.csv`: {str(e)}"

def run_solver():
    # Paths to the files saved above
    COMMON_P_PATH = "new_orders.csv"
    J_COMPATIBILITY_PATH = "articoli_macchine.json"
    M_INFO_PATH = "macchine_info.json"
    ARTICLE_LIST_PATH = "lista_articoli.csv"

    # Check if all required files exist
    required_files = [COMMON_P_PATH, J_COMPATIBILITY_PATH, M_INFO_PATH, ARTICLE_LIST_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        return f"❌ Missing files: {', '.join(missing_files)}", None

    try:
        # Initialize data
        common_products, running_products, article_to_machine_comp, base_setup_art_cost, base_load_art_cost, base_unload_art_cost, base_levata_art_cost, standard_levate_art, kg_per_levata_art = init_csv_data(
            COMMON_P_PATH,
            J_COMPATIBILITY_PATH,
            M_INFO_PATH,
            ARTICLE_LIST_PATH,
            costs=(4, 6/256, 2/256)
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
        )

        # Convert the schedule to a string to display
        schedule_str = str(schedule)

        # Generate Gantt chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Collect all tasks for Gantt chart
        tasks = []
        for p, prod in schedule.products:
            for c in prod.setup_beg.keys():
                setup_start = prod.setup_beg[c]
                setup_end = prod.setup_end[c]
                tasks.append({
                    'Product': prod.id,
                    'Operation': 'Setup',
                    'Start': setup_start,
                    'End': setup_end,
                    'Machine': prod.machine[c]
                })
                for l in range(prod.num_levate[c]):
                    if (c, l) in prod.load_beg.keys():
                        load_start = prod.load_beg[c,l]
                        load_end = prod.load_end[c,l]
                        unload_start = prod.unload_beg[c,l]
                        unload_end = prod.unload_end[c,l]
                        tasks.append({
                            'Product': prod.id,
                            'Operation': f'Load Levata {l}',
                            'Start': load_start,
                            'End': load_end,
                            'Machine': prod.machine[c]
                        })
                        tasks.append({
                            'Product': prod.id,
                            'Operation': f'Unload Levata {l}',
                            'Start': unload_start,
                            'End': unload_end,
                            'Machine': prod.machine[c]
                        })

        # Create a DataFrame for easier plotting
        df_tasks = pd.DataFrame(tasks)

        if not df_tasks.empty:
            # Assign each machine to a y-axis position
            machines = sorted(df_tasks['Machine'].unique())
            machine_positions = {machine: idx for idx, machine in enumerate(machines)}
            df_tasks['y_pos'] = df_tasks['Machine'].map(machine_positions)

            # Plot each task as a horizontal bar
            for _, row in df_tasks.iterrows():
                ax.barh(row['y_pos'], row['End'] - row['Start'], left=row['Start'], height=0.4, label=row['Operation'])

            # Set y-axis labels
            ax.set_yticks(list(machine_positions.values()))
            ax.set_yticklabels(list(machine_positions.keys()))
            ax.set_xlabel('Time Units')
            ax.set_ylabel('Machines')
            ax.set_title('Production Schedule Gantt Chart')

            # Create a legend without duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.tight_layout()

            # Save the figure to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(temp_file.name)
            plt.close(fig)
        else:
            # If there are no tasks, create an empty image
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No tasks scheduled.', horizontalalignment='center', verticalalignment='center', fontsize=20)
            ax.axis('off')
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(temp_file.name)
            plt.close(fig)

        # Return the schedule and the Gantt chart image
        return schedule_str, temp_file.name

    except Exception as e:
        return f"❌ An error occurred during solving: {str(e)}", None

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## LeMur Scheduling Web App")
    with gr.Tabs():
        # Tab 1: Common Products
        with gr.TabItem("Upload Common Products (CSV)"):
            gr.Markdown("### Upload and Edit `new_orders.csv`")
            common_products_file = gr.File(label="Upload `new_orders.csv`")
            common_products_content = gr.TextArea(label="Edit `new_orders.csv` content", lines=20)
            common_products_file.change(fn=upload_file, inputs=common_products_file, outputs=common_products_content)
            save_common_products_button = gr.Button("Save `new_orders.csv`")
            save_common_products_status = gr.Textbox(label="Status", interactive=False)
            save_common_products_button.click(
                fn=save_common_products,
                inputs=common_products_content,
                outputs=save_common_products_status
            )

        # Tab 2: Article-Machine Compatibility
        with gr.TabItem("Upload Article-Machine Compatibility (JSON)"):
            gr.Markdown("### Upload and Edit `articoli_macchine.json`")
            article_machine_file = gr.File(label="Upload `articoli_macchine.json`")
            article_machine_content = gr.TextArea(label="Edit `articoli_macchine.json` content", lines=20)
            article_machine_file.change(fn=upload_file, inputs=article_machine_file, outputs=article_machine_content)
            save_article_machine_button = gr.Button("Save `articoli_macchine.json`")
            save_article_machine_status = gr.Textbox(label="Status", interactive=False)
            save_article_machine_button.click(
                fn=save_article_machine_compatibility,
                inputs=article_machine_content,
                outputs=save_article_machine_status
            )

        # Tab 3: Machine Info
        with gr.TabItem("Upload Machine Info (JSON)"):
            gr.Markdown("### Upload and Edit `macchine_info.json`")
            machine_info_file = gr.File(label="Upload `macchine_info.json`")
            machine_info_content = gr.TextArea(label="Edit `macchine_info.json` content", lines=20)
            machine_info_file.change(fn=upload_file, inputs=machine_info_file, outputs=machine_info_content)
            save_machine_info_button = gr.Button("Save `macchine_info.json`")
            save_machine_info_status = gr.Textbox(label="Status", interactive=False)
            save_machine_info_button.click(
                fn=save_machine_info,
                inputs=machine_info_content,
                outputs=save_machine_info_status
            )

        # Tab 4: Article List
        with gr.TabItem("Upload Article List (CSV)"):
            gr.Markdown("### Upload and Edit `lista_articoli.csv`")
            article_list_file = gr.File(label="Upload `lista_articoli.csv`")
            article_list_content = gr.TextArea(label="Edit `lista_articoli.csv` content", lines=20)
            article_list_file.change(fn=upload_file, inputs=article_list_file, outputs=article_list_content)
            save_article_list_button = gr.Button("Save `lista_articoli.csv`")
            save_article_list_status = gr.Textbox(label="Status", interactive=False)
            save_article_list_button.click(
                fn=save_article_list,
                inputs=article_list_content,
                outputs=save_article_list_status
            )

        # Tab 5: Run Solver
        with gr.TabItem("Run Solver"):
            gr.Markdown("### Execute the Scheduler and View Results")
            run_solver_button = gr.Button("Run Solver")
            run_solver_status = gr.Textbox(label="Solver Output", lines=10, interactive=False)
            gantt_output = gr.Image(label="Gantt Chart")
            run_solver_button.click(
                fn=run_solver,
                inputs=None,
                outputs=[run_solver_status, gantt_output]
            )

    gr.Markdown("### Note:")
    gr.Markdown("""
    - Ensure all required files are uploaded and saved before running the solver.
    - The Gantt chart visualizes the scheduling of operations on machines over time.
    - Time units are based on the solver's configuration (e.g., hours).
    """)

# Launch the Gradio app
demo.launch()
