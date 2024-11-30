# LeMur Scheduler - CP-SAT with EA

## Overview

LeMur Scheduler is a Python-based optimization tool designed to solve scheduling problems using Google's OR-Tools CP-SAT solver. The project incorporates evolutionary algorithms (EAs) for refinement, focusing on job scheduling, machine assignments, and constraints like maintenance, operator availability, and production deadlines.

The solution supports both a graphical user interface (GUI) for managing inputs and outputs and a comprehensive backend for performing advanced optimization tasks.

---

## Repository Structure

### Main Folders and Files
- **web_app/**: Contains the Gradio-based web interface and essential logic.
    - `app.py`: The entry point for the web application.
    - Supporting Python files:
        - `data_init.py`: Handles data preprocessing and initialization.
        - `solver.py`: Contains the main optimization solver logic.
        - `ga_refiner.py`: Implements genetic algorithm-based refinement for schedules.
        - `ga_utils.py`: Utility functions for evolutionary algorithms.
        - `utils.py`: General helper functions.
- **code/**: Contains the core solver logic and supporting files.
    - `solver.py`: Main solver logic for scheduling problems.
    - Supporting Python files:
        - `data_init.py`: Data preprocessing and initialization.
        - `ga_refiner.py`: Genetic algorithm-based refinement for schedules.
        - `ga_utils.py`: Utility functions for evolutionary algorithms.
        - `utils.py`: General helper functions.
        - `data_plotting.py`: Functions for generating Gantt charts.
- **data/**: Contains script to process the data provided from Lemur to generate the required input files and simulate orders.

---

## Features

1. **Optimization Goals**:
   - Minimize makespan while balancing workload.
   - Optimize machine and operator utilization.
   
2. **Constraint Handling**:
   - Machine compatibility.
   - Shift and operator scheduling.
   - Maintenance windows and holidays.

3. **Evolutionary Algorithm Refinement**:
   - Genetic algorithm to refine solutions.
   - Handles complex scenarios like overlapping operations.

4. **Interactive GUI**:
   - Upload and edit CSV/JSON input files.
   - Configure solver parameters.
   - Visualize schedules as Gantt charts.

---

## Setup

### Dependencies
Install the required Python packages:
```bash
pip install ortools gradio matplotlib pandas plotly inspyred intervaltree
```

### Files
The following input files are required:
- `new_orders.csv`: Defines the list of jobs to be scheduled.
- `lista_articoli.csv`: Provides production statistics for each article.
- `articoli_macchine.json`: Defines the compatibility between articles and machines.
- `macchine_info.json`: Contains machine-specific details.

---

## How to Use

### Running the Web Interface
Run the following command to start the Gradio-based GUI while inside the `web_app/` directory:
```bash
python app.py
```

### GUI Features
1. **Upload Input Files**:
   - Upload and edit CSV/JSON files.
   - Save modified data back to the appropriate files.

2. **Configure Solver Parameters**:
   - Adjust scheduling horizon, shift times, and operator availability.
   - Define costs for setup, loading, and unloading operations.

3. **Run the Solver**:
   - Execute the scheduler and view outputs in real-time.
   - Visualize results as Gantt charts.

---

## Inputs

### Required Input Files
1. **`new_orders.csv`**:
   - Format: 
        - cliente
        - cod_articolo
        - quantity
        - data inserimento
        - data consegna
   - Example: `SCORTA, 4407DN, 472, 2025-04-20, 2025-08-06`
2. **`running_orders.csv`**:
   - Format: 
         - cliente
         - macchina
         - cod_articolo
         - quantity
         - fine operazione attuale
         - data consegna
         - levate rimanenti ciclo
         - tipo operazione attuale
         - operatore
   - Example: `SCORTA,1,2277DN mini,124.37,2024-12-01 03:30:00,2025-11-19,1,2,0,0`
2. **`lista_articoli.csv`**:
   - Production data for each article:
        - codarticolo
        - kg_ora
        - no_cicli
        - ore_levata
   - Example: `10353ZF, 1.87, 2, 96.0`
3. **`articoli_macchine.json`**:
   - Compatibility between articles and machines.
   - Example:
   ```json
     {
        "2277DN mini": [1,2,4,25,70],
     }
     ```
4. **`macchine_info.json`**:
   - Contains the `n_fusi` for each machine.
   - Example:
    ```json
    {
        "1": {
            "n_fusi": "392",
        },
    }
    ```
---

## Outputs

1. **Schedules**:
   - `output/schedule.txt`: Initial schedule.
   - `output/refined_schedule.txt`: Refined schedule after GA optimization.

2. **Visualizations**:
   - Gantt charts for visualizing machine schedules.

---

## Customization

### Solver Parameters
Configure the solver by adjusting the following:
- **Time Units**:
  - Set time granularity (e.g., 24, 48, or 96 units per day).
- **Operator Configuration**:
  - Define the number of operator groups and members per group.
- **Costs**:
  - Adjust costs for setup, load, and unload phases.

### Exceptions
- Define `broken_machines`, `scheduled_maintenances`, and `festive_days` to customize scheduling constraints.

---

## Error Handling

- **Missing Files**: Ensure all input files are available before running the solver.
- **Empty Rows in CSVs**: The GUI automatically removes empty rows from uploaded files.

---

## Example Usage

1. Upload the following:
   - `new_orders.csv` (Job list).
   - `lista_articoli.csv` (Article statistics).
   - `articoli_macchine.json` (Compatibility data).
   - `macchine_info.json` (Machine details).
   
2. Configure parameters:
   - Set scheduling horizon and operator availability.
   
3. Run the solver:
   - View results in the "Run Solver" tab.

---

## Developers

This repository is created for **LeMur** for the Industrial AI challenge.
Contributors:
- Davide Cavicchini
- Luca Cazzola
- Alessandro Lorenzi
- Emanuele Poiana
- Andrea Decarlo
- Silvano Maddonni
