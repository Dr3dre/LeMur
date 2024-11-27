# LeMur Scheduler - CP-SAT with EA

## Overview

This README provides an introduction to the configuration and input data required for initialize the solver.


## Structure
- `code`
```
code/
┣ __init__.py
┣ data_init.py
┣ data_plotting.py
┣ ga_refiner.py
┣ ga_utils.py
┣ solver.py
┗ utils.py
```

- `data`
```
data/
┣ input/
┃ ┣ xlsx/
┃ ┃ ┣ lista_articoli.xlsx
┃ ┃ ┗ storico_filtrato.xlsx
┃ ┣ Gruppi_Macchine_per_articoli.json
┃ ┣ lista_articoli.csv
┃ ┣ storico_Lemur.csv
┃ ┣ storico_filtrato.csv
┃ ┣ storico_ordini.csv
┃ ┗ storico_pianificazione.csv
┣ utils/
┃ ┣ articoli.json
┃ ┣ articoli_macchine.json
┃ ┣ articoli_macchine_storico.json
┃ ┣ gruppi_macchine.json
┃ ┣ macchine_articoli.json
┃ ┣ macchine_info.json
┃ ┣ ordini_per_client.json
┃ ┣ quantity_per_client.csv
┃ ┣ tot_per_client.json
┃ ┗ uso_per_articolo.json
┣ valid/
┃ ┣ lista_articoli.csv
┃ ┗ storico_ordini.csv
┗ new_orders.csv
```
- `output`
- `plots`

## Features

- **Optimization Goals**: ...
- **Constraint Handling**: Handles advanced constraints such as machine velocities, maintenance windows, and holidays.


## Requirements

### Dependencies

- **Python Packages**:
  - `ortools.sat.python.cp_model`: Google's OR-Tools for constraint programming.
  - `argparse`: For parsing command-line arguments.
  - `google.protobuf.text_format`: For text formatting in protobufs.
  - Custom modules: 
    - `data_init` (Handles data preprocessing and initialization)
    - `data_plotting` (Optional: For visualizing schedules)
    - `utils` (General utility functions)
    - `ga_refiner` (For genetic algorithm-based refinement)

Install required packages using `pip install ortools protobuf`.

## Inputs 

### Files



1. **Path Variables**
    *str containing the path*
    -
    - `COMMON_P_PATH` : csv with the job list to be scheduled 
    - `J_COMPATIBILITY_PATH` : json with the compatibility between jobs and machines
    - `M_COMPATIBILITY_PATH` : json with the compatibility between machines and jobs
    - `M_INFO_PATH` : json containing the info about every machine (like n°fusi)
    - `ARTICLE_LIST_PATH` : csv containing all the production statistics about each article that can be produced

2. **Files for Product-Machine Compatibility**
    - `data/new_orders.csv`: List of new orders.
    - `data/utils/articoli_macchine.json`: Mapping of articles to compatible machines.
    - `data/utils/macchine_articoli.json`: Mapping of machines to compatible articles.
    - `data/utils/macchine_info.json`: Additional machine-specific information.
    - `data/valid/lista_articoli.csv`: A validated list of articles.

3. **Output Files**:
    - `output/schedule.txt`: File to save the initial scheduling output.
    - `output/refined_schedule.txt`: File to save the refined schedule.


### Variables

- `MAKESPAN_SOLVER_TIMEOUT` : For how long the solver can go before early stopping. 60 seconds default
- `CYCLE_OPTIM_SOLVER_TIMEOUT` : For how long the optimizer part of the solver can go before early stopping. 60 seconds default
- `GENETIC_REFINEMENT` : Choose if use the EA refinement or not. True default

- `num_machines` : the number of machines available for the solver. 72 default 

- `horizon_days` : the number of days available for the desired scheduling
- `num_operator_groups` : 2 default
- `num_operators_per_group`: 4 default
- `costs` : tuple containing the specifications for the costs used in the SETTING, LOAD and UNLOAD phases.
   ```
    costs=(
    setup_cost,                                      # Setup time
    load_hours_fuso_person / num_operators_per_group,  # load time for 1 "fuso"
    unload_hours_fuso_person / num_operators_per_group   # Unload time for 1 "fuso"
    )
  ```

### Exceptions 

- `broken_machines`: list of broken machines on top on those no job can be scheduled
- `scheduled_maintenances`: dict of the scheduled maintenances where no job can be scheduled on a specific machine in a specific time.
    ```
    scheduled_maintenances = {
        # machine : [(start, duration)],
        1 : [(50, 150)]
    }
    ```
- `festivities`: list of the days (ordered starting from 0 == today) where no operators are available

## Error Handling

- **Missing Files**: Ensure all input files are correctly placed to avoid `FileNotFoundError`.


## Initialization Details

The `init_csv_data()` function reads input data, processes compatibility matrices, and calculates costs. Products are grouped into:
1. **Common Products**: Products available for immediate scheduling.
2. **Running Products**: Products currently in progress.

These are combined into `all_products` for scheduling.