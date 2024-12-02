import pandas as pd
from datetime import datetime, timedelta
import os
import json
import math

'''
INPUT VARIABLES
'''

# Inputs
DATA_DI_RIFERIMENTO = '25-11-2024'                      # Date representing scheduling start
STATO_MACCHINE_FILE = 'Stato_Macchine_25-11-2024.xlsx'
PIANIFICAZIONE_FILE = 'Pianificazione_25-11-2024.xlsx'
# Kg of cycle 'continuativo' when found one
KG_CONTINUATIVO = 3000     # Better to keep this as low as possible (very haevy on computation)
MAX_DAYS_DELAY = 60         # due date for a product which is found to be already late with respect to its due date
MAX_DAYS_CONTINUATIVO = 365 # continuative productions have maximum this quantity of days to be completed
MAX_DAYS_MISSING = 365      # missing due dates are set to this value (1 year in the future)

# File containing additional common products coming outside 'pianificazione' (if any)
# Such file will be merged into the common products output file
ADDITIONAL_COMMON_PRODUCTS_FILE = None # ex :'additional_common_products.csv' or None => Set to None if no additional common products are present  

'''
OUTPUT VARIABLES
'''

# Outputs
RUNNING_PRODS_OUTPUT_FILE = 'running_products.csv'
COMMON_PRODS_OUTPUT_FILE = 'common_products.csv'

# Merge paths (files are assumed to be in below directories)
STATO_MACCHINE_FULL_PATH = os.path.join('data', 'stato_macchine', STATO_MACCHINE_FILE)
PIANIFICAZIONE_FULL_PATH = os.path.join('data', 'pianificazione', PIANIFICAZIONE_FILE)
LISTA_ARTICOLI_FULL_PATH = os.path.join('..', 'web_app', 'input', 'lista_articoli.csv')
MACHINE_INFO_FULL_PATH = os.path.join('..', 'web_app', 'input', 'macchine_info.json')
ARTICLES_MACHINES_COMPATIBILITY_FULL_PATH = os.path.join('..', 'web_app', 'input', 'articoli_macchine.json')
ADDITIONAL_COMMON_PRODUCTS_FULL_PATH = os.path.join('data', ADDITIONAL_COMMON_PRODUCTS_FILE) if ADDITIONAL_COMMON_PRODUCTS_FILE else None
RUNNING_PRODS_OUTPUT_PATH = os.path.join('output', RUNNING_PRODS_OUTPUT_FILE)
COMMON_PRODS_OUTPUT_PATH = os.path.join('output', COMMON_PRODS_OUTPUT_FILE)

DEBUG = False

def print_warning(warnings):
    '''
    Function to print warnings to the user
    '''
    if len(warnings['missing_articles']) > 0 :
        print("\nERROR : There are articles in the current 'pianificazine' which are not in listino : Output will EXCLUDE orders including such articles")
        for art in warnings['missing_articles'] :
            print(f" * {art}")
        print(f"Reference file => {LISTA_ARTICOLI_FULL_PATH}")

    if len(warnings['article_compatibility_warnings_1']) > 0 :
        print("\nERROR : Some articles were not compatible with the machine they're running on : Output will EXCLUDE orders including such articles")
        for (int_code, art, m) in warnings['article_compatibility_warnings_1'] :
            print(f" * Article {art} on machine {m} (Internal Code : {int_code})")
        print(f"Reference file => {ARTICLES_MACHINES_COMPATIBILITY_FULL_PATH}")

    if len(warnings['article_compatibility_warnings_2']) > 0 :
        print("\nERROR : Some articles were not present in the compatibility list at all : Output will EXCLUDE orders including such articles")
        for (int_code, art, m) in warnings['article_compatibility_warnings_2'] :
            print(f" * Article {art} on machine {m} (Internal Code : {int_code})")
        print(f"Reference file : {ARTICLES_MACHINES_COMPATIBILITY_FULL_PATH}")

    if len(warnings['quantity_cast_warnings']) > 0 :
        print("\nWARNING : Some articles are at their last levata. Automatically casting Kg. request to 1 Levata Kg. (minimum value possible for the solver)")
        for int_code, art, m, prod_type in warnings['quantity_cast_warnings'] :
            print(f" * Article {art} ({prod_type} product) on machine {m} (Internal Code : {int_code})")
        print(f"Reference file : {STATO_MACCHINE_FULL_PATH} and {PIANIFICAZIONE_FULL_PATH}\n")

    if sum([len(w) for w in warnings.values()]) > 0 :
        print("\n=================================================================================")
        print("Please, note that this program DOESN'T MODIFY INPUT FILES content\nConsider updating supporting files to have a more accurate scheduling in the future !")
        print("  * lista_articoli.csv")
        print("  * articoli_macchine.json")
        print("  * macchine_info.json")
        print("=================================================================================")


def data_cleaning(pianificazione, stato_macchine, now=datetime.now()):
    '''
    Cleaning Pianificazione TABLE
    '''
    pianificazione.columns = pianificazione.columns.str.replace('\n', ' ') # remove newline characters from column names

    # Fill NaN values with the previous row's value in the same column (to fix multi-row entries in pianificazione)
    for column in pianificazione.columns:
        pianificazione[column] = pianificazione[column].fillna(method='ffill')
    
    # Group by 'n. ordine interno' and aggregate the data
    pianificazione = pianificazione.groupby('n. ordine interno').agg({
        'Cliente': 'first',
        'kg prodotti': 'sum',
        'quantità': 'first',
        'data di consegna': 'first'
    }).reset_index()

    # Type conversion
    pianificazione['n. ordine interno'] = pd.to_numeric(pianificazione['n. ordine interno'], errors='coerce').astype('Int32')
    
    # Missing 'data di consegna' are set to 1 year in the future
    due_date_filler = now + timedelta(days=MAX_DAYS_MISSING)
    pianificazione.fillna(value={'data di consegna': pd.Timestamp(due_date_filler)}, inplace=True)

    '''
    Cleaning Stato Macchine TABLE
    '''
    stato_macchine.columns = stato_macchine.columns.str.replace('\n', ' ') # remove newline characters from column names
    stato_macchine = stato_macchine.drop([
        'N° fusi eff.',
        'descrizione',
        'anima',
        'spandex',
        'copertura',
        'peso spola int', 
        'ciclo',
        'infilaggio',
        'N° scheda',
        'note',
        'Rotture Elastomero',
        'Rotture Copertura'
    ], axis=1)
    
    # Nans dropped
    stato_macchine = stato_macchine.dropna(subset=['N° codice Interno'])
    # Type conversion
    stato_macchine['N° codice Interno'] = pd.to_numeric(stato_macchine['N° codice Interno'], errors='coerce').astype('Int32')
    stato_macchine['N° levate att.'] = pd.to_numeric(stato_macchine['N° levate att.'], errors='coerce').astype('Int32')

    # modify column names to have matching ones
    stato_macchine = stato_macchine.rename(columns={'N° codice Interno': 'id'})
    pianificazione = pianificazione.rename(columns={'n. ordine interno': 'id'})
    # Merge the two tables
    output_dataframe = stato_macchine.merge(pianificazione, on='id', how='left')
    # Fill NaN values in 'Cliente' column with 'INTERNAL' value (we assume it's an internal order with no client associated)
    output_dataframe = output_dataframe.fillna(value={'Cliente': 'INTERNAL'})

    return output_dataframe


def row_parser(row, art_standard_levate, now) :
    '''
    Parse a row of the dataframe to extract necessary data
    while also applying some at hoc. corrections
    '''
    client = str(row['Cliente'])
    article = str(row['codarticolo'])
    machine = int(row['N° MC'])
    # Check if article is continuative
    is_continuativo = True if row['N° levate ord'] == 'C' else False
    # In the case data di consegna refers to the past (we're having a delay)
    due_date = row['data di consegna'] if row['data di consegna'] > now else (now+timedelta(days=MAX_DAYS_DELAY))
    # Assuming N° levate att. is one indexed (minimum value is 1) correct it
    curr_levata = max(1, int(row['N° levate att.']))
    # In case of continuative productions, set the total number of levate to the standard value
    # the remaining ones to reach production goal will be scheduled as common products (will be done in the future)
    tot_levate = int(row['N° levate ord']) if not is_continuativo else art_standard_levate[article]

    return client, article, machine, is_continuativo, due_date, curr_levata, tot_levate



def extract_current_status_from_xls (curr_status_path: str, curr_plan_path: str, article_list_path : str,  m_info_path : str, a_compatibility_path : str, now : datetime):
    '''
    Process the data from the excel files and return the running products and common products
    '''
    running_products = []
    common_products = []
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)    
    
    '''
    Read and Clean the data from the excel files
    '''
    pianificazione = pd.read_excel(curr_plan_path)
    stato_macchine = pd.read_excel(curr_status_path)
    df = data_cleaning (pianificazione, stato_macchine, now=now)
    
    '''
    Gather Info relative to machines and articles
    '''
    # Retrieve machine info from .json 
    m_info = json.load(open(m_info_path))
    fuses_machines_associations = {int(machine):m_info[machine]['n_fusi'] for machine in m_info}
    # Retrieve article-machine compatibility from .json
    article_compatibility = json.load(open(a_compatibility_path))

    # Retrieve article related costs data from .csv
    art_base_levata_cost = {}
    art_standard_levate = {}
    art_kg_per_levata = {}
    for idx, row in pd.read_csv(article_list_path).iterrows():
        # Skip empty rows
        if pd.isna(row['codarticolo']):
            continue
        # Gather data for the rest
        art_base_levata_cost[row['codarticolo']] = float(row['ore_levata'])
        art_standard_levate[row['codarticolo']] = float(row['no_cicli'])
        if row['codarticolo'] in article_compatibility.keys():
            for m in article_compatibility[row['codarticolo']]:
                art_kg_per_levata[row['codarticolo'],m] = int((float(row['ore_levata']) * float(row['kg_ora']))* float(fuses_machines_associations[m])/256.0)

    '''
    Conformity checks with input .csv and .json
    '''
    warnings = {
        'missing_articles' : [],
        'quantity_cast_warnings' : [],
        'article_compatibility_warnings_1' : [],
        'article_compatibility_warnings_2' : []
    }

    for idx, row in df.iterrows():
        # Gather data
        article = str(row['codarticolo'])
        machine = int(row['N° MC'])

        # Check if article is present in the .csv provided, if not don't schedule it and report the error
        if article not in art_base_levata_cost.keys() or article not in art_standard_levate.keys() or (article,machine) not in art_kg_per_levata.keys() :
            if article not in warnings['missing_articles'] :
                warnings['missing_articles'].append(article)
            df.drop(idx, inplace=True)
            continue
        # Update article_compatibility with articles in running_products (if not already present).
        #     We assume that if a product is running, it can be
        #     surely be produced by the machine it's running on
        if article in article_compatibility.keys() and machine not in article_compatibility[article] :
            warnings['article_compatibility_warnings_1'].append((int_code, article, machine))
            df.drop(idx, inplace=True)
            continue
        elif article not in article_compatibility.keys() :
            warnings['article_compatibility_warnings_2'].append((int_code, article, machine))
            df.drop(idx, inplace=True)
            continue
    
    '''
    Processing
    '''
    # Sort the dataframe by 'id' and group by 'id'
    id_list = sorted(df['id'].unique())
    id_to_df_dict = {id: group_df for id, group_df in df.groupby('id')}
    
    # Start parsing all the articles group by group
    for int_code in id_list:
        '''
        Generate instance of RunningProduct
        (those products that are currently being produced on a machine)
        => An ID refers to a single article requested by a specific client
        '''
        # Initialize due dates (will be minimized in the loop)
        due_date_for_client = now + timedelta(days=1000)
        
        # Goal variables
        asked_quantity = id_to_df_dict[int_code]['quantità'].values[0]
        already_produced_quantity = id_to_df_dict[int_code]['kg prodotti'].values[0]
        production_goal = max(asked_quantity - already_produced_quantity, 0.0) if not pd.isna(asked_quantity) and not pd.isna(already_produced_quantity) else 0.0
        running_kgs = 0.0
        continuative_kgs = 0.0
        remaining_kgs = 0.0

        for i, (idx, row) in enumerate(id_to_df_dict[int_code].iterrows()):
            if DEBUG and i == 0: print(f" \nProcessing code [{int_code}] : which is running on {len(id_to_df_dict[int_code])} machines...")
            # Get necessary row data
            client, article, machine, is_continuativo, due_date, curr_levata, tot_levate = row_parser(row, art_standard_levate, now)
            
            # Track the due date for the client and the continuative product
            if not is_continuativo :
                due_date_for_client = min(due_date_for_client, due_date)

            # Skip products with no levate to do
            if tot_levate == 0 :
                if DEBUG :
                    print("  * This product has no levate to do, Skip it")
                    print(">>>")
                continue 

            # Evaluate how many levate needs to be done to conclude current cycle & how many kg each levata will produce
            remaining_levate = (tot_levate - curr_levata + 1) if not is_continuativo else 1 + (curr_levata % art_standard_levate[article])
            curr_cycle_remaining_levate = min(1 + (curr_levata % art_standard_levate[article]), remaining_levate)

            # If 'data ora partenza' is missing or is set to the future This cycle has not yet started.
            # => treat it as common_product instance so that it will be scheduled in the future (not necessarily on the same machine)
            is_actually_running = True if not pd.isna(row['data ora partenza']) and not pd.isna(row['data ora fine']) and row['data ora partenza'] <= now else False 
            
            # Compute current percentage of levata completion, will be used as discount for Kg. to produce 
            levata_progress_percentage = 0
            if is_actually_running and (row['data ora partenza'] < now < row['data ora fine']) :
                levata_progress_percentage = max(0, (now.timestamp()-row['data ora partenza'].timestamp()) / (row['data ora fine'].timestamp()-row['data ora partenza'].timestamp()))
            
            # Print debug
            already_produced_kg = art_kg_per_levata[article, machine] * ((curr_levata-1)+levata_progress_percentage)
            if DEBUG :
                print(f"  - Article ({article}) on machine [{machine}]")
                if is_continuativo : print("  * This product is Instantiated as Continuativo")
                print(f"  - Status : {int(curr_levata)} of {int(tot_levate)} levate, {int(remaining_levate)} to go ({int(curr_cycle_remaining_levate)} on current cycle)...")
                print(f"  - Article ({article}) performs : {int(art_standard_levate[article])} levate x Cycle")
                print(f"  - Kg. per Levata : {art_kg_per_levata[article, machine]:.3f} Kg.")
                print(f"  - Current levata progress : {int(levata_progress_percentage*100)}%")
                print(f"  - Already Produced : {already_produced_kg:.3f} Kg.")


            # If the product is not running it can be made a Product instance
            if not is_actually_running :
                if DEBUG :
                    print(f"  * This product has not started yet, reschedule (as Common Product)")
                    print(">>>") # indicates row change when seen
                remaining_kgs += remaining_levate * art_kg_per_levata[article, machine]
                continue
            
            # Compute the running production in Kg.
            running_prod_kg = max((curr_cycle_remaining_levate-levata_progress_percentage)*art_kg_per_levata[article, machine], art_kg_per_levata[article, machine])
            if DEBUG :
                print(f"  * Instantiating Running prod. for ({int(curr_cycle_remaining_levate)}) levate : {running_prod_kg:.3f} Kg.")
                print(">>>") # indicates row change when seen
            
            # Initialize a record for the running products dataframe
            running_prod_record = {
                'cliente': client,
                'macchina': machine,
                'cod_articolo': article,
                'quantity' : running_prod_kg,
                'fine operazione attuale' : row['data ora fine'],            # 'fine operazione attuale' refers to time remaining until 'tipo operazione attuale' ends (remineder => current_op_type = {0,1,2,3} : {setup, load, running, unload})
                'data consegna' : due_date,
                'levate rimanenti ciclo': int(curr_cycle_remaining_levate),  # Remaining levate IN ORDER TO COMPLETE CURRENT CYCLE (CURRENT LEVATA INCLUDED)
                'tipo operazione attuale': 2,                                # always set as machine is running (not info on operators, might be added in the future)
                'operatore': 0,                                              # As op_type is fixed, operator is actually irrelevant
                'velocità': 0,                                               # Always assume standard velocity
            }
            # Insert to list
            running_products.append(running_prod_record)

            # Kgs produced by running products are always added to the total
            running_kgs += running_prod_kg
            production_goal -= running_prod_kg
            
            # Remaining Kgs are basically Kg. produced remaining levate, excluding those in the current cycle
            if not is_continuativo :
                remaining_kgs += (remaining_levate-curr_cycle_remaining_levate)*art_kg_per_levata[article, machine]
            else :
                continuative_kgs += running_prod_kg

        
        '''
        Generate instance of (common) Product according to remaining kgs
        (those products that are not currently being produced on any machine, but will be in the future)
        '''
        associated_article = id_to_df_dict[int_code]['codarticolo'].values[0]
        associated_client = id_to_df_dict[int_code]['Cliente'].values[0]
        min_kgs_per_levata = min([art_kg_per_levata[associated_article, m] for m in article_compatibility[associated_article]])
        
        if production_goal > 0 :
            if DEBUG :
                print(f"  * Kg. remaining for client {associated_client} are : {max(0, production_goal):.3f} Kg.")
                print(f"  * Instantiating Common prod. to satisfy client's request with {max(min_kgs_per_levata, production_goal):.3f} Kg.")
            common_prod_record = {
                'id' : int_code,
                'cliente': associated_client,
                'cod_articolo': associated_article,
                'quantity' : max(min_kgs_per_levata, production_goal),
                'data inserimento': midnight,    # can be set to midnight (zero value in time units) as it was scheduled to be running
                'data consegna' : due_date_for_client,
            }
            common_products.append(common_prod_record)

        # remove from kgs to reschedule those specific at satifying the client request (if any)
        remaining_kgs -= max(0, production_goal)

        # Associate the remaining kgs which are relative to 'stato macchine' for consistency, with relaxed due date (as it's not urgent)
        if remaining_kgs > 0 :
            actual_kgs = max([min_kgs_per_levata, remaining_kgs])
            if DEBUG :
                print(f"  * Kg. requested by client is achieved.")
                print(f"  * Instantiating Common prod. for : {actual_kgs:.3f} Kg. with relaxed due date (same as continuative)")
            common_prod_record = {
                'cliente': associated_client,
                'cod_articolo': associated_article,
                'quantity' : actual_kgs,
                'data inserimento': midnight,    # can be set to midnight (zero value in time units) as it was scheduled to be running
                'data consegna' : now + timedelta(days=MAX_DAYS_CONTINUATIVO),
            }
            common_products.append(common_prod_record)

        # If Continuative production were present
        if continuative_kgs > 0 :
            # Compute how much is missing to reach the KG_CONTINUATIVO specified
            remaining_kgs_continuativi = 0 if continuative_kgs >= KG_CONTINUATIVO else KG_CONTINUATIVO - continuative_kgs
            # Schedule the remaining kgs as a continuative product (if any)
            if remaining_kgs_continuativi > 0 :
                if DEBUG : print(f"  * Instantiating Continuative production : {remaining_kgs_continuativi:.3f} Kg.")
                common_prod_record = {
                    'cliente': associated_client,
                    'cod_articolo': associated_article,
                    'quantity' : max([min_kgs_per_levata, remaining_kgs_continuativi]),
                    'data inserimento': midnight,    # can be set to midnight (zero value in time units) as it was scheduled to be running
                    'data consegna' : now + timedelta(days=MAX_DAYS_CONTINUATIVO),
                }
                common_products.append(common_prod_record)
        if DEBUG : print(">>>") # indicates row change when seen

 
    # Ensure asked quantities are at least the corresponding Kg. of 1 levata
    for idx, prod in enumerate(running_products+common_products) :
        min_valid_quantity = min([art_kg_per_levata[prod['cod_articolo'],m] for m in article_compatibility[prod['cod_articolo']]])
        if prod['quantity'] < min_valid_quantity :
            warnings['quantity_cast_warnings'].append((int_code, article, machine, 'running' if idx < len(running_products) else 'common'))
            prod['quantity'] = max(min_valid_quantity, prod['quantity'])
        # Round to 2 decimal precision
        prod['quantity'] = math.floor(prod['quantity'] * 100) / 100

    # Convert to DataFrames and return if initialized (else returns None object)
    running_prod_columns = list(running_products[0].keys()) if len(running_products) > 0 else []
    common_prod_columns = list(common_products[0].keys()) if len(common_products) > 0 else []
    running_products = pd.DataFrame(running_products, columns=running_prod_columns) if len(running_prod_columns) > 0 else None
    common_products = pd.DataFrame(common_products, columns=common_prod_columns) if len(common_prod_columns) > 0 else None

    # sort running_products by 'macchina' and 'data consegna' for readability
    if running_products is not None:
        running_products = running_products.sort_values(by=['macchina']).reset_index(drop=True)
    # sort common_products by 'data consegna' for readability
    if common_products is not None:
        common_products = common_products.sort_values(by=['data consegna']).reset_index(drop=True)

    print_warning(warnings)

    return running_products, common_products


if __name__ == "__main__":

    now = datetime.strptime(DATA_DI_RIFERIMENTO, '%d-%m-%Y')
    running_products_df, common_products_df = extract_current_status_from_xls(
        STATO_MACCHINE_FULL_PATH,
        PIANIFICAZIONE_FULL_PATH,
        LISTA_ARTICOLI_FULL_PATH,
        MACHINE_INFO_FULL_PATH,
        ARTICLES_MACHINES_COMPATIBILITY_FULL_PATH,
        now
    )
    print("\nData extraction completed !")
    # Save running products to .csv
    if running_products_df is not None:
        running_products_df.to_csv(RUNNING_PRODS_OUTPUT_PATH, index=False)
        print(f"  * Saved running products to {RUNNING_PRODS_OUTPUT_PATH}.")
    else:
        print("  * No running products found. Output file not created.")

    # Handle common products and merge with additional_common_products
    if common_products_df is not None:
        if ADDITIONAL_COMMON_PRODUCTS_FILE and os.path.exists(ADDITIONAL_COMMON_PRODUCTS_FULL_PATH):
            additional_common_products_df = pd.read_csv(ADDITIONAL_COMMON_PRODUCTS_FULL_PATH)
            # Ensure consistent column names for merging
            additional_common_products_df.columns = additional_common_products_df.columns.str.strip()
            common_products_df = pd.concat([common_products_df, additional_common_products_df], ignore_index=True)
            print(f"  * Merged {ADDITIONAL_COMMON_PRODUCTS_FULL_PATH} with common {COMMON_PRODS_OUTPUT_PATH}.")
        else :
            print("  * No additional common products specified")
        # Save common products to .csv
        common_products_df.to_csv(COMMON_PRODS_OUTPUT_PATH, index=False)
        print(f"  * Saved common products to {COMMON_PRODS_OUTPUT_PATH}.")
    else:
        print("  * No common products found. Output file not created.")
    print("\nClosing...")
