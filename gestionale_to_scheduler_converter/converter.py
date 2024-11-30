import pandas as pd
from datetime import datetime, timedelta
import os
import json
import csv
import math

# Inputs
DATA_DI_RIFERIMENTO = '19-11-2024'                      # Date representing scheduling start
STATO_MACCHINE_FILE = 'Stato_Macchine_19-11-2024.xlsx'
PIANIFICAZIONE_FILE = 'Pianificazione_19-11-2024.xlsx'

LISTA_ARTICOLI_FILE = 'lista_articoli.csv'
MACHINE_INFO_FILE = 'macchine_info.json'
ARTICLES_MACHINES_COMPATIBILITY_FILE = 'articoli_macchine.json'

# File containing additional common products coming outside 'pianificazione' (if any)
ADDITIONAL_COMMON_PRODUCTS_FILE = None # ex :'additional_common_products.csv' or None => Set to None if no additional common products are present  
# Kg of cycle 'continuativo' when found one
KG_CONTINUATIVO = 2000  # Better to keep this as low as possible (very haevy on computation)

# Outputs
RUNNING_PRODS_OUTPUT_FILE = 'running_products.csv'
COMMON_PRODS_OUTPUT_FILE = 'common_products.csv'

# Merge paths (files are assumed to be in below directories)
STATO_MACCHINE_FULL_PATH = os.path.join('data', 'stato_macchine', STATO_MACCHINE_FILE)
PIANIFICAZIONE_FULL_PATH = os.path.join('data', 'pianificazione', PIANIFICAZIONE_FILE)
LISTA_ARTICOLI_FULL_PATH = os.path.join('data', LISTA_ARTICOLI_FILE)
MACHINE_INFO_FULL_PATH = os.path.join('data', MACHINE_INFO_FILE)
ARTICLES_MACHINES_COMPATIBILITY_FULL_PATH = os.path.join('data', ARTICLES_MACHINES_COMPATIBILITY_FILE)
ADDITIONAL_COMMON_PRODUCTS_FULL_PATH = os.path.join('data', ADDITIONAL_COMMON_PRODUCTS_FILE) if ADDITIONAL_COMMON_PRODUCTS_FILE else None

RUNNING_PRODS_OUTPUT_PATH = os.path.join('output', RUNNING_PRODS_OUTPUT_FILE)
COMMON_PRODS_OUTPUT_PATH = os.path.join('output', COMMON_PRODS_OUTPUT_FILE)


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
        print("\nWARNING : Some articles were not compatible with the machine they're running on. Automatically adding machine to compatibility list")
        for (int_code, art, m) in warnings['article_compatibility_warnings_1'] :
            print(f" * Article {art} on machine {m} (Internal Code : {int_code})")
        print(f"Reference file => {ARTICLES_MACHINES_COMPATIBILITY_FULL_PATH}")

    if len(warnings['article_compatibility_warnings_2']) > 0 :
        print("\nWARNING : Some articles were not present in the compatibility list at all. Automatically adding both the article with associated machine to compatibility list")
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

def extract_current_status_from_xls (curr_status_path: str, curr_plan_path: str, article_list_path : str,  m_info_path : str, a_compatibility_path : str, now : datetime):
    running_products = []
    common_products = []
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)    
    '''
    Read and Clean the data from the excel files
    '''
    m_stat = pd.read_excel(curr_status_path)
    m_plan = pd.read_excel(curr_plan_path)
    # clean typose on column names
    m_stat.columns = m_stat.columns.str.replace('\n', ' ')
    m_plan.columns = m_plan.columns.str.replace('\n', ' ')
    # Join the two dataframes based on 'n. ordine interno' column
    running_prods_df = m_stat.join(m_plan.set_index('n. ordine interno'), on='N° codice Interno', how='left', rsuffix='_plan')
    # Data Cleaning
    running_prods_df.dropna(subset=['N° codice Interno'], inplace=True)
    
    due_date_filler = now + timedelta(days=365)
    running_prods_df.fillna(value={
        'data inserimento produzione': pd.Timestamp(now) - pd.Timedelta(days=1), # set 'inserimento' to yesterday if missing
        'data di consegna': pd.Timestamp(due_date_filler), # set 'consegna' to 1 year in the future date if missing
    }, inplace=True)

    '''
    Gather Info relative to machines and articles
    '''
    # Retrieve machine info from .json 
    m_info = json.load(open(m_info_path))
    fuses_machines_associations = {int(machine):m_info[machine]['n_fusi'] for machine in m_info}
    # Retrieve article-machine compatibility from .json
    article_compatibility = json.load(open(a_compatibility_path))
    # Retrieving article base costs data from .csv
    art_base_levata_cost = {}
    art_standard_levate = {}
    art_kg_per_levata = {}
    art_kg_per_hour = {}
    with open(article_list_path, newline='') as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(csv_data):
            # row[0] is the article code, row[10] is the hours needed for a single "levata"
            if idx > 0 :
                art_base_levata_cost[row[0]] = float(row[10])
                art_standard_levate[row[0]] = float(row[9])
                art_kg_per_hour[row[0]] = float(row[6])
                if row[0] in article_compatibility.keys() :
                    for m in article_compatibility[row[0]] :
                        art_kg_per_levata[row[0],m] = (float(row[10]) * float(row[6]) * float(fuses_machines_associations[m]) / 256.0)
    
    '''
    Transform into cleaned dataframes the .xls obtained joining 'curr_status' & 'curr_plan'
    '''
    common_prod_ids_so_far = {} # keeps track of common products already scheduled
    warnings = {
        'missing_articles' : [],
        'quantity_cast_warnings' : [],
        'article_compatibility_warnings_1' : [],
        'article_compatibility_warnings_2' : []
    }    
    for idx, row in running_prods_df.iterrows():
        '''
        Row Parse and Dict Updates
        '''
        client = str(row['Cliente']) if not pd.isna(row['Cliente']) else "INTERNAL"
        article = str(row['codarticolo'])
        machine = int(row['N° MC'])
        due_date = row['data di consegna'] if row['data di consegna'] > now else due_date_filler # We're Already Late !!! (should happen, but add placeholder of 1 year ahead)
        curr_levata = int(row['N° levate att.'])
        tot_levate = int(row['N° levate ord']) if row['N° levate ord'] != 'C' else art_standard_levate[article] # Consider number of levate in a single cycle for 'C' productions
        is_continuativo = True if row['N° levate ord'] == 'C' else False
        int_code = int(row['N° codice Interno'])

        # Check if article is present in the .csv provided, if not don't schedule it and report the error
        if article not in art_standard_levate.keys() or article not in art_standard_levate.keys() :
            if article not in warnings['missing_articles'] :
                warnings['missing_articles'].append(article)
            continue
        
        # Update article_compatibility with articles in running_products (if not already present).
        #     We assume that if a product is running, it can be
        #     surely be produced by the machine it's running on
        if article in article_compatibility.keys() and machine not in article_compatibility[article] :
            article_compatibility[article].append(machine) # Add machine to the list of machines able to produce the article if not already present
            warnings['article_compatibility_warnings_1'].append((int_code, article, machine))
        else :
            if article not in article_compatibility.keys() :
                article_compatibility[article] = [machine] # Create a new entry if article is not present in the compatibility list at all
                warnings['article_compatibility_warnings_2'].append((int_code, article, machine))
        # Update also the art_kg_per_levata dictionary with the new article-machine association if not already present
        if (article, machine) not in art_kg_per_levata.keys() :
            art_kg_per_levata[article, machine] = (art_kg_per_hour[article]*art_base_levata_cost[article]) * float(fuses_machines_associations[machine]) / 256.0

        '''
        Start Processing the current row
        '''

        # Evaluate how many levate needs to be done to conclude current cycle & how many kg each levata will produce
        remaining_levate = (tot_levate - curr_levata + 1) if not is_continuativo else 1 + (curr_levata % art_standard_levate[article])
        curr_cycle_remaining_levate = min(1 + (curr_levata % art_standard_levate[article]), remaining_levate)
        # If 'data ora partenza' is missing or is set to the future This cycle is not yet started, treat it as common_product instance
        is_actually_running = True if not pd.isna(row['data ora partenza']) and not pd.isna(row['data ora fine']) and row['data ora partenza'] <= now else False 
        
        levata_progress_percentage = 0
        if is_actually_running and (row['data ora partenza'] < now < row['data ora fine']) :
            # Compute current percentage of levata completion
            levata_progress_percentage = max(0, (now.timestamp()-row['data ora partenza'].timestamp()) / (row['data ora fine'].timestamp()-row['data ora partenza'].timestamp()))

        # Handle running products
        if is_actually_running :
            running_prod_kg = (curr_cycle_remaining_levate - levata_progress_percentage) * art_kg_per_levata[article, machine]
            # Initialize a record for the running products dataframe
            running_prod_record = {
                'cliente': client,
                'macchina': machine,
                'cod_articolo': article,
                'quantity' : max(running_prod_kg, art_kg_per_levata[article, machine]),
                'fine operazione attuale' : row['data ora fine'],       # 'fine operazione attuale' refers to time remaining until 'tipo operazione attuale' ends (remineder => current_op_type = {0,1,2,3} : {setup, load, running, unload})
                'data consegna' : due_date,
                'levate rimanenti ciclo': int(curr_cycle_remaining_levate),  # Remaining levate IN ORDER TO COMPLETE CURRENT CYCLE (CURRENT LEVATA INCLUDED)
                'tipo operazione attuale': 2,                           # always set as machine is running (not info on operators, might be added in the future)
                'operatore': 0,                                         # As op_type is fixed, operator is actually irrelevant
                'velocità': 0,                                          # Always assume standard velocity
            }
            running_products.append(running_prod_record)
        
        # if current cycle doesn't end programmed levate, instntiate the rest as common product
        further_levate_to_schedule = remaining_levate - curr_cycle_remaining_levate if is_actually_running else remaining_levate
        # If continuative, set quantity to 2000 - amount scheduled in running_products
        kg_for_continuative = KG_CONTINUATIVO - running_prod_record['quantity'] if is_actually_running else KG_CONTINUATIVO
        
        if further_levate_to_schedule > 0 or (is_continuativo and kg_for_continuative > 0):
            # Get amount of Kg. to be scheduled for common product
            common_prod_kg = further_levate_to_schedule * art_kg_per_levata[article, machine] if not is_continuativo else kg_for_continuative

            # Initialize a record for the common products if it's the first time it's found
            if int_code not in common_prod_ids_so_far.keys() :
                common_prod_ids_so_far[int_code] = len(common_products) # track position of common product in the list if it's the first time it's found
                common_prod_record = {
                    'cliente': client,
                    'cod_articolo': article,
                    'quantity' : common_prod_kg if not is_continuativo else kg_for_continuative,
                    'data inserimento': midnight,    # can be set to midnight (zero value in time units) as it was scheduled to be running
                    'data consegna' : due_date,
                }
                common_products.append(common_prod_record)
            else :
                # Update only the quantity of common product if it's already present in the list
                common_products[common_prod_ids_so_far[int_code]]['quantity'] += common_prod_kg if not is_continuativo else 0.0                
    
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
