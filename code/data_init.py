import math
import random
import copy
import csv
import json
import pandas as pd

from datetime import datetime
from datetime import timedelta


class Product (object) :
    '''
    A Product is a request for a certain amount of an certain article
    '''
    def __init__(self, id, article, kg_request, start_date, due_date):
        # Input level Data
        self.id = id
        self.article = article
        self.kg_request = kg_request
        self.start_date = start_date
        self.due_date = due_date

        # Output data (will be filled by the scheduler output)
        self.machine = {}
        self.velocity = {}
        self.num_levate = {}
        # setup
        self.setup_operator = {}
        self.setup_beg = {}
        self.setup_end = {}
        self.cycle_end = {}
        # load
        self.load_operator = {}
        self.load_beg = {}
        self.load_end = {}
        # unload
        self.unload_operator = {}
        self.unload_beg = {}
        self.unload_end = {}
    
    def __str__(self) :
        return f"Product {self.id}\n    Article : {self.article}\n    Request : {self.kg_request} Kg\n    Start date : {self.start_date}\n    Due date : {self.due_date}\n---"

class RunningProduct (Product) :
    '''
    A RunningProduct is a some product which is already being processed by some machine
    when the scheduling operation begins
    '''
    def __init__(self, id, article, kg_request, start_date, due_date, machine, operator, velocity, remaining_levate, current_op_type, remaining_time) :
        super().__init__(id, article, kg_request, start_date, due_date)
        # Data related to settings associated to the running product
        self.operator = operator
        self.machine[0] = machine
        self.velocity[0] = velocity
        self.remaining_levate = remaining_levate # amount of levate to finish the cycle

        # states in which phase the product is when the scheduling starts
        #   can have 4 possible values {0, 1, 2, 3} 
        #   associated to a all cycle's possible phases {setup, load, running, unload}
        self.current_op_type = current_op_type
        # remaining time to finish the active production phase
        self.remaining_time = remaining_time

    def __str__(self) :
        return f"RunningProduct {self.id}\n    Article : {self.article}\n    Request : {self.kg_request} Kg\n    Start date : {self.start_date}\n    Due date : {self.due_date}\n    Machine : {self.machine}\n    Operator : {self.operator}\n    Velocity : {self.velocity}\n    Remaining levate : {self.remaining_levate}\n    Current operation type : {self.current_op_type}\n    Remaining time : {self.remaining_time}\n---"


class Schedule(object):
    '''
    A schedule is the final result of the Job Scheduling Problem (JSP).
    '''
    def __init__(self, products):
        # Reference to products class instances, ensuring proper types
        self.products = products
        for _, prod in self.products:
            if not isinstance(prod, (Product, RunningProduct)):
                raise ValueError("Invalid product type")

    def __str__(self):
        output = "Production Schedule:\n"
        for p, prod in self.products :
            output += f"Product : {prod.id} Article : {prod.article} Request : {prod.kg_request} Kg\n"
            for c in prod.setup_beg.keys():
                output += f"    Cycle {c} :\n"
                output += f"        Machine   : {prod.machine[c]}:\n"
                output += f"        Velocity  : {prod.velocity[c]}\n"
                output += f"        Cycle End : {prod.cycle_end[c]}\n"
                output += f"        Setup     : ({prod.setup_beg[c]}, {prod.setup_end[c]})\n"
                for l in range(prod.num_levate[c]) :
                    if (c,l) in prod.load_beg.keys() : 
                        # compute load and unload times from the current day at 00:00 (remove the current time of the day)
                        load_beg_datetime = datetime.now() + timedelta(hours=prod.load_beg[c,l]) - timedelta(hours=datetime.now().hour, minutes=datetime.now().minute, seconds=datetime.now().second)
                        load_end_datetime = datetime.now() + timedelta(hours=prod.load_end[c,l]) - timedelta(hours=datetime.now().hour, minutes=datetime.now().minute, seconds=datetime.now().second)
                        unload_beg_datetime = datetime.now() + timedelta(hours=prod.unload_beg[c,l]) - timedelta(hours=datetime.now().hour, minutes=datetime.now().minute, seconds=datetime.now().second)
                        unload_end_datetime = datetime.now() + timedelta(hours=prod.unload_end[c,l]) - timedelta(hours=datetime.now().hour, minutes=datetime.now().minute, seconds=datetime.now().second)
                        output += f"            Levata [{l}] : LOAD({load_beg_datetime}, {load_end_datetime}) UNLOAD({unload_beg_datetime}, {unload_end_datetime})\n"
            output += "\n"
        
        return output
    
    def __len__(self):
        return len(self.products)


def date_hours_parser(start_date, due_date, current_datetime=datetime.now()):
    """
    Parser for start and due date from strings to hours
    """
    
    # Parse input dates
    print(f'start_date:  {start_date} - type: {type(start_date)}, due_date: {due_date} - type: {type(due_date)}')
    
    if pd.isna(start_date) or pd.isna(due_date):
        return None, None
    
    if isinstance(start_date, str):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        due = datetime.strptime(due_date, "%Y-%m-%d")
    elif isinstance(start_date, pd.Timestamp):
        start = start_date.to_pydatetime()
        due = due_date.to_pydatetime()
    else:
        return None, None

    # Calculate differences in hours
    start_hours = int((start - current_datetime).total_seconds() // 3600)
    due_hours = int((due - current_datetime).total_seconds() // 3600)

    if start_hours < 0:
        start_hours = 0

    return start_hours, due_hours

def get_input_data(stato_path, plan_path):
    """
    -> pd.DataFrame, pd.DataFrame
    """

    stato = pd.read_excel(stato_path)
    plan = pd.read_excel(plan_path)

    running = stato.join(plan.set_index('n. ordine interno'), on='N° codice Interno', how='left', rsuffix='_plan')
    running.dropna(subset=['N° codice Interno'], inplace=True)

    running.fillna(value={
        'data ora\npartenza': pd.Timestamp.now(),
        'data ora\nfine': pd.Timestamp(2027, 1, 1),
    }, inplace=True)

    return running


def get_running_products(df, time_units_in_a_day=24, current_datetime=datetime.now()):
    '''
    This function reads the production and machine state files and returns a list of RunningProduct objects
    '''
    
    running_products = []

    for index, row in df.iterrows():

        id = row['N° codice Interno']
        article = row['codarticolo']
        kg_request = row['quantità']
        # start_date = row['data ora\npartenza'].date()
        # due_date = row['data ora\nfine'].date()
        print(id)
        start_date, due_date = date_hours_parser(row['data ora\npartenza'], row['data ora\nfine'], current_datetime)
        machine = row['N° MC']
        operator = pd.NA

        giorni_passati = ((pd.Timestamp.now() - pd.to_datetime(start_date)).days) * time_units_in_a_day
        tempo_levata = row['N° levate att.'] / giorni_passati

        velocity = tempo_levata             #NOTE: not sure about this
        if row['N° levate ord'] == 'C':     #continuato
            remaining_levate = 1000
        else:
            remaining_levate = int(row['N° levate ord']) - row['N° levate att.']
        current_op_type = 3      #running   #NOTE: possiamo capire gli altri stati da qui?
        remaining_time = remaining_levate * tempo_levata

        running_products.append(RunningProduct(id, article, kg_request, start_date, due_date, machine, operator, velocity, remaining_levate, current_op_type, remaining_time))
    
    return running_products

def get_orders(common_p_path):
        with open(common_p_path, newline='') as csvfile:
            
            csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')
            orders = []
            str_out = ""
            for idx, row in enumerate(csv_data):
                if idx > 0:
                    str_out = str_out + ', '.join(row) 
                    str_out += '\n'
                    # print(f'start_date:  {row[3]} - type: {type(row[3])}')
                    start_date, due_date = date_hours_parser(row[3], row[4])
                    orders.append(Product(idx, row[1], int(float(row[2])), start_date, due_date))
        return orders

def check_consistency_compatibility(articoli_macchine, macchine_articoli):
    #check consistency between articoli_macchine and macchine_articoli
    print(articoli_macchine)
    print(macchine_articoli)
    for a in articoli_macchine:
        for m in articoli_macchine[a]:
            if a not in macchine_articoli[m]:
                print(f'Inconsistency between articoli_macchine and macchine_articoli: {a} not in {m}')
    
def get_infos(articoli_macchine_path, macchine_info_path, article_list_path, costs = (2,1,1)):

    const_setup_cost, const_load_cost, const_unload_cost = costs

    base_setup_cost = {}
    base_load_cost = {}
    base_unload_cost = {}

    m_info = json.load(open(macchine_info_path))
    articoli_macchine = json.load(open(articoli_macchine_path))
    fuses_machines_associations = {int(machine):m_info[machine]['n_fusi'] for machine in m_info}

    for a in articoli_macchine:
        for m in fuses_machines_associations:
            base_setup_cost[a,int(m)] = int(math.ceil(float(const_setup_cost)))
            base_load_cost[a,int(m)] = int(math.ceil(float(const_load_cost) * float(fuses_machines_associations[m])))
            base_unload_cost[a,int(m)] = int(math.ceil(float(const_unload_cost) * float(fuses_machines_associations[m])))


    # 8-9-10. base_levata_cost, standard_levate, kg_per_levata
    base_levata_cost = {}
    standard_levate = {}
    kg_per_levata = {}

    with open(article_list_path, newline='') as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(csv_data):
            # row[0] is the article code, row[10] is the hours needed for a single "levata"
            if idx > 0:
                # print(f'article: {row[0]}')
                base_levata_cost[row[0]] = int(float(row[10]))
                standard_levate[row[0]] = int(float(row[9]))
                # print(f'costo levata: {int(float(row[10]))}')
                if row[0] in articoli_macchine:
                    for m in articoli_macchine[row[0]]:
                        # print(f'kg_ora: {float(row[6])} - ore_levata: {float(row[10])} - fusi: {fuses_machines_associations[m]} - kg_per_levata: {int((float(row[10]) * float(row[6])) * float(fuses_machines_associations[m]) / 256.0)} - kg_ciclo: {int((float(row[10]) * float(row[6])) * float(fuses_machines_associations[m]) / 256.0) * int(float(row[9]))}')
                        kg_per_levata[m,row[0]] = int((float(row[10]) * float(row[6])) * float(fuses_machines_associations[m]) / 256.0)
    
    return base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata

def init_data(
        orders_path: str, 
        articoli_macchine_path: str, 
        macchine_articoli_path: str, 
        macchine_info_path: str, 
        article_list_path: str, 
        status_path: str,
        plan_path: str,
        costs = (2,1,1), 
        running_products =  {},
        current_datetime = datetime.now()
    ):
    """

    - `costs:` a tuple representing the costs of the operation made by the human operators, cost expressed *per fuse*:
        - `base_setup_cost:` the cost of setting up a machine before a cicle (see 5)
        - `base_load_cost:` the cost of the load operation before a levata (see 6)
        - `base_unload_cost:` the cost of the unload operation after a levata (see 7)
    1.`common_products: [Product]`
        The csv gave us:
        - Client name (useless)
        - Product code 
        - Product quantity
        - Insertion date
        - Due date
        for every row, which is a "Product" object and the list of those form the "common_products"
    
    2.`running_products: {Product}`
        Assuming an empty list right now

    3.`articoli_macchine[article]: {str : [int]}`
        From "articoli_macchine.json" we retrieve the associations, is the dict containing which article goes on which machine
    
    4.`macchine_articoli[machine]: {int: [str]}`
        From "macchine_articoli.json" we retrieve the associations, is the dict containing which machine can produce which article
    
    5.`base_setup_cost[article, machine]: {(str,int) : float}`
        Costant taken from input, machine dependent (number of fuses)
    
    6.`base_load_cost[article, machine]: {(str,int) : float}`
        For every product is machine dependent (number of fuses) 
    
    7.`base_unload_cost[article, machine]: {(str,int) : float}`
        Same as point 6 
    
    8.`base_levata_cost[article]: {str: int}`
        For every product, from "lista_articoli" see "ore_levata" , is the dict with the associations of the cost(in hours) of a levata for a product
    
    9.`standard_levate[article]: {str: int}`
        For every product, form "lista_articoli" see "no_cicli" but is leavta, is the dict with the associations of the number of levate in a cicle for a product
    
    10.`kg_per_levata[machine, article]: {(int,str): int}`
        See from "lista_articoli" the "kg_ora" and the "ore_levata", ASSUMING THAT the obtained data have been made in a 256 fuses machine
    
"""

    # 1. common_products
    orders = get_orders(orders_path)
    
    # 2. running_products
    df = get_input_data(status_path, plan_path)
    running_products = get_running_products(df, current_datetime)

    # 3. articoli_macchine      #TODO: has to be updated
    articoli_macchine = json.load(open(articoli_macchine_path))

    # 4. macchine_articoli
    # macchine_articoli = {}
    # for a in articoli_macchine:
    #     for m in articoli_macchine[a]:
    #         if m not in macchine_articoli:
    #             macchine_articoli[m] = []
    #         macchine_articoli[m].append(a)
    # print(len(macchine_articoli))
    macchine_articoli = json.load(open(macchine_articoli_path))

    # check_consistency_compatibility(articoli_macchine, macchine_articoli)

    # 5-6-7. base_setup_cost, base_load_cost, base_unload_cost
    base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata = get_infos(articoli_macchine_path, macchine_info_path, article_list_path, costs)
    
    # breakpoint()
    return orders, running_products, articoli_macchine, macchine_articoli, base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata
