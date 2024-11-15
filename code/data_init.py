import math
import random
import copy
import csv
import json
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
        self.setup_beg = {}
        self.setup_end = {}
        self.cycle_end = {}
        self.load_beg = {}
        self.load_end = {}
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


def init_data(num_common_jobs, num_running_jobs, num_machines, num_articles, num_op_groups, horizon) :
    """
    Job data randomly generated
    """
    random.seed(123)
    num_total_jobs = num_common_jobs + num_running_jobs

    # Fake list of articles
    articles = [a for a in range(num_articles)]

    # Fake Machine - Job & Job - Machine compatibility
    article_compatibility = {}
    machine_compatibility = {}
    for a in range(num_articles):
        article_compatibility[a] = random.sample(list(range(num_machines)), random.randint(num_machines//2, num_machines))
    for m in range(num_machines):
        machine_compatibility[m] = [a for a in range(num_articles) if m in article_compatibility[a]]

    # Fake costs
    base_setup_cost = {}
    base_load_cost = {}
    base_unload_cost = {}
    base_levata_cost = {}
    for a in range(num_articles):
        base_levata_cost[a] = 24*random.randint(2,4)
        for m in article_compatibility[a]:
            base_setup_cost[a,m] = 2
            base_load_cost[a,m] = 3
            base_unload_cost[a,m] = 3

    # Fake specs
    kg_per_levata = {}
    for m in range(num_machines) :
        for a in machine_compatibility[m] :
            kg_per_levata[m,a] = random.randint(2, 5)
    standard_levate = {}
    for a in range(num_articles) :
        standard_levate[a] = random.randint(2, 4)

    # Initialize objects
    common_products = []
    product_id = 0
    for _ in range(num_common_jobs) :
        article = random.randint(0, len(articles)-1)
        kg_request = random.randint(10, 40)
        start_date = 0
        due_date = horizon
        common_products.append(Product(product_id, article, kg_request, start_date, due_date))
        product_id += 1

    running_products = []
    avail_machines = [m for m in range(num_machines)]
    avail_operators = num_op_groups-1
    for _ in range(num_running_jobs): 
        # choose an article for which there is at least one machine available
        compatible_machines = 0
        while compatible_machines == 0:
            article = random.randint(0, len(articles)-1)
            for m in article_compatibility[article]:
                if m in avail_machines:
                    compatible_machines += 1
        start_date = 0
        due_date = horizon
        machine = random.choice(article_compatibility[article])
        while machine not in avail_machines:
            machine = random.choice(article_compatibility[article])
        operator = avail_operators
        velocity = 0
        remaining_levate = random.randint(1, standard_levate[article])
        kg_request = kg_per_levata[machine,article] * remaining_levate
        current_op_type = random.randint(0, 3)
        if current_op_type in [0,1,3] and avail_operators < 0 :
            current_op_type = 2
        remaining_time = 3 # Note that this can't exceed the amount of hours in a working day, or the GAP mechanism fails
        running_products.append(RunningProduct(product_id, article, kg_request, start_date, due_date, machine, operator, velocity, remaining_levate, current_op_type, remaining_time))
        product_id += 1
        avail_machines.remove(machine)
        avail_operators -= 1


    return common_products, running_products, article_compatibility, machine_compatibility, base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata

def date_hours_parser(start_date, due_date):
    """
    Parser for start and due date from strings to hours
    """
    # Parse input dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    due = datetime.strptime(due_date, "%Y-%m-%d")
    
    # Get current date and time
    current_datetime = datetime.now()

    # Calculate differences in hours
    start_hours = int((start - current_datetime).total_seconds() // 3600)
    due_hours = int((due - current_datetime).total_seconds() // 3600)

    if start_hours < 0:
        start_hours = 0

    return start_hours, due_hours

def init_csv_data(common_p_path: str, j_compatibility_path: str, m_compatibility_path: str, m_info_path: str, article_list_path: str, costs = (2,1,1), running_products =  {}):
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

    3.`job_compatibility[article]: {str : [int]}`
        From "articoli_macchine.json" we retrieve the associations, is the dict containing which article goes on which machine
    
    4.`machine_compatibility[machine]: {int: [str]}`
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
    const_setup_cost, const_load_cost, const_unload_cost = costs    

    # 1. common_products
    with open(common_p_path, newline='') as csvfile:
        
        csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        common_products = []
        str_out = ""
        for idx, row in enumerate(csv_data):
            if idx > 0:
                str_out = str_out + ', '.join(row) 
                str_out += '\n'
                # print(f'start_date:  {row[3]} - type: {type(row[3])}')
                start_date, due_date = date_hours_parser(row[3], row[4])
                common_products.append(Product(idx, row[1], int(float(row[2])), start_date, due_date))
    # 2. running_products
    # there is nothing here for now

    # 3. job_compatibility
    job_compatibility = json.load(open(j_compatibility_path))

    # 4. machine_compatibility
    machine_temp = json.load(open(m_compatibility_path))
    machine_compatibility = {}
    for i in machine_temp:
        if int(i) < 73:
            machine_compatibility[int(i)] = machine_temp[i]
    print(len(machine_compatibility))

    # 5-6-7. base_setup_cost, base_load_cost, base_unload_cost
    base_setup_cost = {}
    base_load_cost = {}
    base_unload_cost = {}

    m_info = json.load(open(m_info_path))
    fuses_machines_associations = {int(machine):m_info[machine]['n_fusi'] for machine in m_info}

    for a in job_compatibility:
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
                print(f'costo levata: {int(float(row[10]))}')
                for m in machine_compatibility:
                    kg_per_levata[m,row[0]] = int((float(row[10]) * float(row[6])) * float(fuses_machines_associations[m]) / 256.0)
          
    return common_products, running_products, job_compatibility, machine_compatibility, base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata
