import random
import copy
import csv

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
            output += f"Product : {chr(p+65)}\n"
            for c in prod.setup_beg.keys():
                output += f"    Cycle {c} :\n"
                output += f"        Machine   : {prod.machine[c]}:\n"
                output += f"        Velocity  : {prod.velocity[c]}\n"
                output += f"        Cycle End : {prod.cycle_end[c]}\n"
                output += f"        Setup     : ({prod.setup_beg[c]}, {prod.setup_end[c]})\n"
                for l in range(prod.num_levate[c]) :
                    if (c,l) in prod.load_beg.keys() : 
                        output += f"            Levata [{l}] : ({prod.load_beg[c,l]}, {prod.load_end[c,l]}) => ({prod.unload_beg[c,l]}, {prod.unload_end[c,l]})\n"
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
    job_compatibility = {}
    machine_compatibility = {}
    for m in range(num_machines):
        machine_compatibility[m] = [a for a in range(num_articles)]  # All other machines are compatible with all jobs
    for a in range(num_articles):
        job_compatibility[a] = [i for i in range(num_machines)]  # All jobs are compatible with all machines except machine 0

    # Fake costs
    base_setup_cost = {}
    base_load_cost = {}
    base_unload_cost = {}
    base_levata_cost = {}
    for a in range(num_articles):
        base_levata_cost[a] = 24*random.randint(2,4)
        for m in job_compatibility[a]:
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
    avail_machines = num_machines-1
    avail_operators = num_op_groups-1
    for _ in range(num_running_jobs): 
        article = random.randint(0, len(articles)-1)
        start_date = 0
        due_date = horizon
        machine = avail_machines
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
        avail_machines -= 1
        avail_operators -= 1


    return common_products, running_products, job_compatibility, machine_compatibility, base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata

def init_csv_data(csv_path: str):
    with open(csv_path, newline='') as csvfile:

        csv_data = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in csv_data:
            print(', '.join(row))

    # return common_products, running_products, job_compatibility, machine_compatibility, base_setup_cost, base_load_cost, base_unload_cost, base_levata_cost, standard_levate, kg_per_levata
