import math
import csv
import json
from datetime import datetime
from datetime import timedelta
import pandas as pd


class Product(object):
    """
    A Product is a request for a certain amount of an certain article
    """

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
        self.setup_base_cost = {}
        self.setup_gap = {}
        # load
        self.load_operator = {}
        self.load_beg = {}
        self.load_end = {}
        self.load_base_cost = {}
        self.load_gap = {}
        # unload
        self.unload_operator = {}
        self.unload_beg = {}
        self.unload_end = {}
        self.unload_base_cost = {}
        self.unload_gap = {}

    def __str__(self):
        return f"Product {self.id}\n    Article : {self.article}\n    Request : {self.kg_request} Kg\n    Start date : {self.start_date}\n    Due date : {self.due_date}\n---"


class RunningProduct(Product):
    """
    A RunningProduct is a some product which is already being processed by some machine
    when the scheduling operation begins
    """

    def __init__(
        self,
        id,
        article,
        kg_request,
        start_date,
        due_date,
        machine,
        operator,
        velocity,
        remaining_levate,
        current_op_type,
        remaining_time,
    ):
        super().__init__(id, article, kg_request, start_date, due_date)
        # Data related to settings associated to the running product
        self.operator = operator
        self.machine[0] = machine
        self.velocity[0] = velocity
        self.remaining_levate = remaining_levate  # amount of levate to finish the cycle

        # states in which phase the product is when the scheduling starts
        #   can have 4 possible values {0, 1, 2, 3}
        #   associated to a all cycle's possible phases {setup, load, running, unload}
        self.current_op_type = current_op_type
        # remaining time to finish the active production phase
        self.remaining_time = remaining_time

    def __str__(self):
        return f"RunningProduct {self.id}\n    Article : {self.article}\n    Request : {self.kg_request} Kg\n    Start date : {self.start_date}\n    Due date : {self.due_date}\n    Machine : {self.machine}\n    Operator : {self.operator}\n    Velocity : {self.velocity}\n    Remaining levate : {self.remaining_levate}\n    Current operation type : {self.current_op_type}\n    Remaining time : {self.remaining_time}\n---"


class Schedule(object):
    """
    A schedule is the final result of the Job Scheduling Problem (JSP).
    """

    def __init__(self, products, invalid_intervals=[]):
        # Reference to products class instances, ensuring proper types
        self.products = products
        self.invalid_intervals = invalid_intervals
        for _, prod in self.products:
            if not isinstance(prod, (Product, RunningProduct)):
                raise ValueError("Invalid product type")

    def __str__(self):
        output = "Production Schedule:\n"
        for p, prod in self.products:
            output += f"Product : {prod.id} - Article : {prod.article} - Request : {prod.kg_request} Kg - Start Date : {prod.start_date} - Due Date : {prod.due_date}\n\tLevate: {prod.num_levate}\n"
            for c in prod.setup_beg.keys():
                output += f"    Cycle {c} :\n"
                output += f"        Machine   : {prod.machine[c]}:\n"
                output += f"        Velocity  : {prod.velocity[c]}\n"
                output += f"        Cycle End : {prod.cycle_end[c]}\n"
                output += f"        Setup     : ({prod.setup_beg[c]}, {prod.setup_end[c]}) COST: {prod.setup_base_cost[c]} GAP: {prod.setup_gap[c]}\n"
                for l in range(prod.num_levate[c]):
                    if (c, l) in prod.load_beg.keys():
                        # compute load and unload times from the current day at 00:00 (remove the current time of the day)
                        load_beg_datetime = (
                            datetime.now()
                            + timedelta(hours=prod.load_beg[c, l])
                            - timedelta(
                                hours=datetime.now().hour,
                                minutes=datetime.now().minute,
                                seconds=datetime.now().second,
                            )
                        )
                        load_end_datetime = (
                            datetime.now()
                            + timedelta(hours=prod.load_end[c, l])
                            - timedelta(
                                hours=datetime.now().hour,
                                minutes=datetime.now().minute,
                                seconds=datetime.now().second,
                            )
                        )
                        unload_beg_datetime = (
                            datetime.now()
                            + timedelta(hours=prod.unload_beg[c, l])
                            - timedelta(
                                hours=datetime.now().hour,
                                minutes=datetime.now().minute,
                                seconds=datetime.now().second,
                            )
                        )
                        unload_end_datetime = (
                            datetime.now()
                            + timedelta(hours=prod.unload_end[c, l])
                            - timedelta(
                                hours=datetime.now().hour,
                                minutes=datetime.now().minute,
                                seconds=datetime.now().second,
                            )
                        )
                        output += f"            Levata [{l}] : LOAD({load_beg_datetime}, {load_end_datetime}, COST: {prod.load_base_cost[c,l]} GAP: {prod.load_gap[c,l]}) UNLOAD({unload_beg_datetime}, {unload_end_datetime}, COST: {prod.unload_base_cost[c,l]} GAP: {prod.unload_gap[c,l]})\n"
            output += "\n"

        return output

    def __len__(self):
        return len(self.products)


def seconds_to_time_units(seconds, time_units_in_a_day):
    if time_units_in_a_day == 24:
        return seconds // (60 * 60)
    elif time_units_in_a_day == 48:
        return seconds // (60 * 30)
    elif time_units_in_a_day == 96:
        return seconds // (60 * 15)
    else:
        assert ValueError("Invalid choice for time units")


def get_prods_from_csv(
    running_products_path: str,
    common_products_path: str,
    now: datetime,
    time_units_in_a_day: int,
):
    """
    Get the products from the csv files
    
    Args:
    - `running_products_path`: path to the csv file containing the running products
    - `common_products_path`: path to the csv file containing the common products
    - `now`: current datetime
    - `time_units_in_a_day`: number of time units in a day
    
    Returns:
    - `running_products`: list of RunningProduct objects
    - `common_products`: list of Product objects
    """
    running_products = []
    common_products = []
    base_id = 1
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    """
    Initialize the RunningProduct objects
    """
    running_products_df = pd.read_csv(running_products_path)
    for _, row in running_products_df.iterrows():
        # Gather Row Data
        this_prod_id = base_id
        kg_request = int(row["quantity"])
        article = str(row["cod_articolo"])
        machine = int(row["macchina"])
        operator = int(row["operatore"])
        velocity = int(0)  # In this version of the scheduler, velocity is not used
        remaining_levate = int(row["levate rimanenti ciclo"])
        current_op_type = int(row["tipo operazione attuale"])
        end_curr_op = pd.to_datetime(row["fine operazione attuale"])
        start_date = (
            0  # As the product is already running, the start date can set to be 0
        )
        due_date = pd.to_datetime(row["data consegna"])

        # Adjust inputs to the scheduler input format (time units & other assumptions)
        remaining_time = seconds_to_time_units(
            (end_curr_op - now).total_seconds(), time_units_in_a_day
        )
        remaining_time = int(
            max(0, remaining_time)
        )  # if the remaining time is negative, set it to 0

        due_date = seconds_to_time_units(
            (due_date - midnight).total_seconds(), time_units_in_a_day
        )
        due_date = int(max(0, due_date))

        running_products.append(
            RunningProduct(
                this_prod_id,
                article,
                kg_request,
                start_date,
                due_date,
                machine,
                operator,
                velocity,
                remaining_levate,
                current_op_type,
                remaining_time,
            )
        )
        base_id += 1
    """
    Initialize the CommonProduct objects
    """
    running_products_df = pd.read_csv(common_products_path)
    for _, row in running_products_df.iterrows():
        # Gather Row Data
        this_prod_id = base_id
        kg_request = int(row["quantity"])
        article = str(row["cod_articolo"])
        start_date = pd.to_datetime(row["data inserimento"])
        due_date = pd.to_datetime(row['data consegna'])

        # Adjust inputs to the scheduler input format (time units & other assumptions)
        start_date = seconds_to_time_units(
            (start_date - midnight).total_seconds(), time_units_in_a_day
        )
        start_date = int(max(0, start_date))
        due_date = seconds_to_time_units(
            (due_date - midnight).total_seconds(), time_units_in_a_day
        )
        due_date = int(max(0, due_date))

        # Initialize instance of 'Product'
        common_products.append(
            Product(this_prod_id, article, kg_request, start_date, due_date)
        )
        base_id += 1

    return running_products, common_products


def init_csv_data(
    common_p_path: str,
    running_p_path: str,
    j_compatibility_path: str,
    m_info_path: str,
    article_list_path: str,
    costs=(2, 1, 1),
    now=datetime.now(),
    time_units_in_a_day=24,
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

    """
    START DATA INITIALIZATION
    """
    try:
        # 1. Get products from csv
        running_products, common_products = get_prods_from_csv(
            running_p_path, common_p_path, now, time_units_in_a_day
        )
    except Exception as e:
        raise ValueError(f"Error while reading csv files: {e}")
    
    try:
        # common_products = []
        # 2. extract job_compatibility from .json file
        job_compatibility = json.load(open(j_compatibility_path))

        # 2.5 Update job_compatibility with articles in running_products (if not already present).
        #     We assume that if a product is running, it can be
        #     surely be produced by the machine it's running on
        for prod in running_products:
            # Add key to dictionary if not present and set the machine as the only compatible one
            if prod.article not in job_compatibility.keys():
                raise ValueError(f"Article {prod.article} not found in job_compatibility")
            else:
                # Add machine to the list of compatible machines if not already present
                if prod.machine[0] not in job_compatibility[prod.article]:
                    raise ValueError(f"Machine {prod.machine[0]} not found in job_compatibility for article {prod.article}")
                    
    except Exception as e:
        raise ValueError(f"Error while reading json files for job compatibility: {e}")

    try:
        # Retrieve machine info from .json
        m_info = json.load(open(m_info_path))
        fuses_machines_associations = {
            int(machine): m_info[machine]["n_fusi"] for machine in m_info
        }
    except Exception as e:
        raise ValueError(f"Error while reading json files for machine info: {e}")

    # 4-5-6. base_setup_cost, base_load_cost, base_unload_cost
    const_setup_cost, const_load_cost, const_unload_cost = costs
    base_setup_cost = {}
    base_load_cost = {}
    base_unload_cost = {}
    for a in job_compatibility:
        for m in fuses_machines_associations:
            base_setup_cost[a, int(m)] = int(math.ceil(float(const_setup_cost)))
            base_load_cost[a, int(m)] = int(
                math.ceil(
                    float(const_load_cost) * float(fuses_machines_associations[m])
                )
            )
            base_unload_cost[a, int(m)] = int(
                math.ceil(
                    float(const_unload_cost) * float(fuses_machines_associations[m])
                )
            )

    try:
        # 8-9-10. base_levata_cost, standard_levate, kg_per_levata
        base_levata_cost = {}
        standard_levate = {}
        kg_per_levata = {}

        df = pd.read_csv(article_list_path)

        for idx, row in df.iterrows():
            # row['codarticolo'] is the article code, row['ore_levata'] is the hours needed for a single "levata"
            if pd.isna(row['codarticolo']):
                continue
            base_levata_cost[row['codarticolo']] = int(float(row['ore_levata']))
            standard_levate[row['codarticolo']] = int(float(row['no_cicli']))
            
            if row['codarticolo'] in job_compatibility:
                for m in job_compatibility[row['codarticolo']]:
                    kg_per_levata[m, row['codarticolo']] = int(
                        (float(row['ore_levata']) * float(row['kg_ora']))
                        * float(fuses_machines_associations[m])
                        / 256.0
                    )
        # breakpoint()
    except Exception as e:
        raise ValueError(f"Error while reading csv files for article list: {e}")

    return (
        common_products,
        running_products,
        job_compatibility,
        base_setup_cost,
        base_load_cost,
        base_unload_cost,
        base_levata_cost,
        standard_levate,
        kg_per_levata,
    )
