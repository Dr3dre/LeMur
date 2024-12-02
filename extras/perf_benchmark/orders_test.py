import pandas as pd
import time
from utils import run_solver

NUM_RUNS = 5

def read_orders(file_path):
    df = pd.read_csv(file_path)
    return df

def write_orders(file_path, orders, num_orders):
    orders = orders[:num_orders]
    df = pd.DataFrame(orders)
    df.to_csv(file_path, index=False)

def benchmark_solve(orders):
    means = []
    stds = []
    
    # get the results from the previous run and start from the last order
    try:
        df = pd.read_csv('incremental_orders/results.csv')
        means = df['mean'].tolist()
        stds = df['std_dev'].tolist()
        start_index = len(means)
    except Exception as e:
        start_index = 1
    
    for i in range(start_index, len(orders) + 1):
        write_orders('incremental_orders/orders.csv', orders, i)
        times = []
        for j in range(NUM_RUNS):
            start_time = time.time()
            
            res = run_solver(120,"","",24,8,16,"",2,4,4,6,2,6000,0,0)
            print(res)
            
            end_time = time.time()
            times.append(end_time - start_time)
        # compute std deviation and mean of times
        mean = sum(times) / len(times)
        std_dev = (sum([(t - mean) ** 2 for t in times]) / len(times)) ** 0.5
        
        means.append(mean)
        stds.append(std_dev)
        
        # write results to file
        df = pd.DataFrame({'mean': means, 'std_dev': stds, 'num_orders': list(range(1, i + 1))})
        df.to_csv('incremental_orders/results.csv', index=False)
        
    return times

def main():
    orders = read_orders('incremental_orders/new_orders.csv')
    times = benchmark_solve(orders)
    for i, t in enumerate(times):
        print(f"Time for solve {i + 1}: {t:.4f} seconds")

if __name__ == "__main__":
    main()
