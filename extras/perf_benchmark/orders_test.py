import pandas as pd
import time
from utils import run_solver
from tqdm.auto import tqdm

"""
Script to benchmark the solve time of the solver for incremental orders.
"""

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
        df = pd.read_csv("incremental_orders/results.csv")
        means = df["mean"].tolist()
        stds = df["std_dev"].tolist()
        start_index = len(means) + 1
    except Exception as e:
        start_index = 1

    for i in tqdm(range(start_index, len(orders) + 1), desc="Running benchmarks"):
        write_orders("incremental_orders/orders.csv", orders, i)
        times = []
        for j in tqdm(range(NUM_RUNS), desc=f"Running for {i} orders"):
            start_time = time.time()

            print()
            print()
            res = run_solver(
                "incremental_orders/orders.csv",
                "incremental_orders/running_products.csv",
                120,
                "",
                "",
                24,
                8,
                16,
                "",
                2,
                4,
                4,
                6,
                2,
                6000,
                0,
                0,
            )

            end_time = time.time()
            times.append(end_time - start_time)

        # compute std deviation and mean of times
        mean = sum(times) / len(times)
        std_dev = (sum([(t - mean) ** 2 for t in times]) / len(times)) ** 0.5

        means.append(mean)
        stds.append(std_dev)

        # write results to file
        df = pd.DataFrame(
            {"mean": means, "std_dev": stds, "num_orders": list(range(1, i + 1))}
        )
        df.to_csv("incremental_orders/results.csv", index=False)


def main():
    orders = read_orders("incremental_orders/new_orders.csv")
    benchmark_solve(orders)


if __name__ == "__main__":
    main()
