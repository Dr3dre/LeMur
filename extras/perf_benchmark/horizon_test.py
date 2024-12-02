import pandas as pd
import time
from utils import run_solver
from tqdm.auto import tqdm

"""
Script to benchmark the solve time of the solver for incremental orders.
"""

NUM_RUNS = 5
HORIZONS = [ 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400 ]

def benchmark_solve():
    means = []
    stds = []

    # get the results from the previous run and start from the last order
    try:
        df = pd.read_csv("incremental_horizon/results.csv")
        means = df["mean"].tolist()
        stds = df["std_dev"].tolist()
        start_index = len(means)
    except Exception as e:
        start_index = 0

    for i in tqdm(range(start_index, len(HORIZONS)), desc="Running benchmarks"):
        times = []
        for j in tqdm(range(NUM_RUNS), desc=f"Running for {HORIZONS[i]} horizon"):
            start_time = time.time()

            res = run_solver(
                "incremental_horizon/new_orders.csv",
                "incremental_horizon/running_products.csv",
                HORIZONS[i],
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
            print(res)

            end_time = time.time()
            times.append(end_time - start_time)
        # compute std deviation and mean of times
        mean = sum(times) / len(times)
        std_dev = (sum([(t - mean) ** 2 for t in times]) / len(times)) ** 0.5

        means.append(mean)
        stds.append(std_dev)

        # write results to file
        df = pd.DataFrame(
            {"mean": means, "std_dev": stds, "horizon": HORIZONS[: i + 1]}
        )
        df.to_csv("incremental_orders/results.csv", index=False)


def main():
    benchmark_solve()


if __name__ == "__main__":
    main()
