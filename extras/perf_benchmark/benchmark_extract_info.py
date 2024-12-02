import re
import matplotlib.pyplot as plt

def extract_bounds_from_text(raw_text):
    # Regular expressions to extract data
    bound_pattern = re.compile(r"#[^ ]*\s+([\d.]+)s best:(\S+)\s+next:\[(\d+),(\d+)\]")
    first_solution_pattern = re.compile(r"#(\d+)\s+([\d.]+)s best:(\d+)\s+next:\[.*\]")

    timestamps = [0]
    lower_bounds = [0]
    upper_bounds = [1e10]
    first_solution_time = None

    # Process lines in the raw text
    for line in raw_text.splitlines():
        # Match bounds
        bound_match = bound_pattern.search(line)
        if bound_match:
            timestamps.append(float(bound_match.group(1)))
            lower_bounds.append(int(bound_match.group(3)))
            upper_bounds.append(int(bound_match.group(4)))

        # Match first solution
        if not first_solution_time:
            solution_match = first_solution_pattern.search(line)
            if solution_match:
                first_solution_time = float(solution_match.group(2))

    # Collapse upper and lower bounds to the final solution at the end
    final_solution = int(input("Enter the final solution: "))
    upper_bounds.append(final_solution)
    lower_bounds.append(final_solution)
    timestamps.append(timestamps[-1])

    return {
        "timestamps": timestamps,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "first_solution_time": first_solution_time
    }

# Example usage
raw_text_file = "benchmark_output.txt"
with open(raw_text_file, "r") as file:
    raw_text = file.read()
data = extract_bounds_from_text(raw_text)

# Displaying extracted data
print("Timestamps:", data["timestamps"])
print("Lower Bounds:", data["lower_bounds"])
print("Upper Bounds:", data["upper_bounds"])
print("Time of First Solution:", data["first_solution_time"])

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(data["timestamps"], data["lower_bounds"], label='Lower Bound', color='blue', marker='o')
plt.plot(data["timestamps"], data["upper_bounds"], label='Upper Bound', color='red', marker='o')

# Highlight the area between lower and upper bounds
plt.fill_between(data["timestamps"], data["lower_bounds"], data["upper_bounds"], color='gray', alpha=0.2)

# Mark the time of first solution
if data["first_solution_time"] is not None:
    plt.axvline(x=data["first_solution_time"], color='green', linestyle='--', label='First Solution')

# y log scale
plt.yscale('log')

# set limits as max and min of bounds excluding the first value
plt.ylim(min(data["lower_bounds"][1:])-100, max(data["upper_bounds"][1:])+100)

plt.xlabel('Time (s)')
plt.ylabel('Bound Values')
plt.title('Evolution of Bounds Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
