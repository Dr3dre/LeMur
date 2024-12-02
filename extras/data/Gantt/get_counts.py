import pandas as pd
import matplotlib.pyplot as plt
import locale
import numpy as np

solver_counts = [0, 0, 5, 3, 1, 1, 0, 3, 2, 3, 2, 2, 3, 0, 7, 2, 2, 0, 1, 4, 0, 5, 3, 6, 1, 2, 4, 0, 7, 4, 3, 2, 2, 5, 0, 5, 6, 4, 2, 2, 3, 0, 7, 4, 2, 0, 2, 3, 0, 5, 3, 3, 1, 3, 5, 0, 4, 3, 2, 3, 1, 3, 0, 5, 2, 4, 3, 3, 1, 0, 4, 2, 0, 2, 2, 1, 0, 2, 2, 3, 4, 2, 3, 0, 2, 3, 3, 1, 3, 3]


if __name__ == '__main__':
    # Set locale for Italian month names
    try:
        locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')  # Adjust if necessary for your system
    except locale.Error:
        raise RuntimeError("Italian locale not supported on this system. Ensure 'it_IT.UTF-8' is installed.")

    # Load the dataset
    df = pd.read_csv('25-11-2024_counts.csv', sep='\t')

    # Parse column names as dates
    try:
        dates = pd.to_datetime(df.columns, format='%d-%b', errors='coerce', dayfirst=True)
    except Exception as e:
        raise ValueError(f"Error parsing dates from column names: {e}")

    # Handle missing year (assign year based on assumption)
    dates = dates.map(lambda x: x.replace(year=2024) if x.month >= 11 else x.replace(year=2025))

    # Assign parsed dates back to columns
    df.columns = dates

    # Count non-NaN values per day (column)
    daily_non_nan_counts = df.notna().sum()

    # Align `solver_counts` with dates
    if len(solver_counts) < len(dates):
        # Pad with zeros if there are fewer `solver_counts` than dates
        solver_counts.extend([0] * (len(dates) - len(solver_counts)))
    elif len(solver_counts) > len(dates):
        # Truncate `solver_counts` to match the length of dates
        solver_counts = solver_counts[:len(dates)]

    # Convert `solver_counts` to a Series with date index
    solver_counts_series = pd.Series(solver_counts, index=dates)

    # Calculate mean and standard deviation for the dataset counts
    mean_count = daily_non_nan_counts.mean()
    std_dev_count = daily_non_nan_counts.std()

    # Plot the results
    fig, ax = plt.subplots(figsize=(16, 8))

    # Line and marker styling for the daily counts
    ax.plot(
        daily_non_nan_counts.index,
        daily_non_nan_counts.values,
        marker='o',
        markersize=4.5,
        linestyle='-',
        linewidth=2,
        color='salmon',  # Softer red for the line
        markerfacecolor='lightblue',  # Softer blue for markers
        markeredgecolor='darkblue',
        label='LeMur (from Gantt)'
    )

    # Line styling for the solver_counts data
    ax.plot(
        solver_counts_series.index,
        solver_counts_series.values,
        marker='o',
        markersize=4.5,
        linestyle='-',
        linewidth=2,
        color='#3CB371',
        markerfacecolor='#B0F2B6',
        markeredgecolor='darkgreen',
        label='Our Solver'
    )

    # Crop the X-axis
    ax.set_xlim(dates.min(), dates.max())

    # Aesthetic improvements
    ax.set_title("Scheduled Levate (25-11-2024)", fontsize=18, fontweight='bold', pad=40)
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.grid(visible=True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=12)

    # Highlight weekends for visual context
    weekends = dates[dates.weekday >= 5]
    for weekend in weekends:
        ax.axvline(x=weekend, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)

    # Add the LaTeX-style annotation for mean and standard deviation between the title and the graph
    plt.figtext(
        0.44, 0.865,  # Position in figure coordinates (x, y)
        f"${mean_count:.2f} \pm {std_dev_count:.2f}$",  # LaTeX formatted text
        ha='center',  # Horizontal alignment to center
        va='top',  # Align text to the top
        fontsize=16,
        fontweight='bold',  # Make the text bold
        color='red',  # Match the red color of the graph line
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    # Add the LaTeX-style annotation for mean and standard deviation between the title and the graph
    plt.figtext(
        0.59, 0.865,  # Position in figure coordinates (x, y)
        f"${np.mean(solver_counts):.2f} \pm {np.std(solver_counts):.2f}$",  # LaTeX formatted text
        ha='center',  # Horizontal alignment to center
        va='top',  # Align text to the top
        fontsize=16,
        fontweight='bold',  # Make the text bold
        color='green',  # Match the red color of the graph line
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
    )


    # Adjust layout for better spacing and centering
    plt.subplots_adjust(top=0.85, bottom=0.15)

    # Show the plot
    plt.show()
