import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from data_init import RunningProduct

def plot_gantt_chart(production_schedule, max_cycles, num_machines, horizon, prohibited_intervals, time_units_from_midnight):
    """
    Plot a Gantt chart for the given job schedule
    """
    # Initialize plot
    _, gnt = plt.subplots(figsize=(10, num_machines * 0.75))

    # Add indicators for night shifts, holidays, weekends, etc.
    for x_start, x_end in prohibited_intervals:
        gnt.axvspan(x_start, x_end, color='gray', alpha=0.4, zorder=10, linewidth=0)
    # Mark in red area x < 0 to indicate past
    gnt.axvspan(0, time_units_from_midnight, color='red', alpha=0.3, zorder=5)

    # Colormap
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(production_schedule)))
    rectangle_height = 7

    # Plot each job
    for p, prod in production_schedule.products:
        color = colors[p % len(production_schedule)]

        for c in range(max_cycles[p]):
            if c in prod.machine.keys():
                machine_position = 10 * prod.machine[c] + 5
                # Cycle patch
                edgecolor = 'red' if isinstance(prod, RunningProduct) and c == 0 else 'black'
                linewidth = 0.8 if isinstance(prod, RunningProduct) and c == 0 else 0.25
                gnt.broken_barh([(prod.setup_beg[c], prod.cycle_end[c] - prod.setup_beg[c])], (machine_position - rectangle_height / 2, rectangle_height), facecolors=(color,), edgecolors=(edgecolor,), linewidth=linewidth)
                gnt.text((prod.setup_beg[c] + (prod.cycle_end[c] - prod.setup_beg[c]) / 2), machine_position, f'JOB {chr(p+65)}{c}, V[{prod.velocity[c]}]', color='black', ha='center', va='center', fontsize=8)
                # Setup patch
                gnt.add_patch(patches.Rectangle((prod.setup_beg[c], machine_position - rectangle_height / 2), prod.setup_end[c]-prod.setup_beg[c], rectangle_height, linewidth=0.8, edgecolor='red', facecolor='none', hatch='///', linestyle='-', alpha=0.75))
                # Load / Unload patches
                for l in range(prod.num_levate[c]):
                    gnt.add_patch(patches.Rectangle((prod.load_beg[c,l], machine_position - rectangle_height / 2), (prod.load_end[c,l] - prod.load_beg[c,l]), rectangle_height, linewidth=0.8, edgecolor='black', facecolor='none', hatch='///', linestyle='-', alpha=0.75))
                    gnt.add_patch(patches.Rectangle((prod.unload_beg[c,l], machine_position - rectangle_height / 2), (prod.unload_end[c,l] - prod.unload_beg[c,l]), rectangle_height, linewidth=0.8, edgecolor='black', facecolor='none', hatch='///', linestyle='-', alpha=0.75))

    # X-axis appearence
    gnt.set_xlabel('Time (hours)')
    gnt.set_xlim(0, horizon)
    gnt.set_xticks(np.linspace(0, horizon, 10))        
    # Y-axis appearence
    gnt.set_ylabel('Machine ID')
    gnt.set_ylim(0, num_machines * 10)
    gnt.set_yticks([10 * i + 5 for i in range(num_machines)])
    gnt.set_yticklabels([f'[{i}]' for i in range(num_machines)])
    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='red', hatch='///', label='Job Start'),
        mpatches.Patch(facecolor='none', edgecolor='black', hatch='///', label='Levata Start'),
        mpatches.Patch(facecolor='gray', alpha=0.4, label='Non-working hours'),
        mpatches.Patch(facecolor='red', alpha=0.3, label='past')
    ]
    gnt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.125), fontsize=8)
    # Show
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

