import matplotlib.pyplot as plt
import plotly.graph_objects as go
from data_init import RunningProduct

def plot_gantt_chart(
    production_schedule,
    max_cycles,
    num_machines,
    horizon,
    prohibited_intervals,
    time_units_from_midnight,
):
    """
    Plot an interactive Gantt chart for the given job schedule with enhanced readability and navigation.
    """
    # Define start time (hour 0) for the x-axis
    start_time = 0  # Assuming time is measured in hours

    # Identify used machines
    used_machines = set()
    for p, prod in production_schedule.products:
        for c in range(max_cycles[prod.id]):
            if c in prod.machine.keys():
                used_machines.add(prod.machine[c])
    used_machines = sorted(used_machines)
    machine_id_map = {machine: idx for idx, machine in enumerate(used_machines)}
    updated_num_machines = len(used_machines)

    # Initialize Plotly figure
    fig = go.Figure()

    # Add prohibited intervals (e.g., non-working hours) as shapes
    for interval in prohibited_intervals:
        x_start, x_end = interval
        fig.add_vrect(
            x0=x_start,
            x1=x_end,
            fillcolor="gray",
            opacity=0.4,
            layer="below",
            line_width=0,
            annotation_text="Non-working Hours",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="black",
            annotation_bgcolor="rgba(0,0,0,0)",
        )

    # Mark past time as a red shaded area
    fig.add_vrect(
        x0=start_time,
        x1=time_units_from_midnight,
        fillcolor="red",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="Past",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="black",
        annotation_bgcolor="rgba(0,0,0,0)",
    )

    # Colormap for different products
    cmap = plt.get_cmap("tab20")
    colors = [
        f"rgba{tuple(int(c * 255) for c in cmap(i % 20)[:3]) + (0.8,)}"
        for i in range(len(production_schedule.products))
    ]

    # Dictionaries to group operations by type for performance
    setup_traces = {}
    cycle_traces = {}
    load_traces = {}
    unload_traces = {}

    # Plot each job
    for idx, (p, prod) in enumerate(production_schedule.products):
        color = colors[idx]
        for c in range(max_cycles[prod.id]):
            if c in prod.machine.keys():
                machine = prod.machine[c]
                machine_pos = machine_id_map[machine]

                # Setup times
                setup_start = prod.setup_beg[c]  # Assuming in hours
                setup_end = prod.setup_end[c]
                setup_duration = setup_end - setup_start

                # Cycle times
                cycle_start = setup_end  # Assuming cycle starts after setup
                cycle_end = prod.cycle_end[c]
                cycle_duration = cycle_end - cycle_start

                # Add Production Cycle Bar
                cycle_key = f"Cycle_P{prod.id}_C{c}"
                if cycle_key not in cycle_traces:
                    cycle_traces[cycle_key] = {
                        "x": [],
                        "y": [],
                        "base": [],
                        "hovertext": [],
                        "color": color,
                        "line_color": (
                            "red"
                            if isinstance(prod, RunningProduct) and c == 0
                            else "black"
                        ),
                        "line_width": (
                            2 if isinstance(prod, RunningProduct) and c == 0 else 1
                        ),
                    }
                cycle_traces[cycle_key]["x"].append(cycle_duration)
                cycle_traces[cycle_key]["y"].append(machine_pos)
                cycle_traces[cycle_key]["base"].append(cycle_start)
                cycle_traces[cycle_key]["hovertext"].append(
                    f"Product ID: {prod.id}<br>"
                    f"Article: {prod.article}<br>"
                    f"Velocity: {prod.velocity[c]}<br>"
                    f"Cycle: {c}"
                )

                # Add Setup Bar
                setup_key = f"Setup_P{prod.id}_C{c}"
                if setup_key not in setup_traces:
                    setup_traces[setup_key] = {
                        "x": [],
                        "y": [],
                        "base": [],
                        "hovertext": [],
                        "color": "rgba(255,0,0,0)",  # Transparent fill
                        "line_color": "red",
                        "line_width": 2,
                    }
                setup_traces[setup_key]["x"].append(setup_duration)
                setup_traces[setup_key]["y"].append(machine_pos)
                setup_traces[setup_key]["base"].append(setup_start)
                setup_traces[setup_key]["hovertext"].append(
                    f"Job Setup for Product ID: {prod.id}, Cycle: {c}"
                )

                # Add Load/Unload Bars
                for l in range(prod.num_levate[c]):
                    # Load
                    load_key = f"Load_P{prod.id}_C{c}_L{l}"
                    if load_key not in load_traces:
                        load_traces[load_key] = {
                            "x": [],
                            "y": [],
                            "base": [],
                            "hovertext": [],
                            "color": "rgba(0, 128, 0, 0.8)",  # Green with opacity
                            "line_color": "green",
                            "line_width": 1.5,
                        }
                    load_start = prod.load_beg[c, l]
                    load_end = prod.load_end[c, l]
                    load_duration = load_end - load_start
                    load_traces[load_key]["x"].append(load_duration)
                    load_traces[load_key]["y"].append(machine_pos)
                    load_traces[load_key]["base"].append(load_start)
                    load_traces[load_key]["hovertext"].append(
                        f"Load for Product ID: {prod.id}, Cycle: {c}, Load: {l}"
                    )

                    # Unload
                    unload_key = f"Unload_P{prod.id}_C{c}_U{l}"
                    if unload_key not in unload_traces:
                        unload_traces[unload_key] = {
                            "x": [],
                            "y": [],
                            "base": [],
                            "hovertext": [],
                            "color": "rgba(255, 165, 0, 0.8)",  # Orange with opacity
                            "line_color": "orange",
                            "line_width": 1.5,
                        }
                    unload_start = prod.unload_beg[c, l]
                    unload_end = prod.unload_end[c, l]
                    unload_duration = unload_end - unload_start
                    unload_traces[unload_key]["x"].append(unload_duration)
                    unload_traces[unload_key]["y"].append(machine_pos)
                    unload_traces[unload_key]["base"].append(unload_start)
                    unload_traces[unload_key]["hovertext"].append(
                        f"Unload for Product ID: {prod.id}, Cycle: {c}, Unload: {l}"
                    )

                # Add Text Annotation
                # Using Scatter for annotations
                fig.add_trace(
                    go.Scatter(
                        x=[setup_start + setup_duration + cycle_duration / 2],
                        y=[machine_pos],
                        text=[f"JOB {prod.id} C {c} V[{prod.velocity[c]}]"],
                        mode="text",
                        textposition="middle center",
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

    # Add all Cycle Traces
    for trace_key, trace_data in cycle_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Production Cycle",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                showlegend=False,  # We'll add a custom legend later
            )
        )

    # Add all Setup Traces
    for trace_key, trace_data in setup_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Job Setup",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                width=0.4,
                opacity=0.75,
                showlegend=False,
            )
        )

    # Add all Load Traces
    for trace_key, trace_data in load_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Load",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                width=0.3,
                opacity=1.0,
                showlegend=False,
            )
        )

    # Add all Unload Traces
    for trace_key, trace_data in unload_traces.items():
        fig.add_trace(
            go.Bar(
                x=trace_data["x"],
                y=trace_data["y"],
                base=trace_data["base"],
                orientation="h",
                marker=dict(
                    color=trace_data["color"],
                    line=dict(
                        color=trace_data["line_color"], width=trace_data["line_width"]
                    ),
                ),
                name="Unload",
                hoverinfo="text",
                hovertext=trace_data["hovertext"],
                width=0.3,
                opacity=1.0,
                showlegend=False,
            )
        )

    # Update layout for better readability and navigation
    fig.update_layout(
        barmode="overlay",  # Overlay to stack bars correctly
        height=400
        + 40 * updated_num_machines,  # Dynamic height based on number of machines
        title="Production Schedule Gantt Chart",
        xaxis=dict(
            title="Time (hours)",
            type="linear",
            tickmode="linear",
            tick0=0,
            dtick=max(1, horizon // 10),  # Dynamic tick interval
            range=[start_time, horizon],
            showgrid=True,
            zeroline=False,
            rangeslider=dict(
                visible=True,
                thickness=0.1,
                range=[start_time, min(horizon, start_time + 50)],  # Initial view range
            ),
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=10, label="10h", step="hour", stepmode="backward"),
                        dict(count=20, label="20h", step="hour", stepmode="backward"),
                        dict(count=50, label="50h", step="hour", stepmode="backward"),
                        dict(step="all"),
                    ]
                ),
                x=0,
                y=-0.2,  # Positioning the selector below the chart
                xanchor="left",
                yanchor="top",
            ),
        ),
        yaxis=dict(
            title="Machine ID",
            tickvals=list(range(updated_num_machines)),
            ticktext=[f"[{m}]" for m in used_machines],
            autorange="reversed",  # To have the first machine at the top
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(
            l=100, r=40, t=80, b=150
        ),  # Increased left margin for machine labels
        plot_bgcolor="white",
    )

    # Add Legend Manually using Scatter (since individual traces have showlegend=False)
    legend_elements = [
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="rgba(255,0,0,0)", line=dict(color="red", width=2)),
            name="Job Setup",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(
                color="rgba(0, 128, 0, 0.8)", line=dict(color="green", width=1.5)
            ),
            name="Load",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(
                color="rgba(255, 165, 0, 0.8)", line=dict(color="orange", width=1.5)
            ),
            name="Unload",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="gray", opacity=0.4),
            name="Non-working Hours",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="red", opacity=0.3),
            name="Past",
            hoverinfo="none",
        ),
        go.Bar(
            x=[0],
            y=[0],
            marker=dict(color="blue"),
            name="Production Cycle",
            hoverinfo="none",
        ),
    ]

    for elem in legend_elements:
        fig.add_trace(elem)

    # Update layout to include the custom legend
    fig.update_layout(
        legend=dict(
            itemsizing="constant",
            title="Legend",
            traceorder="normal",
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1,
        )
    )

    # Final Touches: Ensure legend entries are unique and not duplicated
    fig.update_traces(showlegend=False)

    return fig

