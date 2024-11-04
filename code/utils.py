
from datetime import datetime

def get_time_intervals (horizon_days, time_units_in_a_day, start_shift, end_shift) :
    # Compute horizon
    horizon = horizon_days * time_units_in_a_day

    # Current time data
    now = datetime.now()
    days_in_week = 7
    minutes_from_midnight = now.hour * 60 + now.minute # Current time in minutes from midnight
    current_day_of_week = now.weekday() # 0: Monday, 1: Tuesday, ..., 6: Sunday

    # Time steps convertion
    time_units_from_midnight = minutes_from_midnight
    if time_units_in_a_day == 24 :
        time_units_from_midnight =  minutes_from_midnight // 60
    elif time_units_in_a_day == 48 :
        time_units_from_midnight = minutes_from_midnight // 30
    elif time_units_in_a_day == 96 :
        time_units_from_midnight = minutes_from_midnight // 15

    # Define worktime intervals according to current daytime
    worktime_intervals = []
    # first interval (scheduling startin in or out workday)
    if (time_units_from_midnight < end_shift) and (time_units_from_midnight > start_shift) and (current_day_of_week % days_in_week not in [6]):
        worktime_intervals.append((time_units_from_midnight,end_shift))
    # Handle remaining cases
    for day in range(1,horizon_days) :
        if (day + current_day_of_week) % days_in_week not in [6]:
            workday_start = day*time_units_in_a_day + start_shift
            workday_end = day*time_units_in_a_day + end_shift
            worktime_intervals.append((workday_start, workday_end))

    # Define prohibited intervals as complementar set of worktime
    prohibited_intervals = []
    # first interval (scheduling startin in or out workday)
    if time_units_from_midnight < worktime_intervals[0][0] :
        prohibited_intervals.append((time_units_from_midnight, worktime_intervals[0][0]))
    # handle remaining cases
    for i in range(len(worktime_intervals)-1):
        _, gap_start = worktime_intervals[i]
        gap_end, _ = worktime_intervals[i+1]
        prohibited_intervals.append((gap_start, gap_end))
    # Append last interval (from last worktime end to horizon)
    prohibited_intervals.append((worktime_intervals[-1][1], horizon))

    # List of gap size relative to day index (0 being today)
    gap_at_day = []
    gap_idx = 0 if prohibited_intervals[0][0] > 0 else 1
    for g in range(horizon_days):
        time_step = g * time_units_in_a_day + start_shift
        # Check if need to advance in the prohibited intervals
        while gap_idx < len(prohibited_intervals) and time_step >= prohibited_intervals[gap_idx][1]:
            gap_idx += 1
        
        gap_start, gap_end = prohibited_intervals[gap_idx]
        if gap_start <= time_step <= gap_end:
            gap_at_day.append(-1)  # Day inside a prohibited interval
        else :
            gap_at_day.append(gap_end-gap_start)

    return worktime_intervals, prohibited_intervals, gap_at_day, time_units_from_midnight, worktime_intervals[0][0]
