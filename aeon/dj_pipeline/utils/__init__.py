def _make_maintenance_filter(position_df, maintenance_filter, start, stop):
    """Make a boolean filter for eliminating maintenance period"""

    maintenance_filter += (position_df.index >= start) & (position_df.index <= stop)
    return maintenance_filter
