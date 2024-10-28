"""Utility functions for plotting visit data."""

import datajoint as dj
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

from aeon.dj_pipeline import acquisition, analysis, lab
from aeon.dj_pipeline.analysis.visit import VisitEnd
from aeon.dj_pipeline.analysis.visit_analysis import (
    VisitForagingBout,
    VisitSummary,
    VisitTimeDistribution,
    filter_out_maintenance_periods,
    get_maintenance_periods,
)

# pio.renderers.default = 'png'
# pio.orca.config.executable = '~/.conda/envs/aeon_env/bin/orca'
# pio.orca.config.use_xvfb = True
# pio.orca.config.save()


def plot_reward_rate_differences(subject_keys):
    """Plotting the reward rate differences between food patches
    (Patch 2 - Patch 1) for all sessions from all subjects specified in "subject_keys".

    Examples:
    ```
    subject_keys =
    (acquisition.Experiment.Subject & 'experiment_name = "exp0.1-r0"').fetch('KEY')

    fig = plot_reward_rate_differences(subject_keys)
    ```
    """
    subj_names, sess_starts, rate_timestamps, rate_diffs = (
        analysis.InArenaRewardRate & subject_keys
    ).fetch(
        "subject", "in_arena_start", "pellet_rate_timestamps", "patch2_patch1_rate_diff"
    )

    nSessions = len(sess_starts)
    longest_rateDiff = np.max([len(t) for t in rate_timestamps])

    max_session_idx = np.argmax([len(t) for t in rate_timestamps])
    max_session_elapsed_times = (
        rate_timestamps[max_session_idx] - rate_timestamps[max_session_idx][0]
    )
    x_labels = [t.total_seconds() / 60 for t in max_session_elapsed_times]

    y_labels = [
        f'{subj_name}_{sess_start.strftime("%m/%d/%Y")}'
        for subj_name, sess_start in zip(subj_names, sess_starts, strict=False)
    ]

    rateDiffs_matrix = np.full((nSessions, longest_rateDiff), np.nan)
    for row_index, rate_diff in enumerate(rate_diffs):
        rateDiffs_matrix[row_index, : len(rate_diff)] = rate_diff

    absZmax = np.nanmax(np.absolute(rateDiffs_matrix))

    fig = px.imshow(
        img=rateDiffs_matrix,
        x=x_labels,
        y=y_labels,
        zmin=-absZmax,
        zmax=absZmax,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        labels={"color": "Reward Rate<br>Patch2-Patch1"},
    )
    fig.update_layout(
        xaxis_title="Time (min)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def plot_wheel_travelled_distance(session_keys):
    """Plotting the wheel travelled distance for different patches
    for all sessions specified in "session_keys".

    Examples:
    ```
    session_keys = (acquisition.Session & acquisition.SessionEnd
     & {'experiment_name': 'exp0.1-r0', 'subject': 'BAA-1099794'}).fetch('KEY')

    fig = plot_wheel_travelled_distance(session_keys)
    ```
    """
    distance_travelled_query = (
        analysis.InArenaSummary.FoodPatch
        * acquisition.ExperimentFoodPatch.proj("food_patch_description")
        & session_keys
    )

    distance_travelled_df = (
        distance_travelled_query.proj(
            "food_patch_description", "wheel_distance_travelled"
        )
        .fetch(format="frame")
        .reset_index()
    )

    distance_travelled_df["in_arena"] = [
        f'{subj_name}_{sess_start.strftime("%m/%d/%Y")}'
        for subj_name, sess_start in zip(
            distance_travelled_df.subject, distance_travelled_df.in_arena_start, strict=False
        )
    ]

    distance_travelled_df.rename(
        columns={
            "food_patch_description": "Patch",
            "wheel_distance_travelled": "Travelled Distance (m)",
        },
        inplace=True,
    )

    title = "|".join((acquisition.Experiment.Subject & session_keys).fetch("subject"))
    fig = px.bar(
        distance_travelled_df,
        x="in_arena",
        y="Travelled Distance (m)",
        color="Patch",
        title=title,
    )
    fig.update_xaxes(tickangle=45)

    return fig


def plot_average_time_distribution(session_keys):
    """Plotting the average time spent in different regions."""
    subject_list, arena_location_list, avg_time_spent_list = [], [], []

    # Time spent in arena and corridor
    subjects, avg_in_corridor, avg_in_arena = (
        (acquisition.Experiment.Subject & session_keys)
        .aggr(
            analysis.InArenaTimeDistribution,
            avg_in_corridor="AVG(time_fraction_in_corridor)",
            avg_in_arena="AVG(time_fraction_in_arena)",
        )
        .fetch("subject", "avg_in_corridor", "avg_in_arena")
    )
    subject_list.extend(subjects)
    arena_location_list.extend(["corridor"] * len(avg_in_corridor))
    avg_time_spent_list.extend(avg_in_corridor)
    subject_list.extend(subjects)
    arena_location_list.extend(["arena"] * len(avg_in_arena))
    avg_time_spent_list.extend(avg_in_arena)

    # Time spent in food-patch
    subjects, patches, avg_in_patch = (
        (
            dj.U("experiment_name", "subject", "food_patch_description")
            & acquisition.Experiment.Subject * acquisition.ExperimentFoodPatch
            & session_keys
        )
        .aggr(
            analysis.InArenaTimeDistribution.FoodPatch
            * acquisition.ExperimentFoodPatch,
            avg_in_patch="AVG(time_fraction_in_patch)",
        )
        .fetch("subject", "food_patch_description", "avg_in_patch")
    )
    subject_list.extend(subjects)
    arena_location_list.extend(patches)
    avg_time_spent_list.extend(avg_in_patch)

    # Time spent in nest
    subjects, nests, avg_in_nest = (
        (acquisition.Experiment.Subject * lab.ArenaNest & session_keys)
        .aggr(
            analysis.InArenaTimeDistribution.Nest,
            avg_in_nest="AVG(time_fraction_in_nest)",
        )
        .fetch("subject", "nest", "avg_in_nest")
    )
    subject_list.extend(subjects)
    arena_location_list.extend([f"Nest{n}" for n in nests])
    avg_time_spent_list.extend(avg_in_nest)

    # Average time distribution
    avg_time_df = pd.DataFrame(
        {
            "Subject": subject_list,
            "Location": arena_location_list,
            "Time Fraction": avg_time_spent_list,
        }
    )

    title = "|".join((acquisition.Experiment & session_keys).fetch("experiment_name"))
    fig = px.bar(
        avg_time_df,
        x="Subject",
        y="Time Fraction",
        color="Location",
        barmode="group",
        title=title,
    )
    fig.update_xaxes(tickangle=45)

    return fig


def plot_visit_daily_summary(
    visit_key,
    attr,
    per_food_patch=False,
):
    """Plot results from VisitSummary per visit.

    Args:
        visit_key (dict) : Key from the VisitSummary table
        attr (str): Name of the attribute to plot (e.g., 'pellet_count',
                    'wheel_distance_travelled', 'total_distance_travelled')
        per_food_patch (bool, optional): Separately plot results from
                    different food patches. Defaults to False.

    Returns:
        fig: Figure object

    Examples:
        >>> fig = plot_visit_daily_summary(visit_key, attr='pellet_count',
        per_food_patch=True)
        >>> fig = plot_visit_daily_summary(visit_key,
        attr='wheel_distance_travelled', per_food_patch=True)
        >>> fig = plot_visit_daily_summary(visit_key,
        attr='total_distance_travelled')
    """
    per_food_patch = not attr.startswith("total")
    color = "food_patch_description" if per_food_patch else None

    if per_food_patch:  # split by food patch
        visit_per_day_df = (
            (
                (VisitSummary.FoodPatch & visit_key)
                * acquisition.ExperimentFoodPatch.proj("food_patch_description")
            )
            .fetch(format="frame")
            .reset_index()
        )
    else:
        visit_per_day_df = (
            (VisitSummary & visit_key).fetch(format="frame").reset_index()
        )
        if not attr.startswith("total"):
            attr = "total_" + attr

    visit_per_day_df["day"] = (
        visit_per_day_df["visit_date"] - visit_per_day_df["visit_date"].min()
    )
    visit_per_day_df["day"] = visit_per_day_df["day"].dt.days

    fig = px.bar(
        visit_per_day_df,
        x="visit_date",
        y=attr,
        color=color,
        labels={
            attr: attr.replace("_", " "),
            "visit_date": "date",
        },
        hover_name="visit_date",
        hover_data=["day"],
        width=700,
        height=400,
        template="simple_white",
        title=visit_key["subject"] + "<br><i>" + attr.replace("_", " ") + " (daily)",
    )

    fig.update_layout(
        legend={
            "title": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.98,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x",
        yaxis_tickformat="digits",
        yaxis_range=[0, None],
    )

    return fig


def plot_foraging_bouts_count(
    visit_key,
    freq="D",
    per_food_patch=False,
    min_bout_duration=0,
    min_pellet_count=0,
    min_wheel_dist=0,
):
    """Plot the number of foraging bouts per visit.

    Args:
        visit_key (dict): Key from the Visit table
        freq (str): Frequency level at which the visit time
                    distribution is plotted. Corresponds to pandas freq.
        per_food_patch (bool, optional): Separately plot results from
                    different food patches. Defaults to False.
        min_bout_duration (int): Minimum foraging bout duration (in seconds)
        min_pellet_count (int): Minimum number of pellets
        min_wheel_dist (int): Minimum wheel distance travelled (in cm)

    Returns:
        fig: Figure object

    Examples:
        >>> fig = plot_foraging_bouts_count(visit_key, freq="D",
        per_food_patch=True, min_bout_duration=1, min_wheel_dist=1)
    """
    # Get all foraging bouts for the visit
    foraging_bouts = (
        (
            dj.U(
                "bout_start",
                "bout_end",
                "bout_duration",
                "food_patch_description",
                "pellet_count",
                "wheel_distance_travelled",
            )
            & (VisitForagingBout * acquisition.ExperimentFoodPatch & visit_key)
        )
        .fetch(order_by="bout_start", format="frame")
        .reset_index()
    )

    # Apply filter
    foraging_bouts = foraging_bouts[
        (foraging_bouts["bout_duration"] >= min_bout_duration)
        & (foraging_bouts["pellet_count"] >= min_pellet_count)
        & (foraging_bouts["wheel_distance_travelled"] >= min_wheel_dist)
    ]

    group_by_attrs = (
        [foraging_bouts["bout_start"].dt.floor(freq), "food_patch_description"]
        if per_food_patch
        else [foraging_bouts["bout_start"].dt.floor("D")]
    )

    foraging_bouts_count = (
        foraging_bouts.groupby(group_by_attrs).size().reset_index(name="count")
    )

    visit_start = (VisitEnd & visit_key).fetch1("visit_start")
    foraging_bouts_count["day"] = (
        foraging_bouts_count["bout_start"].dt.date - visit_start.date()
    ).dt.days

    fig = px.bar(
        foraging_bouts_count,
        x="bout_start",
        y="count",
        color="food_patch_description" if per_food_patch else None,
        labels={
            "bout_start": "date" if freq == "D" else "time",
        },
        hover_data=["day"],
        width=700,
        height=400,
        template="simple_white",
        title=visit_key["subject"]
        + "<br><i>Foraging bouts: count (freq='"
        + freq
        + "')",
    )

    fig.update_layout(
        legend={
            "title": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.98,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x",
        yaxis_tickformat="digits",
        yaxis_range=[0, None],
    )

    return fig


def plot_foraging_bouts_distribution(
    visit_key,
    attr,
    per_food_patch=False,
    min_bout_duration=0,
    min_pellet_count=0,
    min_wheel_dist=0,
):
    """Plot distribution of foraging bout attributes.

    Args:
        visit_key (dict): Key from the Visit table
        attr (str): Options include: pellet_count, bout_duration,
                    wheel_distance_travelled
        per_food_patch (bool, optional): Separately plot results from
                    different food patches. Defaults to False.
        min_bout_duration (int): Minimum foraging bout duration (in seconds)
        min_pellet_count (int): Minimum number of pellets
        min_wheel_dist (int): Minimum wheel distance travelled (in cm)

    Returns:
        fig: Figure object

    Examples:
        >>> fig = plot_foraging_bouts_distribution(visit_key, "pellet_count", True, 0, 3, 0)
        >>> fig = plot_foraging_bouts_distribution(visit_key, "wheel_distance_travelled")
        >>> fig = plot_foraging_bouts_distribution(visit_key, "bout_duration")
    """
    # Get all foraging bouts for the visit
    foraging_bouts = (
        (
            dj.U(
                "bout_start",
                "bout_end",
                "bout_duration",
                "food_patch_description",
                "pellet_count",
                "wheel_distance_travelled",
            )
            & (VisitForagingBout * acquisition.ExperimentFoodPatch & visit_key)
        )
        .fetch(order_by="bout_start", format="frame")
        .reset_index()
    )

    # Apply filter
    foraging_bouts = foraging_bouts[
        (foraging_bouts["bout_duration"] >= min_bout_duration)
        & (foraging_bouts["pellet_count"] >= min_pellet_count)
        & (foraging_bouts["wheel_distance_travelled"] >= min_wheel_dist)
    ]

    fig = go.Figure()
    if per_food_patch:
        patch_names = (acquisition.ExperimentFoodPatch & visit_key).fetch(
            "food_patch_description"
        )
        for patch in patch_names:
            bouts = foraging_bouts[foraging_bouts["food_patch_description"] == patch]
            fig.add_trace(
                go.Violin(
                    x=bouts["bout_start"].dt.date,
                    y=bouts[attr],
                    legendgroup=patch,
                    scalegroup=patch,
                    name=patch,
                    side="negative" if patch == "Patch1" else "positive",
                )
            )
    else:
        fig.add_trace(
            go.Violin(
                x=foraging_bouts["bout_start"].dt.date,
                y=foraging_bouts[attr],
            )
        )

    fig.update_traces(
        box_visible=True,
        meanline_visible=True,
    )

    fig.update_layout(
        title_text=visit_key["subject"]
        + "<br><i>Foraging bouts: "
        + attr.replace("_", " "),
        xaxis_title="date",
        yaxis_title=attr.replace("_", " "),
        violingap=0,
        violingroupgap=0,
        violinmode="overlay",
        width=700,
        height=400,
        template="simple_white",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig


def plot_visit_time_distribution(visit_key, freq="D"):
    """Plot fraction of time spent in each region per visit.

    Args:
        visit_key (dict): Key from the Visit table
        freq (str): Frequency level at which the visit time distribution
                    is plotted. Corresponds to pandas freq.

    Returns:
        fig: Figure object

    Examples:
        >>> fig = plot_visit_time_distribution(visit_key, freq="D")
        >>> fig = plot_visit_time_distribution(visit_key, freq="H")
    """
    region = _get_region_data(visit_key)

    # Compute time spent per region
    time_spent = (
        region.groupby([region.index.floor(freq), "region"])
        .size()
        .reset_index(name="count")
    )
    time_spent["time_fraction"] = time_spent["count"] / time_spent.groupby(
        "timestamps"
    )["count"].transform("sum")
    time_spent["day"] = (
        time_spent["timestamps"] - time_spent["timestamps"].min()
    ).dt.days

    fig = px.bar(
        time_spent,
        x="timestamps",
        y="time_fraction",
        color="region",
        hover_data=["day"],
        labels={
            "time_fraction": "time fraction",
            "timestamps": "date" if freq == "D" else "time",
        },
        title=visit_key["subject"]
        + "<br><i>Fraction of time spent in each region (freq='"
        + freq
        + "')",
        width=700,
        height=400,
        template="simple_white",
    )

    fig.update_layout(
        hovermode="x",
        yaxis_tickformat="digits",
        yaxis_range=[0, None],
        legend={
            "title": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.98,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig


def _get_region_data(visit_key, attrs=None):
    """Retrieve region data from VisitTimeDistribution tables.

    Args:
        visit_key (dict): Key from the Visit table
        attrs (list, optional): List of column names (in VisitTimeDistribution tables) to retrieve.
        Defaults is None, which will create a new list with the desired default values inside the function.

    Returns:
        region (pd.DataFrame): Timestamped region info
    """
    if attrs is None:
        attrs = ["in_nest", "in_arena", "in_corridor", "in_patch"]

    visit_start, visit_end = (VisitEnd & visit_key).fetch1("visit_start", "visit_end")
    region = pd.DataFrame()

    # Get region timestamps
    for attr in attrs:
        if attr == "in_nest":  # Nest
            in_nest = np.concatenate(
                (VisitTimeDistribution.Nest & visit_key).fetch(
                    attr, order_by="visit_date"
                )
            )
            region = pd.concat(
                [
                    region,
                    pd.DataFrame(
                        [[attr.replace("in_", "")]] * len(in_nest),
                        columns=["region"],
                        index=in_nest,
                    ),
                ]
            )
        elif attr == "in_patch":  # Food patch
            # Find all patches
            patches = np.unique(
                (
                    VisitTimeDistribution.FoodPatch * acquisition.ExperimentFoodPatch
                    & visit_key
                ).fetch("food_patch_description")
            )
            for patch in patches:
                in_patch = np.concatenate(
                    (
                        VisitTimeDistribution.FoodPatch
                        * acquisition.ExperimentFoodPatch
                        & visit_key
                        & f"food_patch_description = '{patch}'"
                    ).fetch("in_patch", order_by="visit_date")
                )
                region = pd.concat(
                    [
                        region,
                        pd.DataFrame(
                            [[patch.lower()]] * len(in_patch),
                            columns=["region"],
                            index=in_patch,
                        ),
                    ]
                )
        else:  # corridor, arena
            in_other = np.concatenate(
                (VisitTimeDistribution & visit_key).fetch(attr, order_by="visit_date")
            )
            region = pd.concat(
                [
                    region,
                    pd.DataFrame(
                        [[attr.replace("in_", "")]] * len(in_other),
                        columns=["region"],
                        index=in_other,
                    ),
                ]
            )
    region = region.sort_index().rename_axis("timestamps")

    # Exclude data during maintenance
    maintenance_period = get_maintenance_periods(
        visit_key["experiment_name"], visit_start, visit_end
    )
    region = filter_out_maintenance_periods(
        region, maintenance_period, visit_end, dropna=True
    )

    return region


def plot_weight_patch_data(
    visit_key, freq="H", smooth_weight=True, min_weight=0, max_weight=35
):
    """Plot subject weight and patch data (pellet trigger count) per visit.

    Args:
        visit_key (dict): Key from the Visit table
        freq (str): Frequency level at which patch data is plotted. Corresponds to pandas freq.
        smooth_weight (bool, optional): Apply savgol filter to subject weight. Defaults to True
        min_weight (bool, optional): Lower bound of subject weight. Defaults to 0
        max_weight (bool, optional): Upper bound of subject weight. Defaults to 35

    Returns:
        fig: Figure object

    Examples:
        >>> fig = plot_weight_patch_data(visit_key, freq="H", smooth_weight=True)
        >>> fig = plot_weight_patch_data(visit_key, freq="D")
    """
    subject_weight = _get_filtered_subject_weight(
        visit_key, smooth_weight, min_weight, max_weight
    )

    # Count pellet trigger per patch per day/hour/...
    patch = _get_patch_data(visit_key)
    patch_summary = (
        patch.groupby(
            [
                # group by freq and patch
                patch.index.to_series().dt.floor(freq),
                "food_patch_description",
            ]
        )
        .count()
        .unstack(fill_value=0)  # fill none count with 0s
        .stack()
        .reset_index()
    )

    # Get patch names
    patch_names = patch["food_patch_description"].unique()
    patch_names.sort()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add trace for each patch
    for p in patch_names:
        fig.add_trace(
            go.Bar(
                x=patch_summary[patch_summary["food_patch_description"] == p][
                    "event_time"
                ],
                y=patch_summary[patch_summary["food_patch_description"] == p][
                    "event_type"
                ],
                name=p,
            ),
            secondary_y=False,
        )

    # Add subject weight trace
    fig.add_trace(
        go.Scatter(
            x=subject_weight.index,
            y=subject_weight["weight_subject"],
            mode="lines+markers",
            opacity=0.5,
            name="subject weight",
            marker={
                "size": 3,
            },
            legendrank=1,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="stack",
        hovermode="x",
        title_text=visit_key["subject"]
        + "<br><i>Weight and pellet count (freq='"
        + freq
        + "')",
        xaxis_title="date" if freq == "D" else "time",
        yaxis={"title": "pellet count"},
        yaxis2={"title": "weight"},
        width=700,
        height=400,
        template="simple_white",
        legend={
            "title": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.98,
            "xanchor": "right",
            "x": 1,
            "traceorder": "normal",
        },
    )

    return fig


def _get_filtered_subject_weight(
    visit_key, smooth_weight=True, min_weight=0, max_weight=35
):
    """Retrieve subject weight from WeightMeasurementFiltered table.

    Args:
        visit_key (dict): Key from the Visit table
        smooth_weight (bool, optional): Apply savgol filter to subject weight. Defaults to True
        min_weight (bool, optional): Lower bound of subject weight. Defaults to 0
        max_weight (bool, optional): Upper bound of subject weight. Defaults to 35

    Returns:
        subject_weight (pd.DataFrame): Timestamped weight data
    """
    visit_start, visit_end = (VisitEnd & visit_key).fetch1("visit_start", "visit_end")

    chunk_keys = (
        acquisition.Chunk
        & f'chunk_start BETWEEN "{pd.Timestamp(visit_start).floor("H")}" AND "{visit_end}"'
    ).fetch("KEY")

    # Create subject_weight dataframe
    subject_weight = (
        pd.DataFrame(
            (
                dj.U("weight_subject_timestamps", "weight_subject")
                & (acquisition.WeightMeasurementFiltered & chunk_keys)
            ).fetch(order_by="chunk_start")
        )
        .explode(["weight_subject_timestamps", "weight_subject"])
        .set_index("weight_subject_timestamps")
        .dropna()
    )

    # Return empty dataframe if no weight data
    if subject_weight.empty:
        return subject_weight

    subject_weight = subject_weight.loc[visit_start:visit_end]

    # Exclude data during maintenance
    maintenance_period = get_maintenance_periods(
        visit_key["experiment_name"], visit_start, visit_end
    )
    subject_weight = filter_out_maintenance_periods(
        subject_weight, maintenance_period, visit_end, dropna=True
    )

    # Drop rows where weight is out of specified range
    subject_weight = subject_weight.drop(
        subject_weight[
            (subject_weight["weight_subject"] < min_weight)
            | (subject_weight["weight_subject"] > max_weight)
        ].index
    )

    # Downsample data to every minute
    subject_weight = subject_weight.resample("1T").mean().dropna()

    if smooth_weight:
        subject_weight["weight_subject"] = savgol_filter(
            subject_weight["weight_subject"], 10, 3
        )

    return subject_weight


def _get_patch_data(visit_key):
    """Retrieve all patch (pellet trigger) data from FoodPatchEvent table.

    Args:
        visit_key (dict): Key from the Visit table

    Returns:
        patch (pd.DataFrame): Timestamped pellet trigger events
    """
    visit_start, visit_end = (VisitEnd & visit_key).fetch1("visit_start", "visit_end")

    # Get pellet trigger dataframe for all patches
    patch = (
        (
            dj.U("event_time", "event_type", "food_patch_description")
            & (
                acquisition.FoodPatchEvent
                * acquisition.EventType
                * acquisition.ExperimentFoodPatch
                & f'event_time BETWEEN "{visit_start}" AND "{visit_end}"'
                & 'event_type = "TriggerPellet"'
            )
        )
        .fetch(order_by="event_time", format="frame")
        .reset_index()
        .set_index("event_time")
    )

    # TODO: handle repeat attempts (pellet delivery trigger and beam break)

    # Exclude data during maintenance
    maintenance_period = get_maintenance_periods(
        visit_key["experiment_name"], visit_start, visit_end
    )
    patch = filter_out_maintenance_periods(
        patch, maintenance_period, visit_end, dropna=True
    )

    return patch
