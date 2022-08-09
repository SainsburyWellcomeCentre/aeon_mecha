import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns

from aeon.dj_pipeline import acquisition, analysis, lab
from aeon.dj_pipeline.analysis.visit import Visit, VisitEnd
from aeon.dj_pipeline.analysis.visit_analysis import VisitSummary, VisitTimeDistribution

# pio.renderers.default = 'png'
# pio.orca.config.executable = '~/.conda/envs/aeon_env/bin/orca'
# pio.orca.config.use_xvfb = True
# pio.orca.config.save()


def plot_reward_rate_differences(subject_keys):
    """
    Plotting the reward rate differences between food patches (Patch 2 - Patch 1)
    for all sessions from all subjects specified in "subject_keys"
    Example usage:
    ```
    subject_keys = (acquisition.Experiment.Subject & 'experiment_name = "exp0.1-r0"').fetch('KEY')

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
        for subj_name, sess_start in zip(subj_names, sess_starts)
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
        labels=dict(color="Reward Rate<br>Patch2-Patch1"),
    )
    fig.update_layout(
        xaxis_title="Time (min)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def plot_wheel_travelled_distance(session_keys):
    """
    Plotting the wheel travelled distance for different patches
        for all sessions specified in "session_keys"
    Example usage:
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
            distance_travelled_df.subject, distance_travelled_df.in_arena_start
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
    """plot results from VisitSummary per visit

    Args:
        visit_key (dict) : Key from the VisitSummary table
        attr (str): Name of the attribute to plot (e.g., 'pellet_count', 'wheel_distance_travelled', 'total_distance_travelled')
        per_food_patch (bool, optional): Separately plot results from different food patches. Defaults to False.

    Returns:
        fig: Figure object

    Examples:
        >>> fig = plot_visit_daily_summary(visit_key, attr='pellet_count', per_food_patch=True)
        >>> fig = plot_visit_daily_summary(visit_key, attr='wheel_distance_travelled', per_food_patch=True)
        >>> fig = plot_visit_daily_summary(visit_key, attr='total_distance_travelled')
    """

    subject, visit_start = (
        visit_key["subject"],
        visit_key["visit_start"],
    )

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
            ((VisitSummary & visit_key)).fetch(format="frame").reset_index()
        )
        if not attr.startswith("total"):
            attr = "total_" + attr

    visit_per_day_df["subject"] = "_".join([subject, visit_start.strftime("%m%d")])
    visit_per_day_df["day"] = (
        visit_per_day_df["visit_date"] - visit_per_day_df["visit_date"].min()
    )
    visit_per_day_df["day"] = visit_per_day_df["day"].dt.days

    fig = px.line(
        visit_per_day_df,
        x="day",
        y=attr,
        color=color,
        markers=True,
        labels={attr: attr.replace("_", " ")},
        hover_name="visit_date",
        hover_data=["visit_date"],
        width=700,
        height=400,
        template="simple_white",
        title=visit_per_day_df["subject"][0],
    )
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(
        legend_title="", hovermode="x", yaxis_tickformat="digits", yaxis_range=[0, None]
    )

    return fig


def plot_foraging_bouts(
    visit_key,
    wheel_dist_crit=None,
    min_bout_duration=None,
    using_aeon_io=False,
):
    """plot the number of foraging bouts per visit

    Args:
        visit_key (dict) : Key from the VisitTimeDistribution table
        wheel_dist_crit (int) : Minimum wheel distance travelled (in cm)
        min_bout_duration (int) : Minimum foraging bout duration (in seconds)
        using_aeon_io (bool) : Use aeon api to calculate wheel distance. Otherwise use datajoint tables. Defaults to False.

    Returns:
        fig: Figure object

    Examples:
        >>> fig = plot_foraging_bouts(visit_key, wheel_dist_crit=1, min_bout_duration=1)
    """

    subject, visit_start = (
        visit_key["subject"],
        visit_key["visit_start"],
    )

    visit_per_day_df = (
        (
            (VisitTimeDistribution.FoodPatch & visit_key)
            * acquisition.ExperimentFoodPatch.proj("food_patch_description")
        )
        .fetch(format="frame")
        .reset_index()
    )

    visit_per_day_df["subject"] = "_".join([subject, visit_start.strftime("%m%d")])
    visit_per_day_df["day"] = (
        visit_per_day_df["visit_date"] - visit_per_day_df["visit_date"].min()
    )
    visit_per_day_df["day"] = visit_per_day_df["day"].dt.days
    visit_per_day_df["foraging_bouts"] = visit_per_day_df.apply(
        _get_foraging_bouts,
        args=(wheel_dist_crit, min_bout_duration, using_aeon_io),
        axis=1,
    )

    fig = px.line(
        visit_per_day_df,
        x="day",
        y="foraging_bouts",
        color="food_patch_description",
        markers=True,
        labels={"foraging_bouts": "foraging_bouts".replace("_", " ")},
        hover_name="visit_date",
        hover_data=["visit_date"],
        width=700,
        height=400,
        template="simple_white",
        title=visit_per_day_df["subject"][0],
    )
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(
        legend_title="", hovermode="x", yaxis_tickformat="digits", yaxis_range=[0, None]
    )

    return fig


def _get_foraging_bouts(
    visit_per_day_row,
    wheel_dist_crit=None,
    min_bout_duration=None,
    using_aeon_io=False,
):
    """A function that calculates the number of foraging bouts. Works on this table query

            (VisitTimeDistribution.FoodPatch & visit_key)
            * acquisition.ExperimentFoodPatch.proj("food_patch_description")

        This will iterate over this table entries and store results in a new column ('foraging_bouts')

    Args:
        visit_per_day_row (pd.DataFrame): A single row of the pandas dataframe

    Returns:
        nb_bouts (int): Number of foraging bouts
    """

    # Get number of foraging bouts
    nb_bouts = 0

    in_patch = visit_per_day_row["in_patch"]
    if np.size(in_patch) == 0:  # no food patch position timestamps
        return nb_bouts

    change_ind = (
        np.where((np.diff(in_patch) / 1e6) > np.timedelta64(20))[0] + 1
    )  # timestamp index where state changes

    if np.size(change_ind) == 0:  # one contiguous block

        wheel_start, wheel_end = in_patch[0], in_patch[-1]
        ts_duration = (wheel_end - wheel_start) / np.timedelta64(1, "s")  # in seconds
        if ts_duration < min_bout_duration:
            return nb_bouts

        wheel_data = acquisition.FoodPatchWheel.get_wheel_data(
            experiment_name=visit_per_day_row["experiment_name"],
            start=wheel_start,
            end=wheel_end,
            patch_name=visit_per_day_row["food_patch_description"],
            using_aeon_io=using_aeon_io,
        )
        if wheel_data.distance_travelled[-1] > wheel_dist_crit:
            return nb_bouts + 1
        else:
            return nb_bouts

    # fetch contiguous timestamp blocks
    for i in range(len(change_ind) + 1):
        if i == 0:
            ts_array = in_patch[: change_ind[i]]
        elif i == len(change_ind):
            ts_array = in_patch[change_ind[i - 1] :]
        else:
            ts_array = in_patch[change_ind[i - 1] : change_ind[i]]

        ts_duration = (ts_array[-1] - ts_array[0]) / np.timedelta64(
            1, "s"
        )  # in seconds
        if ts_duration < min_bout_duration:
            continue

        wheel_start, wheel_end = ts_array[0], ts_array[-1]
        if wheel_start > wheel_end:  # skip if timestamps were misaligned
            continue

        wheel_data = acquisition.FoodPatchWheel.get_wheel_data(
            experiment_name=visit_per_day_row["experiment_name"],
            start=wheel_start,
            end=wheel_end,
            patch_name=visit_per_day_row["food_patch_description"],
            using_aeon_io=using_aeon_io,
        )

        if wheel_data.distance_travelled[-1] > wheel_dist_crit:
            nb_bouts += 1

    return nb_bouts
