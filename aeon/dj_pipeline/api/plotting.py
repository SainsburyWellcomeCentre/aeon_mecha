import datajoint as dj
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

from aeon.dj_pipeline import lab, acquisition, analysis


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
        "subject", "session_start", "pellet_rate_timestamps", "patch2_patch1_rate_diff"
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

    distance_travelled_df["session"] = [
        f'{subj_name}_{sess_start.strftime("%m/%d/%Y")}'
        for subj_name, sess_start in zip(
            distance_travelled_df.subject, distance_travelled_df.session_start
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
        x="session",
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
