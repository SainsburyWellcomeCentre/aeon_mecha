import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go
import plotly.subplots
import plotly.express as px

from aeon.dj_pipeline import acquisition, analysis


def plot_reward_rate_differences(subject_keys, save_figure=False):
    """
    Plotting the reward rate differences between food patches (Patch 2 - Patch 1)
    for all sessions from all subjects specified in "subject_keys"
    Example usage:
    ```
    subject_keys = acquisition.Experiment.Subject.fetch('KEY')

    fig = plot_reward_rate_differences(subject_keys)
    ```
    """
    subj_names, sess_starts, rate_timestamps, rate_diffs = (analysis.SessionRewardRate
                                                         & subject_keys).fetch(
        'subject', 'session_start', 'pellet_rate_timestamps', 'patch2_patch1_rate_diff')

    nSessions = len(sess_starts)
    longest_rateDiff = np.max([len(t) for t in rate_timestamps])

    max_session_idx = np.argmax([len(t) for t in rate_timestamps])
    max_session_elapsed_times = rate_timestamps[max_session_idx] - rate_timestamps[max_session_idx][0]
    x_labels = [t.total_seconds() / 60 for t in max_session_elapsed_times]

    y_labels = [f'{subj_name}_{sess_start.strftime("%m/%d/%Y")}' for subj_name, sess_start in
                zip(subj_names, sess_starts)]

    rateDiffs_matrix = np.full((nSessions, longest_rateDiff), np.nan)
    for row_index, rate_diff in enumerate(rate_diffs):
        rateDiffs_matrix[row_index, :len(rate_diff)] = rate_diff

    absZmax = np.nanmax(np.absolute(rateDiffs_matrix))
    fig = px.imshow(img=rateDiffs_matrix, x=x_labels, y=y_labels,
                    zmin=-absZmax, zmax=absZmax, aspect="auto",
                    color_continuous_scale='RdBu_r',
                    labels=dict(color="Reward Rate<br>Patch2-Patch1"))
    fig.update_layout(
        xaxis_title="Time (min)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    if save_figure:
        fig_filename_pattern = 'reward_rate_diffsPatch1MinusPatch2.{:s}'
        fig.write_image(fig_filename_pattern.format("png"))
        fig.write_html(fig_filename_pattern.format("html"))
    fig.show()
    