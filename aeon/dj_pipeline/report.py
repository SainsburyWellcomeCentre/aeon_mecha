import os
import datajoint as dj
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import re
import datetime
import json

from aeon.preprocess import api as aeon_api
from aeon.util import plotting as aeon_plotting

from . import lab, acquisition, tracking, analysis
from . import get_schema_name


schema = dj.schema(get_schema_name('report'))
os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"


"""
    DataJoint schema dedicated for tables containing figures
"""


@schema
class SessionSummaryPlot(dj.Computed):
    definition = """
    -> analysis.SessionTimeDistribution
    -> analysis.SessionSummary
    ---
    summary_plot_png: filepath@djstore
    """

    key_source = acquisition.Session & analysis.SessionTimeDistribution & analysis.SessionSummary

    color_code = {'Patch1': 'b', 'Patch2': 'r', 'arena': 'g', 'corridor': 'gray', 'nest': 'k'}

    def make(self, key):
        raw_data_dir = acquisition.Experiment.get_data_directory(key)

        session_start, session_end = (acquisition.Session * acquisition.SessionEnd & key).fetch1(
            'session_start', 'session_end')

        # subject's position data in the time_slices
        position = tracking.SubjectPosition.get_session_position(key)

        position_minutes_elapsed = (position.index - session_start).total_seconds() / 60

        # figure
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(21, 6)
        rate_ax = fig.add_subplot(gs[:10, :4])
        distance_ax = fig.add_subplot(gs[10:20, :4])
        ethogram_ax = fig.add_subplot(gs[20, :4])
        position_ax = fig.add_subplot(gs[10:, 4:])
        pellet_ax = fig.add_subplot(gs[:10, 4])
        time_dist_ax = fig.add_subplot(gs[:10, 5:])

        # position plot
        non_nan = np.logical_and(~np.isnan(position.x), ~np.isnan(position.y))
        aeon_plotting.heatmap(position[non_nan], 50, ax=position_ax, bins=500, alpha=0.5)

        # event rate plots
        session_food_patches = (
                acquisition.Session
                * acquisition.ExperimentFoodPatch.join(
            acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_start < IFNULL(food_patch_remove_time, "2200-01-01")').proj(
            'food_patch_description')

        for food_patch_key in session_food_patches.fetch(as_dict=True):
            pellet_times_df = (acquisition.FoodPatchEvent * acquisition.EventType
                               & food_patch_key
                               & 'event_type = "TriggerPellet"'
                               & f'event_time BETWEEN "{session_start}" AND "{session_end}"').proj(
                'event_time').fetch(format='frame', order_by='event_time').reset_index()
            pellet_times_df.set_index('event_time', inplace=True)
            aeon_plotting.rateplot(pellet_times_df, window='600s',
                                   frequency=500, ax=rate_ax, smooth='120s',
                                   start=session_start, end=session_end,
                                   color=self.color_code[food_patch_key['food_patch_description']])

            # wheel data
            wheel_data = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                              device=food_patch_key['food_patch_description'],
                                              start=pd.Timestamp(session_start),
                                              end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(wheel_data.angle).values

            minutes_elapsed = (wheel_data.index - session_start).total_seconds() / 60
            distance_ax.plot(minutes_elapsed, wheel_distance_travelled,
                             color=self.color_code[food_patch_key['food_patch_description']])

        # ethogram
        in_arena, in_corridor, arena_time, corridor_time = (
                analysis.SessionTimeDistribution & key).fetch1(
            'in_arena', 'in_corridor', 'time_fraction_in_arena', 'time_fraction_in_corridor')
        nest_keys, in_nests, nests_times = (analysis.SessionTimeDistribution.Nest & key).fetch(
            'KEY', 'in_nest', 'time_fraction_in_nest')
        patch_names, in_patches, patches_times = (
                analysis.SessionTimeDistribution.FoodPatch
                * acquisition.ExperimentFoodPatch & key).fetch(
            'food_patch_description', 'in_patch', 'time_fraction_in_patch')

        ethogram_ax.plot(position_minutes_elapsed[in_arena],
                         np.full_like(position_minutes_elapsed[in_arena], 0),
                         '.', color=self.color_code['arena'], markersize=0.5, alpha=0.6,
                         label=f'Times in arena')
        ethogram_ax.plot(position_minutes_elapsed[in_corridor],
                         np.full_like(position_minutes_elapsed[in_corridor], 1),
                         '.', color=self.color_code['corridor'], markersize=0.5, alpha=0.6,
                         label=f'Times in corridor')
        for in_nest in in_nests:
            ethogram_ax.plot(position_minutes_elapsed[in_nest],
                             np.full_like(position_minutes_elapsed[in_nest], 2),
                             '.', color=self.color_code['nest'], markersize=0.5, alpha=0.6,
                             label=f'Times in nest')
        for patch_idx, (patch_name, in_patch) in enumerate(zip(patch_names, in_patches)):
            ethogram_ax.plot(position_minutes_elapsed[in_patch],
                             np.full_like(position_minutes_elapsed[in_patch], (patch_idx + 3)),
                             '.', color=self.color_code[patch_name], markersize=0.5, alpha=0.6,
                             label=f'Times in {patch_name}')

        # pellet
        patch_names, patches_pellet = (
                analysis.SessionSummary.FoodPatch
                * acquisition.ExperimentFoodPatch & key).fetch(
            'food_patch_description', 'pellet_count')
        pellet_ax.bar(range(len(patches_pellet)), patches_pellet,
                      color=[self.color_code[n] for n in patch_names])

        # time distribution
        time_fractions = [arena_time, corridor_time]
        colors = [self.color_code['arena'], self.color_code['corridor']]
        time_fractions.extend(nests_times)
        colors.extend([self.color_code['nest'] for _ in nests_times])
        time_fractions.extend(patches_times)
        colors.extend([self.color_code[n] for n in patch_names])
        time_dist_ax.bar(range(len(time_fractions)), time_fractions, color=colors)

        # cosmetic
        rate_ax.legend()
        rate_ax.sharex(distance_ax)
        fig.subplots_adjust(hspace=0.1)
        rate_ax.set_ylabel('pellets / min')
        rate_ax.set_title('foraging rate (bin size = 10 min)')
        distance_ax.set_ylabel('distance travelled (m)')
        ethogram_ax.set_xlabel('time (min)')
        aeon_plotting.set_ymargin(distance_ax, 0.2, 0.1)
        for ax in (rate_ax, distance_ax, pellet_ax, time_dist_ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)

        ethogram_ax.spines['top'].set_visible(False)
        ethogram_ax.spines['right'].set_visible(False)
        ethogram_ax.spines['left'].set_visible(False)
        ethogram_ax.tick_params(left=False, labelleft=False)
        aeon_plotting.set_ymargin(ethogram_ax, 0.4, 0)

        position_ax.set_aspect('equal')
        position_ax.set_axis_off()

        pellet_ax.set_ylabel('pellets delivered')
        time_dist_ax.set_ylabel('Fraction of session duration')

        # ---- Save fig and insert ----
        session_dir = _make_session_path(key)
        fig_dict = _save_figs((fig,), ('summary_plot_png',),
                              save_dir=session_dir, prefix=session_dir.name)

        self.insert1({**key, **fig_dict})


# ---- Dynamically updated tables for plots ----


@schema
class SubjectRewardRateDifference(dj.Computed):
    definition = """
    -> acquisition.Experiment.Subject
    ---
    session_count: int
    reward_rate_difference_plotly: longblob  # dictionary storing the plotly object (from fig.to_plotly_json())
    """

    key_source = acquisition.Experiment.Subject & analysis.SessionRewardRate

    def make(self, key):
        from aeon.dj_pipeline.api.plotting import plot_reward_rate_differences

        fig = plot_reward_rate_differences(key)

        fig_json = json.loads(fig.to_json())

        self.insert1({**key, 'session_count': len(analysis.SessionRewardRate & key),
                      'reward_rate_difference_plotly': fig_json})

    @classmethod
    def delete_outdated_entries(cls):
        """
        Each entry in this table correspond to one subject. However the plot is capturing
            data for all sessions.
        Hence a dynamic update routine is needed to recompute the plot as new sessions
            become available
        """
        outdated_entries = (cls * (acquisition.Experiment.Subject.aggr(
            analysis.SessionRewardRate, current_session_count='count(session_start)'))
                            & 'session_count != current_session_count')
        with dj.config(safemode=False):
            (cls & outdated_entries.fetch('KEY')).delete()


@schema
class SubjectWheelTravelledDistance(dj.Computed):
    definition = """
    -> acquisition.Experiment.Subject
    ---
    session_count: int
    wheel_travelled_distance_plotly: longblob  # dictionary storing the plotly object (from fig.to_plotly_json())
    """

    key_source = acquisition.Experiment.Subject & analysis.SessionSummary

    def make(self, key):
        from aeon.dj_pipeline.api.plotting import plot_wheel_travelled_distance
        session_keys = (analysis.SessionSummary & key).fetch('KEY')

        fig = plot_wheel_travelled_distance(session_keys)

        fig_json = json.loads(fig.to_json())

        self.insert1({**key, 'session_count': len(session_keys),
                      'wheel_travelled_distance_plotly': fig_json})

    @classmethod
    def delete_outdated_entries(cls):
        """
        Each entry in this table correspond to one subject. However the plot is capturing
            data for all sessions.
        Hence a dynamic update routine is needed to recompute the plot as new sessions
            become available
        """
        outdated_entries = (cls * (acquisition.Experiment.Subject.aggr(
            analysis.SessionSummary, current_session_count='count(session_start)'))
                            & 'session_count != current_session_count')
        with dj.config(safemode=False):
            (cls & outdated_entries.fetch('KEY')).delete()


@schema
class ExperimentTimeDistribution(dj.Computed):
    definition = """
    -> acquisition.Experiment
    ---
    session_count: int
    time_distribution_plotly: longblob  # dictionary storing the plotly object (from fig.to_plotly_json())
    """

    def make(self, key):
        from aeon.dj_pipeline.api.plotting import plot_average_time_distribution
        session_keys = (analysis.SessionTimeDistribution & key).fetch('KEY')

        fig = plot_average_time_distribution(session_keys)

        fig_json = json.loads(fig.to_json())

        self.insert1({**key, 'session_count': len(session_keys),
                      'time_distribution_plotly': fig_json})

    @classmethod
    def delete_outdated_entries(cls):
        """
        Each entry in this table correspond to one subject. However the plot is capturing
            data for all sessions.
        Hence a dynamic update routine is needed to recompute the plot as new sessions
            become available
        """
        outdated_entries = (cls * (acquisition.Experiment.aggr(
            analysis.SessionTimeDistribution, current_session_count='count(session_start)'))
                            & 'session_count != current_session_count')
        with dj.config(safemode=False):
            (cls & outdated_entries.fetch('KEY')).delete()


def delete_outdated_plot_entries():
    for tbl in (SubjectRewardRateDifference,
                SubjectWheelTravelledDistance,
                ExperimentTimeDistribution):
        tbl.delete_outdated_entries()


# ---------- HELPER FUNCTIONS --------------


def _make_session_path(session_key):
    store_stage = pathlib.Path(dj.config['stores']['djstore']['stage'])
    experiment_name, subject, session_start = (acquisition.Session & session_key).fetch1(
        'experiment_name', 'subject', 'session_start')
    session_dir = store_stage / experiment_name / subject / session_start.strftime('%y%m%d_%H%M%S_%f')
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _save_figs(figs, fig_names, save_dir, prefix, extension='.png'):
    fig_dict = {}
    for fig, figname in zip(figs, fig_names):
        fig_fp = save_dir / (prefix + figname + extension)
        fig.tight_layout()
        fig.savefig(fig_fp, dpi=300)
        fig_dict[figname] = fig_fp.as_posix()

    return fig_dict