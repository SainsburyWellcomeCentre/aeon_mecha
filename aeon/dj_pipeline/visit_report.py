import datetime
import json
import os
import pathlib
import re

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeon.analysis import plotting as analysis_plotting
from aeon.dj_pipeline.analysis.visit import Visit, VisitEnd
from aeon.dj_pipeline.analysis.visit_analysis import *

from . import acquisition, analysis, get_schema_name

# schema = dj.schema(get_schema_name("report"))
schema = dj.schema(get_schema_name("u_jaeronga_test"))
os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

experiment_name = "exp0.2-r0"
MIN_VISIT_DURATION = 24  # in hours (minimum duration of visit for analysis)
WHEEL_DIST_CRIT = 1  # in cm (minimum wheel distance travelled)
MIN_BOUT_DURATION = 1  # in seconds (minimum foraging bout duration)


"""
DataJoint schema dedicated for tables containing figures
"""


@schema
class VisitSummaryPlot(dj.Computed):
    definition = """
    -> VisitSummary
    ---
    pellet_count_png:             attach
    wheel_distance_travelled_png: attach
    total_distance_travelled_png: attach
    """

    key_source = dj.U("experiment_name", "subject", "visit_start", "visit_end") & (
        VisitEnd
        & f'experiment_name="{experiment_name}"'
        & f"visit_duration > {MIN_VISIT_DURATION}"
    )

    def make(self, key):
        from aeon.dj_pipeline.utils.plotting import plot_summary_per_visit

        fig = plot_summary_per_visit(
            key,
            attr="pellet_count",
            per_food_patch=True,
        )

        fig = plot_summary_per_visit(
            key,
            attr="wheel_distance_travelled",
            per_food_patch=True,
        )

        fig = plot_summary_per_visit(
            key,
            attr="total_distance_travelled",
            per_food_patch=False,
        )

        # ---- Save fig and insert ----
        save_dir = _make_path(key)
        fig_dict = _save_figs(
            (fig,), ("summary_plot_png",), save_dir=save_dir, prefix=save_dir.name
        )

        self.insert1({**key, **fig_dict})


# ---- Dynamically updated tables for plots ----


@schema
class SubjectRewardRateDifference(dj.Computed):
    definition = """
    -> acquisition.Experiment.Subject
    ---
    in_arena_count: int
    reward_rate_difference_plotly: longblob  # dictionary storing the plotly object (from fig.to_plotly_json())
    """

    key_source = acquisition.Experiment.Subject & analysis.InArenaRewardRate

    def make(self, key):
        from aeon.dj_pipeline.utils.plotting import plot_reward_rate_differences

        fig = plot_reward_rate_differences(key)

        fig_json = json.loads(fig.to_json())

        self.insert1(
            {
                **key,
                "in_arena_count": len(analysis.InArenaRewardRate & key),
                "reward_rate_difference_plotly": fig_json,
            }
        )

    @classmethod
    def delete_outdated_entries(cls):
        """
        Each entry in this table correspond to one subject. However, the plot is capturing
            data for all sessions.
        Hence a dynamic update routine is needed to recompute the plot as new sessions
            become available
        """
        outdated_entries = (
            cls
            * (
                acquisition.Experiment.Subject.aggr(
                    analysis.InArenaRewardRate,
                    current_in_arena_count="count(in_arena_start)",
                )
            )
            & "in_arena_count != current_in_arena_count"
        )
        with dj.config(safemode=False):
            (cls & outdated_entries.fetch("KEY")).delete()


@schema
class SubjectWheelTravelledDistance(dj.Computed):
    definition = """
    -> acquisition.Experiment.Subject
    ---
    in_arena_count: int
    wheel_travelled_distance_plotly: longblob  # dictionary storing the plotly object (from fig.to_plotly_json())
    """

    key_source = acquisition.Experiment.Subject & analysis.InArenaSummary

    def make(self, key):
        from aeon.dj_pipeline.utils.plotting import plot_wheel_travelled_distance

        in_arena_keys = (analysis.InArenaSummary & key).fetch("KEY")

        fig = plot_wheel_travelled_distance(in_arena_keys)

        fig_json = json.loads(fig.to_json())

        self.insert1(
            {
                **key,
                "in_arena_count": len(in_arena_keys),
                "wheel_travelled_distance_plotly": fig_json,
            }
        )

    @classmethod
    def delete_outdated_entries(cls):
        """
        Each entry in this table correspond to one subject. However the plot is capturing
            data for all sessions.
        Hence a dynamic update routine is needed to recompute the plot as new sessions
            become available
        """
        outdated_entries = (
            cls
            * (
                acquisition.Experiment.Subject.aggr(
                    analysis.InArenaSummary,
                    current_in_arena_count="count(in_arena_start)",
                )
            )
            & "in_arena_count != current_in_arena_count"
        )
        with dj.config(safemode=False):
            (cls & outdated_entries.fetch("KEY")).delete()


@schema
class ExperimentTimeDistribution(dj.Computed):
    definition = """
    -> acquisition.Experiment
    ---
    in_arena_count: int
    time_distribution_plotly: longblob  # dictionary storing the plotly object (from fig.to_plotly_json())
    """

    def make(self, key):
        from aeon.dj_pipeline.utils.plotting import plot_average_time_distribution

        in_arena_keys = (analysis.InArenaTimeDistribution & key).fetch("KEY")

        fig = plot_average_time_distribution(in_arena_keys)

        fig_json = json.loads(fig.to_json())

        self.insert1(
            {
                **key,
                "in_arena_count": len(in_arena_keys),
                "time_distribution_plotly": fig_json,
            }
        )

    @classmethod
    def delete_outdated_entries(cls):
        """
        Each entry in this table correspond to one subject. However the plot is capturing
            data for all sessions.
        Hence a dynamic update routine is needed to recompute the plot as new sessions
            become available
        """
        outdated_entries = (
            cls
            * (
                acquisition.Experiment.aggr(
                    analysis.InArenaTimeDistribution,
                    current_in_arena_count="count(in_arena_start)",
                )
            )
            & "in_arena_count != current_in_arena_count"
        )
        with dj.config(safemode=False):
            (cls & outdated_entries.fetch("KEY")).delete()


def delete_outdated_plot_entries():
    for tbl in (
        SubjectRewardRateDifference,
        SubjectWheelTravelledDistance,
        ExperimentTimeDistribution,
    ):
        tbl.delete_outdated_entries()


# ---------- HELPER FUNCTIONS --------------


def _make_path(in_arena_key):
    store_stage = pathlib.Path(dj.config["stores"]["djstore"]["stage"])
    experiment_name, subject, in_arena_start = (analysis.InArena & in_arena_key).fetch1(
        "experiment_name", "subject", "in_arena_start"
    )
    output_dir = (
        store_stage
        / experiment_name
        / subject
        / in_arena_start.strftime("%y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_figs(figs, fig_names, save_dir, prefix, extension=".png"):
    fig_dict = {}
    for fig, figname in zip(figs, fig_names):
        fig_fp = save_dir / (prefix + "_" + figname + extension)
        fig.tight_layout()
        fig.savefig(fig_fp, dpi=300)
        fig_dict[figname] = fig_fp.as_posix()

    return fig_dict
