"""DataJoint schema dedicated for tables containing figures. """

import datetime
import json
import os
import pathlib

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeon.analysis import plotting as analysis_plotting
from aeon.dj_pipeline.analysis.visit import Visit, VisitEnd
from aeon.dj_pipeline.analysis.visit_analysis import *

from . import acquisition, analysis, get_schema_name

# schema = dj.schema(get_schema_name("report"))
schema = dj.schema()
os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"


@schema
class InArenaSummaryPlot(dj.Computed):
    definition = """
    -> analysis.InArenaTimeDistribution
    -> analysis.InArenaSummary
    ---
    summary_plot_png: attach
    """

    key_source = (
        analysis.InArena & analysis.InArenaTimeDistribution & analysis.InArenaSummary
    )

    color_code = {
        "Patch1": "b",
        "Patch2": "r",
        "arena": "g",
        "corridor": "gray",
        "nest": "k",
    }

    def make(self, key):
        in_arena_start, in_arena_end = (
            analysis.InArena * analysis.InArenaEnd & key
        ).fetch1("in_arena_start", "in_arena_end")

        # subject's position data in the time_slices
        position = analysis.InArenaSubjectPosition.get_position(key)
        position.rename(columns={"position_x": "x", "position_y": "y"}, inplace=True)

        position_minutes_elapsed = (
            position.index - in_arena_start
        ).total_seconds() / 60

        # figure
        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(22, 5)
        threshold_ax = fig.add_subplot(gs[:3, :3])
        rate_ax = fig.add_subplot(gs[5:13, :3])
        distance_ax = fig.add_subplot(gs[14:20, :3])
        ethogram_ax = fig.add_subplot(gs[20, :3])
        position_ax = fig.add_subplot(gs[13:, 3:])
        pellet_ax = fig.add_subplot(gs[2:12, 3])
        time_dist_ax = fig.add_subplot(gs[2:12, 4:])

        # position plot
        non_nan = np.logical_and(~np.isnan(position.x), ~np.isnan(position.y))
        analysis_plotting.heatmap(
            position[non_nan], 50, ax=position_ax, bins=500, alpha=0.5
        )

        # event rate plots
        in_arena_food_patches = (
            analysis.InArena
            * acquisition.ExperimentFoodPatch.join(
                acquisition.ExperimentFoodPatch.RemovalTime, left=True
            )
            & key
            & "in_arena_start >= food_patch_install_time"
            & 'in_arena_start < IFNULL(food_patch_remove_time, "2200-01-01")'
        ).proj("food_patch_description")

        for food_patch_key in in_arena_food_patches.fetch(as_dict=True):
            pellet_times_df = (
                (
                    acquisition.FoodPatchEvent * acquisition.EventType
                    & food_patch_key
                    & 'event_type = "TriggerPellet"'
                    & f'event_time BETWEEN "{in_arena_start}" AND "{in_arena_end}"'
                )
                .proj("event_time")
                .fetch(format="frame", order_by="event_time")
                .reset_index()
            )
            pellet_times_df.set_index("event_time", inplace=True)
            analysis_plotting.rateplot(
                pellet_times_df,
                window="600s",
                frequency=500,
                ax=rate_ax,
                smooth="120s",
                start=in_arena_start,
                end=in_arena_end,
                color=self.color_code[food_patch_key["food_patch_description"]],
                label=food_patch_key["food_patch_serial_number"],
            )

            # wheel data
            wheel_data = acquisition.FoodPatchWheel.get_wheel_data(
                experiment_name=key["experiment_name"],
                start=pd.Timestamp(in_arena_start),
                end=pd.Timestamp(in_arena_end),
                patch_name=food_patch_key["food_patch_description"],
                using_aeon_io=True,
            )

            minutes_elapsed = (wheel_data.index - in_arena_start).total_seconds() / 60
            distance_ax.plot(
                minutes_elapsed,
                wheel_data.distance_travelled.values,
                color=self.color_code[food_patch_key["food_patch_description"]],
            )

            # plot wheel threshold
            wheel_time, wheel_threshold = (
                acquisition.WheelState.Time
                & food_patch_key
                & f'state_timestamp between "{in_arena_start}" and "{in_arena_end}"'
            ).fetch("state_timestamp", "threshold")
            wheel_time -= in_arena_start
            wheel_time /= datetime.timedelta(minutes=1)

            wheel_time = np.append(wheel_time, position_minutes_elapsed[-1])

            for i in range(len(wheel_time) - 1):
                threshold_ax.hlines(
                    y=wheel_threshold[i],
                    xmin=wheel_time[i],
                    xmax=wheel_time[i + 1],
                    linewidth=2,
                    color=self.color_code[food_patch_key["food_patch_description"]],
                    alpha=0.3,
                )
            threshold_change_ind = np.where(
                wheel_threshold[:-1] != wheel_threshold[1:]
            )[0]
            threshold_ax.vlines(
                wheel_time[threshold_change_ind + 1],
                ymin=wheel_threshold[threshold_change_ind],
                ymax=wheel_threshold[threshold_change_ind + 1],
                linewidth=1,
                linestyle="dashed",
                color=self.color_code[food_patch_key["food_patch_description"]],
                alpha=0.4,
            )

        # ethogram
        in_arena, in_corridor, arena_time, corridor_time = (
            analysis.InArenaTimeDistribution & key
        ).fetch1(
            "in_arena",
            "in_corridor",
            "time_fraction_in_arena",
            "time_fraction_in_corridor",
        )
        nest_keys, in_nests, nests_times = (
            analysis.InArenaTimeDistribution.Nest & key
        ).fetch("KEY", "in_nest", "time_fraction_in_nest")
        patch_names, in_patches, patches_times = (
            analysis.InArenaTimeDistribution.FoodPatch * acquisition.ExperimentFoodPatch
            & key
        ).fetch("food_patch_description", "in_patch", "time_fraction_in_patch")

        ethogram_ax.plot(
            position_minutes_elapsed[in_arena],
            np.full_like(position_minutes_elapsed[in_arena], 0),
            ".",
            color=self.color_code["arena"],
            markersize=0.5,
            alpha=0.6,
            label="arena",
        )
        ethogram_ax.plot(
            position_minutes_elapsed[in_corridor],
            np.full_like(position_minutes_elapsed[in_corridor], 1),
            ".",
            color=self.color_code["corridor"],
            markersize=0.5,
            alpha=0.6,
            label="corridor",
        )
        for in_nest in in_nests:
            ethogram_ax.plot(
                position_minutes_elapsed[in_nest],
                np.full_like(position_minutes_elapsed[in_nest], 2),
                ".",
                color=self.color_code["nest"],
                markersize=0.5,
                alpha=0.6,
                label="nest",
            )
        for patch_idx, (patch_name, in_patch) in enumerate(
            zip(patch_names, in_patches)
        ):
            ethogram_ax.plot(
                position_minutes_elapsed[in_patch],
                np.full_like(position_minutes_elapsed[in_patch], (patch_idx + 3)),
                ".",
                color=self.color_code[patch_name],
                markersize=0.5,
                alpha=0.6,
                label=f"{patch_name}",
            )

        # pellet
        patch_names, patches_pellet = (
            analysis.InArenaSummary.FoodPatch * acquisition.ExperimentFoodPatch & key
        ).fetch("food_patch_description", "pellet_count")
        pellet_ax.bar(
            range(len(patches_pellet)),
            patches_pellet,
            color=[self.color_code[n] for n in patch_names],
        )

        # time distribution
        time_fractions = [arena_time, corridor_time]
        colors = [
            self.color_code["arena"],
            self.color_code["corridor"],
        ]
        time_fractions.extend(nests_times)
        colors.extend([self.color_code["nest"] for _ in nests_times])
        time_fractions.extend(patches_times)
        colors.extend([self.color_code[n] for n in patch_names])
        time_dist_ax.bar(range(len(time_fractions)), time_fractions, color=colors)

        # cosmetic
        rate_ax.legend()
        rate_ax.sharex(distance_ax)
        fig.subplots_adjust(hspace=0.1)
        rate_ax.set_ylabel("pellets / min")
        rate_ax.set_title("foraging rate (bin size = 10 min)")
        distance_ax.set_ylabel("distance travelled (m)")
        threshold_ax.set_ylabel("threshold")
        threshold_ax.set_ylim(
            [threshold_ax.get_ylim()[0] - 100, threshold_ax.get_ylim()[1] + 100]
        )
        ethogram_ax.set_xlabel("time (min)")
        analysis_plotting.set_ymargin(distance_ax, 0.2, 0.1)
        for ax in (rate_ax, distance_ax, pellet_ax, time_dist_ax, threshold_ax):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)

        ethogram_ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 2.5),
            prop={"size": 12},
            markerscale=40,
        )
        ethogram_ax.spines["top"].set_visible(False)
        ethogram_ax.spines["right"].set_visible(False)
        ethogram_ax.spines["left"].set_visible(False)
        ethogram_ax.tick_params(left=False, labelleft=False)
        analysis_plotting.set_ymargin(ethogram_ax, 0.4, 0)

        position_ax.set_aspect("equal")
        position_ax.set_axis_off()

        pellet_ax.set_ylabel("pellets delivered")
        time_dist_ax.set_ylabel("Fraction of session duration")

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
        """Each entry in this table correspond to one subject. However, the plot is capturing data for all sessions.Hence a dynamic update routine is needed to recompute the plot as new sessions become available."""
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
        """Each entry in this table correspond to one subject. However the plot is capturing data for all sessions. Hence a dynamic update routine is needed to recompute the plot as new sessions become available."""
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
        """Each entry in this table correspond to one subject. However the plot is capturing data for all sessions. Hence a dynamic update routine is needed to recompute the plot as new sessions become available."""
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


@schema
class VisitDailySummaryPlot(dj.Computed):
    definition = """
    -> Visit
    ---
    pellet_count_plotly:             longblob  # Dictionary storing the plotly object (from fig.to_plotly_json())
    wheel_distance_travelled_plotly: longblob
    total_distance_travelled_plotly: longblob
    weight_patch_plotly:                longblob
    foraging_bouts_plotly:              longblob
    foraging_bouts_pellet_count_plotly: longblob
    foraging_bouts_duration_plotly:     longblob
    region_time_fraction_daily_plotly:  longblob
    region_time_fraction_hourly_plotly: longblob
    """

    key_source = (
        Visit
        & analysis.VisitSummary
        & (VisitEnd & "visit_duration > 24")
        & "experiment_name= 'exp0.2-r0'"
    )

    def make(self, key):
        from aeon.dj_pipeline.utils.plotting import (
            plot_foraging_bouts_count,
            plot_foraging_bouts_distribution,
            plot_visit_daily_summary,
            plot_visit_time_distribution,
            plot_weight_patch_data,
        )

        # bout criteria
        min_wheel_dist = 1  # in cm (minimum wheel distance travelled)
        min_bout_duration = 1  # in seconds (minimum foraging bout duration)
        min_pellet_count = 3  # minimum number of pellets

        fig = plot_visit_daily_summary(
            key,
            attr="pellet_count",
            per_food_patch=True,
        )
        fig_pellet = json.loads(fig.to_json())

        fig = plot_visit_daily_summary(
            key,
            attr="wheel_distance_travelled",
            per_food_patch=True,
        )
        fig_wheel_dist = json.loads(fig.to_json())

        fig = plot_visit_daily_summary(
            key,
            attr="total_distance_travelled",
        )
        fig_total_dist = json.loads(fig.to_json())

        fig = plot_weight_patch_data(
            key,
        )
        fig_weight_patch = json.loads(fig.to_json())

        fig = plot_foraging_bouts_count(
            key,
            per_food_patch=True,
            min_bout_duration=min_bout_duration,
            min_pellet_count=min_pellet_count,
            min_wheel_dist=min_wheel_dist,
        )
        fig_foraging_bouts = json.loads(fig.to_json())

        fig = plot_foraging_bouts_distribution(
            key,
            "pellet_count",
            per_food_patch=True,
            min_bout_duration=min_bout_duration,
            min_pellet_count=min_pellet_count,
            min_wheel_dist=min_wheel_dist,
        )
        fig_foraging_bouts_pellet_count = json.loads(fig.to_json())

        fig = plot_foraging_bouts_distribution(
            key,
            "bout_duration",
            per_food_patch=False,
            min_bout_duration=min_bout_duration,
            min_pellet_count=min_pellet_count,
            min_wheel_dist=min_wheel_dist,
        )
        fig_foraging_bouts_duration = json.loads(fig.to_json())

        fig = plot_visit_time_distribution(
            key,
            freq="D",
        )
        fig_region_time_fraction_daily = json.loads(fig.to_json())

        fig = plot_visit_time_distribution(
            key,
            freq="H",
        )
        fig_region_time_fraction_hourly = json.loads(fig.to_json())

        self.insert1(
            {
                **key,
                "pellet_count_plotly": fig_pellet,
                "wheel_distance_travelled_plotly": fig_wheel_dist,
                "total_distance_travelled_plotly": fig_total_dist,
                "weight_patch_plotly": fig_weight_patch,
                "foraging_bouts_plotly": fig_foraging_bouts,
                "foraging_bouts_pellet_count_plotly": fig_foraging_bouts_pellet_count,
                "foraging_bouts_duration_plotly": fig_foraging_bouts_duration,
                "region_time_fraction_daily_plotly": fig_region_time_fraction_daily,
                "region_time_fraction_hourly_plotly": fig_region_time_fraction_hourly,
            }
        )


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
