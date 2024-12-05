import argparse
import os
from datetime import timedelta
from enum import IntFlag

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.time import Time
from lsst.summit.utils.efdUtils import getEfdData, getDayObsForTime
from lsst.summit.utils.tmaUtils import TMAEvent, TMAEventMaker
from lsst.ts.xml.tables.m1m3 import (
    FATable,
    HP_COUNT,
    FAOrientation,
)
from lsst.ts.m1m3.utils.force_actuator_forces import ForceActuatorForces
import warnings


HAS_EFD_CLIENT = True
try:
    from lsst_efd_client import EfdClient
except ImportError:
    EfdClient = None  # this is currently just for mypy
    HAS_EFD_CLIENT = False


class SlewState(IntFlag):
    EL_SLEW = 1  # 0b0001
    AZ_SLEW = 2  # 0b0010
    EL_POSITIVE = 4  # 0b0100
    AZ_POSITIVE = 8  # 0b1000


def compute_state(row):
    """
    Compute the state of a row based on the azimuth and elevation distances.

    Parameters
    ----------
    row : `pd.Series`
        A row of the DataFrame containing azimuth (`az_distance`) and
        elevation (`el_distance`) values.

    Returns
    -------
    state : `int`
        An integer representing the state of the row using bitwise flags.
    """
    state = 0
    if abs(row["el_distance"]) > 2:
        state |= SlewState.EL_SLEW
        if row["el_distance"] > 0:
            state |= SlewState.EL_POSITIVE

    if abs(row["az_distance"]) > 2:
        state |= SlewState.AZ_SLEW
        if row["az_distance"] > 0:
            state |= SlewState.AZ_POSITIVE
    return state


class M1M3Query:
    """
    Evaluate the M1M3 Inertia Compensation System's performance by calculating
    the minima, maxima and peak-to-peak values during a slew. In addition,
    calculates the mean, median and standard deviation when the slew has
    contant velocity or zero acceleration.

    Parameters
    ----------
    event : `lsst.summit.utils.tmaUtils.TMAEvent`
        Abtract representation of a slew event.
    efd_client : `EfdClient`
        Client to access the EFD.
    inner_pad : `float`, optional
        Time padding inside the stable time window of the slew.
    outer_pad : `float`, optional
        Time padding outside the slew time window.
    n_sigma : `float`, optional
        Number of standard deviations to use for the stable region.
    log : `logging.Logger`, optional
        Logger object to use for logging messages.
    """

    def __init__(
        self,
        event: TMAEvent,
        efd_client: EfdClient,
        outer_pad: float = 10.0,
    ):

        self.event = event
        self.outer_pad = outer_pad * u.second
        self.client = efd_client

        self.number_of_hardpoints = HP_COUNT
        self.measured_forces_topics = [
            f"measuredForce{i}" for i in range(self.number_of_hardpoints)
        ]
        self.measured_forces_topics += [f"f{i}" for i in "xyz"]
        self.measured_forces_topics += [f"m{i}" for i in "xyz"]

        self.force_actuator_topics = [
            f"primaryCylinderFollowingError{i}" for i in range(10, 156)
        ]
        self.force_actuator_topics += [
            f"secondaryCylinderFollowingError{i}" for i in range(10, 156)
        ]

    async def query_force_actuator_following_error_dataset(
        self,
    ) -> pd.DataFrame:
        evt = self.event
        pad = self.outer_pad
        faf = ForceActuatorForces(
            start=evt.begin - pad,
            end=evt.end + pad,
            client=self.client,
        )
        # Query datasets
        following_error_frame = await faf.following_errors()

        fa_sel_dict = {
            "primary": {key: [] for key in ["all", "fa_quadrant", "fa_orientation"]},
            "secondary": {key: [] for key in ["all", "fa_quadrant", "fa_orientation"]},
        }

        fa_sel_dict["primary"]["all"] = [i.index for i in FATable if i.index > 10]
        fa_sel_dict["secondary"]["all"] = [
            i.s_index for i in FATable if (i.s_index is not None) and (i.index > 10)
        ]
        summary_stats_frames = []
        for act_type in ["primary", "secondary"]:
            for i in fa_sel_dict[act_type]["all"]:
                fa_info = FATable[i]
                fa_sel_dict[act_type]["fa_quadrant"].append(fa_info.quadrant)
                fa_sel_dict[act_type]["fa_orientation"].append(fa_info.orientation)

            fa_idx = fa_sel_dict[act_type]["all"]
            summary_stats_frames.append(
                self.compute_following_error_summary_stats(
                    following_error_frame, fa_idx, "all", act_type
                )
            )

            for quadrant in [1, 2, 3, 4]:
                fa_idx = [
                    i
                    for i, q in zip(
                        fa_sel_dict[act_type]["all"],
                        fa_sel_dict[act_type]["fa_quadrant"],
                    )
                    if q == quadrant
                ]
                summary_stats_frames.append(
                    self.compute_following_error_summary_stats(
                        following_error_frame, fa_idx, f"quadrant_{quadrant}", act_type
                    )
                )
            fao_list = np.unique(
                [
                    FAOrientation(i).name
                    for i in fa_sel_dict[act_type]["fa_orientation"]
                    if i > 0
                ]
            )
            for orientation in fao_list:
                fa_idx = [
                    i
                    for i, o in zip(
                        fa_sel_dict[act_type]["all"],
                        fa_sel_dict[act_type]["fa_orientation"],
                    )
                    if o == FAOrientation[orientation].value
                ]

                summary_stats_frames.append(
                    self.compute_following_error_summary_stats(
                        following_error_frame,
                        fa_idx,
                        f"orientation_{orientation}",
                        act_type,
                    )
                )
        following_error_frame = pd.concat(
            [following_error_frame.copy()] + summary_stats_frames, axis=1
        )
        return following_error_frame

    def compute_following_error_summary_stats(
        self,
        following_error_frame: pd.DataFrame,
        hp_idx_list: list[int],
        col_key: str,
        act_type: str,
    ) -> pd.DataFrame:
        if act_type == "primary":
            cols = [f"{act_type}CylinderFollowingError" + str(i) for i in hp_idx_list]
        if act_type == "secondary":
            cols = [f"{act_type}CylinderFollowingError" + str(i) for i in hp_idx_list]
        following_error_dict = {}
        following_error_dict[f"{col_key}_{act_type}_max_val"] = (
            following_error_frame[cols].max(axis=1).values
        )
        # min
        following_error_dict[f"{col_key}_{act_type}_min_val"] = (
            following_error_frame[cols].min(axis=1).values
        )
        # abs max
        following_error_dict[f"{col_key}_{act_type}_absmax_val"] = (
            following_error_frame[cols].abs().max(axis=1).values
        )
        # median
        following_error_dict[f"{col_key}_{act_type}_median_val"] = (
            following_error_frame[cols].median(axis=1).values
        )
        # std
        following_error_dict[f"{col_key}_{act_type}_std_val"] = (
            following_error_frame[cols].std(axis=1).values
        )
        # confidence interval
        following_error_dict[f"{col_key}_{act_type}_q1_val"] = (
            following_error_frame[cols].quantile(0.16, axis=1).values
        )
        following_error_dict[f"{col_key}_{act_type}_q3_val"] = (
            following_error_frame[cols].quantile(0.84, axis=1).values
        )
        return pd.DataFrame(following_error_dict, index=following_error_frame.index)

    async def query_dataset(self) -> pd.DataFrame:
        """
        Queries all the relevant data, resampling them to have the same
        frequency, and merges them into a single dataframe.

        Returns
        -------
        data : `pd.DataFrame`
            The data.
        """
        query_config = {
            "hp_measured_forces": {
                "topic": "lsst.sal.MTM1M3.hardpointActuatorData",
                "columns": self.measured_forces_topics,
            },
            "tma_az": {
                "topic": "lsst.sal.MTMount.azimuth",
                "columns": [
                    "timestamp",
                    "actualPosition",
                    "actualVelocity",
                    "actualTorque",
                ],
                "reset_index": True,
                "rename_columns": {
                    "actualTorque": "az_actual_torque",
                    "actualVelocity": "az_actual_velocity",
                    "actualPosition": "az_actual_position",
                },
            },
            "tma_el": {
                "topic": "lsst.sal.MTMount.elevation",
                "columns": [
                    "timestamp",
                    "actualPosition",
                    "actualVelocity",
                    "actualTorque",
                ],
                "reset_index": True,
                "rename_columns": {
                    "actualPosition": "el_actual_position",
                    "actualTorque": "el_actual_torque",
                    "actualVelocity": "el_actual_velocity",
                },
            },
        }

        # Query datasets
        queries = {key: self.query_efd_data(**cfg) for key, cfg in query_config.items()}
        queries["fa_following_errors"] = (
            await self.query_force_actuator_following_error_dataset()
        )
        queries["slew"] = self.event

        return queries

    def query_efd_data(
        self,
        topic: str,
        columns: list[str],
        reset_index: bool = False,
        rename_columns: dict | None = None,
    ) -> pd.DataFrame:
        """
        Query the EFD data for a given topic and return a dataframe.

        Parameters
        ----------
        topic : `str`
            The topic to query.
        columns : `List[str]`
            The columns to query.
        err_msg : `str`, optional
            The error message to raise if no data is found. If None, it creates
            a dataframe padded with zeros.
        reset_index : `bool`, optional
            Whether to reset the index of the dataframe.
        rename_columns : `dict`, optional
            A dictionary of column names to rename.
        resample : `float`, optional
            The resampling frequency in seconds.

        Returns
        -------
        df : `pd.DataFrame`
            A dataframe containing the queried data. If no data is found and
            `err_msg` is None, returns a dataframe padded with zeros.
        """
        # self.log.info(f"Querying dataset: {topic}")
        df = getEfdData(
            self.client,
            topic,
            columns=columns,
            event=self.event,
            prePadding=self.outer_pad,
            postPadding=self.outer_pad,
            warn=False,
            raiseIfTopicNotInSchema=False,
        )

        # self.log.debug(f"Queried {df.index.size} rows from {topic}")
        if df.index.size == 0:
            begin_timestamp = pd.Timestamp(self.event.begin.unix, unit="s")
            end_timestamp = pd.Timestamp(self.event.end.unix, unit="s")
            index = pd.DatetimeIndex(
                pd.date_range(begin_timestamp, end_timestamp, freq="1s")
            )
            df = pd.DataFrame(
                columns=columns,
                index=index,
                data=np.zeros((index.size, len(columns))),
            )

        if rename_columns is not None:
            df = df.rename(columns=rename_columns)

        if reset_index:
            df["timestamp"] = Time(
                df["timestamp"], format="unix_tai", scale="utc"
            ).datetime
            df.set_index("timestamp", inplace=True)
            df.index = df.index.tz_localize("UTC")

        return df


ax_map_dict = {
    0: SlewState.EL_SLEW | SlewState.AZ_SLEW | SlewState.EL_POSITIVE,
    1: SlewState.EL_SLEW | SlewState.EL_POSITIVE,
    2: SlewState.EL_SLEW
    | SlewState.AZ_SLEW
    | SlewState.EL_POSITIVE
    | SlewState.AZ_POSITIVE,
    3: SlewState.AZ_SLEW,
    4: 0,
    5: SlewState.AZ_SLEW | SlewState.AZ_POSITIVE,
    6: SlewState.AZ_SLEW | SlewState.EL_SLEW,
    7: SlewState.EL_SLEW,
    8: SlewState.AZ_SLEW | SlewState.AZ_POSITIVE | SlewState.EL_SLEW,
}

ax_label_dict = {
    0: {
        "title": "az negative",
        "ylabel": "el positive",
    },
    1: {
        "title": "no az",
    },
    2: {
        "title": "az positive",
    },
    3: {
        "ylabel": "no el",
    },
    4: {},
    5: {},
    6: {"ylabel": "el negative", "xlabel": "az negative"},
    7: {"xlabel": "no az"},
    8: {
        "xlabel": "az positive",
    },
}


def create_color_dict(values, cmap_name="viridis"):
    """
    Create a color dictionary from a list of values using a colormap.

    Parameters
    ----------
    values : list
        A list of values to assign colors.
    cmap_name : str, optional
        The name of the Matplotlib colormap to use. Default is "viridis".

    Returns
    -------
    dict
        A dictionary where keys are values from the list and values are colors from the colormap.
    """
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)
    if (len(values) > 10) | (cmap_name != "tab10"):
        # Normalize the values to the range [0, 1]
        norm = plt.Normalize(vmin=0, vmax=len(values) - 1)
        # Assign a color to each value
        color_dict = {value: cmap(norm(i)) for i, value in enumerate(values)}
    else:
        color_dict = {value: cmap(i) for i, value in enumerate(values)}
    return color_dict


class SlewPlotter:
    """
    Class for plotting slew data.
    """

    def __init__(self, plot_dir="./plots/"):
        """
        Parameters:
            plot_dir (str): Directory to save plots.
            If None, plots are not saved.
        """
        self.plot_dir = plot_dir

    def single_test_plot(
        self,
        col_key,
        stats_frame,
        hp_forces_df,
        fa_following_errors_df,
        time_align,
        duration,
        block="T293",
        block_info="",
        xmin=None,
        xmax=None,
        day_obs=None,
    ):
        """
        Plot a single slew based on the provided parameters.

        Parameters:
        col_key (str): Column key for the data to plot.
        dynamic_test_info_dict (dict): Dict with dynamic test info.
        dynamic_test_data_dict (dict): Dict with dynamic test data.
        time_align (str): Time alignment for the plot ("start" or "stop").
        duration (int): Duration for the plot.
        xmin (float): Minimum x-axis value.
        xmax (float): Maximum x-axis value.
        block (str): Block identifier.
        block_info (str): Block information.
        day_obs (str): Observation day.
        plot_fa_following (bool): Whether to plot following errors.

        Returns:
        matplotlib.figure.Figure: The generated plot figure.
        """
        if day_obs is None:
            day_obs = np.unique(stats_frame["day_obs"])[0]

        if time_align == "start":
            x_label = "time since slew start [s]"
            file_suffix = "slew_start"
        elif time_align == "stop":
            x_label = "time before slew stop [s]"
            file_suffix = "slew_stop"
        else:
            print(f"bad time align {time_align}")

        plt.rcParams["axes.labelsize"] = 12
        fig, axs = plt.subplots(
            3, 3, dpi=125, figsize=(12, 10), sharex=True, sharey=True
        )
        axs = axs.flatten()
        max_val = 10

        for ax_val in range(9):
            ax = axs[ax_val]
            if ax_val == 4:
                ax.axis("off")
                continue
            slew_sel = stats_frame["state"] == int(ax_map_dict[ax_val])
            for seq_num in stats_frame["seq_num"][slew_sel].values:
                if col_key in hp_forces_df.columns:
                    plot_frame = hp_forces_df
                if col_key in fa_following_errors_df.columns:
                    plot_frame = fa_following_errors_df
                row_sel = plot_frame["seq_num"] == seq_num
                ydata = plot_frame.loc[row_sel, [col_key]].copy()
                ydata.index = plot_frame["time"][row_sel]

                t0 = Time(ydata.index[0] if time_align == "start" else ydata.index[-1])
                times = Time(ydata.index) - t0
                time_sel = (
                    times < duration * u.second
                    if time_align == "start"
                    else times > duration * u.second
                )
                times = times[time_sel]
                ydata = ydata[time_sel]
                if len(ydata) == 0:
                    continue
                label = f"{seq_num}"

                if "median" in col_key:
                    median = ydata.values
                    q1 = plot_frame.loc[
                        row_sel, [col_key.replace("_median", "_q1")]
                    ].values.flatten()[time_sel]

                    q3 = plot_frame.loc[
                        row_sel, [col_key.replace("_median", "_q3")]
                    ].values.flatten()[time_sel]

                    ax.plot(times.sec, median, label=label)
                    ax.fill_between(times.sec, q1, q3, alpha=0.3)
                    ax.plot(times.sec, q1, ls="dashed", alpha=0.3)
                    ax.plot(times.sec, q3, ls="dashed", alpha=0.3)

                    max_val = max(
                        max_val,
                        np.max(abs(median)),
                        np.max(abs(q1)),
                        np.max(abs(q3)),
                    )
                elif "_max" in col_key:
                    # if plotting max also plot min
                    y2data = plot_frame.loc[
                        row_sel, [col_key.replace("_max", "_min")]
                    ].values.flatten()[time_sel]
                    ax.plot(times.sec, ydata, label=label)
                    ax.plot(times.sec, y2data, label=label)

                    max_val = max(
                        max_val,
                        np.max(abs(ydata)),
                        np.max(abs(y2data)),
                    )
                else:
                    ax.plot(times.sec, ydata, label=label)
                    max_val = max(max_val, np.max(abs(ydata)))

                ax.set(**ax_label_dict[ax_val])
            if xmin is None and xmax is None:
                (
                    ax.set_xlim(0, duration)
                    if time_align == "start"
                    else ax.set_xlim(duration, 0)
                )
            else:
                ax.set_xlim(xmin, xmax)

            ax.tick_params(direction="in")

            if ax_val == 4:
                ax.axis("off")
            else:
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(facecolor="none", edgecolor="none", title="seq_num")
        for ax in axs:
            if "absmax" in col_key:
                ax.set_ylim(0, max_val * 1.1)
            else:
                ax.set_ylim(-max_val * 1.1, max_val * 1.1)

        fig.text(
            0.5,
            0.04,
            x_label,
            ha="center",
            va="center",
            fontsize=16,
        )  # x-label

        col_label = (
            col_key.replace("_max_val", "max/min").replace("_val", "").replace("_", " ")
        )
        if ("primary" in col_label) or ("secondary" in col_label):
            col_label = "FA following error " + col_label

        fig.text(
            0.04,
            0.5,
            col_label,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=16,
        )  # y-label
        plt.suptitle(
            f"dynamic test: '{col_label}'\n{day_obs} - {file_suffix}\nBLOCK-{block}: {block_info} ",
            y=0.96,
        )
        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        if self.plot_dir:
            plt.savefig(
                self.plot_dir + f"{day_obs}_{col_key}_{block}_{file_suffix}.png"
            )
            plt.close()
        return fig

    def multi_test_plot(
        self,
        col_key,
        dynamic_test_info_dict,
        dynamic_test_data_dict,
        color_dict,
        time_align="start",
        duration=2,
        xmin=None,
        xmax=None,
    ):
        plt.rcParams["axes.labelsize"] = 12
        fig, axs = plt.subplots(
            3, 3, dpi=125, figsize=(12, 10), sharex=True, sharey=True
        )
        axs = axs.flatten()
        max_val = 10
        exclude_list = []
        duration = duration
        time_align = time_align
        if time_align == "stop":
            duration *= -1
        for ax_val in range(9):
            ax = axs[ax_val]
            if ax_val == 4:
                ax.axis("off")
                continue
            for key in dynamic_test_data_dict.keys():
                stats_frame = dynamic_test_data_dict[key]["stats_frame"]
                hp_forces_df = dynamic_test_data_dict[key]["hp_forces_df"]
                fa_following_errors_df = dynamic_test_data_dict[key][
                    "fa_following_errors_df"
                ]
                slew_sel = stats_frame["state"] == int(ax_map_dict[ax_val])
                for seq_num in stats_frame["seq_num"][slew_sel].values:
                    if seq_num in exclude_list:
                        continue
                    if col_key in hp_forces_df.columns:
                        plot_frame = hp_forces_df
                    if col_key in fa_following_errors_df.columns:
                        plot_frame = fa_following_errors_df
                    row_sel = plot_frame["seq_num"] == seq_num
                    ydata = plot_frame.loc[row_sel, [col_key]].copy()
                    ydata.index = plot_frame["time"][row_sel]

                    t0 = Time(
                        ydata.index[0] if time_align == "start" else ydata.index[-1]
                    )
                    times = Time(ydata.index) - t0
                    time_sel = (
                        times < duration * u.second
                        if time_align == "start"
                        else times > duration * u.second
                    )
                    times = times[time_sel]
                    ydata = ydata[time_sel]

                    label = (
                        dynamic_test_info_dict[key]["block_info"]
                        + " "
                        + str(np.unique(stats_frame["day_obs"])[0])
                    )

                    if "median" in col_key:
                        median = ydata.values
                        q1 = plot_frame.loc[
                            row_sel, [col_key.replace("_median", "_q1")]
                        ].values.flatten()[time_sel]

                        q3 = plot_frame.loc[
                            row_sel, [col_key.replace("_median", "_q3")]
                        ].values.flatten()[time_sel]

                        ax.plot(times.sec, median, label=label, c=color_dict[key])
                        ax.fill_between(
                            times.sec, q1, q3, alpha=0.3, color=color_dict[key]
                        )
                        ax.plot(times.sec, q1, ls="dashed", alpha=0.3)
                        ax.plot(times.sec, q3, ls="dashed", alpha=0.3)

                        max_val = max(
                            max_val,
                            np.max(abs(median)),
                            np.max(abs(q1)),
                            np.max(abs(q3)),
                        )
                    elif "_max" in col_key:
                        # if plotting max also plot min
                        y2data = plot_frame.loc[
                            row_sel, [col_key.replace("_max", "_min")]
                        ].values.flatten()[time_sel]
                        ax.plot(times.sec, ydata, label=label, c=color_dict[key])
                        ax.plot(times.sec, y2data, label=label, c=color_dict[key])
                        max_val = max(
                            max_val,
                            np.max(abs(ydata)),
                            np.max(abs(y2data)),
                        )
                    else:
                        ax.plot(times.sec, ydata, label=label, c=color_dict[key])
                        max_val = max(max_val, np.max(abs(ydata)))

                ax.set(**ax_label_dict[ax_val])
            if xmin is None and xmax is None:
                (
                    ax.set_xlim(0, duration)
                    if time_align == "start"
                    else ax.set_xlim(duration, 0)
                )
            else:
                ax.set_xlim(xmin, xmax)
            ax.tick_params(direction="in")

            # else:
            #     ax.legend(facecolor="none", edgecolor="none", title="seq_num")
        handles, labels = axs[1].get_legend_handles_labels()
        unique_labels = dict(sorted(zip(labels, handles), key=lambda pair: pair[0]))
        axs[4].legend(
            unique_labels.values(), unique_labels.keys(), title="%speed settings"
        )

        for ax in axs:
            if "absmax" in col_key:
                ax.set_ylim(0, max_val * 1.1)
            else:
                ax.set_ylim(-max_val * 1.1, max_val * 1.1)

        if time_align == "start":
            x_label = "time since slew start [s]"
            file_suffix = "slew_start"
        elif time_align == "stop":
            x_label = "time before slew stop [s]"
            file_suffix = "slew_stop"
        else:
            print(f"bad time align {time_align}")

        fig.text(
            0.5,
            0.04,
            x_label,
            ha="center",
            va="center",
            fontsize=16,
        )  # x-label

        col_label = (
            col_key.replace("_max_val", "max/min").replace("_val", "").replace("_", " ")
        )
        if ("primary" in col_label) or ("secondary" in col_label):
            col_label = "FA following error " + col_label
        fig.text(
            0.04,
            0.5,
            col_label,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=16,
        )  # y-label
        plt.suptitle(
            f"dynamic test compare: '{col_label}'\nslew {time_align} aligned",
            y=0.96,
            fontsize=20,
        )
        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        if self.plot_dir:
            plt.savefig(
                self.plot_dir + f"dynamic_test_compare_{col_key}_{file_suffix}.png"
            )
            plt.close()
        return fig


def compute_stats_frame(query_dict, day_obs, block, block_info):
    """
    Compute stats for the given query dictionary.

    Parameters:
    query_dict (dict): Dictionary with query results.
    day_obs (str): Observation day.
    block (str): Block identifier.
    block_info (str): Block information.

    Returns:
    pd.DataFrame: DataFrame with computed stats.
    """
    stats_dict = {
        key: []
        for key in [
            "seq_num",
            "el_start",
            "el_end",
            "el_distance",
            "az_start",
            "az_end",
            "az_distance",
            "total_distance",
            "max_hp_force",
            "min_hp_force",
            "max_fa_following_error",
            "min_fa_following_error",
        ]
    }
    for seq_num in query_dict.keys():
        stats_dict["seq_num"].append(seq_num)
        for axis in ["az", "el"]:
            vals = query_dict[seq_num][f"tma_{axis}"][f"{axis}_actual_position"].values
            stats_dict[f"{axis}_start"].append(vals[0])
            stats_dict[f"{axis}_end"].append(vals[-1])
            stats_dict[f"{axis}_distance"].append((vals[-1] - vals[0]))
        total_distance = np.sqrt(
            stats_dict["az_distance"][-1] ** 2 + stats_dict["el_distance"][-1] ** 2
        )
        stats_dict["total_distance"].append(total_distance)
        stats_dict["max_hp_force"].append(
            query_dict[seq_num]["hp_measured_forces"]
            .filter(like="measuredForce")
            .max()
            .max()
        )
        stats_dict["min_hp_force"].append(
            query_dict[seq_num]["hp_measured_forces"]
            .filter(like="measuredForce")
            .min()
            .min()
        )

        stats_dict["max_fa_following_error"].append(
            query_dict[seq_num]["fa_following_errors"]["all_primary_max_val"].max()
        )
        stats_dict["min_fa_following_error"].append(
            query_dict[seq_num]["fa_following_errors"]["all_primary_min_val"].min()
        )
    stats_frame = pd.DataFrame(stats_dict)
    stats_frame["day_obs"] = day_obs
    stats_frame["block"] = block
    stats_frame["block_info"] = block_info.replace("%", "")
    stats_frame["state"] = stats_frame.apply(compute_state, axis=1)

    return stats_frame


def load_config(config_file=None, **kwargs):
    """
    Load config from a YAML file and/or keyword arguments.

    Parameters:
    config_file (str): Path to the YAML config file.
    **kwargs: Additional config parameters.

    Returns:
    dict: Configuration dictionary.
    """
    # Define default values
    default_config = {
        "begin_seq_num": None,
        "end_seq_num": None,
        "begin_time": None,
        "end_time": None,
        "day_obs": None,
        "block_info": None,
        "block": None,
        "tmax": None,
        "tmin": None,
        "exclude_list": [],
        "data_dir": "./data/",
        "plot_dir": "./plots/",
        "save_data": True,
        "make_plots": True,
        "hp_col_keys": [],  # would like to specify measuredForce, f, or m and have it populate
        "act_groups": [],
        "act_types": ["primary", "secondary"],
    }
    # set hp_col_keys
    default_config["hp_col_keys"] += [f"measuredForce{i}" for i in range(HP_COUNT)]
    default_config["hp_col_keys"] += [f"f{i}" for i in "xyz"]
    default_config["hp_col_keys"] += [f"m{i}" for i in "xyz"]

    default_config["act_groups"] += ["all", "quadrant", "orientation"]

    # Load config from file if specified
    if config_file:
        with open(config_file, "r") as file:
            file_config = yaml.safe_load(file)
        default_config.update(file_config)

    # Override with any provided keyword arguments
    default_config.update(kwargs)
    if default_config["plot_dir"]:
        os.makedirs(default_config["plot_dir"], exist_ok=True)
    if default_config["data_dir"]:
        os.makedirs(default_config["data_dir"], exist_ok=True)

    act_groups = []

    if "all" in default_config["act_groups"]:
        act_groups += ["all"]
    if "quadrant" in default_config["act_groups"]:
        act_groups += [f"quadrant_{i}" for i in range(1, 5)]
    if "orientation" in default_config["act_groups"]:
        act_groups += [f"orientation_{FAOrientation(i).name}" for i in range(1, 5)]
    default_config["act_groups"] = act_groups

    fa_col_keys = []
    for act_type in ["primary", "secondary"]:
        for act_group in act_groups:
            fa_col_keys += [f"{act_group}_{act_type}_max_val"]
            fa_col_keys += [f"{act_group}_{act_type}_min_val"]
            fa_col_keys += [f"{act_group}_{act_type}_median_val"]

    default_config["fa_col_keys"] = fa_col_keys

    return default_config


async def run_single_dynamic_test(
    begin_time=None,
    end_time=None,
    begin_seq_num=None,
    end_seq_num=None,
    day_obs=None,
    block_info=None,
    block=None,
    plot_dir=None,
    tmax=3.2,
    tmin=-3.2,
    data_dir="./data/",
    save_data=True,
    make_plots=True,
    hp_col_keys=[],
    fa_col_keys=[],
    **kwargs,
):
    """
    Run a single dynamic test based on the provided config.

    Parameters:
    **config: Configuration parameters.

    Returns:
    tuple: A tuple with stats_frame, hp_forces_df, and fa_following_errors_df.
    """
    if not EfdClient:
        raise RuntimeError("EFD client is not available.")

    if begin_time is not None and end_time is not None:
        begin_time = Time(begin_time, format="iso")
        end_time = Time(end_time, format="iso")
        if day_obs is None:
            day_obs_begin = getDayObsForTime(begin_time)
            day_obs_end = getDayObsForTime(end_time)
            if day_obs_begin != day_obs_end:
                raise ValueError(
                    "begin_time and end_time span multiple day_obs values."
                )
            day_obs = day_obs_begin
        event_maker = TMAEventMaker()
        events = event_maker.getEvents(day_obs)
        slews = [
            e
            for e in events
            if (e.begin.unix >= begin_time.unix) & (e.end.unix <= end_time.unix)
        ]
        print(
            (
                f"Using time range: {begin_time} to {end_time},"
                f"found {len(slews)} slews"
            )
        )
    else:
        if day_obs is None:
            raise ValueError(
                "day_obs must be specified if begin_time and end_time are not."
            )
        event_maker = TMAEventMaker()
        events = event_maker.getEvents(day_obs)
        slews = [
            e for e in events if (e.seqNum >= begin_seq_num) & (e.seqNum <= end_seq_num)
        ]
        print(
            (
                f"dayobs {day_obs}"
                f"Using sequence numbers: {begin_seq_num} to {end_seq_num}"
                f"found {len(slews)} slews"
            )
        )

    query_dict = {}

    hp_forces_list = []
    fa_following_errors_list = []
    for slew in np.asarray(slews):
        seq_num = slew.seqNum
        # with warnings.catch_warnings():
        #     # this warning should be safe to ignore
        #     warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        query_result = await M1M3Query(
            slew, event_maker.client, outer_pad=0
        ).query_dataset()
        query_result["hp_measured_forces"]["seq_num"] = seq_num
        query_result["hp_measured_forces"]["day_obs"] = day_obs
        query_result["fa_following_errors"]["seq_num"] = seq_num
        query_result["fa_following_errors"]["day_obs"] = day_obs
        query_dict[seq_num] = query_result

        hp_forces_data = query_result["hp_measured_forces"].copy()
        fa_following_errors_data = query_result["fa_following_errors"].copy()
        hp_forces_list.append(hp_forces_data)
        fa_following_errors_list.append(fa_following_errors_data)

    # Concatenate all forces data into a single DataFrame
    hp_forces_df = pd.concat(hp_forces_list, axis=0)
    hp_forces_df = hp_forces_df.reset_index()
    hp_forces_df.rename(columns={"index": "time"}, inplace=True)

    fa_following_errors_df = pd.concat(fa_following_errors_list, axis=0)
    fa_following_errors_df = fa_following_errors_df.reset_index()
    fa_following_errors_df.rename(columns={"index": "time"}, inplace=True)

    stats_frame = compute_stats_frame(query_dict, day_obs, block, block_info)

    if save_data:
        # save data frames
        hp_forces_csv_path = os.path.join(
            data_dir,
            config_file.split("/")[-1].replace(".yaml", "_hp_efd_frame.csv"),
        )

        fa_following_error_csv_path = os.path.join(
            data_dir,
            config_file.split("/")[-1].replace(".yaml", "_fa_efd_frame.csv"),
        )

        hp_forces_df.to_csv(hp_forces_csv_path)
        fa_following_errors_df.to_csv(fa_following_error_csv_path)

        stats_csv_path = os.path.join(
            data_dir,
            config_file.split("/")[-1].replace(".yaml", "_stats_frame.csv"),
        )
        stats_frame.to_csv(stats_csv_path)

        print(
            f"Saved concatenated forces data to {hp_forces_csv_path}, fa_efd_frame.csv & stats_frame.csv"
        )

    if make_plots:
        sp = SlewPlotter(plot_dir=plot_dir)
        for col_key in hp_col_keys + fa_col_keys:
            sp.single_test_plot(
                col_key=col_key,
                stats_frame=stats_frame,
                hp_forces_df=hp_forces_df,
                fa_following_errors_df=fa_following_errors_df,
                time_align="start",
                duration=tmax,
                block="T293",
                block_info="",
                xmin=None,
                xmax=None,
                day_obs=None,
            )
            sp.single_test_plot(
                col_key=col_key,
                stats_frame=stats_frame,
                hp_forces_df=hp_forces_df,
                fa_following_errors_df=fa_following_errors_df,
                time_align="stop",
                duration=abs(tmin),
                block="T293",
                block_info="",
                xmin=None,
                xmax=None,
                day_obs=None,
            )

    return stats_frame, hp_forces_df, fa_following_errors_df


if __name__ == "__main__":
    """
    example call:
        python dynamic_test_analysis.py ./config/20241201_T293_1.yaml
    """
    import asyncio

    config_file = "./config/20241201_T293_1.yaml"
    parser = argparse.ArgumentParser(description="Load configuration and run analysis.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--begin_time",
        type=str,
        help="iso begin time",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="iso end time",
    )

    args = parser.parse_args()
    if args.config_file:
        config_file = args.config_file
        config = load_config(config_file=config_file, save_data=False, make_plots=True)
    elif args.begin_time and args.end_time:
        config = load_config(begin_time=args.begin_time, end_time=args.end_time)
    stats_frame, hp_forces_df, fa_following_errors_df = asyncio.run(
        run_single_dynamic_test(**config)
    )
