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
from lsst.summit.utils.efdUtils import getEfdData
from lsst.summit.utils.tmaUtils import TMAEvent, TMAEventMaker
from lsst.ts.xml.tables.m1m3 import HP_COUNT

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
            f"primaryCylinderFollowingError{i}" for i in range(10,156)
        ]
        self.force_actuator_topics += [
            f"secondaryCylinderFollowingError{i}" for i in range(10,156)
        ]
        
    def query_force_actuator_following_error_dataset(self) -> pd.DataFrame:
        evt = self.event
        query_config = {
            "force_actuator_forces": {
                "topic": "lsst.sal.MTM1M3.forceActuatorData",
                "columns": self.force_actuator_topics,
                "err_msg": (
                    "No hard-point data found for event" f"{evt.seqNum} on {evt.dayObs}"
                ),
            },
            "tma_az": {
                "topic": "lsst.sal.MTMount.azimuth",
                "columns": [
                    "timestamp",
                    "actualPosition",
                    "actualVelocity",
                    "actualTorque",
                ],
                "err_msg": (
                    "No TMA azimuth data found for event"
                    f"{evt.seqNum} on {evt.dayObs}"
                ),
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
                "err_msg": (
                    "No TMA elevation data found for event"
                    f"{evt.seqNum} on {evt.dayObs}"
                ),
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
        queries["slew"] = self.event
        queries["force_actuator_forces"] = compute_summary_stats(queries["force_actuator_forces"])
        return queries
    def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
        return df 
    def query_dataset(self) -> pd.DataFrame:
        """
        Queries all the relevant data, resampling them to have the same
        frequency, and merges them into a single dataframe.

        Returns
        -------
        data : `pd.DataFrame`
            The data.
        """
        evt = self.event
        query_config = {
            "hp_measured_forces": {
                "topic": "lsst.sal.MTM1M3.hardpointActuatorData",
                "columns": self.measured_forces_topics,
                "err_msg": (
                    "No hard-point data found for event" f"{evt.seqNum} on {evt.dayObs}"
                ),
            },
            "tma_az": {
                "topic": "lsst.sal.MTMount.azimuth",
                "columns": [
                    "timestamp",
                    "actualPosition",
                    "actualVelocity",
                    "actualTorque",
                ],
                "err_msg": (
                    "No TMA azimuth data found for event"
                    f"{evt.seqNum} on {evt.dayObs}"
                ),
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
                "err_msg": (
                    "No TMA elevation data found for event"
                    f"{evt.seqNum} on {evt.dayObs}"
                ),
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
        queries["slew"] = self.event

        return queries

    def merge_datasets(self, queries: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple datasets based on their timestamps.

        Parameters
        ----------
        queries (dict[str, pd.DataFrame]):
            A dictionary of dataframes to be merged.

        Returns
        -------
        df : `pd.DataFrame`
            A merged dataframe.
        """
        merge_cfg = {
            "left_index": True,
            "right_index": True,
            "tolerance": timedelta(seconds=1),
            "direction": "nearest",
        }

        # self.log.info("Merging datasets")
        df_list = [df for _, df in queries.items()]
        merged_df = df_list[0]

        for df in df_list[1:]:
            merged_df = pd.merge_asof(merged_df, df, **merge_cfg)

        return merged_df

    def query_efd_data(
        self,
        topic: str,
        columns: list[str],
        err_msg: str | None = None,
        reset_index: bool = False,
        rename_columns: dict | None = None,
        resample: float | None = None,
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


def make_slew_start_plot(
    stats_frame,
    query_dict,
    day_obs,
    col_key="measuredForce2",
    exclude_list=[],
    block="T227",
    block_info="",
    out_dir="./plots/",
    tmax=3.2,
):
    """
    Generate and save a plot showing telemetry data
    for a time range starting from the beginning of a slew.
    The slews will be grouped by their SlewState.

    Parameters
    ----------
    stats_frame : `pd.DataFrame`
        A DataFrame containing statistics for each slew.
    query_dict : `dict`
        A dictionary containing telemetry data for each slew.
    day_obs : `int`
        The observation day identifier (YYYYMMDD).
    col_key : `str`, optional
        The telemetry data column to plot. Default is "measuredForce2".
    exclude_list : `list`, optional
        A list of sequence numbers to exclude
        from the plot. Default is an empty list.
    block : `str`, optional
        The block identifier for the test. Default is "T227".
    block_info : `str`, optional
        Additional information about the block. Default is an empty string.
    out_dir : `str`, optional
        The output directory for saving the plot. Default is "./plots/".
    tmax : `float`, optional
        The maximum time after the start of the slew to plot,
        in seconds. Default is 3.2.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The figure object of the generated plot.
    """
    fig, axs = plt.subplots(3, 3, dpi=125, figsize=(12, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    max_val = 200

    for ax_val in range(9):
        ax = axs[ax_val]
        slew_sel = stats_frame["state"] == int(ax_map_dict[ax_val])
        for seq_num in stats_frame["seq_num"][slew_sel].values:
            if seq_num in exclude_list:
                continue
            ydata = query_dict[seq_num]["hp_measured_forces"][col_key]
            t0 = Time(ydata.index[0])
            times = Time(ydata.index) - t0
            time_sel = times < tmax * u.second
            times = times[time_sel]
            ydata = ydata[time_sel]

            if np.max(abs(ydata)) > max_val:
                max_val = np.max(abs(ydata)) * 1.1

            ax.plot(times.sec, ydata, label=f"{seq_num}")
            ax.set(**ax_label_dict[ax_val])
        ax.set_xlim(0, tmax)

        ax.tick_params(direction="in")

        if ax_val == 4:
            ax.axis("off")
        else:
            ax.legend(facecolor="none", edgecolor="none")
    for ax in axs:
        ax.set_ylim(-max_val, max_val)
    fig.text(
        0.5,
        0.04,
        "Time after slew start [s]",
        ha="center",
        va="center",
        fontsize=16,
    )  # x-label
    fig.text(
        0.04,
        0.5,
        col_key,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16,
    )  # y-label
    plt.suptitle(f"{day_obs} - slew starts\nBLOCK-{block}: {block_info} ", y=0.96)
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    plt.savefig(out_dir + f"{day_obs}_{col_key}_{block}_slew_start.png")
    plt.close()
    return fig


def make_slew_stop_plot(
    stats_frame,
    query_dict,
    day_obs,
    col_key="measuredForce2",
    exclude_list=[],
    block="T227",
    block_info="",
    out_dir="./plots/",
    tmin=-3.2,
):
    """
    Generate and save a plot showing telemetry data
    for a time range starting before the end of a slew.
    The slews will be grouped by their SlewState.

    Parameters
    ----------
    stats_frame : `pd.DataFrame`
        A DataFrame containing statistics for each slew.
    query_dict : `dict`
        A dictionary containing telemetry data for each slew.
    day_obs : `int`
        The observation day identifier (YYYYMMDD).
    col_key : `str`, optional
        The telemetry data column to plot. Default is "measuredForce2".
    exclude_list : `list`, optional
        A list of sequence numbers to exclude from the plot.
        Default is an empty list.
    block : `str`, optional
        The block identifier for the test. Default is "T227".
    block_info : `str`, optional
        Additional information about the block. Default is an empty string.
    out_dir : `str`, optional
        The output directory for saving the plot. Default is "./plots/".
    tmin : `float`, optional
        The minimum time before the end of the slew to plot,
        in seconds. Default is -3.2.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The figure object of the generated plot.
    """

    fig, axs = plt.subplots(3, 3, dpi=125, figsize=(12, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    max_val = 200

    for ax_val in range(9):
        ax = axs[ax_val]
        slew_sel = stats_frame["state"] == int(ax_map_dict[ax_val])
        for seq_num in stats_frame["seq_num"][slew_sel].values:
            if seq_num in exclude_list:
                continue
            ydata = query_dict[seq_num]["hp_measured_forces"][col_key]
            t0 = Time(ydata.index[-1])
            times = Time(ydata.index) - t0
            time_sel = times > tmin * u.second
            times = times[time_sel]
            ydata = ydata[time_sel]

            if np.max(abs(ydata)) > max_val:
                max_val = np.max(abs(ydata)) * 1.1

            ax.plot(times.sec, ydata, label=f"{seq_num}")
            ax.set(**ax_label_dict[ax_val])
        ax.set_xlim(tmin, 0)

        ax.tick_params(direction="in")

        if ax_val == 4:
            ax.axis("off")
        else:
            ax.legend(facecolor="none", edgecolor="none")
    for ax in axs:
        ax.set_ylim(-max_val, max_val)
    fig.text(
        0.5,
        0.04,
        "Time before slew stops [s]",
        ha="center",
        va="center",
        fontsize=16,
    )  # x-label
    fig.text(
        0.04,
        0.5,
        col_key,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16,
    )  # y-label
    plt.suptitle(f"{day_obs} - slew stops\nBLOCK-{block}: {block_info} ", y=0.96)
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    plt.savefig(out_dir + f"{day_obs}_{col_key}_{block}_slew_stop.png")
    plt.close()
    return fig

def main_following_error():
    """
    Main function to load the configuration file, query telemetry data,
    and generate slew start and stop plots.

    Command-line Arguments
    ----------------------
    config_file : `str`
        Path to the YAML configuration file.

    Example config:
    begin_seq_num: 35
    end_seq_num: 50
    day_obs: 20241128
    block_info: "20% GGRR"
    block: "T293"
    out_dir: "./plots/20241128_T293_1/"
    tmax: 3.2
    tmin: -3.2
    """
    parser = argparse.ArgumentParser(description="Load configuration and run analysis.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    if not EfdClient:
        raise RuntimeError("EFD client is not available.")

    # Load configuration from YAML
    config_file = args.config_file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    begin_seq_num = config.get("begin_seq_num", None)
    end_seq_num = config.get("end_seq_num", None)
    begin_time = config.get("begin_time", None)
    end_time = config.get("end_time", None)
    day_obs = config.get("day_obs")
    block_info = config.get("block_info")
    block = config.get("block")
    out_dir = config.get("out_dir")
    tmax = config.get("tmax")
    tmin = config.get("tmin")
    exclude_list = config.get("exclude_list", [])
    data_dir = config.get("data_dir", "./data/")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    col_keys = ["median_primary_following_error", "median_secondary_following_error"]

    event_maker = TMAEventMaker()
    events = event_maker.getEvents(day_obs)

    if begin_time is not None and end_time is not None:

        begin_time = Time(begin_time, format="iso")
        end_time = Time(end_time, format="iso")

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

    all_forces_list = []
    for slew in np.asarray(slews):
        seq_num = slew.seqNum
        query_result = M1M3Query(slew, event_maker.client, outer_pad=0).query_force_actuator_following_error_dataset()
        query_result["force_actuator_forces"]["seq_num"] = seq_num
        query_result["force_actuator_forces"]["day_obs"] = day_obs
        query_dict[seq_num] = query_result

        # Add seq_num column to forces data and store it
        forces_data = query_result["hp_measured_forces"].copy()
        forces_data["seq_num"] = seq_num
        all_forces_list.append(forces_data)
    # Concatenate all forces data into a single DataFrame
    all_forces_df = pd.concat(all_forces_list, axis=0)

    forces_csv_path = os.path.join(
        data_dir, config_file.split("/")[-1].replace(".yaml", "_efd_frame.csv")
    )
    all_forces_df = all_forces_df.reset_index()
    all_forces_df.rename(columns={"index": "time"}, inplace=True)

    all_forces_df.to_csv(forces_csv_path)

    print(f"Saved concatenated forces data to {forces_csv_path}")

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

    stats_frame = pd.DataFrame(stats_dict)
    stats_frame["day_obs"] = day_obs
    stats_frame["block"] = block
    stats_frame["block_info"] = block_info.replace("%", "")
    stats_frame["state"] = stats_frame.apply(compute_state, axis=1)
    stats_csv_path = os.path.join(
        data_dir, config_file.split("/")[-1].replace(".yaml", "_stats_frame.csv")
    )
    stats_frame.to_csv(stats_csv_path)

    for col_key in col_keys:
        _ = make_slew_start_plot(
            stats_frame,
            query_dict,
            col_key=col_key,
            exclude_list=exclude_list,
            day_obs=day_obs,
            block=block,
            block_info=block_info,
            out_dir=out_dir,
            tmax=tmax,
        )
        _ = make_slew_stop_plot(
            stats_frame,
            query_dict,
            col_key=col_key,
            exclude_list=exclude_list,
            day_obs=day_obs,
            block=block,
            block_info=block_info,
            out_dir=out_dir,
            tmin=tmin,
        )
def main():
    """
    Main function to load the configuration file, query telemetry data,
    and generate slew start and stop plots.

    Command-line Arguments
    ----------------------
    config_file : `str`
        Path to the YAML configuration file.

    Example config:
    begin_seq_num: 35
    end_seq_num: 50
    day_obs: 20241128
    block_info: "20% GGRR"
    block: "T293"
    out_dir: "./plots/20241128_T293_1/"
    tmax: 3.2
    tmin: -3.2
    """
    parser = argparse.ArgumentParser(description="Load configuration and run analysis.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    if not EfdClient:
        raise RuntimeError("EFD client is not available.")

    # Load configuration from YAML
    config_file = args.config_file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    begin_seq_num = config.get("begin_seq_num", None)
    end_seq_num = config.get("end_seq_num", None)
    begin_time = config.get("begin_time", None)
    end_time = config.get("end_time", None)
    day_obs = config.get("day_obs")
    block_info = config.get("block_info")
    block = config.get("block")
    out_dir = config.get("out_dir")
    tmax = config.get("tmax")
    tmin = config.get("tmin")
    exclude_list = config.get("exclude_list", [])
    data_dir = config.get("data_dir", "./data/")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    col_keys = [f"measuredForce{i}" for i in range(HP_COUNT)]
    col_keys += [f"f{i}" for i in "xyz"]
    col_keys += [f"m{i}" for i in "xyz"]

    event_maker = TMAEventMaker()
    events = event_maker.getEvents(day_obs)

    if begin_time is not None and end_time is not None:

        begin_time = Time(begin_time, format="iso")
        end_time = Time(end_time, format="iso")

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

    all_forces_list = []
    for slew in np.asarray(slews):
        seq_num = slew.seqNum
        query_result = M1M3Query(slew, event_maker.client, outer_pad=0).query_dataset()
        query_result["hp_measured_forces"]["seq_num"] = seq_num
        query_result["hp_measured_forces"]["day_obs"] = day_obs
        query_dict[seq_num] = query_result

        # Add seq_num column to forces data and store it
        forces_data = query_result["hp_measured_forces"].copy()
        forces_data["seq_num"] = seq_num
        all_forces_list.append(forces_data)
    # Concatenate all forces data into a single DataFrame
    all_forces_df = pd.concat(all_forces_list, axis=0)

    forces_csv_path = os.path.join(
        data_dir, config_file.split("/")[-1].replace(".yaml", "_efd_frame.csv")
    )
    all_forces_df = all_forces_df.reset_index()
    all_forces_df.rename(columns={"index": "time"}, inplace=True)

    all_forces_df.to_csv(forces_csv_path)

    print(f"Saved concatenated forces data to {forces_csv_path}")

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

    stats_frame = pd.DataFrame(stats_dict)
    stats_frame["day_obs"] = day_obs
    stats_frame["block"] = block
    stats_frame["block_info"] = block_info.replace("%", "")
    stats_frame["state"] = stats_frame.apply(compute_state, axis=1)
    stats_csv_path = os.path.join(
        data_dir, config_file.split("/")[-1].replace(".yaml", "_stats_frame.csv")
    )
    stats_frame.to_csv(stats_csv_path)

    for col_key in col_keys:
        _ = make_slew_start_plot(
            stats_frame,
            query_dict,
            col_key=col_key,
            exclude_list=exclude_list,
            day_obs=day_obs,
            block=block,
            block_info=block_info,
            out_dir=out_dir,
            tmax=tmax,
        )
        _ = make_slew_stop_plot(
            stats_frame,
            query_dict,
            col_key=col_key,
            exclude_list=exclude_list,
            day_obs=day_obs,
            block=block,
            block_info=block_info,
            out_dir=out_dir,
            tmin=tmin,
        )


if __name__ == "__main__":
    main()
