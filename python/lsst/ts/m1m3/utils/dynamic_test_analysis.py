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
from lsst.ts.xml.tables.m1m3 import (
    FATable,
    ForceActuatorData,
    FAOrientation,
    HP_COUNT,
    FAOrientation,
)
from lsst.ts.m1m3.utils.force_actuator_forces import ForceActuatorForces


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
            "primary": {
                key: [] for key in ["all", "fa_quadrant", "fa_orientation"]
            },
            "secondary": {
                key: [] for key in ["all", "fa_quadrant", "fa_orientation"]
            },
        }

        fa_sel_dict["primary"]["all"] = [
            i.index for i in FATable if i.index > 10
        ]
        fa_sel_dict["secondary"]["all"] = [
            i.s_index
            for i in FATable
            if (i.s_index is not None) and (i.index > 10)
        ]

        for act_type in ["primary", "secondary"]:
            for i in fa_sel_dict[act_type]["all"]:
                fa_info = FATable[i]
                fa_sel_dict[act_type]["fa_quadrant"].append(fa_info.quadrant)
                fa_sel_dict[act_type]["fa_orientation"].append(
                    fa_info.orientation
                )

            fa_idx = fa_sel_dict[act_type]["all"]
            following_error_frame = self.compute_following_error_summary_stats(
                following_error_frame, fa_idx, "all", act_type
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
            following_error_frame = self.compute_following_error_summary_stats(
                following_error_frame, fa_idx, f"quadrant_{quadrant}", act_type
            )
        fao_list = [
            FAOrientation(i).name
            for i in fa_sel_dict[act_type]["fa_orientation"]
            if i > 0
        ]
        for orientation in fao_list:
            fa_idx = [
                i
                for i, o in zip(
                    fa_sel_dict[act_type]["all"],
                    fa_sel_dict[act_type]["fa_orientation"],
                )
                if o == FAOrientation[orientation].value
            ]
            following_error_frame = self.compute_following_error_summary_stats(
                following_error_frame,
                fa_idx,
                f"orientation_{orientation}",
                act_type,
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
            cols = [
                f"{act_type}CylinderFollowingError" + str(i)
                for i in hp_idx_list
            ]
        if act_type == "secondary":
            cols = [
                f"{act_type}CylinderFollowingError" + str(i)
                for i in hp_idx_list
            ]

        following_error_frame[f"{col_key}_{act_type}_max_val"] = (
            following_error_frame[cols].max(axis=1).values
        )
        # min
        following_error_frame[f"{col_key}_{act_type}_min_val"] = (
            following_error_frame[cols].min(axis=1).values
        )
        # median
        following_error_frame[f"{col_key}_{act_type}_median_val"] = (
            following_error_frame[cols].median(axis=1).values
        )
        following_error_frame[f"{col_key}_{act_type}_std_val"] = (
            following_error_frame[cols].std(axis=1).values
        )
        # confidence interval
        following_error_frame[f"{col_key}_{act_type}_q1_val"] = (
            following_error_frame[cols].quantile(0.16, axis=1).values
        )
        following_error_frame[f"{col_key}_{act_type}_q3_val"] = (
            following_error_frame[cols].quantile(0.84, axis=1).values
        )
        # std
        following_error_frame[f"{act_type}_std_val"] = (
            following_error_frame.filter(
                like=f"{act_type}CylinderFollowingError", axis=1
            )
            .std(axis=1)
            .values
        )
        return following_error_frame

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
        queries = {
            key: self.query_efd_data(**cfg)
            for key, cfg in query_config.items()
        }
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


class SlewPlotter:
    def __init__(
        self,
        stats_frame,
        query_dict,
        query_key,
        day_obs,
        col_key,
        exclude_list=[],
        block="T227",
        block_info="",
        out_dir="./plots/",
    ):
        self.stats_frame = stats_frame
        self.query_dict = query_dict
        self.query_key = query_key
        self.day_obs = day_obs
        self.col_key = col_key
        self.exclude_list = exclude_list
        self.block = block
        self.block_info = block_info
        self.out_dir = out_dir

    def _generate_plot(
        self,
        time_selector,
        x_label,
        file_suffix,
        t_limit,
        plot_fa_following=False,
    ):
        plt.rcParams["axes.labelsize"] = 12
        fig, axs = plt.subplots(
            3, 3, dpi=125, figsize=(12, 10), sharex=True, sharey=True
        )
        axs = axs.flatten()
        max_val = 200

        for ax_val in range(9):
            ax = axs[ax_val]
            slew_sel = self.stats_frame["state"] == int(ax_map_dict[ax_val])
            for seq_num in self.stats_frame["seq_num"][slew_sel].values:
                if seq_num in self.exclude_list:
                    continue
                ydata = self.query_dict[seq_num][self.query_key][self.col_key]
                t0 = Time(
                    ydata.index[0]
                    if time_selector == "start"
                    else ydata.index[-1]
                )
                times = Time(ydata.index) - t0
                time_sel = (
                    times < t_limit * u.second
                    if time_selector == "start"
                    else times > t_limit * u.second
                )
                times = times[time_sel]
                ydata = ydata[time_sel]

                if plot_fa_following:
                    median = ydata
                    q1 = self.query_dict[seq_num][self.query_key][
                        self.col_key.replace("_median", "_q1")
                    ][time_sel]
                    q3 = self.query_dict[seq_num][self.query_key][
                        self.col_key.replace("_median", "_q3")
                    ][time_sel]
                    ax.plot(times.sec, median, label=f"{seq_num}")
                    ax.fill_between(times.sec, q1, q3, alpha=0.3)
                    max_val = max(
                        max_val,
                        np.max(abs(median)),
                        np.max(abs(q1)),
                        np.max(abs(q3)),
                    )
                else:
                    ax.plot(times.sec, ydata, label=f"{seq_num}")
                    max_val = max(max_val, np.max(abs(ydata)))

                ax.set(**ax_label_dict[ax_val])
            (
                ax.set_xlim(0, t_limit)
                if time_selector == "start"
                else ax.set_xlim(t_limit, 0)
            )

            ax.tick_params(direction="in")

            if ax_val == 4:
                ax.axis("off")
            else:
                ax.legend(facecolor="none", edgecolor="none", title="seq_num")
        for ax in axs:
            max_val = max_val * 1.1
            ax.set_ylim(-max_val, max_val)
        fig.text(
            0.5,
            0.04,
            x_label,
            ha="center",
            va="center",
            fontsize=16,
        )  # x-label
        fig.text(
            0.04,
            0.5,
            self.col_key,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=16,
        )  # y-label
        plt.suptitle(
            f"{self.col_key}\n{self.day_obs} - {file_suffix}\nBLOCK-{self.block}: {self.block_info} ",
            y=0.96,
        )
        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        plt.savefig(
            self.out_dir
            + f"{self.day_obs}_{self.col_key}_{self.block}_{file_suffix}.png"
        )
        plt.close()
        return fig

    def make_slew_start_plot(self, tmax=3.2):
        return self._generate_plot(
            "start", "Time after slew start [s]", "slew_start", tmax
        )

    def make_slew_stop_plot(self, tmin=-3.2):
        return self._generate_plot(
            "stop", "Time before slew stops [s]", "slew_stop", tmin
        )

    def make_fa_following_slew_start_plot(self, tmax=3.2):
        return self._generate_plot(
            "start",
            "Time after slew start [s]",
            "fa_following_slew_start",
            tmax,
            plot_fa_following=True,
        )

    def make_fa_following_slew_stop_plot(self, tmin=-3.2):
        return self._generate_plot(
            "stop",
            "Time before slew stops [s]",
            "fa_following_slew_stop",
            tmin,
            plot_fa_following=True,
        )


def compute_stats_frame(query_dict, day_obs, block, block_info):
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
            vals = query_dict[seq_num][f"tma_{axis}"][
                f"{axis}_actual_position"
            ].values
            stats_dict[f"{axis}_start"].append(vals[0])
            stats_dict[f"{axis}_end"].append(vals[-1])
            stats_dict[f"{axis}_distance"].append((vals[-1] - vals[0]))
        total_distance = np.sqrt(
            stats_dict["az_distance"][-1] ** 2
            + stats_dict["el_distance"][-1] ** 2
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
            query_dict[seq_num]["fa_following_errors"][
                "all_primary_max_val"
            ].max()
        )
        stats_dict["min_fa_following_error"].append(
            query_dict[seq_num]["fa_following_errors"][
                "all_primary_min_val"
            ].min()
        )
    stats_frame = pd.DataFrame(stats_dict)
    stats_frame["day_obs"] = day_obs
    stats_frame["block"] = block
    stats_frame["block_info"] = block_info.replace("%", "")
    stats_frame["state"] = stats_frame.apply(compute_state, axis=1)

    return stats_frame


async def main():
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
    parser = argparse.ArgumentParser(
        description="Load configuration and run analysis."
    )
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
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found."
        )

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

    hp_col_keys = [f"measuredForce{i}" for i in range(HP_COUNT)]
    hp_col_keys += [f"f{i}" for i in "xyz"]
    hp_col_keys += [f"m{i}" for i in "xyz"]

    fa_col_keys = []
    for act_type in ["primary", "secondary"]:
        fa_col_keys += [
            f"{col_key}_{act_type}_max_val" for col_key in ["all"]
        ]  # , "quadrant", "orientation"]]
        fa_col_keys += [
            f"{col_key}_{act_type}_median_val" for col_key in ["all"]
        ]
    event_maker = TMAEventMaker()
    events = event_maker.getEvents(day_obs)

    if begin_time is not None and end_time is not None:

        begin_time = Time(begin_time, format="iso")
        end_time = Time(end_time, format="iso")

        slews = [
            e
            for e in events
            if (e.begin.unix >= begin_time.unix)
            & (e.end.unix <= end_time.unix)
        ]
        print(
            (
                f"Using time range: {begin_time} to {end_time},"
                f"found {len(slews)} slews"
            )
        )
    else:

        slews = [
            e
            for e in events
            if (e.seqNum >= begin_seq_num) & (e.seqNum <= end_seq_num)
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

    print(
        f"Saved concatenated forces data to {hp_forces_csv_path} & fa_efd_frame.csv"
    )

    stats_frame = compute_stats_frame(query_dict, day_obs, block, block_info)
    stats_csv_path = os.path.join(
        data_dir,
        config_file.split("/")[-1].replace(".yaml", "_stats_frame.csv"),
    )
    stats_frame.to_csv(stats_csv_path)

    for col_key in hp_col_keys:
        query_key = "hp_measured_forces"
        sp = SlewPlotter(
            stats_frame=stats_frame,
            query_dict=query_dict,
            query_key=query_key,
            day_obs=day_obs,
            col_key=col_key,
            exclude_list=exclude_list,
            block=block,
            block_info=block_info,
            out_dir=out_dir,
        )
        sp.make_slew_start_plot(tmax=tmax)
        sp.make_slew_stop_plot(tmin=tmin)

    for col_key in fa_col_keys:
        query_key = "fa_following_errors"
        sp = SlewPlotter(
            stats_frame=stats_frame,
            query_dict=query_dict,
            query_key=query_key,
            day_obs=day_obs,
            col_key=col_key,
            exclude_list=exclude_list,
            block=block,
            block_info=block_info,
            out_dir=out_dir,
        )
        if "median" in col_key:
            sp.make_fa_following_slew_start_plot(tmax=tmax)
            sp.make_fa_following_slew_stop_plot(tmin=tmin)
        else:
            sp.make_slew_start_plot(tmax=tmax)
            sp.make_slew_stop_plot(tmin=tmin)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
