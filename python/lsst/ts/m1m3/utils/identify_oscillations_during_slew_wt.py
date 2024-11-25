"""
added as part of SITCOM-1759
example call python identify_oscillations_during_slew_wt.py  --day_obs 20241121
or python identify_oscillations_during_slew_wt.py
--begin_day_obs 20241101 --end_day_obs 20241124
"""

import argparse
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from astropy import units as u
from astropy.time import Time
from lsst.summit.utils.efdUtils import calcNextDay, getEfdData
from lsst.summit.utils.tmaUtils import TMAEvent, TMAEventMaker
from lsst.ts.xml.tables.m1m3 import HP_COUNT
from scipy.signal import find_peaks


HAS_EFD_CLIENT = True
try:
    from lsst_efd_client import EfdClient
except ImportError:
    EfdClient = None  # this is currently just for mypy
    HAS_EFD_CLIENT = False

__all__ = [
    "M1M3EFDQuery",
    "M1M3IdentifyOscillations",
]


class M1M3EFDQuery:
    """
    Query M1M3 telemetry data from the EFD and process it.

    This class provides methods for retrieving and analyzing telemetry data
    related to the M1M3 Inertia Compensation System during a slew event.

    Attributes:
        event (TMAEvent): Representation of a slew event.
        outer_pad (float): Time padding around the slew event in seconds.
        client (EfdClient): EFD client for data retrieval.
        number_of_hardpoints (int): Number of hardpoints in the system.
        measured_forces_topics (list): Topics for measured forces data.
        applied_forces_topics (list): Topics for applied forces data.
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

    def query_dataset(self) -> pd.DataFrame:
        """
        Query and all relevant telemetry data.

        Returns:
            dict of dataframes containing telemetry data.
        """
        evt = self.event
        query_config = {
            "hp_measured_forces": {
                "topic": "lsst.sal.MTM1M3.hardpointActuatorData",
                "columns": self.measured_forces_topics,
                "err_msg": (
                    "No hard-point data found for event"
                    f"{evt.seqNum} on {evt.dayObs}"
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
                    f" {evt.seqNum} on {evt.dayObs}"
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
        queries = {
            key: self.query_efd_data(**cfg)
            for key, cfg in query_config.items()
        }  # type: ignore
        queries["slew"] = self.event
        # Merge datasets
        # df = self.merge_datasets(queries)

        # Convert torque from Nm to kNm
        # cols = ["az_actual_torque", "el_actual_torque"]
        # df.loc[:, cols] *= 1e-3

        return queries

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
        Query telemetry data from the EFD for a given topic.

        Parameters:
            topic (str): Topic name to query.
            columns (list[str]): Columns to retrieve from the topic.
            err_msg (str, optional): Error message for missing data.
            reset_index (bool, optional): Whether to reset the dataframe index.
            rename_columns (dict, optional): Column renaming mapping.
            resample (float, optional): Resampling frequency in seconds.

        Returns:
            pd.DataFrame: Dataframe containing the queried data.
        """
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
        if df.index.size == 0:
            # no data return an empty dataframe
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


class M1M3IdentifyOscillations:
    """
    Identify oscillations in M1M3 telemetry data using wavelet transforms.

    Attributes:
        day_obs (int): Observation day identifier (YYYYMMDD).
        eventMaker (TMAEventMaker): Event maker instance.
        events (list[TMAEvent]): List of telemetry events.
        slews (list[TMAEvent]): List of slewing events.
        peak_height (int): Minimum height of detected peaks.
        save_results (bool): Whether to save results as CSV.
    """

    def __init__(self, day_obs, peak_height=3000, save_results=True) -> None:
        self.day_obs = day_obs
        self.eventMaker = TMAEventMaker()
        self.events = self.eventMaker.getEvents(self.day_obs)
        self.slews = [e for e in self.events]
        self.peak_height = peak_height
        self.save_results = save_results

    def run(self):
        """
        Run oscillation analysis on all slews for the observation day.

        Returns:
            pd.DataFrame | None: Dataframe of detected peaks, or
            None if no peaks are found.
        """

        peaks_df_list = []
        for slew in self.slews:
            query_result = M1M3EFDQuery(
                slew, self.eventMaker.client, outer_pad=0
            ).query_dataset()
            peaks_df = self.run_single_slew(
                slew.seqNum,
                slew.dayObs,
                query_result=query_result,
                peak_height=self.peak_height,
            )

            if peaks_df is not None:
                peaks_df = peaks_df[peaks_df["count"] >= 2]
                if len(peaks_df) > 0:
                    print(
                        slew.dayObs,
                        slew.seqNum,
                        f"found {len(peaks_df)} peaks",
                    )
                    peaks_df = self.add_telemetry(peaks_df, query_result)
                    peaks_df_list.append(peaks_df)
            del query_result
        if len(peaks_df_list) == 0:
            print(f"No peaks found for {self.day_obs}")
            return None
        peaks_df = pd.concat(peaks_df_list, ignore_index=True)
        if self.save_results:
            peaks_df.to_csv(f"peaks_frame_{slew.dayObs}.csv")

        return peaks_df

    def identify_peaks_in_wt(
        self, coeffs, freqs, times, data, peak_height=1000, time_window=1
    ):
        """
        Identify peaks in the wavelet transform of telemetry data.

        Parameters:
            coeffs (np.ndarray): Wavelet coefficients.
            freqs (np.ndarray): Frequencies corresponding to the coefficients.
            times (np.ndarray): Time indices for the data.
            data (np.ndarray): Original telemetry data.
            peak_height (int): Minimum height for peak detection.
            time_window (int): Time tolerance for grouping peaks (seconds).

        Returns:
            pd.DataFrame | None: Dataframe of detected peaks,
            or None if no peaks are found.
        """

        # Compute Wavelet Power Spectrum
        power = (
            np.abs(coeffs) ** 2
        )  # Power is the square of the wavelet coefficients

        # Detect peaks in the wavelet power spectrum
        peaks_time = []
        for freq_idx in range(len(freqs)):
            peaks, properties = find_peaks(
                power[freq_idx], height=peak_height
            )  # Adjust `height` for sensitivity
            for peak in peaks:
                peaks_time.append(
                    (
                        times[peak],
                        freqs[freq_idx],
                        properties["peak_heights"][
                            np.where(peaks == peak)[0][0]
                        ],
                    )
                )
        if len(peaks_time) == 0:
            return None
        # Convert peaks to a DataFrame for easier processing
        peaks_df = pd.DataFrame(
            peaks_time, columns=["time", "frequency", "power"]
        )

        # Sort by time and group nearby peaks
        time_tolerance = timedelta(
            seconds=time_window
        )  # Tolerance for grouping peaks in seconds
        peaks_df = peaks_df.sort_values("time")
        grouped_peaks = []
        current_group = [peaks_df.iloc[0]]
        for i in range(1, len(peaks_df)):
            if (
                peaks_df.iloc[i]["time"] - current_group[-1]["time"]
                <= time_tolerance
            ):
                current_group.append(peaks_df.iloc[i])
            else:
                # Keep only the peak with the maximum power in the group
                max_peak = max(current_group, key=lambda x: x["power"])
                grouped_peaks.append(max_peak)
                current_group = [peaks_df.iloc[i]]

        # Add the last group
        if current_group:
            max_peak = max(current_group, key=lambda x: x["power"])
            grouped_peaks.append(max_peak)

        # Convert grouped peaks back to a DataFrame
        grouped_peaks_df = pd.DataFrame(grouped_peaks)
        # remove peaks where data is near zero before
        # and near +/- 3k after (breakaway)
        grouped_peaks_df = self.remove_breakaway(
            grouped_peaks_df, data, time_window=3
        )
        if len(grouped_peaks_df) == 0:
            return None
        return grouped_peaks_df

    def group_across_hp(self, peaks_df, time_window=1):
        """
        Group peaks across nearby times and retain the one with the
        maximum power.
        Additionally, count the number of peaks grouped together.

        Parameters:
        - peaks_df: DataFrame containing 'time' and 'power' columns.
        - time_window: Time tolerance (in seconds) for grouping peaks.

        Returns:
        - grouped_peaks_df: DataFrame with grouped peaks and
        counts of grouped occurrences.
        """
        time_tolerance = timedelta(
            seconds=time_window
        )  # Tolerance for grouping peaks in seconds
        peaks_df = peaks_df.sort_values("time")
        grouped_peaks = []
        current_group = [peaks_df.iloc[0]]

        for i in range(1, len(peaks_df)):
            if (
                peaks_df.iloc[i]["time"] - current_group[-1]["time"]
                <= time_tolerance
            ):
                current_group.append(peaks_df.iloc[i])
            else:
                current_group_df = pd.DataFrame(current_group)
                max_peak = current_group_df.loc[
                    current_group_df["power"].idxmax()
                ].copy()
                max_peak["count"] = len(
                    current_group
                )  # Count of peaks in the group
                grouped_peaks.append(max_peak)
                current_group = [peaks_df.iloc[i]]

        # Add the last group
        if current_group:
            current_group_df = pd.DataFrame(current_group)
            max_peak = current_group_df.loc[
                current_group_df["power"].idxmax()
            ].copy()
            max_peak["count"] = len(
                current_group
            )  # Count of peaks in the last group
            grouped_peaks.append(max_peak)

        # Convert grouped peaks back to a DataFrame
        grouped_peaks_df = pd.DataFrame(grouped_peaks)
        return grouped_peaks_df

    def run_single_slew(
        self,
        seq_num,
        day_obs,
        query_result,
        peak_height=1000,
        show_plots=False,
    ):
        """
        Analyze oscillations for a single slew.

        Parameters:
            seq_num (int): Sequence number of the slew.
            day_obs (int): Observation day identifier.
            query_result (dict): Retrieved telemetry data for the slew.
            peak_height (int): Minimum height for peak detection.
            show_plots (bool): Whether to show plots of detected peaks.

        Returns:
            pd.DataFrame | None: Dataframe of detected peaks,
            or None if no peaks are found.
        """
        peaks_df_list = []
        for hp in range(HP_COUNT):
            data = query_result["hp_measured_forces"][f"measuredForce{hp}"]
            times = query_result["hp_measured_forces"].index.values
            coeffs, freqs = self.compute_wt(data)
            peaks_df = self.identify_peaks_in_wt(
                coeffs, freqs, times, data, peak_height=peak_height
            )
            if peaks_df is None:
                continue
            if show_plots:
                self.plot_results_wt(
                    coeffs, times, freqs, peaks_df, hp, seq_num, day_obs
                )
            peaks_df["seq_num"] = seq_num
            peaks_df["max_power_hp_num"] = hp
            peaks_df["day_obs"] = day_obs
            peaks_df_list.append(peaks_df)

        if len(peaks_df_list) == 0 | (np.all(peaks_df_list is None)):
            return None

        peaks_df = pd.concat(peaks_df_list, ignore_index=True)

        peaks_df = self.group_across_hp(peaks_df)

        return peaks_df

    def add_telemetry(self, peaks_df, query_result):
        """
        Add telemetry data to the detected peaks dataframe.

        Parameters:
            peaks_df (pd.DataFrame): Dataframe of detected peaks.
            query_result (dict): Retrieved telemetry data for the slew.

        Returns:
            pd.DataFrame: Dataframe with added telemetry information.
        """
        cols = []
        for mt_ax in ["az", "el"]:
            cols += [
                f"{mt_ax}_actual_position",
                f"{mt_ax}_actual_velocity",
                f"{mt_ax}_actual_torque",
            ]
        telem_dict = {key: [] for key in cols}

        for i, row in peaks_df.iterrows():
            t0 = Time(row["time"], scale="utc").datetime64
            for mt_ax in ["az", "el"]:
                dat = query_result[f"tma_{mt_ax}"]
                idxmin = np.argmin(abs(dat.index.values - t0))
                for key in [
                    f"{mt_ax}_actual_position",
                    f"{mt_ax}_actual_velocity",
                    f"{mt_ax}_actual_torque",
                ]:
                    telem_dict[key].append(dat[key].iloc[idxmin])
        telem_df = pd.DataFrame(telem_dict)
        peaks_df = pd.concat(
            [peaks_df.reset_index(drop=True), telem_df.reset_index(drop=True)],
            axis=1,
        )
        return peaks_df

    def remove_breakaway(self, peaks_df, data, time_window=3):
        """
        Remove breakaway peaks from the dataset.

        Parameters:
            peaks_df (pd.DataFrame): Dataframe of detected peaks.
            data (pd.Series): Original telemetry data.
            time_window (int): Time window for breakaway detection (seconds).

        Returns:
            pd.DataFrame: Filtered dataframe without breakaway peaks.
        """
        # returns false if there is a breakaway
        time_window = timedelta(seconds=time_window)
        breakaway_sel = []
        for peak_time in peaks_df["time"]:
            sel_before = (data.index.values < peak_time) & (
                data.index.values > peak_time - time_window
            )
            sel_after = (data.index.values > peak_time) & (
                data.index.values < peak_time + time_window
            )
            median_before = np.median(data[sel_before])
            median_after = np.median(data[sel_after])
            if (abs(median_before) < 10) & (abs(median_after) > 2000):
                breakaway_sel.append(False)
                print(f"breakaway detected removing {peak_time}")
            else:
                breakaway_sel.append(True)

        return peaks_df[breakaway_sel].reset_index(drop=True)

    def plot_results_wt(
        self, coeffs, times, freqs, grouped_peaks_df, hp, seq_num, day_obs
    ):
        """
        Plot wavelet transform results and detected peaks.

        Parameters:
            coeffs (np.ndarray): Wavelet coefficients.
            times (np.ndarray): Time indices for the data.
            freqs (np.ndarray): Frequencies corresponding to the coefficients.
            grouped_peaks_df (pd.DataFrame): Dataframe of grouped peaks.
            hp (int): Hardpoint identifier.
            seq_num (int): Sequence number of the slew.
            day_obs (int): Observation day identifier.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        power = np.abs(coeffs) ** 2
        plt.imshow(
            power,
            extent=[times.min(), times.max(), freqs.max(), freqs.min()],
            aspect="auto",
            cmap="viridis",
            vmax=1000,
        )
        plt.colorbar(label="Wavelet Power")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title(f"measuredForce{hp}, seq_num:{seq_num}, day_obs:{day_obs}")

        # Plot detected peaks (grouped)
        plt.scatter(
            grouped_peaks_df["time"],
            grouped_peaks_df["frequency"],
            color="red",
            s=20,
            label="Grouped Peaks",
        )

        plt.legend()
        plt.show()

    @staticmethod
    def compute_wt(
        data,
        sampling_period=0.02,
        frequency_scales=np.arange(5, 20, 0.2),
        wavelet="cmor1.5-1.0",
    ):
        """
        Compute the wavelet transform of the telemetry data.

        Parameters:
            data (np.ndarray): Input telemetry data.
            sampling_period (float): Sampling period in seconds.
            frequency_scales (np.ndarray): Frequency scales for the
            wavelet transform.
            wavelet (str): Wavelet type.

        Returns:
            tuple[np.ndarray, np.ndarray]: Wavelet coefficients
            and corresponding frequencies.
        """

        scales = pywt.frequency2scale(
            wavelet, frequency_scales * sampling_period
        )
        coeffs, freqs = pywt.cwt(
            data,
            scales=scales,
            wavelet="cmor1.5-1.0",
            sampling_period=sampling_period,
        )
        return coeffs, freqs


def main():
    """
    Main function to analyze M1M3 slew telemetry.

    This script identifies oscillations in the M1M3 system's telemetry data
    for a given observation day or a range of days. It outputs results as
    a CSV file.

    Command-line Arguments:
    --day_obs (int): Single observation day to process (format: YYYYMMDD).
    --begin_day_obs (int): Start of observation day range (format: YYYYMMDD).
    --end_day_obs (int): End of observation day range (format: YYYYMMDD).
    --peak_height (int): Minimum peak height for detection (default: 3000).
    --force (bool): Force reprocessing of data, even if results exist.

    Raises:
        RuntimeError: If the EFD client is not available.
        ValueError: If invalid arguments are provided.
    """

    parser = argparse.ArgumentParser(
        description="Analyze M1M3 slew telemetry."
    )
    parser.add_argument(
        "--day_obs", type=int, help="Single day to process (YYYYMMDD)."
    )
    parser.add_argument(
        "--begin_day_obs", type=int, help="begin of day_obs range (YYYYMMDD)."
    )
    parser.add_argument(
        "--end_day_obs", type=int, help="End of day_obs range (YYYYMMDD)."
    )
    parser.add_argument(
        "--peak_height",
        type=int,
        default=3000,
        help="Minimum peak height for detection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of data.",
        default=False,
    )
    args = parser.parse_args()

    if not EfdClient:
        raise RuntimeError("EFD client is not available.")

    if args.day_obs and not args.begin_day_obs and not args.end_day_obs:
        day_obs_list = [args.day_obs]
    elif args.begin_day_obs and args.end_day_obs:
        current_day_obs = args.begin_day_obs
        day_obs_list = []
        while int(current_day_obs) <= int(args.end_day_obs):
            day_obs_list.append(current_day_obs)
            current_day_obs = calcNextDay(current_day_obs)
    else:
        raise ValueError(
            (
                "Either --day_obs or --begin_day_obs"
                "and --end_day_obs must be provided."
            )
        )

    for day_obs in day_obs_list:
        if os.path.exists(f"peaks_frame_{day_obs}.csv") and not args.force:
            print(f"result found peaks_frame_{day_obs}.csv skipping {day_obs}")
            continue
        analysis = M1M3IdentifyOscillations(day_obs, args.peak_height)
        analysis.run()


if __name__ == "__main__":
    main()
