# This file is part of ts_m1m3_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["AccelerationAndVelocity"]

import argparse
import asyncio
import logging
import os
import pathlib

import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.ts.xml.enums.MTM1M3 import DetailedStates
from lsst.ts.xml.tables.m1m3 import FATABLE_XFA, FATABLE_YFA, FATABLE_ZFA
from lsst_efd_client import EfdClient
from tqdm import tqdm

from .acceleration_and_velocity_fitter import AccelerationAndVelocityFitter
from .force_calculator import ForceCalculator

tqdm.pandas()

RAD2D = np.degrees(1)


class AccelerationAndVelocity:
    """Compute fit of the M1M3 accelerations and velocities forces.

    Parameters
    ----------
    config : pathlib.Path
        Path to configuration files.
    load_from : pathlib.Path
        Optional path to load data (from hd5 file). If None, data are
        collected/computed.

    Attributes
    ----------
    azimuths : pd.DataFrame
        Mount azimuth. Contains actual and demand Position, Velocity and
        Acceleration.
    accelerometers : pd.DataFrame
        DC Accelerometers angular acceleration values.
    calculated_fam : pd.DataFrame
        Forces and moments calculated from the fit.
    coefficients : pd.DataFrame
        Coefficients for velocity and acceleration transformation.
    detailed_states : pd.DataFrame
        Mirror detailed states before and during selected time interval.
    elevations : pd.DataFrame
        Mount elevation. Contains actual and demand Position, Velocity and
        Acceleration.
    fitter : AccelerationAndVelocityFitter
        Fitter doing the actual work - fitting data.
    interpolated : pd.DataFrame
        Interpolated raw data.
    intervals : pd.DataFrame
        Contains start and end times of intervals suitable for fitting.
    mirror : pd.DataFrame
        Mirror forces. Contains interpolated data, with extra X0..11, Y0..99
        and Z0..155 columns - those contain data derived from hardpoint forces
        (f[xyz], m[xyz]). Those are the per-mirror (XYZ) forces the algorithm
        needs to counter-act.
    raw : pd.DataFrame
        Raw data. Concatenation of azimuths, elevations and hardpoint forces.
        Contains null/NaNs, as the three tables are merged together.
    residuals : pd.DataFrame
        Residuals of six forces/components.
    """

    def __init__(
        self, efd_name: str, config: pathlib.Path, load_from: pathlib.Path | None
    ):
        self.efd_name = efd_name
        self.force_calculator = ForceCalculator(config)

        self.actual_velocity_limit = 0.01
        self.demand_velocity_limit = 0.01

        self.azimuths: pd.DataFrame | None = None
        self.accelerometers: pd.DataFrame | None = None
        self.calculated_fam: pd.DataFrame | None = None
        self.coefficients: pd.DataFrame | None = None
        self.detailed_states: pd.DataFrame | None = None
        self.elevations: pd.DataFrame | None = None
        self.fitter: AccelerationAndVelocityFitter | None = None
        self.gyroscope: pd.DataFrame | None = None
        self.interpolated: pd.DataFrame | None = None
        self.intervals: pd.DataFrame | None = None
        self.mirror: pd.DataFrame | None = None
        self.raw: pd.DataFrame | None = None
        self.residuals: pd.DataFrame | None = None

        if load_from is not None:

            def try_load(name: str) -> pd.DataFrame | None:
                try:
                    logging.debug(f"Loading {name}...")
                    return pd.read_hdf(load_from, name)
                except KeyError as ke:
                    logging.warning(f"Cannot load {name} from {load_from}: {ke}")
                    return None

            for loads in [
                "azimuths",
                "gyroscope",
                "accelerometers",
                "calculated_fam",
                "coefficients",
                "elevations",
                "intervals",
                "raw",
                "mirror",
                "residuals",
            ]:
                setattr(self, loads, try_load(loads))

    async def load_detailed_state(self, start_time: Time, end_time: Time) -> None:
        """Find M1M3's detailedState events. The last event entry pre-dating
        start_time, and all events in start_time till end_time are returned.
        Those are primary used to verify mirror wasn't lowered during slew -
        was either in ACTIVE or ACTIVEENGINEERING detailedState.

        Fills self.detailed_states.

        Parameters
        ----------
        start_time: Time
            Time to look from.
        end_time: Time
            Time to look till.
        """

        # find last state before start_time
        logging.info(f"Retrieve last detailedState before {start_time.isot}.")
        query = (
            "SELECT detailedState "
            'FROM "efd"."autogen"."lsst.sal.MTM1M3.logevent_detailedState" '
            f"WHERE time <= '{start_time.isot}+00:00' ORDER BY time DESC LIMIT 1"
        )
        ret = await self.client.influx_client.query(query)
        detailed_states = await self.client.select_time_series(
            "lsst.sal.MTM1M3.logevent_detailedState",
            "detailedState",
            start_time,
            end_time,
        )
        self.detailed_states = pd.concat([ret, detailed_states])
        self.detailed_states.set_index(
            pd.DatetimeIndex(self.detailed_states.index), inplace=True
        )

    def was_raised(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Return whenever mirror ws raised (in either ACTIVE or
        ACTIVEENGINEERING states) from start till end parameters.

        Parameters
        ----------
        start : Time
            Starting time of interval to query for mirror state.
        end : Time
            End time of interval to query for mirror state.

        Returns
        -------
        raised : bool
            True if mirror was raised (in ACTIVE or ACTIVEENGINEERING
            detailedState) from start till end.
        """
        assert self.detailed_states is not None
        last_state = self.detailed_states[
            self.detailed_states.index < pd.to_datetime(start, utc=True)
        ]["detailedState"].iloc[-1]
        if DetailedStates(last_state) not in (
            DetailedStates.ACTIVE,
            DetailedStates.ACTIVEENGINEERING,
        ):
            return False
        states = self.detailed_states[
            (Time(self.detailed_states.index) >= start)
            & (Time(self.detailed_states.index) <= end)
        ]
        return (
            len(
                states[
                    ~states.detailedState.isin(
                        [
                            DetailedStates.ACTIVE,
                            DetailedStates.ACTIVEENGINEERING,
                        ]
                    )
                ].index
            )
            == 0
        )

    async def find_az_el(self, start: Time, end: Time) -> None:
        """Find times when elevations and azimuth MTMount axis moved. Store
        elevations and azimuth to elevations and azimuths attributes.

        Parameters
        ----------
        start : Time
            Start time. Azimuths and elevations data will be available from
            that time.
        end : Time
            End time. Azimuths and elevations data will be available till that
            time.
        """

        async def find_axis(axis: str) -> pd.DataFrame:
            """Retrieve axis data.

            Parameters
            ----------
            axis : `str`
                Axis to retrieve. Either elevation or azimuth.

            Returns
            -------
            ret : pd.DataFrame
                DataFrame with axis data. demand and actual Accelerations are
                calculated as difference in speed from the last datapoint,
                divided by interval between the two measurements.
            """
            logging.info(f"Querying EFD for {axis} data")
            ret = await self.client.select_time_series(
                f"lsst.sal.MTMount.{axis}",
                [
                    "timestamp",
                    "actualAcceleration",
                    "demandPosition",
                    "actualPosition",
                    "demandVelocity",
                    "actualVelocity",
                ],
                start,
                end,
            )

            ret.set_index(
                pd.DatetimeIndex(
                    Time(Time(ret["timestamp"], format="unix_tai"), scale="utc").isot
                ),
                inplace=True,
            )
            ret["timediff"] = ret["timestamp"].diff()

            # calculate derivatives - demand acceleration
            den = ret["timediff"]
            ret["demandAcceleration"] = ret["demandVelocity"].diff().div(den, axis=0)
            logging.info(
                f"Retrieved {axis} {len(ret.index)} rows. "
                f"Ranges: {ret.demandPosition.min():+.2f} .. "
                f"{ret.demandPosition.max():+.2f}"
                f", {ret.actualPosition.min():+.2f} .. "
                f"{ret.actualPosition.max():+.2f}"
            )
            return ret

        self.elevations = await find_axis("elevation")
        self.azimuths = await find_axis("azimuth")

    async def find_applicable_times(self) -> None:
        """Find start and end times of any axis movents. Uses self.elevations
        and self.azimuths data retrieved in find_az_el method.

        Fills self.intervals with start and stop times useful for fitting.
        """
        assert self.azimuths is not None
        assert self.elevations is not None

        ret: list[tuple[float, float]] = []

        # timestamps of all measurements when TMA was moving
        # add last row from elevations - it will not show in results, but is
        # needed to show last movement in results
        movements = pd.concat(
            [
                self.elevations[
                    (abs(self.elevations.actualVelocity) > self.actual_velocity_limit)
                    | (abs(self.elevations.demandVelocity) > self.demand_velocity_limit)
                ],
                self.azimuths[
                    (abs(self.azimuths.actualVelocity) > self.actual_velocity_limit)
                    | (abs(self.azimuths.demandVelocity) > self.demand_velocity_limit)
                ],
                self.elevations.tail(1),
            ]
        )

        movements.sort_index(inplace=True)
        movements["timediff"] = movements.index.to_series().diff()

        start = movements.index[0]
        end = None
        for index, row in movements[
            (movements.timediff > np.timedelta64(1, "s")) | (movements.timediff is None)
        ].iterrows():
            end = index - row["timediff"] + np.timedelta64(500, "ms")
            if (end - start) > pd.Timedelta(seconds=0.75):
                if self.was_raised(start, end):
                    logging.debug(f"Found interval {start} - {end}")
                    ret.append((start, end))
                else:
                    logging.warning(
                        f"Interval {start} - {end} " "mirror was not raised, ignoring."
                    )
            else:
                logging.warning(f"Short slew? {start} {end} {end - start}")
            start = index

        # if no movement, use full range - this is for short times useful for
        # tests
        if len(ret) == 0:
            logging.warning("No movements detected, using full range")
            assert isinstance(movements.index[0], float)
            assert isinstance(movements.index[-1], float)
            ret = [(movements.index[0], movements.index[-1])]

        self.intervals = pd.DataFrame(ret, columns=["start", "end"])
        self.intervals.index = self.intervals["start"]

    def calculate_forces_and_moments(self) -> None:
        """Calculate forces and moments from velocity and accelerations.

        Fills self.calculated_fam with calculated forces and moments.

        """
        assert self.mirror is not None
        assert self.fitter is not None

        def vect_to_fam(row: pd.Series) -> pd.Series:
            applied = self.force_calculator.velocity(
                row.values[:3]
            ) + self.force_calculator.acceleration(row.values[3:])

            return pd.Series(
                [
                    applied.fx,
                    applied.fy,
                    applied.fz,
                    applied.mx,
                    applied.my,
                    applied.mz,
                    applied.forceMagnitude,
                ],
                index=[f"calculated_f{a}" for a in "xyz"]
                + [f"calculated_m{a}" for a in "xyz"]
                + ["forceMagnitude"],
            )

        acceleration_and_velocity = pd.concat(
            [self.fitter.velocities, self.fitter.accelerations], axis=1
        )

        logging.info("Calculating new mirror forces - backpropagation")
        self.calculated_fam = acceleration_and_velocity.progress_apply(
            vect_to_fam, axis=1
        )
        self.calculated_fam.set_index(self.fitter.aav.index, inplace=True)

    async def load_gyroscope(self, start: Time, end: Time) -> None | pd.DataFrame:
        """Load Gyroscope data in given interval. Those are used for velocities
        corrections.

        Parameters
        ----------
        start : Time
            Search interval start time.
        end : Time
            Search interval end time.

        Returns
        -------
        ret : pd.DataFrame
            Time indexed dataframe with measured gyroscope angular velocity
            values (angularVelocity[XYZ]).
        """
        logging.debug(f"Retrieving gyroscope data for {start.isot} - {end.isot}..")
        ret = await self.client.select_time_series(
            "lsst.sal.MTM1M3.gyroData",
            ["timestamp"] + [f"angularVelocity{a}" for a in "XYZ"],
            start,
            end,
        )
        if ret.empty:
            logging.debug("empty, ignored")
            return None
        ret.set_index(
            pd.DatetimeIndex(
                Time(Time(ret["timestamp"], format="unix_tai"), scale="utc").isot
            ),
            inplace=True,
        )
        logging.debug(f"..OK ({len(ret.index)} records)")
        return ret

    async def collect_gyroscope_data(self) -> None:
        """
        Collect gyroscope data for intervals specified in self.intervals
        DataFrame.

        Fills self.gyroscope DataFrame.
        """
        assert self.intervals is not None

        self.gyroscope = pd.DataFrame()
        for index, row in self.intervals.iterrows():
            block_start = row["start"]
            block_end = row["end"]
            velocities = await self.load_gyroscope(Time(block_start), Time(block_end))
            self.gyroscope = pd.concat([self.gyroscope, velocities])

    async def load_accelerometers(self, start: Time, end: Time) -> None | pd.DataFrame:
        """Load DC accelerometers data in given interval. Those are used for
        acceleration corrections.

        Parameters
        ----------
        start : Time
            Search interval start time.
        end : Time
            Search interval end time.

        Returns
        -------
        ret : pd.DataFrame
            Time indexed dataframe with measured DC accelerometers values
            (angularAcceleration[XYZ]).
        """
        logging.debug(f"Retrieving accelerometer data for {start.isot} - {end.isot}..")
        ret = await self.client.select_time_series(
            "lsst.sal.MTM1M3.accelerometerData",
            ["timestamp"]
            + [f"rawAccelerometer{acc}" for acc in range(8)]
            + [f"accelerometer{acc}" for acc in range(8)]
            + [f"angularAcceleration{a}" for a in "XYZ"],
            start,
            end,
        )
        if ret.empty:
            logging.debug("empty, ignored")
            return None
        ret.set_index(
            pd.DatetimeIndex(
                Time(Time(ret["timestamp"], format="unix_tai"), scale="utc").isot
            ),
            inplace=True,
        )
        logging.debug(f"..OK ({len(ret.index)} records)")
        return ret

    async def collect_accelerometers_data(self) -> None:
        """
        Collect DC accelerometers data for intervals specified in
        self.intervals DataFrame.

        Fills self.accelerometers DataFrame.
        """
        assert self.intervals is not None

        self.accelerometers = pd.DataFrame()
        for index, row in self.intervals.iterrows():
            block_start = row["start"]
            block_end = row["end"]
            accels = await self.load_accelerometers(Time(block_start), Time(block_end))
            self.accelerometers = pd.concat([self.accelerometers, accels])

    async def load_hardpoints(self, start: Time, end: Time) -> None | pd.DataFrame:
        """Load hardpoint forces and moments in given interval. Those are used
        to calculate (through standard distribution matrices) per-actuator
        offset forces. Those are target values for fitting of the acceleration
        and velocity forces.

        Parameters
        ----------
        start : Time
            Search interval start time.
        end : Time
            Search interval end time.

        Returns
        -------
        ret : pd.DataFrame
            Time indexed dataframe with measured hardpoint forces (f[xyz]) and
            moments (m[xyz]).
        """
        logging.debug(f"Retrieving HP data for {start.isot} - {end.isot}..")
        ret = await self.client.select_time_series(
            "lsst.sal.MTM1M3.hardpointActuatorData",
            ["timestamp"]
            + [f"measuredForce{hp}" for hp in range(6)]
            + [f"f{a}" for a in "xyz"]
            + [f"m{a}" for a in "xyz"],
            start,
            end,
        )
        if ret.empty:
            logging.debug("empty, ignored")
            return None
        ret.set_index(
            pd.DatetimeIndex(
                Time(Time(ret["timestamp"], format="unix_tai"), scale="utc").isot
            ),
            inplace=True,
        )
        logging.debug(f"..OK ({len(ret.index)} records)")

        ret.drop(columns="timestamp", inplace=True)
        return ret

    async def collect_hardpoint_data(self) -> None:
        """Collect hardpoint forces for intervals specified in self.intervals
        DataFrame.

        Fills self.raw DataFrame.
        """
        assert self.azimuths is not None
        assert self.elevations is not None
        assert self.intervals is not None

        self.raw = pd.DataFrame()
        for index, row in self.intervals.iterrows():
            block_start = row["start"]
            block_end = row["end"]
            hardpoints = await self.load_hardpoints(Time(block_start), Time(block_end))
            if hardpoints is None:
                continue
            elevations_hardpoints = self.elevations[
                (self.elevations.index >= block_start)
                & (self.elevations.index <= block_end)
            ]
            azimuths_hardpoints = self.azimuths[
                (self.azimuths.index >= block_start)
                & (self.azimuths.index <= block_end)
            ]
            data = hardpoints.join(
                [azimuths_hardpoints, elevations_hardpoints], how="outer", sort=True
            )

            self.raw = pd.concat([self.raw, data])

    def mirror_forces_and_moments(self) -> None:
        """Calculate mirror forces from hardpoint forces and moments.
        Hardpoint values (retrieved in load_hardpoints) are distributed to
        mirror forces. Mirror forces are fit target values.

        Fills self.mirror DataFrame.
        """
        assert self.interpolated is not None

        fa_names = (
            [f"X{x}" for x in range(FATABLE_XFA)]
            + [f"Y{y}" for y in range(FATABLE_YFA)]
            + [f"Z{z}" for z in range(FATABLE_ZFA)]
        )

        def fa_forces(row: pd.Series) -> pd.Series:
            forces = self.force_calculator.forces_and_moments_forces(
                [row["fx"], row["fy"], row["fz"], row["mx"], row["my"], row["mz"]]
            )
            return pd.Series(
                np.concatenate((forces.xForces, forces.yForces, forces.zForces)),
                index=fa_names,
            )

        logging.info("Calculating mirror forces")
        applied = self.interpolated.progress_apply(fa_forces, axis=1)
        applied.set_index(self.interpolated.index, inplace=True)

        self.mirror = self.interpolated.merge(
            applied, how="left", left_index=True, right_index=True
        )
        self.mirror.dropna(inplace=True)
        logging.info(f"Processing {len(self.mirror.index)} records")

    def plot_acc_forces(self) -> None:
        """Plot together velocity, acceleration and calculated forces."""
        assert self.calculated_fam is not None
        assert self.fitter is not None
        assert self.mirror is not None
        assert self.raw is not None

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True)

        for r, axis in enumerate(["elevation", "azimuth"]):
            self.mirror.plot(
                y=[
                    f"{axis}_demandVelocity",
                    f"{axis}_actualVelocity",
                    f"{axis}_demandAcceleration",
                    f"{axis}_actualAcceleration",
                ],
                ax=axes[r, 0],
                style=".-",
            )

        for axis in "xyz":
            self.fitter.aav[f"A_{axis}"].mul(RAD2D).plot(
                ax=axes[2, 0],
                style=".-",
            )
            self.fitter.aav[f"V_{axis}2"].mul(RAD2D).plot(
                ax=axes[2, 0],
                style=".-",
            )

        for r, m_ax in enumerate("xyz"):
            axis = "elevation" if m_ax == "x" else "azimuth"

            self.raw.plot(
                y=f"f{m_ax}",
                ax=axes[r, 1],
                style=".-",
                secondary_y=True,
            )
            self.raw.plot(
                y=f"m{m_ax}",
                ax=axes[r, 2],
                style=".-",
                secondary_y=True,
            )

            self.calculated_fam.plot(
                y=f"calculated_f{m_ax}",
                ax=axes[r, 1],
                style=".-",
                secondary_y=True,
            )
            self.calculated_fam.plot(
                y=f"calculated_m{m_ax}",
                ax=axes[r, 2],
                style=".-",
                secondary_y=True,
            )

        plt.show()

    def plot_axes(self) -> None:
        """Plot axes (azimuth and elevation) data."""
        assert self.mirror is not None

        import matplotlib.pyplot as plt

        for ax in ["azimuth", "elevation"]:
            fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
            self.mirror.plot(
                y=[f"{ax}_demandPosition", f"{ax}_actualPosition"],
                ax=axes[0],
                style=".-",
            )
            self.mirror.plot(
                y=[f"{ax}_demandVelocity", f"{ax}_actualVelocity"],
                ax=axes[1],
                style=".-",
            )
            self.mirror.plot(
                y=[f"{ax}_demandAcceleration", f"{ax}_actualAcceleration"],
                ax=axes[2],
                style=".-",
            )

            plt.show()

    def plot_acceleration_residuals(self) -> None:
        """Plot acceleration and fitted residuals."""
        assert self.fitter is not None
        assert self.residuals is not None

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=9, ncols=1, sharex=True)

        for r, ax in enumerate("xyz"):
            self.fitter.aav[f"A_{ax}"].mul(RAD2D).plot(
                ax=axes[r],
                style=".",
            )

        for r, ax in enumerate([f"f{a}" for a in "xyz"] + [f"m{a}" for a in "xyz"]):
            self.residuals.plot(
                y=[ax, f"residuals_{ax}"],
                ax=axes[r + 3],
                style=".",
            )

        plt.show()

    def plot_velocity_residuals(self) -> None:
        """Plot velocity and fitted residuals."""
        assert self.fitter is not None
        assert self.residuals is not None

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=12, ncols=1, sharex=True)

        for r, ax in enumerate("xyz"):
            self.fitter.aav[f"V_{ax}2"].mul(RAD2D).plot(
                ax=axes[r],
                style=".",
            )

        for r, ax in enumerate(["xz", "yz", "xy"]):
            self.fitter.aav[f"V_{ax}"].mul(RAD2D).plot(
                ax=axes[r + 3],
                style=".",
            )

        for r, ax in enumerate([f"f{a}" for a in "xyz"] + [f"m{a}" for a in "xyz"]):
            self.residuals.plot(
                y=[ax, f"residuals_{ax}"],
                ax=axes[r + 6],
                style=".",
            )

        plt.show()

    async def fit_aav(
        self,
        start_time: Time,
        end_time: Time,
        out_dir: pathlib.Path,
        fit_values: str,
        no_gyroscope: bool,
        no_accelerometers: bool,
        set_new: bool,
        plot: bool,
        hd5_debug: None | pathlib.Path,
    ) -> None:
        """Fit acceleration and velocity (aav) forces.

        Parameters
        ----------
        start_time : Time
            Fit start time interval.
        end_time : Time
            Fit end time interval.
        out_dir : pathlib.Path
            Directory where output files will be stored.
        fit_values : str
            Whenever to fit actual or demand values.
        no_gyroscope : bool
            If set to true, don't use Gyroscope values as angular velocities
            input. Use angular velocities provided by TMA.
        no_accelerometers : bool
            If set to true, don't use DC accelerometer's values as angular
            acceleration input. Use angular accelerations calculated from TMA
            velocities.
        set_new : bool
            Don't add fit to existing values. If true, it's assumed data were
            collected without acceleration and velocity compensations.
        plot : bool
            If true, produce various plots as data are retrieved and processed.
        hd5_debug : str
            If not None, store intermediate data into this hd5 file.
        """
        self.client = EfdClient(self.efd_name)

        if self.azimuths is None or self.elevations is None:
            await self.find_az_el(start_time, end_time)

        assert self.azimuths is not None
        assert self.elevations is not None

        if self.intervals is None:
            await self.load_detailed_state(start_time, end_time)
            await self.find_applicable_times()

        assert self.intervals is not None

        if hd5_debug is not None:
            self.azimuths.to_hdf(hd5_debug, key="azimuths")
            self.elevations.to_hdf(hd5_debug, key="elevations")
            self.intervals.to_hdf(hd5_debug, key="intervals")

        self.elevations.rename(columns=lambda n: f"elevation_{n}", inplace=True)
        self.azimuths.rename(columns=lambda n: f"azimuth_{n}", inplace=True)

        if self.raw is None:
            await self.collect_hardpoint_data()

        logging.info("Hardpoint data retrieved")

        assert self.raw is not None

        self.raw.sort_index(inplace=True)

        if no_gyroscope is False:
            if self.gyroscope is None:
                await self.collect_gyroscope_data()

            assert self.gyroscope is not None

            self.gyroscope.sort_index(inplace=True)

            if hd5_debug is not None:
                self.gyroscope.to_hdf(hd5_debug, key="gyroscope")

            logging.info(
                f"Gyroscope data retrieved, has {len(self.gyroscope.index)} rows."
            )

            self.raw = self.raw.merge(
                self.gyroscope.rename(columns=lambda n: f"gyroscope_{n}"),
                how="outer",
                left_index=True,
                right_index=True,
            )

        if no_accelerometers is False:
            if self.accelerometers is None:
                await self.collect_accelerometers_data()

            assert self.accelerometers is not None

            self.accelerometers.sort_index(inplace=True)

            if hd5_debug is not None:
                self.accelerometers.to_hdf(hd5_debug, key="accelerometers")

            logging.info(
                f"Accelerometers data retrieved, has {len(self.accelerometers.index)} rows."
            )

            self.raw = self.raw.merge(
                self.accelerometers.rename(columns=lambda n: f"accelerometers_{n}"),
                how="outer",
                left_index=True,
                right_index=True,
            )

        if hd5_debug is not None:
            self.raw.to_hdf(hd5_debug, key="raw")

        logging.info(f"Raw data: {len(self.raw.index)} rows")

        if self.mirror is None:
            self.interpolated = self.raw.interpolate(method="time")
            self.interpolated = self.interpolated[
                self.interpolated.index.to_series().diff() < np.timedelta64(1, "s")
            ]
            self.mirror_forces_and_moments()

            assert self.mirror is not None

            if hd5_debug is not None:
                self.mirror.to_hdf(hd5_debug, key="mirror")

            logging.info(
                f"Mirror data: {len(self.mirror.index)} rows, "
                f"azimuth {self.mirror.azimuth_demandPosition.min():.2f}"
                f" .. {self.mirror.azimuth_demandPosition.max():.2f}, "
                f"elevation {self.mirror.elevation_demandPosition.min():.2f}"
                f" .. {self.mirror.elevation_demandPosition.max():.2f}"
            )

        if plot:
            self.plot_axes()

        old_coef = pd.concat(
            [t.data for t in self.force_calculator.velocity_tables]
            + [t.data for t in self.force_calculator.acceleration_tables],
        ).mean()

        # prepare for fit A @ x = B
        self.fitter = AccelerationAndVelocityFitter(
            self.mirror,
            fit_values if no_gyroscope else "meters",
            fit_values if no_accelerometers else "meters",
        )
        if hd5_debug is not None:
            self.fitter.aav.to_hdf(hd5_debug, key="A")

        if self.coefficients is None:
            self.coefficients, self.fit_residuals = self.fitter.do_fit(self.mirror)

        assert self.coefficients is not None

        if hd5_debug is not None:
            self.coefficients.to_hdf(hd5_debug, key="coefficients")

        if set_new:
            self.force_calculator.set_acceleration_and_velocity(self.coefficients)
        else:
            self.force_calculator.update_acceleration_and_velocity(self.coefficients)

        new_coef = pd.concat(
            [t.data for t in self.force_calculator.velocity_tables]
            + [t.data for t in self.force_calculator.acceleration_tables],
        ).mean()

        logging.info(
            f"Coefficients mean - old {old_coef.iloc[0]:.2f}, "
            f"current {self.coefficients.values.mean():.2f}, "
            f"new {new_coef.iloc[0]:.2f}"
        )

        if self.calculated_fam is None:
            self.calculate_forces_and_moments()

        assert self.calculated_fam is not None

        if hd5_debug is not None:
            self.calculated_fam.to_hdf(hd5_debug, key="calculated_fam")

        if plot:
            self.plot_acc_forces()

        if self.residuals is None:
            self.mirror_res = pd.DataFrame(
                {
                    "fx": self.mirror["fx"] + self.calculated_fam["calculated_fx"],
                    "fy": self.mirror["fy"] + self.calculated_fam["calculated_fy"],
                    "fz": self.mirror["fz"] + self.calculated_fam["calculated_fz"],
                    "mx": self.mirror["mx"] + self.calculated_fam["calculated_mx"],
                    "my": self.mirror["my"] + self.calculated_fam["calculated_my"],
                    "mz": self.mirror["mz"] + self.calculated_fam["calculated_mz"],
                }
            )

            for axis in [f"f{a}" for a in "xyz"] + [f"m{a}" for a in "xyz"]:
                logging.info(
                    f"Residuals {axis} "
                    f"min {np.min(self.mirror_res[axis]):.2f} "
                    f"mean {np.mean(self.mirror_res[axis]):.2f} "
                    f"max {np.max(self.mirror_res[axis]):.2f} "
                    f"std {np.std(self.mirror_res[axis]):.2f}"
                )

            self.mirror_res.rename(columns=lambda n: f"residuals_{n}", inplace=True)

            self.residuals = self.mirror.join([self.calculated_fam, self.mirror_res])

        assert self.residuals is not None

        if hd5_debug is not None:
            self.residuals.to_hdf(hd5_debug, key="residuals")

        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass

        if set_new:
            self.force_calculator.save(
                out_dir,
                f"Calculated from slews {start_time.isot} - {end_time.isot}",
                reset_comments=True,
            )
        else:
            self.force_calculator.save(
                out_dir, f"Updated with slews {start_time.isot} - {end_time.isot}"
            )

        logging.info(f"Saved new tables into {out_dir} directory")

        if plot:
            self.plot_acceleration_residuals()
            self.plot_velocity_residuals()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update Acceleration And Velocities tables based on registered hardpoint load cells loads"
    )
    parser.add_argument(
        "start_time",
        type=Time,
        default=None,
        nargs="?",
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "end_time",
        type=Time,
        default=None,
        nargs="?",
        help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )

    parser.add_argument(
        "--axes",
        choices=["X", "Y", "Z", "XY", "XZ", "YZ", "XZY"],
        default="XYZ",
        help="Axis of the force balance to be updated. Defaults to all axis.",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )
    parser.add_argument(
        "--fit-values",
        choices=["actual", "demand"],
        default="actual",
        help="Fit actual or demand accelerations and velocities",
    )
    parser.add_argument(
        "--no-gyroscope",
        action="store_true",
        help="Don't use Gyroscope velocities, use TMA values for velocities",
    )
    parser.add_argument(
        "--no-accelerometers",
        action="store_true",
        help="Don't use DC accelerometers, use TMA values for accelerations",
    )
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Plot graphs during processing",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(os.environ["TS_CONFIG_MTTCS_DIR"]) / "MTM1M3" / "v1",
        help="M1M3 configuration directory",
    )
    parser.add_argument(
        "--set-new",
        default=False,
        action="store_true",
        help="Don't add to existing forces, assuming acceleration and velocity "
        "forces were disabled when acquiring data",
    )
    parser.add_argument(
        "--hd5-debug",
        type=pathlib.Path,
        default=None,
        help="Save debug outputs into provided HDF5 file",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=Time.now().isot,
    )
    parser.add_argument(
        "--load",
        type=pathlib.Path,
        default=None,
        help="Load data from given HD5 file.",
    )
    parser.add_argument(
        "-d",
        default=False,
        action="store_true",
        help="Print debug messages",
    )

    return parser.parse_args()


async def run_loop() -> None:
    args = parse_arguments()

    level = logging.INFO

    if args.d:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    if args.load is None:
        if args.start_time is None or args.end_time is None:
            print("Either start and end times or --load argument must be provided.")
            return

    aav = AccelerationAndVelocity(args.efd, args.config, args.load)
    await aav.fit_aav(
        args.start_time,
        args.end_time,
        args.out_dir,
        args.fit_values,
        args.no_gyroscope,
        args.no_accelerometers,
        args.set_new,
        args.plot,
        args.hd5_debug,
    )


def run() -> None:
    asyncio.run(run_loop())


if __name__ == "__main__":
    run()
