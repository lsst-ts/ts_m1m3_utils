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

import argparse
import asyncio
import logging
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient
from lsst.ts.m1m3.utils import ForceCalculator
from lsst.ts.m1m3.utils.correlate_timeseries import compute_time_delay

N_PRIMARY = 156
N_SECONDARY = 112
warnings.simplefilter(action="ignore", category=FutureWarning)


async def get_data(client: EfdClient, field: str, t_start: Time, t_end: Time) -> pd.DataFrame:
    """
    Returns data from Efd

    Parameters
    ----------
    client : `EfdClient`
    field : `str`
    t_start : `Time`
    t_end : `Time`

    Returns
    -------
    data : `pd.DataFrame`
    """
    logging.debug("Retrieving %s - %s to %s.", field, str(t_start), str(t_end))
    data = await client.select_time_series("lsst.sal.MTM1M3.forceActuatorData", field, t_start, t_end)
    logging.debug("Retrieved %i records.", len(data.index))
    return data


async def compute_forces(
    efd_name: str,
    t1: Time,
    t2: Time,
    delta_t: pd.Timedelta,
    time_delay: float,
) -> list:
    """
    Returns computed forces.

    Parameters
    ----------
    efd_name : `str`
    t1 : `Time`
    t2 : `Time`
    delta_t : `pd.Timedelta`
    time_delay: `float`

    Returns
    -------
    forces_s1 : `ForceCalculator.CylinderForces`
    forces_s2 : `ForceCalculator.CylinderForces`
    Computed forces (xForce, yForce, zForce) on each setting,
    each axis a dataframe through all Cylinders
    """

    client = EfdClient(efd_name)

    [df_s1, df_s2] = await asyncio.gather(
        get_data(client, "*", t1, t1 + delta_t),
        get_data(
            client,
            "*",
            t2 + pd.to_timedelta(time_delay, unit="s"),
            t2 + pd.to_timedelta(time_delay, unit="s") + delta_t,
        ),
    )

    # we take both dataframes to start at 0
    rebased_index1 = df_s1.index - df_s1.index[0]
    s1 = df_s1.copy()
    s1.index = rebased_index1

    rebased_index2 = df_s2.index - df_s2.index[0]
    s2 = df_s2.copy()
    s2.index = rebased_index2

    # however we need to get the indices at the same times in both,
    # so we need to interpolate one of them to the indices of the other one
    s2_interp = s2.reindex(s1.index.union(s2.index)).interpolate("time").reindex(s1.index)

    data1_primary = [None] * N_PRIMARY
    data1_secondary = [None] * N_SECONDARY
    data2_primary = [None] * N_PRIMARY
    data2_secondary = [None] * N_SECONDARY

    for i in range(N_PRIMARY):
        data1_primary[i] = s1[f"primaryCylinderFollowingError{i}"]
        data2_primary[i] = s2_interp[f"primaryCylinderFollowingError{i}"]
    for j in range(N_SECONDARY):
        data1_secondary[j] = s1[f"secondaryCylinderFollowingError{j}"]
        data2_secondary[j] = s2_interp[f"secondaryCylinderFollowingError{j}"]

    forces_s1 = ForceCalculator.CylinderForces(data1_primary, data1_secondary)
    forces_s2 = ForceCalculator.CylinderForces(data2_primary, data2_secondary)

    return [forces_s1, forces_s2]


async def plot_forces(
    axis: str,
    forces_s1: pd.Series,
    forces_s2: pd.Series,
    show: bool,
    save: pathlib.Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    if axis == "x":
        forces_s1 = forces_s1.fx
        forces_s2 = forces_s2.fx
    elif axis == "y":
        forces_s1 = forces_s1.fy
        forces_s2 = forces_s2.fy
    else:
        forces_s1 = forces_s1.fz
        forces_s2 = forces_s2.fz

    N = 200
    residuals = forces_s2 - forces_s1
    df_residuals = residuals.to_frame(name="value")
    groups = pd.Series(np.arange(len(residuals)) // N, index=residuals.index)
    df_residuals["t_seconds"] = (df_residuals.index - df_residuals.index[0]).total_seconds()
    grouped_stats = (
        df_residuals.groupby(groups)
        .agg(x_mean=("t_seconds", "mean"), y_mean=("value", "mean"), y_std=("value", "std"))
        .reset_index()
    )

    df_s1 = forces_s1.to_frame(name="value")
    df_s1["t_seconds"] = (df_s1.index - df_s1.index[0]).total_seconds()
    df_s2 = forces_s2.to_frame(name="value")
    df_s2["t_seconds"] = (df_s2.index - df_s2.index[0]).total_seconds()

    ax1.plot(df_s1["t_seconds"], df_s1["value"], label=f"Setting 1 - Force {axis}")
    ax1.set_ylabel("Force (N)")
    ax1.plot(df_s2["t_seconds"], df_s2["value"], label=f"Setting 2 - Force {axis}")
    plt.errorbar(
        grouped_stats["x_mean"],
        grouped_stats["y_mean"],
        yerr=grouped_stats["y_std"],
        fmt="o-",
        capsize=3,
        label="Residuals",
        color="black",
    )
    ax2.set_xlabel("Rebased time (s)")
    ax2.set_ylabel("Force (N)")
    fig.legend()

    if save is not None:
        filename = save.stem + "_" + axis + save.suffix
        plt.savefig(filename)
        logging.info("Plots saved to %s.", filename)

    if show:
        plt.show()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="M1M3 forces comparison given two similar TMA displacements under different settings"
    )
    parser.add_argument("--efd", default="usdf_efd", help="EFD name. Defaults to usdf_efd.")
    parser.add_argument(
        "--t1",
        type=str,
        default="2025-09-12T09:42:30Z",
        help="Start time for first setting in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "--t2",
        type=str,
        default="2025-09-12T09:59:50Z",
        help="Start time for second setting in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=200,
        help="Number of seconds to sample starting at t1/t2",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=20,
        help="Resampling frequency in Hz. Defaults to 20.",
    )
    parser.add_argument(
        "--overlay",
        type=float,
        default=60,
        help="Allowed overlay - the signal correlation will be done"
        " with starts in t2 to t2 + overlay ranges.",
    )
    parser.add_argument(
        "-d",
        default=False,
        action="store_true",
        help="Print debug messages.",
    )
    parser.add_argument(
        "--show",
        default=False,
        action="store_true",
        help="Show plots.",
    )
    parser.add_argument(
        "--save",
        type=pathlib.Path,
        help="Save graphs to given file.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.d else logging.INFO

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    logging.info(
        "Using t1=%s, t2=%s, delta_t=%.3fs, overlay=%.3fs.", args.t1, args.t2, args.delta_t, args.overlay
    )

    delay, signal1, signal2 = await compute_time_delay(
        args.efd,
        Time(args.t1, scale="utc"),
        Time(args.t2, scale="utc"),
        pd.to_timedelta(args.delta_t, unit="s"),
        args.sampling_frequency,
        TimeDelta(args.overlay, format="sec"),
    )

    logging.info("Computed delay: %.3fs.", delay)

    forces_s1, forces_s2 = await compute_forces(
        args.efd,
        Time(args.t1, scale="utc"),
        Time(args.t2, scale="utc"),
        pd.to_timedelta(args.delta_t, unit="s"),
        delay / args.sampling_frequency,
    )

    axes = ["x", "y", "z"]

    if args.show or args.save is not None:
        for axis in axes:
            logging.debug("Plotting %s axis.", axis)
            await plot_forces(axis, forces_s1, forces_s2, args.show, args.save)


def run() -> None:
    asyncio.run(main())
