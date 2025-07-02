# This file is part of ts_m1m3_utils
#
# Developed for the LSST Telescope and Site.
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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst.ts.xml.tables.m1m3 import FCUTable, fcu_from_address
from lsst_efd_client import EfdClient

from .chronograf import M1M3FCUStats
from .duration_time import DurationTime
from .fcu_stats import FCUStats


def plot_fcu_temperature(
    df: pd.DataFrame, fcu_index: int, start_time: Time, end_time: Time
) -> None:
    """Plot the FCU temperature data from the DataFrame."""
    title = f"FCU{fcu_index} Temperature Data\n From {start_time.iso} to {end_time.iso}"
    fig, ax = plt.subplots(num=1, clear=True)

    ax.plot(
        df[f"absoluteTemperature{fcu_index}"],
        label=f"FCU{fcu_index} Temperature",
        color="blue",
        linewidth=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (deg C)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.legend(loc="upper right", fontsize=10)

    fig.autofmt_xdate(rotation=45, ha="right")
    fig.savefig(
        f"fcu{fcu_index:02d}_temperature_{start_time.isot.replace(':', '-')}_"
        f"{end_time.isot.replace(':', '-')}.png",
        dpi=300,
    )
    plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    now = Time.now()

    parser = argparse.ArgumentParser(
        description="Queries M1M3 Thermal System's FCU statistics."
    )
    parser.add_argument(
        "start_time",
        type=DurationTime(now),
        default=now - TimeDelta(7, format="jd"),
        nargs="?",
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "end_time",
        type=DurationTime(now),
        default=now,
        nargs="?",
        help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "fcus",
        nargs="*",
        type=int,
        help="Fan Coil Unit (FCU) address(es) to analyze (1-96). "
        "If not specified, all FCUs will be analyzed.",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )
    parser.add_argument(
        "-sp",
        "--set_point",
        type=float,
        default=9.0,
        help="Temperature set point for the FCU in degrees Celsius. Default is 9.0 deg C.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        default=False,
        action="store_true",
        help="Plot the temperature data.",
    )
    parser.add_argument(
        "-u",
        "--show_url",
        default=False,
        action="store_true",
        help="Show the URL for Summit Chronograf.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="Print debug messages",
    )

    return parser.parse_args()


async def main() -> None:
    """Main function to run the FCU quick analysis."""
    args = parse_arguments()

    start_t, end_t = DurationTime.pair(args.start_time, args.end_time)

    level = logging.INFO

    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    logging.debug("Parsed arguments: %s", str(args))

    # Create using the EFD client
    client = EfdClient(args.efd)

    stats = FCUStats(client)

    fcus = [fcu_from_address(address) for address in args.fcus]

    if len(fcus) == 0:
        fcus = FCUTable

    logging.info(f"Looking for bump test times in {start_t} to {end_t}")

    await stats.fcu_quick_analysis(
        [fcu.index for fcu in fcus],
        start_t,
        end_t,
        args.set_point,
        args.plot,
        args.show_url,
    )

    assert stats.data is not None

    for fcu in fcus:
        rms = (
            (stats.data[f"absoluteTemperature{fcu.index}"] - args.set_point) ** 2
        ).mean() ** 0.5

        statistics = stats.statistics[fcu.index]
        print(
            f"""
FCU {fcu.name} (index={fcu.index}) Temperature Stats:
    Min:       {statistics['min']:.3f} \u00b0C
    Mean:      {statistics['mean']:.3f} \u00b0C
    Median:    {statistics['median']:.3f} \u00b0C
    Max:       {statistics['max']:.3f} \u00b0C
    Std:       {statistics['std']:.3f} \u00b0C
    Set Point: {args.set_point:.3f} \u00b0C
    RMS:       {rms:.3f} \u00b0C"""
        )

        if args.show_url:
            print(
                f"  View the data in Chronograf:\n    {M1M3FCUStats(args.efd).url(start_t, end_t)}\n"
            )

        if args.plot:
            plot_fcu_temperature(stats.data, fcu.index, start_t, end_t)

    # Ensure to close the EFD client connection
    if client.influx_client is None:
        await client._influx_client.close()
    else:
        await client.influx_client.close()


def run() -> None:
    asyncio.run(main())
