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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient


def create_url_for_summit_chronograf(t_start, t_end):
    """Create a URL for the Summit Chronograf to visualize FCU data."""
    base_url = "https://summit-lsp.lsst.codes/chronograf"
    dashboard = "sources/1/dashboards/390"

    t_start_str = t_start.isot.replace(":", "%3A")
    t_end_str = t_end.isot.replace(":", "%3A")

    url = (
        f"{base_url}/{dashboard}?refresh=Paused&lower={t_start_str}Z&upper={t_end_str}Z"
    )
    return url


def get_time_window(timestamp: str, delta_t: str):
    """Given a timestamp and a duration string, return (t_start, t_end) as
    astropy Time objects."""
    delta_t_seconds = parse_duration(delta_t)
    if delta_t_seconds <= 0:
        t_end = parse_timestamp(timestamp)
        t_start = t_end + delta_t_seconds
    else:
        t_start = parse_timestamp(timestamp)
        t_end = t_start + delta_t_seconds
    return t_start, t_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Queries M1M3 Thermal System's FCU statistics."
    )
    parser.add_argument(
        "-rt",
        "--reference_time",
        default="now",
        nargs="?",
        help="Reference time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "-dt",
        "--delta_time",
        default="-10s",
        nargs="?",
        help="Delta time string (e.g., '-10s' for 10 seconds before the reference time). "
        "If the value is negative, use an equal sign (e.g., -df=-30s) to avoid "
        "confusion with argparse.",
    )
    parser.add_argument(
        "-fcu",
        "--fcu_indexes",
        type=int,
        default=None,
        nargs="*",
        help="Fan Coil Unit (FCU) indexes to analyze (0-95). If not specified, all FCUs will be analyzed.",
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
        help="Plot the temperature data. Default is False.",
    )
    parser.add_argument(
        "-u",
        "--show_url",
        default=True,
        action="store_true",
        help="Show the URL for Summit Chronograf. Default is True.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="Print debug messages",
    )

    return parser.parse_args()


# Function copied from lsst-ts/ts_m1m3_utils
def parse_duration(duration: str) -> TimeDelta:
    """Accept string depicting duration.

    Numbers can be suffixed with character, denomination their lengths.

    Length denominators
    -------------------
    D : days (86400 seconds)
    h : hours (3600 seconds)
    m : minutes (60 seconds)
    s : seconds (1 second)

    Examples
    --------
    '1D 1m' = 86460 seconds
    '1h 1m 30s' = 3690 seconds

    Parameters
    ----------
    duration : `str`
        Duration string. Numbers with know suffixed. Non-sufficed number will
        be treated as seconds.

    Returns
    -------
    seconds : float
        Number of seconds in string.
    """
    if not duration:
        raise ValueError("Duration string cannot be empty.")

    muls = {"D": 86400, "h": 3600, "m": 60, "s": 1, "u": 0.001, "n": 0.000001}
    ret: float = 0.0
    current: float = 0.0
    duration = duration.strip()
    sign = 1
    fraction = 0

    if duration[0] == "-":
        sign = -1
        duration = duration[1:]
    elif duration[0] == "+":
        duration = duration[1:]

    for s in duration.strip():
        if "0" <= s <= "9":
            if fraction > 0:
                current += (0.1**fraction) * int(s)
                fraction += 1
            else:
                current = current * 10 + int(s)
        elif s == ".":
            fraction = 1
        elif s == " ":
            pass
        else:
            try:
                ret += current * muls[s]
                current = 0.0
            except KeyError:
                raise ValueError(f"Unknown suffix: {s}")

    return TimeDelta(sign * (ret + current) * u.s)


def parse_timestamp(timestamp):
    """Parse the timestamp string into an astropy Time object."""

    if timestamp.lower() == "now":
        timestamp = Time.now()
    elif "T" in timestamp:
        timestamp = Time(timestamp, format="isot", scale="utc")
    else:
        timestamp = Time(timestamp, format="iso", scale="utc")

    return timestamp


def plot_fcu_temperature(df, fcu_index, t_start, t_end):
    """Plot the FCU temperature data from the DataFrame."""
    title = f"FCU{fcu_index} Temperature Data\n From {t_start.iso} to {t_end.iso}"
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
        f"fcu{fcu_index:02d}_temperature_{t_start.isot.replace(':', '-')}_{t_end.isot.replace(':', '-')}.png",
        dpi=300,
    )
    plt.show()


async def fcu_quick_analysis(
    client: EfdClient,
    fcu_indexes: int | list[int],
    timestamp,
    delta_t,
    set_point,
    plot: bool = True,
    show_url: bool = True,
):
    """
    Perform a quick analysis of FCU temperature data.

    Parameters
    ----------
    client : EfdClient
        The EFD client to use for querying data.
    fcu_indexes : int or list of int
        The FCU index to analyze (0-95).
        If a list is provided, the analysis will be performed for each index.
    timestamp : str
        The timestamp in ISO format (e.g., "2025-05-10T12:00:00").
    delta_t : str
        The duration string (e.g., "-10s" for 10 seconds before the timestamp).
    set_point : float
        The temperature set point for the FCU in degrees Celsius.
    plot : bool, optional
        Whether to plot the temperature data. Default is True.
    show_url : bool, optional
        Whether to show the URL for Summit Chronograf. Default is True.
    return_output: bool, optional
        Whether return the output or not. Default is False.
    """
    if isinstance(fcu_indexes, int):
        fcu_indexes = [fcu_indexes]
    elif fcu_indexes is None:
        fcu_indexes = list(range(96))
    elif not isinstance(fcu_indexes, list):
        raise ValueError("fcu_index must be an integer or a list of integers.")

    t_start, t_end = get_time_window(timestamp, delta_t)
    print(f"  Time window: {t_start.iso} to {t_end.iso}")

    df = await client.select_time_series(
        "lsst.sal.MTM1M3TS.thermalData",
        ["timestamp"] + [f"absoluteTemperature{i}" for i in fcu_indexes],
        t_start,
        t_end,
    )
    if df.empty:
        print("No data found for the specified time window.")
        return

    for fcu_index in fcu_indexes:
        col = f"absoluteTemperature{fcu_index}"
        stats = df[col].agg(["min", "mean", "median", "max", "std"])
        results = (
            f"  FCU{fcu_index + 1} (index={fcu_index}) Temperature Stats:\n"
            f"    Min:       {stats['min']:.3f} deg_C\n"
            f"    Mean:      {stats['mean']:.3f} deg_C\n"
            f"    Median:    {stats['median']:.3f} deg_C\n"
            f"    Max:       {stats['max']:.3f} deg_C\n"
            f"    Std:       {stats['std']:.3f} deg_C\n"
            f"    Set Point: {set_point:.3f} deg_C\n"
            f"    RMS:       {((df[col] - set_point) ** 2).mean() ** 0.5:.3f} deg_C"
        )
        print(results)

        if show_url:
            url = create_url_for_summit_chronograf(t_start, t_end)
            print(f"  View the data in Summit Chronograf:\n    {url}\n")

        if plot:
            plot_fcu_temperature(df, fcu_index, t_start, t_end)


async def main():
    """Main function to run the FCU quick analysis."""

    # Parse arguments from the command line
    args = parse_arguments()

    if args.debug:
        print(f"Parsed arguments: {args}\n")

    # Create using the EFD client
    efd_client = EfdClient(args.efd)

    # Running analysis
    await fcu_quick_analysis(
        efd_client,
        fcu_indexes=args.fcu_indexes,
        timestamp=args.reference_time,
        delta_t=args.delta_time,
        set_point=args.set_point,
        plot=args.plot,
        show_url=args.show_url,
    )

    # Ensure to close the EFD client connection
    await efd_client.influx_client.close()


def run() -> None:
    asyncio.run(main())
