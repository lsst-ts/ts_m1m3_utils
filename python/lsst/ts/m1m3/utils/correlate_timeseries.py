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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient


async def get_data(
    client: EfdClient, axis: str, t_start: Time, t_end: Time, resample_rate: str
) -> pd.DataFrame:
    """
    Returns upsampled data.

    Parameters
    ----------
    client : `EfdClient`
    axis : `str`
    t_start : `Time`
    t_end : `Time`
    resample_rate : `str`

    Returns
    -------
    data : `pd.DataFrame`
        Upsampled data.
    """
    logging.debug("Retrieving %s - %s to %s.", axis, str(t_start), str(t_end))
    data = await client.select_time_series(
        f"lsst.sal.MTMount.{axis}", ["actualPosition"], t_start, t_end
    )
    logging.debug("Retrieved %i records.", len(data.index))
    # for some reason interpolate doesn't work
    return data["actualPosition"].resample(resample_rate, origin=data.index[0]).ffill()


async def compute_time_delay(
    efd_name: str,
    t1: Time,
    t2: Time,
    delta_t: pd.Timedelta,
    sampling_frequency: float = 20,
    overlay: TimeDelta | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    resample_rate = f"{1.0 / sampling_frequency}s"

    client = EfdClient(efd_name)

    if overlay is None:
        overlay = TimeDelta(60, format="sec")

    max_delay = int(np.floor(overlay.sec * sampling_frequency))

    data = await asyncio.gather(
        get_data(client, "azimuth", t1, t1 + delta_t, resample_rate),
        get_data(
            client, "azimuth", t2 - overlay, t2 + delta_t + overlay, resample_rate
        ),
    )

    signal1 = data[0].values
    signal2 = data[1].values

    length = len(signal1)

    chi2_best = np.inf
    delay_keep = np.nan

    for delay in range(0, max_delay):
        signal2_displaced = signal2[delay : delay + length]
        assert len(signal1) == len(signal2_displaced)
        chi2 = np.sum((signal2_displaced - signal1) ** 2)
        if chi2 < chi2_best:
            delay_keep = delay
            chi2_best = chi2

    assert np.isfinite(chi2_best), "Empty data - cannot find better match?"

    return delay_keep, signal1, signal2


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="M1M3 setting comparison (for now only time delay computation)"
    )
    parser.add_argument(
        "--efd", default="usdf_efd", help="EFD name. Defaults to usdf_efd."
    )
    parser.add_argument(
        "--t1",
        type=str,
        default="2025-09-12T09:42:30Z",
        help="Start time for first setting in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'.",
    )
    parser.add_argument(
        "--t2",
        type=str,
        default="2025-09-12T09:59:50Z",
        help="Start time for second setting in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'.",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=200,
        help="Number of seconds to sample starting at t1/t2.",
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
        f"Using t1={args.t1}, t2={args.t2}, delta_t={args.delta_t}s, overlay={args.overlay}s."
    )

    delay, signal1, signal2 = await compute_time_delay(
        args.efd,
        Time(args.t1, scale="utc"),
        Time(args.t2, scale="utc"),
        pd.to_timedelta(args.delta_t, unit="s"),
        args.sampling_frequency,
        TimeDelta(args.overlay, format="sec"),
    )

    print(f"Time delay: {delay / args.sampling_frequency:.02f}s")

    if args.show or args.save is not None:
        length = len(signal1)

        signal2_delayed = signal2[delay : delay + length]

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(signal1, label="T1")
        ax1.plot(signal2, label="T2")
        ax1.plot(signal2_delayed, label="Delayed")
        ax2.plot(signal1 - signal2_delayed, label="Residuals")

        fig.legend()

        if args.save is not None:
            plt.savefig(args.save)
            logging.info("Plots saved to %s.", args.save)

        if args.show:
            plt.show()


def run() -> None:
    asyncio.run(main())
