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
import typing
from dataclasses import dataclass

import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient

from .duration_time import DurationTime


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    now = Time.now()

    parser = argparse.ArgumentParser(
        description="Find changes in EFD topics. List EFD topics when run without any arguments."
    )
    parser.add_argument(
        "start_time",
        type=DurationTime(now),
        default=DurationTime(now - TimeDelta(7, format="jd")),
        nargs="?",
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "end_time",
        type=DurationTime(now),
        default=DurationTime(now),
        nargs="?",
        help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "topics",
        nargs="*",
        help="Topics to query. If empty, list all topics know to EFD.",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )
    parser.add_argument(
        "--index",
        default=None,
        type=int,
        help="CSC index.",
    )
    parser.add_argument(
        "-d",
        default=False,
        action="store_true",
        help="Print debug messages",
    )

    return parser.parse_args()


@dataclass
class ChangedValue:
    column_name: str
    index: typing.Any
    old: typing.Any
    new: typing.Any


def find_changes(data: pd.DataFrame) -> typing.Iterator[ChangedValue]:
    template = data.head(n=1)
    cols = [c for c in data.columns if not (c.startswith("private_"))]

    for row in data.tail(n=len(data.index) - 1).iterrows():
        for c in cols:
            v = row[1][c]
            t = template[c].item()
            if t != v:
                yield ChangedValue(c, row[0], t, v)
                template.loc[:, c] = v


async def run_loop() -> None:
    args = parse_arguments()

    start_t, end_t = DurationTime.pair(args.start_time, args.end_time)

    level = logging.INFO

    if args.d:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    client = EfdClient(args.efd)

    if len(args.topics) == 0:
        for t in await client.get_topics():
            if t.startswith("lsst.sal."):
                print(t[9:])
            else:
                print(t)
        return

    for t in args.topics:
        data = await client.select_time_series("lsst.sal." + t, "*", start_t, end_t, index=args.index)

        if data.empty:
            print(f"No data found for '{t}' between {start_t.isot} and {end_t.isot}.")
            continue

        for c in find_changes(data):
            print(t, c.column_name, c.index, c.old, c.new)

    if client.influx_client is None:
        await client._influx_client.close()
    else:
        await client.influx_client.close()


def run() -> None:
    asyncio.run(run_loop())
