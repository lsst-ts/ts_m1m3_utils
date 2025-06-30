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

from astropy.time import Time, TimeDelta
from lsst.ts.xml.tables.m1m3 import FATable, force_actuator_from_id
from lsst_efd_client import EfdClient

from .duration_time import DurationTime


def parse_arguments(now: Time) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Queries bump test status.")
    parser.add_argument(
        "start_time",
        type=DurationTime(now),
        default=now - TimeDelta(30, format="sec"),
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
        "actuators",
        nargs="*",
        help="Actuators to query. If empty, list all actuators.",
    )
    parser.add_argument(
        "--toplist",
        type=int,
        default=10,
        help="Number of top actuators to print",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )
    parser.add_argument(
        "-d",
        default=False,
        action="store_true",
        help="Print debug messages",
    )

    return parser.parse_args()


async def run_loop() -> None:
    now = Time.now()

    args = parse_arguments(now)

    level = logging.INFO

    if args.d:
        level = logging.DEBUG

    if len(args.actuators) == 0:
        args.actuators = [fa.actuator_id for fa in FATable]

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    start_t, end_t = DurationTime.pair(args.start_time, args.end_time)

    client = EfdClient(args.efd)
    logging.info(
        "Quering following errors - %s to %s.", start_t.utc.isot, end_t.utc.isot
    )

    forces = await client.select_time_series(
        "lsst.sal.MTM1M3.forceActuatorData", "*", start_t, end_t
    )

    tested = []

    for aid in args.actuators:
        logging.debug("FA %d", aid)
        fa = force_actuator_from_id(aid)
        pe = forces[f"primaryCylinderFollowingError{fa.index}"]
        fa.pfe_max = pe.max()
        fa.pfe_min = pe.min()

        if fa.s_index is not None:
            se = forces[f"secondaryCylinderFollowingError{fa.s_index}"]
            fa.sfe_max = se.max()
            fa.sfe_min = se.min()
        else:
            fa.sfe_max = fa.sfe_min = 0

        tested.append(fa)

    # order..
    print(f"Max primary errors top {args.toplist}")
    tested.sort(reverse=True, key=lambda a: abs(a.pfe_max - a.pfe_min))
    for fa in tested[: args.toplist]:
        print(f"{fa.actuator_id}: {fa.pfe_min:.02f} N - {fa.pfe_max:.02f} N")

    print(f"Max secondary errors top {args.toplist}")
    tested.sort(reverse=True, key=lambda a: abs(a.sfe_max - a.sfe_min))
    for fa in tested[: args.toplist]:
        print(f"{fa.actuator_id}: {fa.sfe_min:.02f} N - {fa.sfe_max:.02f} N")

    await client._influx_client.close()


def run() -> None:

    asyncio.run(run_loop())
