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

import sty
from astropy.time import Time, TimeDelta
from lsst.ts.xml.enums.MTM1M3 import BumpTest as BumpTestStatus
from lsst.ts.xml.tables.m1m3 import FATable, force_actuator_from_id
from lsst_efd_client import EfdClient

from .bump_test_times import BumpTest, BumpTestTimes
from .chronograf import M1M3ForceActuatorForces
from .duration_time import DurationTime
from .force_actuator_forces import ForceActuatorForces


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    now = Time.now()

    parser = argparse.ArgumentParser(description="Queries bump test status.")
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
        "actuators",
        nargs="*",
        help="Actuators to query. If empty, list all actuators.",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )
    parser.add_argument(
        "--details",
        default=False,
        action="store_true",
        help="Print details (average/min/max following errors,..",
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

    start_t, end_t = DurationTime.pair(args.start_time, args.end_time)

    level = logging.INFO

    if args.d:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    client = EfdClient(args.efd)

    btt = BumpTestTimes(client)
    m1m3_forces = M1M3ForceActuatorForces(args.efd)

    if len(args.actuators) == 0:
        args.actuators = [fa.actuator_id for fa in FATable]

    logging.info(f"Looking for bump test times in {start_t} to {end_t}")

    for aid in [int(a) for a in args.actuators]:
        actuator = force_actuator_from_id(aid)
        logging.info(f"** Actuator {aid} type: {actuator.actuator_type}")

        async def print_bump(test: BumpTest) -> None:
            print(
                test.start_time.isot,
                test.end_time.isot,
                test.start_time < test.end_time,
            )
            url = m1m3_forces.fa_url(test.start_time, test.end_time, test.fa)
            print(
                sty.fg.green if test.result == BumpTestStatus.PASSED else sty.fg.red,
                test.start_time.isot,
                test.end_time.isot,
                test.result,
                sty.fg.rs,
                url,
            )
            if args.details:
                faf = ForceActuatorForces(test.start_time, test.end_time, client)
                fa_fe = await faf.actuator_following_error(actuator)
                print(
                    f"Following errors min: {fa_fe.primary.min():.3f} N "
                    f"max: {fa_fe.secondary.max():.3f} N"
                )
                following_errors = await faf.following_errors()
                flat_fe = following_errors.values.reshape(-1)
                print(
                    f"All following errors min: {flat_fe.min():.3f} N "
                    f"max {flat_fe.max():.3f} N"
                )

        print(sty.fg.yellow, "Primary bump tests - FA", actuator.actuator_id, sty.bg.rs)
        async for bump in btt.find_times(actuator, True, start_t, end_t):
            await print_bump(bump)

        if actuator.s_index is not None:
            print(sty.bg.blue, "\u25A9" * 50, sty.bg.rs)
            print(
                sty.fg.yellow,
                "Secondary bump tests - FA",
                actuator.actuator_id,
                sty.fg.rs,
            )
            async for bump in btt.find_times(actuator, False, start_t, end_t):
                await print_bump(bump)

    if client.influx_client is None:
        await client._influx_client.close()
    else:
        await client.influx_client.close()


def run() -> None:
    asyncio.run(run_loop())
