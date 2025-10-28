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

from lsst.ts.m1m3.utils import BumpTestRunner, BumpTestsList
from lsst.ts.salobj import Domain, Remote


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run bump tests.")
    parser.add_argument(
        "--serial",
        type=bool,
        default=False,
        help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "-s",
        default=[],
        action="append",
        dest="skip",
        help="Actuators to skip.",
    )
    parser.add_argument("--timeout", default=20.0, type=float, help="Timeout for async calls in tests.")
    parser.add_argument(
        "-d",
        default=False,
        action="store_true",
        help="Print debug messages",
    )

    return parser.parse_args()


async def do_tests(m1m3: Remote, timeout: float, serial: bool, skip: list[int]) -> None:
    """Run tests on force actuators. Prints out failed and passed actuators.

    Parameters
    ----------
    m1m3 : Remote
        M1M3 remote.
    timeout : float
        Timeout in seconds for various waits for test's finish. Reasonable
        value is 20.
    serial : bool
        If true, run all the tests in parallel.
    skip : list[int]
        List of force actuators to skip.
    """
    runner = BumpTestRunner(BumpTestsList.all_tests(m1m3, skip))

    m1m3.evt_forceActuatorBumpTestStatus.callback = runner.force_actuator_bump_test_status

    distance = 10
    if not (serial):
        distance = m1m3.evt_forceActuatorSettings.get().bumpTestMinimalDistance

    while True:
        test = await runner.next(distance, timeout)
        if test is None:
            break

        primary = test.is_primary()

        print(f"Starting test on {test.actuator.actuator_id} - {'primary' if primary else 'secondary'}")

        await m1m3.cmd_forceActuatorBumpTest.set_start(
            actuatorId=test.actuator.actuator_id,
            testPrimary=primary,
            testSecondary=not (primary),
        )

    await runner.wait_finish(timeout)

    print("Passed:\n", "\n   ".join([str(p.actuator) for p in runner.passed]))
    print("Failed:\n", "\n   ".join([str(f.actuator) for f in runner.failed]))


async def main() -> None:
    """Parse arguments and call do_tests to run the tests."""
    args = parse_arguments()

    level = logging.INFO

    if args.d:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    async with Domain() as domain:
        async with Remote(domain=domain, name="MTM1M3", start=True) as m1m3:
            await do_tests(m1m3, args.timeout, args.serial, args.skip)


def run() -> None:
    asyncio.run(main())
