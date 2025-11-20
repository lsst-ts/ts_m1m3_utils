# This file is part of ts_m1m3_utils.
#
# Developed for the LSST Telescope and Site Systems.  This product includes
# software developed by the LSST Project (https://www.lsst.org).  See the
# COPYRIGHT file at the top-level directory of this distribution for details of
# code ownership.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

__all__ = ["BumpTestKind", "ForceActuatorBumpTest", "BumpTestRunner"]

import asyncio
import typing
from dataclasses import dataclass
from enum import StrEnum

from lsst.ts.salobj import BaseMsgType, Remote
from lsst.ts.xml.enums.MTM1M3 import BumpTest
from lsst.ts.xml.tables.m1m3 import (
    FATABLE_XFA,
    FATABLE_YFA,
    FATABLE_ZFA,
    FATable,
    ForceActuatorData,
)


class BumpTestKind(StrEnum):
    CYLINDER_PRIMARY = "P"
    CYLINDER_SECONDARY = "S"
    AXIS_X = "X"
    AXIS_Y = "Y"
    AXIS_Z = "Z"


@dataclass
class ForceActuatorBumpTest:
    """Record of actuator bump test request."""

    actuator: ForceActuatorData
    kind: BumpTestKind

    def is_primary(self) -> bool:
        return self.kind in [BumpTestKind.CYLINDER_PRIMARY, BumpTestKind.AXIS_Z]


class BumpTestsList:
    """Set of bump tests."""

    def __init__(self, actuators: list[ForceActuatorBumpTest] | None = None):
        self._tests = [] if actuators is None else actuators

    @staticmethod
    def all_tests(m1m3: Remote, skip: list[int] | None = None) -> list[ForceActuatorBumpTest]:
        """Return all tests for all enabled actuators minus actuators to skip.

        Parameters
        ----------
        m1m3 : Remote
            Salobj remote to retrieve a list of enabled force actuatos.
        skip : list[int], optional
            If not specified, no actuator will be skipped.

        Returns
        -------
        tests : list[ForceActuatorBumpTest]
            List of ForceActuatorBumpTest objects for actuators meeting the
            conditions (are enabled and aren't in the skipped list).
        """
        if skip is None:
            skip = []

        enabled = m1m3.evt_enabledForceActuators.get()
        if enabled is None:
            raise RuntimeError("Cannot retrieve list of enabledForceActuators event. Exiting.")

        tests = []
        for fa in FATable:
            if enabled.forceActuatorEnabled[fa.index] and fa.actuator_id not in skip:
                tests.append(ForceActuatorBumpTest(fa, BumpTestKind.AXIS_Z))
                if fa.x_index is not None:
                    tests.append(ForceActuatorBumpTest(fa, BumpTestKind.AXIS_X))
                if fa.y_index is not None:
                    tests.append(ForceActuatorBumpTest(fa, BumpTestKind.AXIS_Y))
        return tests

    def __iter__(self) -> typing.Iterator[ForceActuatorBumpTest]:
        return self._tests.__iter__()

    def __len__(self) -> int:
        return len(self._tests)

    def empty(self) -> bool:
        """Returns true if the set is empty.

        Returns
        -------
        empty : bool
            True if the set is empty.
        """
        return len(self._tests) == 0

    def append(self, test: ForceActuatorBumpTest) -> None:
        """Add new test to the set.

        Parameters
        ----------
        test : ForceActuatorBumpTest
            Test to add.
        """
        self._tests.append(test)

    def contains(self, actuator: ForceActuatorData, primary: bool | None = None) -> bool:
        if primary is None:
            return actuator.actuator_id in [test.actuator.actuator_id for test in self._tests]
        return actuator.actuator_id in [
            test.actuator.actuator_id for test in self._tests if test.is_primary() == primary
        ]

    def remove(self, actuator_id: int, primary: bool) -> ForceActuatorBumpTest | None:
        new_tests = []
        ret: ForceActuatorBumpTest | None = None
        for test in self._tests:
            if test.actuator.actuator_id == actuator_id and test.is_primary() == primary:
                if ret is not None:
                    raise RuntimeError(
                        f"Multiple instances of {actuator_id} primary {primary} in BumpTestsList!"
                    )
                ret = test
            else:
                new_tests.append(test)

        self._tests = new_tests
        return ret

    def clear(self) -> None:
        self._tests = []

    def distance(self, fa: ForceActuatorData) -> float:
        """Calculates minimal distance of the given force actuator to the set.

        Parameters
        ----------
        fa : ForceActuatorData
            Calculate minimal distace to this force actuator.

        Returns
        -------
        distance : float
            Minimal distance to the passed force actuator in meters.
        """
        if self.empty():
            return 201
        return min([fa.distance(test.actuator) for test in self._tests])


class BumpTestRunner:
    """Class managing bump test. Does bump test scheduling and handles bump
    test progress.

    The tests to be performed are placed inside the todo set. When next method
    is called, it either select possible next test to run, or block until it is
    possible to select the next actuator (or a timeout expires). Records for
    tests being performed are placed into runinng set. On text completion, the
    record is moved either to failed or passed set.

    After constructing the object, its next method, and either
    force_actuator_bump_test_status or progress shall be called. The next
    method returns None if there aren't any test to execute, signalling thegq
    scheduler finished all jobs. After that, the wait_finish method shall be
    called to wait for all running tests finishing.

    Please see m1m3-do-bump-tests source code for example how to use the
    runner.

    Attributes
    ----------
    todo : BumpTestsList
        Force actuators flagged to be tested.
    running : BumpTestsList
        Force actuators currently being tested.
    passed : BumpTestsList
        List of the force actuators which passed the bump test.
    failed : BumpTestsList
        List of the force actuators failing the bump test.

    Parameters
    ----------
    actuators : list[ForceActuatorBumpTest]
         List of actuators and axis to test.
    """

    def __init__(self, actuators: list[ForceActuatorBumpTest]):
        self.todo = BumpTestsList(actuators)

        self.running = BumpTestsList()
        self.passed = BumpTestsList()
        self.failed = BumpTestsList()

        self._running_changed = asyncio.Event()

        # last test progress
        self._primary_test = [0] * FATABLE_ZFA
        self._secondary_test = [0] * (FATABLE_XFA + FATABLE_YFA)

    async def next(self, min_distance: float = 200, timeout: float = 10) -> ForceActuatorBumpTest | None:
        """Return next FA to test.

        Parameters
        ----------
        min_distance : float
            Minimal distance (in meters) of the selected force actuator to all
            running force actuators.
        timeout : float
            Timeout (in seconds) to wait will next force actuator can be
            tested.

        Raises
        ------
        TimeoutError
            On timeout.

        Returns
        -------
        test : ForceActuatorBumpTest | None
            Either next bump test to perform, or None if there isn't a bump
            test to run.
        """
        async with asyncio.timeout(timeout):
            while not self.todo.empty():
                possible = [test for test in self.todo if self.running.distance(test.actuator) > min_distance]
                if len(possible) > 0:
                    picked = possible[0]
                    self.running.append(picked)
                    self.todo.remove(picked.actuator.actuator_id, picked.is_primary())
                    return picked

                await self._running_changed.wait()
                self._running_changed.clear()

            return None

    async def wait_finish(self, timeout: float) -> None:
        """Waits untill all execution finishes (todo and running queues are
        empty).

        Parameters
        ----------
        timeout : float
            Timeout in seconds to wait for test finish.

        Raises
        ------
        TimeoutError
            When timeout is reached.
        """
        async with asyncio.timeout(timeout):
            while not self.todo.empty() or not self.running.empty():
                await self._running_changed.wait()
                self._running_changed.clear()

    def force_actuator_bump_test_status(self, data: BaseMsgType) -> None:
        """Process status message describing bump test progress.

        Parameters
        ----------
        data : MTM1M3_logevent_forceActuatorBumpTestStatus
            Logevent data to process. Must contain primaryTest and
            secondaryTest arrays wih lengths equal to the total number of
            actuators and number of dual axis actuators.
        """
        for fa in FATable:
            changed = False
            primary = data.primaryTest[fa.index]
            secondary = None

            if primary != self._primary_test[fa.index]:
                changed = True

            if fa.s_index is not None:
                secondary = data.secondaryTest[fa.s_index]
                if secondary != self._secondary_test[fa.s_index]:
                    changed = True

            if changed:
                self.progress(fa, primary, secondary)

    def progress(self, actuator: ForceActuatorData, primary: int, secondary: int | None) -> None:
        """React to bump test progress of the given force actuator. If the test
        finishes, move the test record from running into either passed or
        failed lists.

        Parameters
        ----------
        actuator : ForceActuatorData
            Actuator being handled.
        primary : int
            Progress (status) of the primary axis bump test.
        secondary : int
            Progress (status) of the secondary axis bump test.
        """

        def passed(state: int) -> bool:
            return state in [BumpTest.PASSED]

        def failed(state: int) -> bool:
            return state in [
                BumpTest.FAILED_TIMEOUT,
                BumpTest.FAILED_TESTEDPOSITIVE_OVERSHOOT,
                BumpTest.FAILED_TESTEDPOSITIVE_UNDERSHOOT,
                BumpTest.FAILED_TESTEDNEGATIVE_OVERSHOOT,
                BumpTest.FAILED_TESTEDNEGATIVE_UNDERSHOOT,
                BumpTest.FAILED_NONTESTEDPROBLEM,
            ]

        def process(state: int, is_primary: bool) -> None:
            if passed(state):
                test = self.running.remove(actuator.actuator_id, is_primary)
                if test is not None:
                    self.passed.append(test)
                    self._running_changed.set()
            elif failed(state):
                test = self.running.remove(actuator.actuator_id, is_primary)
                if test is not None:
                    self.failed.append(test)
                    self._running_changed.set()

        process(primary, True)

        self._primary_test[actuator.index] = primary

        if secondary is None:
            return

        process(secondary, False)

        self._secondary_test[actuator.s_index] = secondary

    @staticmethod
    async def run(
        m1m3: Remote,
        serial: bool = False,
        timeout: float = 20,
        skip: list[int] | None = None,
        start_callback: typing.Callable[[ForceActuatorData, bool], None] | None = None,
    ) -> tuple[BumpTestsList, BumpTestsList]:
        runner = BumpTestRunner(BumpTestsList.all_tests(m1m3, skip))

        try:
            old_callback = m1m3.evt_forceActuatorBumpTestStatus.callback
            m1m3.evt_forceActuatorBumpTestStatus.callback = runner.force_actuator_bump_test_status

            distance = 10
            if not (serial):
                distance = m1m3.evt_forceActuatorSettings.get().bumpTestMinimalDistance

            while True:
                test = await runner.next(distance, timeout)
                if test is None:
                    break

                primary = test.is_primary()

                await m1m3.cmd_forceActuatorBumpTest.set_start(
                    actuatorId=test.actuator.actuator_id,
                    testPrimary=primary,
                    testSecondary=not (primary),
                )

                if start_callback is not None:
                    start_callback(test.actuator, primary)

            await runner.wait_finish(timeout)

            return (runner.passed, runner.failed)

        finally:
            m1m3.evt_forceActuatorBumpTestStatus.callback = old_callback
