# This file is part of ts_m1m3_utils.
#
# Developed for the Rubin Observatory Telescope and Site System.
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

import unittest

from lsst.ts.m1m3.utils import BumpTestKind, BumpTestRunner, ForceActuatorBumpTest
from lsst.ts.xml.enums.MTM1M3 import BumpTest
from lsst.ts.xml.tables.m1m3 import FATable


class ForceActuatorBumpTestTestCase(unittest.IsolatedAsyncioTestCase):
    def test_primary(self) -> None:
        self.assertEqual(
            ForceActuatorBumpTest(
                FATable[0], BumpTestKind.CYLINDER_PRIMARY
            ).is_primary(),
            True,
        )


class BumpTestRunnerTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_next(self) -> None:
        runner = BumpTestRunner(
            [
                ForceActuatorBumpTest(FATable[1], BumpTestKind.AXIS_Z),
                ForceActuatorBumpTest(FATable[1], BumpTestKind.AXIS_Y),
            ]
        )

        test = await runner.next()
        self.assertEqual(test.actuator.actuator_id, 102)
        self.assertEqual(test.kind, BumpTestKind.AXIS_Z)

        runner.progress(
            FATable[1], BumpTest.FAILED_TESTEDPOSITIVE_OVERSHOOT, BumpTest.TRIGGERED
        )

        test = await runner.next()
        self.assertEqual(test.actuator.actuator_id, 102)
        self.assertEqual(test.kind, BumpTestKind.AXIS_Y)

        test = await runner.next()
        self.assertEqual(test, None)

    async def test_distance(self) -> None:
        runner = BumpTestRunner(
            [
                ForceActuatorBumpTest(FATable[143], BumpTestKind.AXIS_Z),
                ForceActuatorBumpTest(FATable[142], BumpTestKind.CYLINDER_PRIMARY),
                ForceActuatorBumpTest(FATable[64], BumpTestKind.AXIS_Z),
            ]
        )

        test = await runner.next(2)
        self.assertEqual(test.actuator.actuator_id, 431)
        self.assertEqual(test.kind, BumpTestKind.AXIS_Z)

        test = await runner.next(2)
        self.assertEqual(test.actuator.actuator_id, 230)
        self.assertEqual(test.kind, BumpTestKind.AXIS_Z)

        runner.progress(FATable[143], BumpTest.FAILED_TESTEDPOSITIVE_OVERSHOOT, None)

        test = await runner.next(2)
        self.assertEqual(test.actuator.actuator_id, 430)
        self.assertEqual(test.kind, BumpTestKind.CYLINDER_PRIMARY)

        test = await runner.next(2)
        self.assertEqual(test, None)

        runner.progress(FATable[64], BumpTest.PASSED, None)

        with self.assertRaises(TimeoutError):
            await runner.wait_finish(1)

        runner.progress(FATable[142], BumpTest.PASSED, None)
        try:
            await runner.wait_finish(1)
        except TimeoutError:
            self.fail("Raised timeout error when all tests were finished.")


if __name__ == "__main__":
    unittest.main()
