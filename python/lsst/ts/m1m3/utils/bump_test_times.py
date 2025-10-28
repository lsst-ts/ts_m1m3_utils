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

from dataclasses import dataclass
from typing import AsyncGenerator

from astropy.time import Time, TimeDelta
from lsst.ts.xml.enums.MTM1M3 import BumpTest as BumpTestStatus
from lsst.ts.xml.tables.m1m3 import ForceActuatorData
from lsst_efd_client import EfdClient


@dataclass
class BumpTest:
    """Represent a single bump test occurence.

    Attributes
    ----------
    fa : `ForceActuatorData`
    start_time : `Time`
    end_time : `Time`
    result : `BumpTestStatus`
    """

    fa: ForceActuatorData
    start_time: Time
    end_time: Time
    result: BumpTestStatus | None


class BumpTestTimes:
    """Returns a set of times for bump tests given the actuator ID and
    a time range.

    Parameters
    ----------
    client : lsst_efd_client.efd_helper.EfdClient object, optional
        This is used to query the EFD. Default to summit_efd client.
    """

    def __init__(self, client: EfdClient | None = None):
        super().__init__()
        self.client = EfdClient("summit_efd") if client is None else client

    async def find_times(
        self,
        fa: ForceActuatorData,
        primary: bool,
        start: Time = (Time.now() - TimeDelta(7, format="jd")),
        end: Time = Time.now(),
        start_delta: TimeDelta = TimeDelta(3, format="sec"),
    ) -> AsyncGenerator[BumpTest, None]:
        """Find bump test query times

        Parameters
        ----------
        fa : `ForceActuatorData`
            Force Actuator identification number. Starting with 101, the first
            number identified segment (1-4). The value ranges up to 443.
        primary : `bool`
            If true, search primary cylinder (Z) bump tests.
        start: 'Time', optional
            Astropy Time of search start. Defaults to week ago.
        end: `Time`
            Astropy Time of search end. Defaults to current time.
        start_delta : `TimeDelta`, optional
            Delta to subtract from start of the tests. Defaults to 3 seconds.

        Returns
        -------
        tests : `[BumpTest]`
            List of performed bump tests.
        """
        # Find the test names
        status = f"primaryTest{fa.index}" if primary else f"secondaryTest{fa.s_index}"

        query = (
            f"SELECT time, {status} "
            'FROM "efd"."autogen"."lsst.sal.MTM1M3.logevent_forceActuatorBumpTestStatus" '
            f"WHERE time >= '{start.isot}Z' AND time <= '{end.isot}Z' "
            f"AND {status} = {BumpTestStatus.TESTINGPOSITIVE}"
        )
        bumps = await self.client._do_query(query)

        end_time: Time | None = None

        # Now find the separate tests
        for time, row in bumps.iterrows():
            start_time = Time(time)
            if end_time is not None and start_time < end_time:
                continue

            end_time = start_time + TimeDelta(60, format="sec")

            ends = await self.client._do_query(
                f"SELECT time, {status} "
                'FROM "efd"."autogen"."lsst.sal.MTM1M3.logevent_forceActuatorBumpTestStatus" '
                f"WHERE time > '{start_time.isot}Z' "
                f"AND time <= '{end_time.isot}Z' "
                f"AND {status} >= {BumpTestStatus.PASSED}"
            )
            start_time -= start_delta
            if len(ends) == 0:
                yield BumpTest(fa, start_time, None, None)
            else:
                end_time = Time(ends.index[0])
                yield BumpTest(fa, start_time, end_time, BumpTestStatus(ends[status].iloc[0]))
