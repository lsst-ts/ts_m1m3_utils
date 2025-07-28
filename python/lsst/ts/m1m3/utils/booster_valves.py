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

import logging
from dataclasses import dataclass
from typing import AsyncGenerator

import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient


@dataclass
class BoosterValveOpened:
    start: Time
    end: Time


class BoosterValves:
    """
    Parameters
        ----------
        client : EfdClient
            The EFD client to use for querying data.
    """

    def __init__(
        self,
        client: EfdClient,
        cache_delta: TimeDelta = TimeDelta(3 * 3600, format="sec"),
    ):
        super().__init__()
        self.client = client
        self.cache_delta = cache_delta
        self.cache: pd.DataFrame | None = None

    async def find_opened(
        self, start: Time, end: Time
    ) -> AsyncGenerator[BoosterValveOpened, None]:
        query_start = start

        self.cache = None
        period_start: Time | None = None

        while query_start < end:
            if (
                self.cache is None
                or len(self.cache.index) == 0
                or Time(self.cache.index[0]) < query_start
            ):
                query_end = min(query_start + self.cache_delta, end)
                logging.info(
                    "Quering %s - %s for booster valve activation.",
                    query_start.isot,
                    query_end.isot,
                )
                self.cache = await self.client.select_time_series(
                    "lsst.sal.MTM1M3.logevent_boosterValveStatus",
                    ["opened"],
                    query_start,
                    query_end,
                )
                query_start = query_end

            for time, row in self.cache.iterrows():
                if row.opened:
                    if period_start is None:
                        period_start = time
                else:
                    if period_start is not None:
                        yield BoosterValveOpened(Time(period_start), Time(time))
                        period_start = None
