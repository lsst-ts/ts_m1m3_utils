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

__all__ = ["FCUStats"]

import pandas as pd
from astropy.time import Time
from lsst_efd_client import EfdClient


class FCUStats:
    """
    Parameters
        ----------
        client : EfdClient
            The EFD client to use for querying data.
    """

    statistics: dict[int, pd.DataFrame] = {}

    def __init__(self, client: EfdClient | None = None):
        super().__init__()
        self.client = EfdClient("summit_efd") if client is None else client
        self.data: pd.DataFrame | None = None

    async def fcu_quick_analysis(
        self,
        fcu_indices: list[int],
        start_time: Time,
        end_time: Time,
        set_point: float,
        plot: bool = True,
        show_url: bool = True,
    ) -> None:
        """
        Perform a quick analysis of FCU temperature data.

        Parameters
        ----------
        fcu_indices : list of int
            The FCU index to analyze (0-95).
            If a list is provided, the analysis will be performed for each
            index.
        timestamp : str
            The timestamp in ISO format (e.g., "2025-05-10T12:00:00").
        delta_t : str
            The duration string (e.g., "-10s" for 10 seconds before the
            timestamp).
        set_point : float
            The temperature set point for the FCU in degrees Celsius.
        plot : bool, optional
            Whether to plot the temperature data. Default is True.
        show_url : bool, optional
            Whether to show the URL for Summit Chronograf. Default is True.
        return_output: bool, optional
            Whether return the output or not. Default is False.
        """
        self.data = await self.client.select_time_series(
            "lsst.sal.MTM1M3TS.thermalData",
            ["timestamp"] + [f"absoluteTemperature{i}" for i in fcu_indices],
            start_time,
            end_time,
        )
        if self.data.empty:
            return

        for fcu_index in fcu_indices:
            col = f"absoluteTemperature{fcu_index}"
            self.statistics[fcu_index] = self.data[col].agg(["min", "mean", "median", "max", "std"])
