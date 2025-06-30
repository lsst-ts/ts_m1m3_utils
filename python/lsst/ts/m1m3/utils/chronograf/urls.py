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

from astropy.time import Time
from urllib.parse import urlencode, urlunparse

__all__ = ["DashboardURL", "M1M3_FA"]

class DashboardURL:
    def __init__(self, efd: str, **kwargs):
        self.efd = efd
        try:
            self.dashboard = kwargs[efd]
        except KeyError:
            raise RuntimeError(f"Unknow dashboard number for EFD {efd}.")

    def hostname(self) -> str:
        if self.efd == "summit_efd":
            return "summit-lsp.lsst.codes"
        if self.efd == "usdf_efd":
            return "usdf-rsp.slac.stanford.edu"


    def url(self, lower: Time, upper: Time, **kwargs) -> str:
        params = {
            "refresh": "Paused",
            "lower": lower.isot + "Z",
            "upper": upper.isot + "Z",
        } | dict([(f"tempVars[{key}]", value) for key, value in kwargs.items()])

        return urlunparse(
            (
                "https",
                self.hostname(),
                f"/chronograf/sources/1/dashboards/{self.dashboard}",
                "",
                urlencode(params),
                "",
            )
        )

class M1M3_FA(DashboardURL):
    def __init__(self, efd: str):
        super().__init__(efd, summit_efd = 199, usdf_efd = 61)
