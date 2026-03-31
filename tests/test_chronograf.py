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

from astropy.time import Time

from lsst.ts.m1m3.utils.chronograf import DashboardURL, M1M3ForceActuatorForces
from lsst.ts.xml.tables.m1m3 import FATable


class TestDashboardURL(unittest.IsolatedAsyncioTestCase):
    def test_url(self) -> None:
        usdf = DashboardURL("usdf_efd", usdf_efd=120, summit_efd=111)
        summit = DashboardURL("summit_efd", summit_efd=101)

        t1 = Time("2025-06-04T23:18:19")
        t2 = Time("2025-06-05T11:18:12")

        assert (
            usdf.url(t1, t2, p1="AAA", p2=10, p3=11.112)
            == "https://usdf-rsp.slac.stanford.edu/chronograf/sources/1/dashboards/120"
            "?refresh=Paused&lower=2025-06-04T23%3A18%3A19.000Z&upper=2025-06-05T11%3A18%3A12.000Z"
            "&tempVars%5Bp1%5D=AAA&tempVars%5Bp2%5D=10&tempVars%5Bp3%5D=11.112"
        )

        assert (
            summit.url(t1, t2, p1="aaa")
            == "https://summit-lsp.lsst.codes/chronograf/sources/1/dashboards/101?refresh=Paused"
            "&lower=2025-06-04T23%3A18%3A19.000Z&upper=2025-06-05T11%3A18%3A12.000Z&tempVars%5Bp1%5D=aaa"
        )


class TestFAs(unittest.IsolatedAsyncioTestCase):
    def test_url(self) -> None:
        usdf = M1M3ForceActuatorForces("usdf_efd")
        assert (
            usdf.url(
                Time("2025-05-05T21:13:17.456"),
                Time("2025-05-05T21:13:18.23"),
                xIndex=101,
            )
            == "https://usdf-rsp.slac.stanford.edu/chronograf/sources/1/dashboards/61?refresh=Paused"
            "&lower=2025-05-05T21%3A13%3A17.456Z&upper=2025-05-05T21%3A13%3A18.230Z&tempVars%5BxIndex%5D=101"
        )

        assert (
            usdf.fa_url(Time("2025-06-12T12:31:12"), Time("2025-06-12T13:45:11"), FATable[2])
            == "https://usdf-rsp.slac.stanford.edu/chronograf/sources/1/dashboards/61?refresh=Paused"
            "&lower=2025-06-12T12%3A31%3A12.000Z&upper=2025-06-12T13%3A45%3A11.000Z"
            "&tempVars%5Bx_index%5D=0&tempVars%5By_index%5D=103&tempVars%5Bs_index%5D=103"
            "&tempVars%5Bz_index%5D=103"
        )
