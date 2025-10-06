# This file is part of ts_m1m3_utils.
#
# Developed for the LSST Data Management System.
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

__all__ = [
    "ThermocoupleAnalysis",
]


import asyncio
import re
from typing import List

import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.ts.xml.tables.m1m3 import (
    AirNozzleTable,
    Scanner,
    ThermocoupleData,
    ThermocoupleTable,
    find_thermocouple,
)
from lsst_efd_client import EfdClient
from scipy.spatial import cKDTree


class ThermocoupleAnalysis:
    """Compute M1M3 thermocouple values and metrics for a given
    time window.

    Parameters
    ----------
    client : `EfdClient`
        EFD client you want to use
    start_time : `Time`
        Astropy Time for beginning of query windo.
    end_time : `Time`
        Astropy Time for end of query window.
    time_bin : `int`
        Time bin in seconds (thermocouple records data every 30s
        so a bin smaller than 60s is not recommended).
        default: 60

    Attributes
    ----------
    efd_client : EfdClient
        EFD client to use for the EFD queries
    cold_junction_dataframe : pd.DataFrame
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are cold junction temperature.
    all_thermocouples_dataframe : pd.DataFrame
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are thermocouple temperature.
    vertical_cell_gradient_dataframe : pd.DataFrame
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are difference between the front and back
        temperature of the mirror.
    nonstandard_thermoocouples : List
        List of thermocouples that are in cells with nonstandard air
        nozzle configurations.
    xyz_r_gradients : pd.DataFrame
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are x, y, z, and radial gradients across
        the mirror in deg C/m.
    mean_vertical_cell_gradient : pd.DataFrame
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are the mean front-back temperature gradient
        across the mirror, excluding nonstandard thermocouples.
    bulk_glass_temperature_metrics : pd.DataFrame
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are different metrics related to the
        bulk glass temperature (mean temp, rate of change, etc.).
    vertical_gradient_temperature_metrics : pd.DataFrame
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are different metrics related to the
        front-back temperature (mean temp, rate of change, etc.).
    """

    def __init__(
        self,
        client: EfdClient,
    ):
        self.efd_client = client
        self.cold_junction_dataframe: pd.DataFrame | None = None
        self.all_thermocouples_dataframe: pd.DataFrame | None = None
        self.vertical_cell_gradient_dataframe: pd.DataFrame | None = None
        self.nonstandard_thermocouples: list = []
        self.xyz_r_gradients: pd.DataFrame | None = None
        self.mean_vertical_cell_gradient: pd.DataFrame | None = None
        self.bulk_glass_temperature_metrics: pd.DataFrame | None = None
        self.vertical_gradient_temperature_metrics: pd.DataFrame | None = None

    async def load(
        self,
        start_time: Time,
        end_time: Time,
        time_bin: int = 60,
        do_remove_cold_junction: bool = True,
        do_remove_offsets: bool = True,
    ) -> None:
        """Get all thermal scanner data within a given time window.

        Parameters
        ----------
        start_time : `Time`
            Astropy Time for beginning of query windo.
        end_time : `Time`
            Astropy Time for end of query window.
        time_bin : `int`
            Time bin in seconds (thermocouple records data every 30s
            so a bin smaller than 30s is not recommended).
            default: 30
        do_remove_cold_junction : `boolean`, optional
            If true, remove the cold junction offset for the thermal scanner.
            Defaults to True.
        do_remove_offsets : `boolean`, optional
            If true, remove offsets for each thermocouple.
            Defaults to True.

        Returns
        -------
        self.full_scanner_dataframe : `DataFrame`
            Time binned dataframe of EFD temperatures where the index
            is time and the columns are thermocouple temperature, including
            cold junction temperatures.
        self.vertical_gradient_dataframe : `DataFrame`
            Time binned dataframe of EFD temperatures where the index
            is time and the columns are difference between the front and back
            temperature of the mirror.
        """

        # bin frequency
        freq = str(time_bin) + "s"

        chunk_re = re.compile(r"m1m3-ts-0\d (\d+)/\d+")

        async def load_chunk(scanner: Scanner) -> list[pd.DataFrame] | None:
            fields = ["sensorName"] + [f"temperatureItem{s}" for s in range(16)]

            data = await self.efd_client.select_time_series(
                "lsst.sal.ESS.temperature",
                fields,
                start_time,
                end_time,
                index=scanner,
            )

            if data.empty:
                return None

            chunks: list[list[pd.Series]] = [[], [], []]

            for row in data.iterrows():
                chunk = chunk_re.match(row[1]["sensorName"])
                if chunk is None:
                    raise RuntimeError(
                        f"Unexpected sensorName for index {scanner}: {row[1]['sensorName']}"
                    )
                chunk_index = int(chunk[1]) - 1
                if chunk_index < len(chunks):
                    chunks[chunk_index].append(row)

            ret = [pd.concat([row[1] for row in chunk], axis=1).T for chunk in chunks]

            # rename columns

            for chunk_index, chunk_data in enumerate(ret):
                to_drop = ["sensorName"]
                for i in range(16):
                    name = f"temperatureItem{i}"
                    tc = find_thermocouple(scanner, chunk_index * 16 + i)
                    if tc is None:
                        if chunk_index == 0 and i == 0:
                            chunk_data.rename(
                                columns={name: f"coldJunction{scanner}"}, inplace=True
                            )
                        else:
                            to_drop.append(name)
                    else:
                        chunk_data.rename(columns={name: tc.name}, inplace=True)

                chunk_data.drop(columns=to_drop, inplace=True)

                ret[chunk_index] = chunk_data.resample(freq).median()

            return ret

        tasks: list[asyncio.Task] = []

        async with asyncio.TaskGroup() as tg:
            # Loop through every thermal scanner
            for scanner in Scanner:
                tasks.append(tg.create_task(load_chunk(scanner)))

        scanner_dataframe = pd.DataFrame()

        for scanner_data in [t.result() for t in tasks]:
            if scanner_data is None:
                continue

            for chunk in scanner_data:
                if scanner_dataframe.empty:
                    scanner_dataframe = chunk
                else:
                    scanner_dataframe = scanner_dataframe.join(chunk)

        if not (scanner_dataframe.empty):
            # Remove the cold junction offset if desired
            if do_remove_cold_junction:
                scanner_dataframe = self.__remove_cold_junction_gradient(
                    scanner_dataframe
                )

            # Remove the individual thermocouple offset if desired
            if do_remove_offsets:
                scanner_dataframe = self.__remove_offsets(scanner_dataframe)

            cold_junction_columns = [
                "coldJunction114",
                "coldJunction115",
                "coldJunction116",
                "coldJunction117",
            ]
            self.cold_junction_dataframe = scanner_dataframe[cold_junction_columns]
            if "coldJunctionMean" in scanner_dataframe.columns:
                cold_junction_columns.append("coldJunctionMean")
            self.all_thermocouples_dataframe = scanner_dataframe.drop(
                columns=cold_junction_columns
            )

            self.find_nonstandard_thermocouples()

            self.calculate_vertical_differences()

            self.xyz_r_gradients = self.calculate_gradients_xy_r()
            _, z_filtered = self.__coordinate_map(
                make_3d_map=False, remove_nonstandard_cells=True
            )
            self.mean_vertical_cell_gradient = z_filtered.mean(axis=1, skipna=True)

            self.bulk_glass_temperature_metrics = self.compute_temp_stats_and_rate(
                data=self.all_thermocouples_dataframe,
            )
            self.vertical_gradient_temperature_metrics = (
                self.compute_temp_stats_and_rate(
                    data=self.vertical_cell_gradient_dataframe
                )
            )

    def __remove_cold_junction_gradient(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove the thermal scanner cold junction offset.

        Parameters
        ----------
        data : pandas dataframe
            Time binned dataframe of EFD temperatures where the index
            is time and the columns are thermocouple temperature.

        Returns
        -------
        data : pandas dataframe
            Time binned dataframe of EFD temperatures where the index
            is time and the columns are thermocouple temperature with the cold
            junction temperature removed.
        """
        data["coldJunctionMean"] = data[
            [
                f"coldJunction{Scanner.TS_01}",
                f"coldJunction{Scanner.TS_02}",
                f"coldJunction{Scanner.TS_03}",
                f"coldJunction{Scanner.TS_04}",
            ]
        ].mean(axis=1)
        for column in data.columns:
            for thermocouple in ThermocoupleTable:
                if thermocouple.name == column:
                    scanner_number = str(thermocouple.scanner.value)
                    data[column] = np.array(data[column]) - np.array(
                        data["coldJunction" + scanner_number] - data.coldJunctionMean
                    )
        return data

    def __remove_offsets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove the individual thermocouple reference offsets.
        This is based on one morning of very stable data in 2025:
        7/21.

        Parameters
        ----------
        data : pandas dataframe
            Time binned dataframe of EFD temperatures where the index is
            time and the columns are thermocouple temperature. Columns names
            are thermocouple names from ThermocoupleData structure - starting
            with MTC.

            The algorithm doesn't check if all names are present.

        Returns
        -------
        data : pandas dataframe
            Time binned dataframe of EFD temperatures where the
            index is time and the columns are thermocouple temperature
            with the cold junction temperature removed.

        Raises
        ------
        KeyError
            Raised if some TC names are missing in supplied data.
        """
        reference_offsets = {
            "MTC039M": -0.09955408293725476,
            "MTC039F": -0.12934326581214606,
            "MTC038B2": -0.055332098921683226,
            "MTC040B1": -0.06205412532277954,
            "MTC040M": -0.08473186478404315,
            "MTC040F": -0.0946209376261109,
            "MTC034B": -0.1804543811618089,
            "MTC036B": -0.08947639450816425,
            "MTC036F": -0.20897644876163568,
            "MTC035B": -0.12934326581214606,
            "MTCOW9B": 0.007123608204510745,
            "MTCOW9M": 0.0077012169488135385,
            "MTCOW9F": -0.029376262943493714,
            "MTC037B": -0.054587491208195615,
            "MTC037F": -0.08176530187714093,
            "MTCOW10B": -0.029354222470402647,
            "MTCOW10M": -0.07327632889537128,
            "MTCOW10F": -0.06555432199161615,
            "MTCIW6B": -0.044743135095187014,
            "MTCIW6M": -0.06975430368106927,
            "MTCIW6F": -0.08903198227671894,
            "MTC039B": -0.07186520879852765,
            "MTC026B2": -0.019443003191431885,
            "MTC030B2": 0.014057032412409853,
            "MTCIW5B": -0.031543222917994385,
            "MTCIW5F": -0.06332102867021838,
            "MTCOW8B": -0.028043026249157776,
            "MTCOW8M": -0.08740967630069818,
            "MTCOW8F": -0.10452092474091046,
            "MTC031B": -0.16197641146237451,
            "MTC031F": -0.15106521168074316,
            "MTC033B": -0.09756535409610834,
            "MTC033M": -0.08742101442868311,
            "MTC033F": -0.08424315967879004,
            "MTC032B": -0.046254179067625856,
            "MTC032F": -0.07856540665416034,
            "MTC029B": -0.03503208145884784,
            "MTC029M": -0.04522081996389282,
            "MTC029F": -0.06226531120195666,
            "MTC028B": -0.03284318697506983,
            "MTCOW7B": 0.015456920345080505,
            "MTCOW7M": -0.04544323800724115,
            "MTCOW7F": -0.05135432334795276,
            "MTC030B1": 0.0010791144233142091,
            "MTC030M": -0.028098657250948733,
            "MTC030F": -0.05679863809269037,
            "MTCOW6F": 0.09873462267662347,
            "MTCIW4B": -0.05119855654294092,
            "MTCIW4F": -0.07483198363305554,
            "MTC023B": 0.009179200211617555,
            "MTC023M": 0.002990277753392334,
            "MTC023F": 0.002967919388863649,
            "MTC024B": -0.016476440284531435,
            "MTC024F": -0.031098704722737125,
            "MTC025B": 0.04817907559817236,
            "MTC025F": -0.004209751407849183,
            "MTC026B1": -0.013698704892277647,
            "MTC026F": -0.034343210710963135,
            "MTC027B": 0.00297904558922113,
            "MTC017B1": -0.04066532968204584,
            "MTC017F": -0.07959855383026593,
            "MTC018B2": -0.10783197600366101,
            "MTC019B": -0.035487619890650635,
            "MTC021B": -0.06317638806556403,
            "MTC021F": -0.0924209169419754,
            "MTC020B": -0.04405426434730231,
            "MTCOW5B": -0.03243204738088501,
            "MTCOW5M": -0.0791762880357254,
            "MTCOW5F": -0.07200984940318378,
            "MTC022B": -0.0388653223917359,
            "MTC022F": -0.07860969952796637,
            "MTCOW6B": 0.1520569908745948,
            "MTCOW6M": 0.07679025876467627,
            "MTCOW4F": 0.02506794414200897,
            "MTCIW3B": 0.05910129561634747,
            "MTCIW3M": 0.045901489402968565,
            "MTCIW3F": 0.035601276966716,
            "MTC016B": 0.05130151127390015,
            "MTC016M": 0.032001156422282406,
            "MTC016F": 0.03281241537410651,
            "MTC017B2": 0.06815717499731555,
            "MTC018B1": 0.004190423898682738,
            "MTC018M": -0.00917638128387921,
            "MTC018F": -0.046887418813719606,
            "MTC010F": 0.04961255405954468,
            "MTC012B": 0.0340679805935622,
            "MTC014B": 0.015179295155192563,
            "MTC014F": -0.029898664541258668,
            "MTC013B": 0.0028124704752894303,
            "MTCOW3B": 0.0832902909776756,
            "MTCOW3M": 0.0810123868910324,
            "MTCOW3F": 0.051968129621068115,
            "MTC015B": 0.09647928888213642,
            "MTC015F": 0.054268074180800596,
            "MTCOW4B": 0.08386811164960584,
            "MTCOW4M": 0.034312651037625486,
            "MTC007B2": -0.0444540658135022,
            "MTC008B2": 0.0663347033786632,
            "MTCIW2B": 0.045101462615317445,
            "MTCIW2F": 0.01472375672338977,
            "MTCOW2B": 0.04367921631811633,
            "MTCOW2M": 0.02490136902807727,
            "MTCOW2F": 0.0011901644992686755,
            "MTC009B": 0.07799040490996845,
            "MTC009F": 0.011979293968398252,
            "MTC011B": 0.04401257847360718,
            "MTC011M": 0.03985710158558575,
            "MTC011F": 0.028901291038710752,
            "MTC010B": 0.10673478458932983,
            "MTC043B": 0.025867970929661865,
            "MTC043F": 0.00037901151125829813,
            "MTC042B": 0.021868048919028382,
            "MTCOW11B": 0.14397915748700818,
            "MTCOW11M": 0.11446812962106812,
            "MTCOW11F": 0.12136806926407928,
            "MTC044B": 0.08823466845299066,
            "MTC044F": 0.039145978436986084,
            "MTCOW12B": -0.011898591638155764,
            "MTCOW12M": -0.02630977616099628,
            "MTCOW12F": -0.06358752765974884,
            "MTC038B1": 0.08579040118004144,
            "MTC038F": 0.05191260458309088,
            "MTC040B2": 0.09525678437231555,
            "MTC041B": 0.05055711548803821,
            "MTC006B": 0.028779326690024476,
            "MTC006M": 0.0033348720730526082,
            "MTC006F": -0.010942904009302978,
            "MTC005B": 0.09879035964222638,
            "MTCOW1B": 0.19741236383330651,
            "MTCOW1M": 0.12783461691219244,
            "MTCOW1F": 0.12997921852216443,
            "MTC008B1": 0.12564582839222638,
            "MTC008M": 0.10631251879478754,
            "MTC008F": 0.07534586603057214,
            "MTCIW1B": 0.11469022977297705,
            "MTCIW1F": 0.09993445093047448,
            "MTC001B": 0.09057911781415662,
            "MTC001M": 0.09715692746584814,
            "MTC001F": 0.08035700070803564,
            "MTC002B": 0.07372366601836511,
            "MTC002F": 0.06742348155655975,
            "MTC003B": 0.024823591589383298,
            "MTC003F": -0.0048543292819278605,
            "MTC007B1": 0.0107459811496593,
            "MTC007F": 0.006834750850451599,
            "MTC004B": 0.09557912629126086,
        }

        for tc_name, offset in reference_offsets.items():
            data[tc_name] = data[tc_name] - offset

        return data

    def find_nonstandard_thermocouples(self) -> None:
        """Make a list of all thermocouples with nonstandard
        air nozzle configurations in their cells.
        """
        nonstandard_thermocouples: List[ThermocoupleData] = []

        for thermocouple in ThermocoupleTable:
            nozzle_status = [
                s.nozzle.value
                for s in AirNozzleTable
                if s.cell == thermocouple.core_location
            ][0]
            if nozzle_status in ["blocked", "sshort", "covered"]:
                nonstandard_thermocouples.append(thermocouple)

        self.nonstandard_thermocouples = nonstandard_thermocouples

    def __coordinate_map(
        self,
        make_3d_map: bool = True,
        remove_nonstandard_cells: bool = False,
    ) -> tuple[np.array, pd.DataFrame]:
        """Return a map

        Parameters
        ----------
        make_3D_map : boolean
            If true, return z positions (0 if B, 0.5 if M, 1 if F). If false,
            assume that the data is 2D.
        remove_nonstandard_cells : boolean
            If false, return xyz positions and dataframe for all thermocouples.
            If true, return xyz positions only for normally
            installed air nozzle locations.

        Returns
        -------
        positions : array positions in x, y, and possibly z
        data : pandas dataframe for only the included thermocouples
        """

        data: pd.DataFrame

        if make_3d_map:
            data = self.all_thermocouples_dataframe
        else:
            data = self.vertical_cell_gradient_dataframe

        thermocouple: List[ThermocoupleData] = []
        the_thermocouple: ThermocoupleData

        coord = {}
        for column in data.columns:
            if make_3d_map:
                thermocouple = [s for s in ThermocoupleTable if s.name == column]
            else:
                thermocouple = [s for s in ThermocoupleTable if s.name == column + "F"]
            if len(thermocouple) == 0:
                continue
            the_thermocouple = thermocouple[0]
            if remove_nonstandard_cells:
                if the_thermocouple not in self.nonstandard_thermocouples:
                    if make_3d_map:
                        coord.update(
                            {
                                the_thermocouple.name: (
                                    the_thermocouple.x_position,
                                    the_thermocouple.y_position,
                                    the_thermocouple.z_position,
                                )
                            }
                        )
                    else:
                        coord.update(
                            {
                                the_thermocouple.name: (
                                    the_thermocouple.x_position,
                                    the_thermocouple.y_position,
                                )
                            }
                        )
            else:
                if make_3d_map:
                    coord.update(
                        {
                            the_thermocouple.name: (
                                the_thermocouple.x_position,
                                the_thermocouple.y_position,
                                the_thermocouple.z_position,
                            )
                        }
                    )
                else:
                    coord.update(
                        {
                            the_thermocouple.name: (
                                the_thermocouple.x_position,
                                the_thermocouple.y_position,
                            )
                        }
                    )
        if make_3d_map:
            common = [c for c in data.columns if c in coord]
            xyz = np.array([coord[n] for n in common])
        else:
            common = [c for c in data.columns if c + "F" in coord]
            xyz = np.array([coord[n + "F"] for n in common])

        return xyz.T, data[common]

    def calculate_vertical_differences(self) -> None:
        """Calculate vertical gradients."""
        # Suffixes
        b_suffixes = ["B", "B1"]
        f_suffix = "F"

        # Store results
        result = {}

        if self.all_thermocouples_dataframe is not None:

            for col in self.all_thermocouples_dataframe.columns:
                for b_suffix in b_suffixes:
                    if col.endswith(b_suffix):
                        base_name = col[: -len(b_suffix)]
                        f_col = base_name + f_suffix
                        if f_col in self.all_thermocouples_dataframe.columns:
                            result[base_name] = (
                                self.all_thermocouples_dataframe[f_col]
                                - self.all_thermocouples_dataframe[col]
                            )

            # Create the new DataFrame
            self.vertical_cell_gradient_dataframe = pd.DataFrame(result)

    def calculate_gradients_xy_r(
        self,
        remove_nonstandard_cells: bool = True,
        use_3d_dataset: bool = True,
        k: int = 20,
        power: float = 1.0,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        """
        Estimate per-point (gx, gy) and radial gradient gr
        using local weighted plane fits.

        Parameters
        ----------
        remove_nonstandard_cells : boolean
            If false, return xyz positions and dataframe for all thermocouples.
            If true, return xyz positions only for normally
            installed air nozzle locations.
        use_3d_dataset : boolean
            If true, use the 3D data.
            If false, use the z difference dataset
        k : int
            Number of nearest neighbors used per fit (>=3). 10–20 is typical.
        power : float
            Distance weighting exponent; w = 1/(d+eps)**power.
            Increase to emphasize local detail.
        eps : float
            Small number to avoid division by zero.

        Returns
        -------
        pd.DataFrame
            Columns:
            - 'x_gradient'   : Estimated mean ∂t/∂x
            - 'y_gradient'    : Estimated mean ∂t/∂y
            - 'radial_gradient'  : Estimated radial gradient
            - 'z_gradient'  :  Estimated mean ∂t/∂z if use_3d_dataset=True
        """

        xyz, temperatures = self.__coordinate_map(
            make_3d_map=use_3d_dataset,
            remove_nonstandard_cells=remove_nonstandard_cells,
        )

        x = np.asarray(xyz[0]).astype(float)
        y = np.asarray(xyz[1]).astype(float)
        n = x.size

        if use_3d_dataset:
            z = np.asarray(xyz[2]).astype(float)
            pts = np.column_stack([x, y, z])
        else:
            pts = np.column_stack([x, y])

        tree = cKDTree(pts)

        # Query k neighbors (include the point itself)
        dists, idxs = tree.query(pts, k=min(k, n))

        gx = np.full(n, np.nan)
        gy = np.full(n, np.nan)
        gr = np.full(n, np.nan)

        if use_3d_dataset:
            gz = np.full(n, np.nan)
            all_gz = []

        all_gx = []
        all_gy = []
        all_gr = []

        for k, row in temperatures.iterrows():
            temperature = row.to_numpy()
            temperature = np.asarray(temperature).astype(float)
            for i in range(n):
                nn = idxs[i]
                if use_3d_dataset:
                    xi, yi, zi, ti = x[nn], y[nn], z[nn], temperature[nn]
                else:
                    xi, yi, ti = x[nn], y[nn], temperature[nn]
                di = dists[i]

                # Weights: closer neighbors count more
                w = 1.0 / np.power(di + eps, power)

                # Design matrix for plane: [x y 1]
                if use_3d_dataset:
                    A = np.column_stack([xi, yi, zi, np.ones_like(xi)])
                else:
                    A = np.column_stack([xi, yi, np.ones_like(xi)])

                # Weighted least squares solve for [a, b, c]
                # Solve (A^T W A) beta = A^T W z
                W = w
                Aw = A * W[:, None]
                M = Aw.T @ A
                rhs = Aw.T @ ti
                if use_3d_dataset:
                    try:
                        a, b, c, d = np.linalg.solve(M, rhs)
                    except np.linalg.LinAlgError:
                        # Fallback to pseudo-inverse if neighbors
                        # are degenerate
                        a, b, c, d = np.linalg.pinv(M) @ rhs
                else:
                    try:
                        a, b, c = np.linalg.solve(M, rhs)
                    except np.linalg.LinAlgError:
                        # Fallback to pseudo-inverse if neighbors
                        # are degenerate
                        a, b, c = np.linalg.pinv(M) @ rhs

                if use_3d_dataset:
                    gx[i], gy[i], gz[i] = a, b, c

                    rvec = np.array([x[i], y[i], z[i]], dtype=float)

                else:
                    gx[i], gy[i] = a, b

                    rvec = np.array([x[i], y[i]], dtype=float)

                rnorm = np.linalg.norm(rvec)
                if rnorm > 0:
                    gvec = np.array(
                        [gx[i], gy[i]] + ([gz[i]] if use_3d_dataset else []),
                        dtype=float,
                    )
                    gr[i] = float(gvec.dot(rvec) / rnorm)  # projection onto r̂

            all_gx.append(np.nanmean(gx))
            all_gy.append(np.nanmean(gy))
            all_gr.append(np.nanmean(gr))
            if use_3d_dataset:
                all_gz.append(np.nanmean(gz))

        if use_3d_dataset:
            return pd.DataFrame(
                data={
                    "x_gradient": all_gx,
                    "y_gradient": all_gy,
                    "z_gradient": all_gz,
                    "radial_gradient": all_gr,
                },
                index=temperatures.index,
            )

        else:
            return pd.DataFrame(
                data={
                    "x_gradient": all_gx,
                    "y_gradient": all_gy,
                    "radial_gradient": all_gr,
                },
                index=temperatures.index,
            )

    def compute_temp_stats_and_rate(
        self,
        data: pd.DataFrame,
        window_minutes: float = 30,
    ) -> pd.DataFrame:
        """
        Compute per-timestamp temperature stats and a windowed rate of change.

        Parameters
        ----------
        data : pd.DataFrame
            Index: pandas.DatetimeIndex (tz-aware or naive).
            Columns: temperature sensors. Values are temperatures (any units).
        window_minutes : float
            Size of the look-back window (e.g., 30 for 30 minutes).

        Returns
        -------
        pd.DataFrame
            Columns:
            - 'mean_temp'   : mean across sensors (row-wise)
            - 'std_temp'    : std across sensors (row-wise)
            - 'range_temp'  : (row-wise max - min)
            - f'rate_per_{window_minutes}min': d(mean)/dt over the window,
                in temperature-units per minute. Rows lacking enough history
                are NaN. Length matches len(df) and index matches df.index.
        """

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pandas.DatetimeIndex.")

        # Row-wise stats (preserve original index order, allow NaNs)
        mean_temp = data.mean(axis=1, skipna=True)
        std_temp = data.std(axis=1, skipna=True)
        range_temp = data.max(axis=1, skipna=True) - data.min(axis=1, skipna=True)

        # Build a time-sorted, duplicate-collapsed mean series
        # for interpolation
        s = mean_temp.groupby(level=0).mean().sort_index()

        # If there aren't at least two valid points, rate is entirely NaN
        if s.notna().sum() < 2:
            rate = pd.Series(np.nan, index=data.index)
        else:
            # Time in minutes since epoch (works for tz-aware or naive)
            t_minutes = s.index.asi8 / (1e9 * 60.0)  # ns -> minutes
            y = s.to_numpy().astype(float)

            valid = ~np.isnan(y)
            t_valid = t_minutes[valid]
            y_valid = y[valid]

            half = float(window_minutes) / 2.0

            # Interpolate mean at t - half and t + half
            y_prev = np.interp(
                t_minutes - half,
                t_valid,
                y_valid,
                left=np.nan,
                right=np.nan,
            )
            y_next = np.interp(
                t_minutes + half,
                t_valid,
                y_valid,
                left=np.nan,
                right=np.nan,
            )

            # Centered rate (temperature per minute) on the collapsed,
            # sorted index
            rate_sorted = pd.Series(
                (y_next - y_prev) / float(window_minutes),
                index=s.index,
            )

            # Map back to original df index/order
            rate = rate_sorted.reindex(data.index)

        rate_col = f"rate_per_{int(window_minutes)}min"
        out = pd.DataFrame(
            {
                "mean_temp": mean_temp,
                "std_temp": std_temp,
                "range_temp": range_temp,
                rate_col: rate,
            }
        )

        return out
