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

import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.ts.xml.tables.m1m3 import ThermocoupleTable
from lsst_efd_client import EfdClient


async def get_scanner_data(
    client: EfdClient,
    start_time: Time,
    end_time: Time,
    time_bin: int = 30,
    do_remove_cold_junction: bool = True,
    do_remove_offsets: bool = True,
) -> pd.DataFrame:
    """Get all thermal scanner data within a given time window.

    Parameters
    ----------
    client : EFD client
        EFD client you want to use
    start_time : Time
        Astropy Time for beginning of query windo.
    end_time : Time
        Astropy Time for end of query window.
    time_bin : int
        Time bin in seconds (thermocouple records data every 30s
        so a bin smaller than 30s is not recommended).
        default: 30
    do_remove_cold_junction : boolean
        If true, remove the cold junction offset for the thermal scanner.
        default: True
    do_remove_offsets : boolean
        If true, remove offsets for each thermocouple.
        default: True

    Return
    ------
    scanner_dataframe : pandas dataframe
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are thermocouple temperature.
    """

    scanner_dataframe = pd.DataFrame()

    # Loop through every thermal scanner
    for scanner in range(4):
        thermocouples = [
            [thermo.name, thermo.channel % 16, int(thermo.channel / 16) + 1]
            for thermo in ThermocoupleTable
            if thermo.scanner.value == (114 + scanner)
        ]

        # Loop through every EFD division in that scanner
        for sensor in range(1, 7):
            data_query = ""
            query_thermocouples = [
                [name, channel]
                for [name, channel, star] in thermocouples
                if star == sensor
            ]
            if query_thermocouples == []:
                continue
            else:
                # Create or add to an EFD query for
                # this EFD divisition of this scanner
                if sensor == 1:
                    data_query = f'''SELECT ("temperatureItem0") AS "coldJunction{scanner + 114}"'''
                for name, channel in query_thermocouples:
                    if data_query == "":
                        data_query = (
                            f'''SELECT ("temperatureItem{channel}") AS "{name}"'''
                        )
                    else:
                        data_query += f''', ("temperatureItem{channel}") AS "{name}"'''

                data_query += f""" FROM "efd"."autogen"."lsst.sal.ESS.temperature"
                    WHERE salIndex = {114 + scanner} AND time > '{start_time.isot}Z'
                    and time <= '{end_time.isot}Z'
                    """
                data = await client.query(data_query)

                if len(scanner_dataframe) == 0:
                    scanner_dataframe = data
                else:
                    # if the query returns results, bin the results
                    # based on the time_bin
                    freq = str(time_bin) + "s"

                    data_bin = data.resample(freq).mean()
                    scanner_bin = scanner_dataframe.resample(freq).mean()

                    # Keep the dataframe where both subdivisions of
                    # thermal scanners have data
                    have1 = data_bin.resample(freq).size() > 0
                    have2 = scanner_bin.resample(freq).size() > 0
                    valid = have1 & have2  # only bins where both had data

                    scanner_dataframe = pd.concat(
                        [data_bin[valid], scanner_bin[valid]], axis=1
                    )

    # Remove the cold junction offset if desired
    if do_remove_cold_junction:
        scanner_dataframe = remove_cold_junction_gradient(scanner_dataframe)

    # Remove the individual thermocouple offset if desired
    if do_remove_offsets:
        scanner_dataframe = remove_offsets(scanner_dataframe)

    return scanner_dataframe


def remove_cold_junction_gradient(data: pd.DataFrame) -> pd.DataFrame:
    """Remove the thermal scanner cold junction offset.

    Parameters
    ----------
    data : pandas dataframe
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are thermocouple temperature.

    Return
    ------
    data : pandas dataframe
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are thermocouple temperature with the cold
        junction temperature removed.
    """
    data["coldJunctionMean"] = data[
        ["coldJunction114", "coldJunction115", "coldJunction116", "coldJunction117"]
    ].mean(axis=1)
    for column in data.columns:
        for thermocouple in ThermocoupleTable:
            if thermocouple.name == column:
                scanner_number = str(thermocouple.scanner.value)
                data[column] = np.array(data[column]) - np.array(
                    data["coldJunction" + scanner_number] - data.coldJunctionMean
                )
    return data


def remove_offsets(data: pd.DataFrame) -> pd.DataFrame:
    """Remove the individual thermocouple reference offsets.
     This is based on three nights and mornings of very stable data in 2025:
     7/21, 7/26, and 8/05.

    Parameters
    ----------
    data : pandas dataframe
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are thermocouple temperature.

    Return
    ------
    data : pandas dataframe
        Time binned dataframe of EFD temperatures where the index
        is time and the columns are thermocouple temperature with the cold
        junction temperature removed.
    """
    reference_offsets = [
        ["MTC039M", 0.028294467528946043],
        ["MTC039F", 0.03376909911936978],
        ["MTC038B2", 0.02088131184407744],
        ["MTC040B1", 0.061522860632183506],
        ["MTC040M", 0.0360289739433904],
        ["MTC040F", 0.00039613402580566395],
        ["MTC034B", 0.028294467528946043],
        ["MTC036B", 0.03376909911936978],
        ["MTC036F", 0.042128462797246746],
        ["MTC035B", 0.019459606682545764],
        ["MTCOW9B", 0.02088131184407744],
        ["MTCOW9M", 0.061522860632183506],
        ["MTCOW9F", 0.0360289739433904],
        ["MTC037B", 0.00039613402580566395],
        ["MTC037F", 0.017248940879281086],
        ["MTCOW10B", 0.06711553301828675],
        ["MTCOW10M", 0.04965980086918029],
        ["MTCOW10F", 0.03353293962805155],
        ["MTCIW6B", 0.03813964061191583],
        ["MTCIW6M", 0.0684208922449727],
        ["MTCIW6F", 0.0535938167519681],
        ["MTC039B", 0.021438448683264225],
        ["MTC026B2", 0.03376909911936978],
        ["MTC030B2", 0.042128462797246746],
        ["MTCIW5B", 0.019459606682545764],
        ["MTCIW5F", 0.02088131184407744],
        ["MTCOW8B", 0.061522860632183506],
        ["MTCOW8M", 0.0360289739433904],
        ["MTCOW8F", 0.00039613402580566395],
        ["MTC031B", 0.017248940879281086],
        ["MTC031F", 0.06711553301828675],
        ["MTC033B", 0.04965980086918029],
        ["MTC033M", 0.03353293962805155],
        ["MTC033F", 0.0684208922449727],
        ["MTC032B", 0.0535938167519681],
        ["MTC032F", 0.021438448683264225],
        ["MTC029B", 0.028294467528946043],
        ["MTC029M", 0.04874993015523106],
        ["MTC029F", -0.047871163260872554],
        ["MTC028B", -0.025303341619109486],
        ["MTCOW7B", 0.00639149813492886],
        ["MTCOW7M", -0.025321798335337885],
        ["MTCOW7F", -0.04100289171740059],
        ["MTC030B1", -0.0673961252440367],
        ["MTC030M", -0.07648497904587878],
        ["MTC030F", -0.03650235041947476],
        ["MTCOW6F", 0.04874993015523106],
        ["MTCIW4B", -0.02381922986467881],
        ["MTCIW4F", -0.047331761157338094],
        ["MTC023B", -0.06214466595234898],
        ["MTC023M", -0.025321798335337885],
        ["MTC023F", -0.04100289171740059],
        ["MTC024B", -0.0673961252440367],
        ["MTC024F", -0.07648497904587878],
        ["MTC025B", -0.03650235041947476],
        ["MTC025F", -0.047871163260872554],
        ["MTC026B1", -0.06190069360009146],
        ["MTC026F", -0.005819818084705547],
        ["MTC027B", -0.01279407311064427],
        ["MTC017B1", -0.02381922986467881],
        ["MTC017F", -0.047331761157338094],
        ["MTC018B2", -0.06214466595234898],
        ["MTC019B", -0.025321798335337885],
        ["MTC021B", -0.04100289171740059],
        ["MTC021F", -0.0673961252440367],
        ["MTC020B", -0.07648497904587878],
        ["MTCOW5B", -0.03650235041947476],
        ["MTCOW5M", -0.047871163260872554],
        ["MTCOW5F", -0.025303341619109486],
        ["MTC022B", -0.06190069360009146],
        ["MTC022F", 0.00639149813492886],
        ["MTCOW6B", -0.005819818084705547],
        ["MTCOW6M", -0.01279407311064427],
        ["MTCOW4F", 0.028294467528946043],
        ["MTCIW3B", 0.023916937786404878],
        ["MTCIW3M", 0.00925847820690178],
        ["MTCIW3F", -0.008495476838882108],
        ["MTC016B", -0.0013832215272076287],
        ["MTC016M", 0.001434538836565441],
        ["MTC016F", -0.0004835478406723659],
        ["MTC017B2", 0.0064408847471719875],
        ["MTC018B1", 0.015604196864430663],
        ["MTC018M", 0.03351489415627735],
        ["MTC018F", -0.033629519925340266],
        ["MTC010F", 0.028294467528946043],
        ["MTC012B", -0.004333903714729512],
        ["MTC014B", 0.00925847820690178],
        ["MTC014F", -0.008495476838882108],
        ["MTC013B", -0.0013832215272076287],
        ["MTCOW3B", 0.001434538836565441],
        ["MTCOW3M", 0.0064408847471719875],
        ["MTCOW3F", 0.015604196864430663],
        ["MTC015B", -0.033629519925340266],
        ["MTC015F", 0.03640655174879911],
        ["MTCOW4B", 0.024314482838752257],
        ["MTCOW4M", 0.020237958230639085],
        ["MTC007B2", -0.004333903714729512],
        ["MTC008B2", 0.023916937786404878],
        ["MTCIW2B", -0.0013832215272076287],
        ["MTCIW2F", 0.001434538836565441],
        ["MTCOW2B", -0.0004835478406723659],
        ["MTCOW2M", 0.0064408847471719875],
        ["MTCOW2F", 0.015604196864430663],
        ["MTC009B", 0.03351489415627735],
        ["MTC009F", -0.033629519925340266],
        ["MTC011B", 0.0292251169656804],
        ["MTC011M", 0.03640655174879911],
        ["MTC011F", 0.024314482838752257],
        ["MTC010B", 0.020237958230639085],
        ["MTC043B", -0.035222856396593276],
        ["MTC043F", -0.030929785260162823],
        ["MTC042B", -0.03372508199374089],
        ["MTCOW11B", 0.03223991992325488],
        ["MTCOW11M", 0.02207644681440252],
        ["MTCOW11F", 0.02887829740318783],
        ["MTC044B", -0.018623295853802364],
        ["MTC044F", -0.019093149919467776],
        ["MTCOW12B", -0.014497566567793932],
        ["MTCOW12M", -0.015887696138401696],
        ["MTCOW12F", -0.049667697570657164],
        ["MTC038B1", -0.014497566567793932],
        ["MTC038F", -0.015887696138401696],
        ["MTC040B2", -0.049667697570657164],
        ["MTC041B", -0.022783132857461634],
        ["MTC006B", 0.028294467528946043],
        ["MTC006M", -0.035222856396593276],
        ["MTC006F", -0.030929785260162823],
        ["MTC005B", -0.03372508199374089],
        ["MTCOW1B", 0.03223991992325488],
        ["MTCOW1M", 0.02207644681440252],
        ["MTCOW1F", 0.02887829740318783],
        ["MTC008B1", -0.019093149919467776],
        ["MTC008M", -0.028693284703892596],
        ["MTC008F", -0.026665935676122066],
        ["MTCIW1B", -0.035222856396593276],
        ["MTCIW1F", -0.030929785260162823],
        ["MTC001B", -0.03372508199374089],
        ["MTC001M", 0.03223991992325488],
        ["MTC001F", 0.02207644681440252],
        ["MTC002B", -0.018623295853802364],
        ["MTC002F", -0.019093149919467776],
        ["MTC003B", -0.028693284703892596],
        ["MTC003F", -0.026665935676122066],
        ["MTC007B1", -0.049667697570657164],
        ["MTC007F", -0.024475232069958053],
        ["MTC004B", -0.022783132857461634],
    ]
    for column in data.columns:
        for row in reference_offsets:
            if row[0] == column:
                data[column] = np.array(data[column]) - row[1]
    return data
