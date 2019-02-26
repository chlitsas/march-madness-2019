import csv
import io
import pkgutil

from src.main.domain.PlayByPlayEvent import PlayByPlayEvent
from src.main.domain.CompactResult import CompactResult
from src.main.domain.DetailedResult import DetailedResult
from src.main.domain.TourneySeed import TourneySeed


def parse_pbp():
    with open('../data/PrelimData2018/Events_Prelim2018.csv') as csv_file:
        data = csv.DictReader(csv_file)
        cnt = 0
        for row in data:
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
            PlayByPlayEvent(
                winner_points=row['WPoints'],
                looser_points=row['LPoints'],
                time=row['ElapsedSeconds'],
                player_id=row['EventPlayerID'],
                team_id=row['EventTeamID'],
                event_type=row['EventType']
            )


def parse_tourney_seeds():
    csv_file = pkgutil.get_data("data.DataFiles", "NCAATourneySeeds.csv")

    data = csv.DictReader(io.StringIO(csv_file.decode('utf-8')))

    result = []
    for row in data:
        result.append(
            TourneySeed(
                Season=int(row['Season']),
                Seed=row['Seed'],
                TeamID=int(row['TeamID'])
            )
        )
    return result


def parse_compact_results(regular_season=False):
    if regular_season:
        csv_file = pkgutil.get_data("data.DataFiles", "RegularSeasonCompactResults.csv")
    else:
        csv_file = pkgutil.get_data("data.DataFiles", "NCAATourneyCompactResults.csv")
    data = csv.DictReader(io.StringIO(csv_file.decode('utf-8')))

    result = []
    for row in data:
        result.append(
            CompactResult(
                Season=int(row['Season']),
                DayNum=int(row['DayNum']),
                WTeamID=row['WTeamID'],
                WScore=int(row['WScore']),
                LTeamID=row['LTeamID'],
                LScore=int(row['LScore']),
                WLoc=row['WLoc'],
                NumOT=int(row['NumOT'])
            )
        )
    return result


def parse_detailed_box():
    csv_file = pkgutil.get_data("data.DataFiles", "NCAATourneyCompactResults.csv")
    data = csv.DictReader(io.StringIO(csv_file.decode('utf-8')))

    result = []
    for row in data:
        result.append(
            DetailedResult(
                Season=int(row['Season']),
                DayNum=int(row['DayNum']),
                WTeamID=row['WTeamID'],
                WScore=int(row['WScore']),
                LTeamID=row['LTeamID'],
                LScore=int(row['LScore']),
                WLoc=row['WLoc'],
                NumOT=int(row['NumOT']),
                WFGM=int(row['WFGM']),
                WFGA=int(row['WFGA']),
                WFGM3=int(row['WFGM3']),
                WFGA3=int(row['WFGA3']),
                WFTM=int(row['WFTM']),
                WFTA=int(row['WFTA']),
                WOR=int(row['WOR']),
                WDR=int(row['WDR']),
                WAst=int(row['WAst']),
                WTO=int(row['WTO']),
                WStl=int(row['WStl']),
                WBlk=int(row['WBlk']),
                WPF=int(row['WPF']),
                LFGM=int(row['LFGM']),
                LFGA=int(row['LFGA']),
                LFGM3=int(row['LFGM3']),
                LFGA3=int(row['LFGA3']),
                LFTM=int(row['LFTM']),
                LFTA=int(row['LFTA']),
                LOR=int(row['LOR']),
                LDR=int(row['LDR']),
                LAst=int(row['LAst']),
                LTO=int(row['LTO']),
                LStl=int(row['LStl']),
                LBlk=int(row['LBlk']),
                LPF=int(row['LPF'])
            )
        )
    return result
