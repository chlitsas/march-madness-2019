import csv
import io
import json
import pkgutil

import pandas as pd

from src.main.domain.CompactResult import CompactResult
from src.main.domain.DetailedResult import DetailedResult
from src.main.domain.GamePrediction import create_valid_game
from src.main.domain.PlayByPlayEvent import PlayByPlayEvent
from src.main.domain.TourneySeed import TourneySeed


def load_pbp():
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


def load_tourney_seeds():
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


def load_massey_ordinals_df():
    csv_file = pkgutil.get_data("data", "MasseyOrdinals.csv")

    return pd.read_csv(io.StringIO(csv_file.decode('utf-8')))


def load_compact_results(regular_season=False):
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
                WTeamID=int(row['WTeamID']),
                WScore=int(row['WScore']),
                LTeamID=int(row['LTeamID']),
                LScore=int(row['LScore']),
                WLoc=row['WLoc'],
                NumOT=int(row['NumOT'])
            )
        )
    return result


def load_tourney_compact_results(season: int):
    return [x for x in load_compact_results(regular_season=False) if x.season == season]


def load_tourney_games(season: int):
    return [
        create_valid_game(x.w_team_id, x.l_team_id)
        for x in load_compact_results(regular_season=False) if x.season == season
    ]


def load_detailed_box(regular_season=False):
    if regular_season:
        csv_file = pkgutil.get_data("data.DataFiles", "RegularSeasonDetailedResults.csv")
    else:
        csv_file = pkgutil.get_data("data.DataFiles", "NCAATourneyDetailedResults.csv")
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


def clutch_wins_df():
    clutch_wins_csv_file = pkgutil.get_data("data.Calculated", "clutch_wins.csv")

    return pd.read_csv(io.StringIO(clutch_wins_csv_file.decode('utf-8')))


def results_sequence_map():
    results_sequence_file = pkgutil.get_data("data.Calculated", "results_sequence.json")
    return json.loads(results_sequence_file.decode())


def detailed_stats_with_seeds_df():
    regular_season_csv_file = pkgutil.get_data("data.DataFiles", "RegularSeasonDetailedResults.csv")
    seeds_csv_file = pkgutil.get_data("data.DataFiles", "NCAATourneySeeds.csv")

    seeds_csv = pd.read_csv(io.StringIO(seeds_csv_file.decode('utf-8')))
    regular_season_csv = pd.read_csv(io.StringIO(regular_season_csv_file.decode('utf-8')))

    w_seeds_csv = seeds_csv.copy().rename(index=str, columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'})

    new_df = pd.merge(regular_season_csv, w_seeds_csv, how='left', on=['Season', 'WTeamID'])
    new_df['WSeed'] = new_df['WSeed'].fillna('?20')

    l_seeds_csv = seeds_csv.copy().rename(index=str, columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'})

    merged = pd.merge(new_df, l_seeds_csv, how='left', on=['Season', 'LTeamID'])
    merged['LSeed'] = merged['LSeed'].fillna('?20')

    final_pt1 = merged.copy()
    final_pt2 = merged.copy()

    final_pt1.columns = final_pt1.columns.map(
        lambda x: 'T1' + x[1:] if x[0] == 'W' else ('T2' + x[1:] if x[0] == 'L' else x))

    final_pt2.columns = final_pt2.columns.map(
        lambda x: 'T1' + x[1:] if x[0] == 'L' else ('T2' + x[1:] if x[0] == 'W' else x))

    final = pd.concat([final_pt1, final_pt2], sort=True)
    final['T2Loc'] = final['T1Loc'].map(lambda x: 'H' if x == 'A' else ('A' if x == 'H' else 'N'))

    final['XAst'] = final['T1Ast'] - final['T2Ast']
    final['XBlk'] = final['T1Blk'] - final['T2Blk']
    final['XDR'] = final['T1DR'] - final['T2DR']
    final['XFGA'] = final['T1FGA'] - final['T2FGA']
    final['XFGA3'] = final['T1FGA3'] - final['T2FGA3']
    final['XFGM'] = final['T1FGM'] - final['T2FGM']
    final['XFGM3'] = final['T1FGM3'] - final['T2FGM3']
    final['XFTA'] = final['T1FTA'] - final['T2FTA']
    final['XFTM'] = final['T1FTM'] - final['T2FTM']
    final['XOR'] = final['T1OR'] - final['T2OR']
    final['XPF'] = final['T1PF'] - final['T2PF']
    final['XScore'] = final['T1Score'] - final['T2Score']
    final['XStl'] = final['T1Stl'] - final['T2Stl']
    final['XTO'] = final['T1TO'] - final['T2TO']
    final['T1Poss'] = final['T1FGA'] + 0.4*final['T1FTA'] - 1.07*\
                      (final['T1OR']/(final['T1OR'] + final['T2DR']))*(final['T1FGA'] - final['T1FGM']) + final['T1TO']
    final['T2Poss'] = final['T2FGA'] + 0.4*final['T2FTA'] - 1.07*\
                      (final['T2OR']/(final['T2OR'] + final['T1DR']))*(final['T2FGA'] - final['T2FGM']) + final['T2TO']
    final['XPoss'] = final['T1Poss'] - final['T2Poss']

    final['T1Region'] = final['T1Seed'].map(lambda x: x[0])
    final['T1Seed'] = final['T1Seed'].map(lambda x: int(x[1:3]))
    final['T2Region'] = final['T2Seed'].map(lambda x: x[0])
    final['T2Seed'] = final['T2Seed'].map(lambda x: int(x[1:3]))

    final['T1SeedBeat'] = final.apply(lambda x: x.T2Seed if x.T1Score > x.T2Score else 100, axis=1)

    final = final.reindex(sorted(final.columns), axis=1)
    return final.sort_values(by=['Season', 'DayNum']).reset_index(drop=True)
