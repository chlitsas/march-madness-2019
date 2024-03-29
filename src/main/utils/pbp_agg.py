import csv
import io
import json
import pkgutil

import pandas as pd

from src.main.domain.data_loaders import load_massey_ordinals_df


def can_not_be_run(starting_moment, current_moment):
    if starting_moment is None or current_moment is None:
        return False
    winner_diff = int(current_moment['WPoints']) - int(starting_moment['WPoints'])
    looser_diff = int(current_moment['LPoints']) - int(starting_moment['LPoints'])
    return looser_diff < winner_diff < looser_diff * 4 or winner_diff < looser_diff < winner_diff * 4


def get_team_run(starting_moment, current_moment):
    if starting_moment is None or current_moment is None:
        return None
    winner_diff = int(current_moment['WPoints']) - int(starting_moment['WPoints'])
    looser_diff = int(current_moment['LPoints']) - int(starting_moment['LPoints'])
    if winner_diff >= 7 and winner_diff > looser_diff * 3.5:
        return current_moment['WTeamID']
    if looser_diff >= 7 and looser_diff > winner_diff * 3.5:
        return current_moment['LTeamID']
    return None


def annotate_run_moments(moments):
    idx = 0
    result = []
    while idx < len(moments):
        run_start_candidate = moments[idx]
        diff_start = int(run_start_candidate['WPoints']) - int(run_start_candidate['LPoints'])
        clutch_start = int(run_start_candidate['ElapsedSeconds']) > 2000 and abs(diff_start) < 7
        idx2 = idx + 1
        run_end_candidate = None
        while idx2 < len(moments):
            current = moments[idx2]
            if can_not_be_run(run_start_candidate, current):
                break
            run_end_candidate = current
            idx2 += 1

        while idx2 < len(moments):
            current = moments[idx2]
            if get_team_run(run_start_candidate, current) is None:
                break
            run_end_candidate = current
            idx2 += 1

        run_team = get_team_run(run_start_candidate, run_end_candidate)
        if run_team is not None:
            diff_end = int(run_end_candidate['WPoints']) - int(run_end_candidate['LPoints'])
            clutch_time = int(run_end_candidate['ElapsedSeconds']) > 2000
            score_flip = diff_start * diff_end <= 0
            clutch_run = clutch_time and (score_flip or abs(diff_end) < 7) or clutch_start
            outscored_team = run_end_candidate['WTeamID'] if run_team == run_end_candidate['LTeamID'] else \
                run_end_candidate['LTeamID']
            run_team_leading = diff_end >= 0 and run_team == run_end_candidate[
                'WTeamID'] or diff_end <= 0 and run_team == run_end_candidate['LTeamID']
            season = int(run_end_candidate['Season'])
            result.append([season, int(run_team), True, clutch_run, score_flip, run_team_leading])
            result.append([season, int(outscored_team), False, clutch_run, score_flip, not run_team_leading])
            idx = idx2 - 1
        idx += 1
    return result


def calculate_runs():
    result = []
    for year in range(2010, 2019):
        events_csv_file = pkgutil.get_data('newdata.PlayByPlay_' + str(year), 'Events_' + str(year) + '.csv')
        data = csv.DictReader(io.StringIO(events_csv_file.decode('utf-8')))
        list_data = list(data)
        final_data = []
        print(len(list_data))
        for x in list_data:
            if (int(x['WPoints']) > 0 or int(x['LPoints']) > 0) and int(x['DayNum']) < 133:
                final_data.append(x)

        print(len(final_data))
        print('annotator is starting for season ' + str(year))
        result += annotate_run_moments(final_data)

    with open('runs_v2.csv', 'w', newline='\n') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')
        wr.writerows(result)


def calculate_clutch_wins():
    result = []
    for year in range(2010, 2019):
        events_csv_file = pkgutil.get_data('newdata.PlayByPlay_' + str(year), 'Events_' + str(year) + '.csv')
        data = csv.DictReader(io.StringIO(events_csv_file.decode('utf-8')))
        final_data = []
        for x in data:
            if (int(x['WPoints']) > 0 or int(x['LPoints']) > 0) and int(x['DayNum']) < 133:
                final_data.append(x)
        print('clutch wins calculator is starting for season ' + str(year))
        last_stored_game = ''
        for row in final_data:
            game_id = row['Season'] + row['DayNum'] + row['WTeamID'] + row['LTeamID']

            if last_stored_game != game_id:
                result.append([int(row['Season']), int(row['WTeamID']), 0, 0, 1, 0])
                result.append([int(row['Season']), int(row['LTeamID']), 0, 0, 0, 1])
                last_stored_game = game_id

            if int(row['ElapsedSeconds']) > 2000 and abs(int(row['WPoints']) - int(row['LPoints'])) < 7:
                idx = len(result)
                result[idx - 2][2] = 1
                result[idx - 2][4] = 0
                result[idx - 1][3] = 1
                result[idx - 1][5] = 0

    with open('clutch_wins.csv', 'w', newline='\n') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')
        wr.writerows(result)


def read_runs_as_df():
    df = pd.read_csv('./runs_v2.csv')
    print(df.head(10))
    df['LeadChangeFor'] = df.apply(lambda row: True if row['TeamRun'] and not row['LeadChange'] else False, axis=1)
    df['LeadChangeAgainst'] = df.apply(lambda row: True if not row['TeamRun'] and row['LeadChange'] else False, axis=1)
    clutch_data = df[df.Clutch] \
        .groupby(['Season', 'TeamID'], as_index=False) \
        .agg({
        'TeamRun': ['sum', 'count'],
        'LeadChange': 'sum',
        'LeadChangeFor': 'sum',
        'LeadChangeAgainst': 'sum',
        'Leading': 'sum',
    })
    clutch_data.columns = ['_Clutch'.join(col).strip() for col in clutch_data.columns.values]
    clutch_data = clutch_data.rename(index=str, columns={'Season_Clutch': 'Season', 'TeamID_Clutch': 'TeamID'})
    print(clutch_data)

    non_clutch_data = df[df.Clutch != True] \
        .groupby(['Season', 'TeamID'], as_index=False) \
        .agg({
        'TeamRun': ['sum', 'count'],
        'LeadChange': 'sum',
        'LeadChangeFor': 'sum',
        'LeadChangeAgainst': 'sum',
        'Leading': 'sum',
    })
    non_clutch_data.columns = ['_NonClutch'.join(col).strip() for col in non_clutch_data.columns.values]
    non_clutch_data = non_clutch_data.rename(index=str,
                                             columns={'Season_NonClutch': 'Season', 'TeamID_NonClutch': 'TeamID'})
    print(non_clutch_data)

    merged = pd.merge(clutch_data, non_clutch_data, how='outer', on=['Season', 'TeamID']).fillna(0)

    final = merged.rename(
        index=str,
        columns={
            'TeamRun_Clutchsum': 'ClutchPositiveRuns', 'TeamRun_Clutchcount': 'ClutchRuns',
            'LeadChange_Clutchsum': 'ClutchLeadChanges', 'LeadChangeFor_Clutchsum': 'ClutchPositiveLeadChanges',
            'LeadChangeAgainst_Clutchsum': 'ClutchNegativeLeadChanges', 'Leading_Clutchsum': 'ClutchLeads',
            'TeamRun_NonClutchsum': 'NonClutchPositiveRuns', 'TeamRun_NonClutchcount': 'NonClutchRuns',
            'LeadChange_NonClutchsum': 'NonClutchLeadChanges',
            'LeadChangeFor_NonClutchsum': 'NonClutchPositiveLeadChanges',
            'LeadChangeAgainst_NonClutchsum': 'NonClutchNegativeLeadChanges', 'Leading_NonClutchsum': 'NonClutchLeads'
        }
    )

    print(final)
    final.to_csv('runs_analysis.csv', sep=',', encoding='utf-8', index=False)


def read_clutch_wins_as_df():
    tourney_csv_file = pkgutil.get_data("newdata.DataFiles", "NCAATourneyCompactResults.csv")
    tourney_csv = pd.read_csv(io.StringIO(tourney_csv_file.decode('utf-8')))

    df = pd.read_csv('./clutch_wins.csv')
    print(df.head(10))
    data = df \
        .groupby(['Season', 'TeamID'], as_index=False) \
        .agg({
        'ClutchWin': 'sum',
        'ClutchLoose': 'sum',
        'EasyWin': 'sum',
        'EasyLoose': 'sum',
    })
    w_data = data.copy().rename(
        index=str,
        columns={'TeamID': 'WTeamID', 'ClutchWin': 'WClutchWin',
                 'ClutchLoose': 'WClutchLoose', 'EasyWin': 'WEasyWin', 'EasyLoose': 'WEasyLoose'})
    l_data = data.copy().rename(
        index=str,
        columns={'TeamID': 'LTeamID', 'ClutchWin': 'LClutchWin',
                 'ClutchLoose': 'LClutchLoose', 'EasyWin': 'LEasyWin', 'EasyLoose': 'LEasyLoose'})

    w_final = pd.merge(
        w_data,
        tourney_csv,
        how='inner', on=['Season', 'WTeamID']
    )

    l_final = pd.merge(
        l_data,
        tourney_csv,
        how='inner', on=['Season', 'LTeamID']
    )

    final = pd.merge(
        w_final,
        l_final,
        how='inner', on=['Season', 'WTeamID', 'LTeamID']
    )
    print(final)
    final.to_csv('clutch_wins_analysis.csv', sep=',', encoding='utf-8', index=False)


def read_runs_analysis_as_df():
    tourney_csv_file = pkgutil.get_data("newdata.DataFiles", "NCAATourneyCompactResults.csv")
    tourney_csv = pd.read_csv(io.StringIO(tourney_csv_file.decode('utf-8')))

    df = pd.read_csv('./runs_analysis.csv')
    print(df.head(10))
    data = df \
        .groupby(['Season', 'TeamID'], as_index=False) \
        .agg({
            'ClutchPositiveRuns': 'sum',
            'ClutchRuns': 'sum',
            'ClutchLeadChanges': 'sum',
            'ClutchPositiveLeadChanges': 'sum',
            'ClutchNegativeLeadChanges': 'sum',
            'ClutchLeads': 'sum',
            'NonClutchPositiveRuns': 'sum',
            'NonClutchRuns': 'sum',
            'NonClutchLeadChanges': 'sum',
            'NonClutchPositiveLeadChanges': 'sum',
            'NonClutchNegativeLeadChanges': 'sum',
            'NonClutchLeads': 'sum'
    })

    w_data = data.copy().rename(
        index=str,
        columns={'TeamID': 'WTeamID', 'ClutchPositiveRuns': 'WClutchPositiveRuns',
                 'ClutchRuns': 'WClutchRuns', 'ClutchLeadChanges': 'WClutchLeadChanges',
                 'ClutchPositiveLeadChanges': 'WClutchPositiveLeadChanges',
                 'ClutchNegativeLeadChanges': 'WClutchNegativeLeadChanges',
                 'ClutchLeads': 'WClutchLeads', 'NonClutchPositiveRuns': 'WNonClutchPositiveRuns',
                 'NonClutchRuns': 'WNonClutchRuns', 'NonClutchLeadChanges': 'WNonClutchLeadChanges',
                 'NonClutchPositiveLeadChanges': 'WNonClutchPositiveLeadChanges',
                 'NonClutchNegativeLeadChanges': 'WNonClutchNegativeLeadChanges',
                 'NonClutchLeads': 'WNonClutchLeads'})
    l_data = data.copy().rename(
        index=str,
        columns={'TeamID': 'LTeamID', 'ClutchPositiveRuns': 'LClutchPositiveRuns',
                 'ClutchRuns': 'LClutchRuns', 'ClutchLeadChanges': 'LClutchLeadChanges',
                 'ClutchPositiveLeadChanges': 'LClutchPositiveLeadChanges',
                 'ClutchNegativeLeadChanges': 'LClutchNegativeLeadChanges',
                 'ClutchLeads': 'LClutchLeads', 'NonClutchPositiveRuns': 'LNonClutchPositiveRuns',
                 'NonClutchRuns': 'LNonClutchRuns', 'NonClutchLeadChanges': 'LNonClutchLeadChanges',
                 'NonClutchPositiveLeadChanges': 'LNonClutchPositiveLeadChanges',
                 'NonClutchNegativeLeadChanges': 'LNonClutchNegativeLeadChanges',
                 'NonClutchLeads': 'LNonClutchLeads'})

    w_final = pd.merge(
        w_data,
        tourney_csv,
        how='right', on=['Season', 'WTeamID']
    )

    l_final = pd.merge(
        l_data,
        tourney_csv,
        how='right', on=['Season', 'LTeamID']
    )

    final = pd.merge(
        w_final,
        l_final,
        how='inner', on=['Season', 'WTeamID', 'LTeamID']
    ).fillna(0)
    print(final)
    final.to_csv('clutch_runs_analysis.csv', sep=',', encoding='utf-8', index=False)


def create_wins_looses_string():
    reg_season_csv_file = pkgutil.get_data("newdata.DataFiles", "RegularSeasonCompactResults.csv")
    data = csv.DictReader(io.StringIO(reg_season_csv_file.decode('utf-8')))
    data_map = {}
    for x in data:
        winner_id = x['Season']+'-'+x['WTeamID']
        looser_id = x['Season']+'-'+x['LTeamID']
        winner = data_map.get(winner_id)
        if winner is None:
            winner = ''
        looser = data_map.get(looser_id)
        if looser is None:
            looser = ''
        if x['WLoc'] == 'A':
            winner += 'W'
            looser += 'L'
        else:
            winner += 'w'
            looser += 'l'
        data_map[winner_id] = winner
        data_map[looser_id] = looser

    with open('results_sequence.json', 'w') as fp:
        json.dump(data_map, fp)


if __name__ == '__main__':
    # calculate_runs()
    # calculate_clutch_wins()
    # read_runs_as_df()
    # read_runs_analysis_as_df()
    # read_clutch_wins_as_df()
    #
    create_wins_looses_string()
    df = load_massey_ordinals_df()

    data = df[(df.Season == 2003) & (df.RankingDayNum == 35) & (df.SystemName == 'SEL')]
    print(data[data.TeamID == 1102].OrdinalRank.iloc[0])
    print(data.TeamID.max())

    final_set = set(df[df.RankingDayNum == 133].SystemName.unique())
    for season in range(2003, 2019):
        print(len(df[(df.Season == season) & (df.RankingDayNum >= 133)].SystemName.unique()))
        final_set = set(final_set) & set(df[(df.Season == season) & (df.RankingDayNum == 133)].SystemName.unique())
    print(final_set)
