import csv
import io
import pkgutil

import pandas as pd


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
        events_csv_file = pkgutil.get_data('data.PlayByPlay_' + str(year), 'Events_' + str(year) + '.csv')
        data = csv.DictReader(io.StringIO(events_csv_file.decode('utf-8')))

        print('annotator is starting for season ' + str(year))
        result += annotate_run_moments(list(data))

    with open('runs_v2.csv', 'w', newline='\n') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')
        wr.writerows(result)


def read_runs_as_df():
    df = pd.read_csv('./runs.csv')
    print(df.head(10))
    clutch_data = df[df.Clutch] \
        .groupby(['Season', 'TeamID'], as_index=False) \
        .agg({
            'TeamRun': ['sum', 'count'],
            'LeadChange': 'sum',
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
            'Leading': 'sum',
    })
    non_clutch_data.columns = ['_NonClutch'.join(col).strip() for col in non_clutch_data.columns.values]
    non_clutch_data = non_clutch_data.rename(index=str, columns={'Season_NonClutch': 'Season', 'TeamID_NonClutch': 'TeamID'})
    print(non_clutch_data)

    merged = pd.merge(clutch_data, non_clutch_data, how='outer', on=['Season', 'TeamID']).fillna(0)
    print(merged)
    merged.to_csv('runs_analysis.csv', sep=',', encoding='utf-8', index=False)


if __name__ == '__main__':
    # calculate_runs()
    read_runs_as_df()
