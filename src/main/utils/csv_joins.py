import io
import pkgutil

import pandas as pd


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
    return final.sort_values(by=['Season', 'DayNum']) \
        .reset_index(drop=True)


def ncaa_results_vs_reg_season_avg_stats_and_seeds_df():
    regular_season_csv_file = pkgutil.get_data("data.DataFiles", "RegularSeasonDetailedResults.csv")
    seeds_csv_file = pkgutil.get_data("data.DataFiles", "NCAATourneySeeds.csv")
    tourney_csv_file = pkgutil.get_data("data.DataFiles", "NCAATourneyCompactResults.csv")

    seeds_csv = pd.read_csv(io.StringIO(seeds_csv_file.decode('utf-8')))
    regular_season_csv = pd.read_csv(io.StringIO(regular_season_csv_file.decode('utf-8')))
    tourney_csv = pd.read_csv(io.StringIO(tourney_csv_file.decode('utf-8')))

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

    fat_df = pd.concat([final_pt1, final_pt2], sort=True)
    fat_df['T2Loc'] = fat_df['T1Loc'].map(lambda x: 'H' if x == 'A' else ('A' if x == 'H' else 'N'))

    fat_df['T1Poss'] = fat_df['T1FGA'] + 0.4*fat_df['T1FTA'] - 1.07*\
                      (fat_df['T1OR']/(fat_df['T1OR'] + fat_df['T2DR']))*(fat_df['T1FGA'] - fat_df['T1FGM']) + fat_df['T1TO']
    fat_df['T2Poss'] = fat_df['T2FGA'] + 0.4*fat_df['T2FTA'] - 1.07*\
                      (fat_df['T2OR']/(fat_df['T2OR'] + fat_df['T1DR']))*(fat_df['T2FGA'] - fat_df['T2FGM']) + fat_df['T2TO']
    fat_df['XPoss'] = fat_df['T1Poss'] - fat_df['T2Poss']

    fat_df['T1Region'] = fat_df['T1Seed'].map(lambda x: x[0])
    fat_df['T1Seed'] = fat_df['T1Seed'].map(lambda x: int(x[1:3]))
    fat_df['T2Region'] = fat_df['T2Seed'].map(lambda x: x[0])
    fat_df['T2Seed'] = fat_df['T2Seed'].map(lambda x: int(x[1:3]))

    fat_df['T1SeedBeat'] = fat_df.apply(lambda x: x.T2Seed if x.T1Score > x.T2Score else 100, axis=1)
    fat_df['T2SeedBeat'] = fat_df.apply(lambda x: x.T1Seed if x.T2Score > x.T1Score else 100, axis=1)

    final = pd.merge(
        fat_df[['Season', 'T1Ast', 'T1Blk', 'T1DR', 'T1FGA', 'T1FGA3', 'T1FGM', 'T1FGM3', 'T1FTA', 'T1FTM', 'T1Loc',
                'T1OR', 'T1PF', 'T1Poss', 'T1Region', 'T1Score', 'T1Seed', 'T1SeedBeat', 'T1Stl', 'T1TO', 'T1TeamID',
                'T2Ast', 'T2Blk', 'T2DR', 'T2FGA', 'T2FGA3', 'T2FGM', 'T2FGM3', 'T2FTA', 'T2FTM', 'T2Loc',
                'T2OR', 'T2PF', 'T2Poss', 'T2Region', 'T2Score', 'T2Seed', 'T2SeedBeat', 'T2Stl', 'T2TO', 'T2TeamID']],
        tourney_csv,
        how='inner', left_on=['Season', 'T1TeamID'], right_on=['Season', 'WTeamID']
    )
    final.columns = final.columns.map(
        lambda x: 'T1XX' + x[2:] if x[0:2] == 'T2' else x)

    fat_df_copy = fat_df.copy()
    fat_df_copy.columns = fat_df_copy.columns.map(
        lambda x: 'T2XX' + x[2:] if x[0:2] == 'T1' else x)

    final = pd.merge(
        final,
        fat_df_copy[['Season', 'T2Ast', 'T2Blk', 'T2DR', 'T2FGA', 'T2FGA3', 'T2FGM', 'T2FGM3', 'T2FTA', 'T2FTM', 'T2Loc',
                     'T2OR', 'T2PF', 'T2Poss', 'T2Region', 'T2Score', 'T2Seed', 'T2SeedBeat', 'T2Stl', 'T2TO', 'T2TeamID',
                     'T2XXAst', 'T2XXBlk', 'T2XXDR', 'T2XXFGA', 'T2XXFGA3', 'T2XXFGM', 'T2XXFGM3', 'T2XXFTA', 'T2XXFTM', 'T2XXLoc',
                     'T2XXOR', 'T2XXPF', 'T2XXPoss', 'T2XXRegion', 'T2XXScore', 'T2XXSeed', 'T2XXSeedBeat', 'T2XXStl', 'T2XXTO', 'T2XXTeamID']],
        how='inner', left_on=['Season', 'LTeamID'], right_on=['Season', 'T2TeamID']
    )

    print(final.head(10))
    print(final.shape)

    final = final.reindex(sorted(final.columns), axis=1)
    return final.sort_values(by=['Season', 'DayNum']) \
        .reset_index(drop=True)


if __name__ == '__main__':
    d = ncaa_results_vs_reg_season_avg_stats_and_seeds_df()
    data = d \
        .groupby(['Season', 'T1TeamID', 'T2TeamID'], as_index=False) \
        .agg({'T1Score': 'mean', 'T2Score': 'mean',
              'T1Ast': 'mean', 'T2Ast': 'mean',
              'T1FGA': 'mean', 'T2FGA': 'mean',
              'T1FGA3': 'mean', 'T2FGA3': 'mean',
              'T1FGM': 'mean', 'T2FGM': 'mean',
              'T1FGM3': 'mean', 'T2FGM3': 'mean',
              'T1FTA': 'mean', 'T2FTA': 'mean',
              'T1FTM': 'mean', 'T2FTM': 'mean',
              'T1OR': 'mean', 'T2OR': 'mean',
              'T1DR': 'mean', 'T2DR': 'mean',
              'T1Stl': 'mean', 'T2Stl': 'mean',
              'T1TO': 'mean', 'T2TO': 'mean',
              'T1Blk': 'mean', 'T2Blk': 'mean',
              'T1PF': 'mean', 'T2PF': 'mean',
              'T1Poss': 'mean', 'T2Poss': 'mean',

              'T1XXScore': 'mean', 'T2XXScore': 'mean',
              'T1XXAst': 'mean', 'T2XXAst': 'mean',
              'T1XXFGA': 'mean', 'T2XXFGA': 'mean',
              'T1XXFGA3': 'mean', 'T2XXFGA3': 'mean',
              'T1XXFGM': 'mean', 'T2XXFGM': 'mean',
              'T1XXFGM3': 'mean', 'T2XXFGM3': 'mean',
              'T1XXFTA': 'mean', 'T2XXFTA': 'mean',
              'T1XXFTM': 'mean', 'T2XXFTM': 'mean',
              'T1XXOR': 'mean', 'T2XXOR': 'mean',
              'T1XXDR': 'mean', 'T2XXDR': 'mean',
              'T1XXStl': 'mean', 'T2XXStl': 'mean',
              'T1XXTO': 'mean', 'T2XXTO': 'mean',
              'T1XXBlk': 'mean', 'T2XXBlk': 'mean',
              'T1XXPF': 'mean', 'T2XXPF': 'mean',
              'T1XXPoss': 'mean', 'T2XXPoss': 'mean',

              'T1Region': 'first', 'T2Region': 'first',
              'T1Seed': 'mean', 'T2Seed': 'mean',
              'T1SeedBeat': 'min', 'T2SeedBeat': 'min'
              })

    # data.columns = ['_'.join(col).strip() for col in data.columns.values]

    data.to_csv('allv2.csv', sep=',', encoding='utf-8', index=False)
    if d is None:
        merged = detailed_stats_with_seeds_df()
        merged.to_csv('merged.csv', sep=',', encoding='utf-8', index=False)
        data = merged \
            .groupby(['Season', 'T1TeamID'], as_index=False) \
            .agg({'T1Score': ['sum', 'max'],
                  'T2Score': 'mean',
                  'T1Blk': 'sum',
                  'T1Seed': 'max',
                  'T1SeedBeat': 'min',
                  'T1OR': lambda x: x.max() - x.min()})

        # data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # data.to_csv('agg.csv', sep=',', encoding='utf-8', index=False)

        # 1380
        for row in data[(data['T1TeamID'] == 1380) & (data['Season'] == 2016)].iterrows():
            index, d = row
            print(d.tolist()[2:])


def main():
    print('Hello, world!')


if __name__ == '__main__':
    main()
