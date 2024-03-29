import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_tourney_compact_results, \
    detailed_stats_with_seeds_df, clutch_wins_df, results_sequence_map, load_tourney_seeds, load_massey_ordinals_df
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability


class LightGBM:
    def __init__(self) -> None:
        super().__init__()
        self.model = lgb

    def fit(self, x_data, y_data):
        #self.scaler = MinMaxScaler(feature_range=(0, 1))
        X = np.array(x_data)
        #X = self.scaler.fit_transform(X)
        Y = np.array(y_data)
        # create model
        x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        #
        # Create the LightGBM data containers
        #
        train_data = lgb.Dataset(x, y)
        test_data = lgb.Dataset(x_test, y_test)

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'binary_logloss',
            'num_leaves': 3,
            'learning_rate': 0.005,
            'feature_fraction': 0.001,
            'bagging_fraction': 0.5,
            'bagging_freq': 15,
            'verbose': 1
        }

        print('Starting training...')
        # train
        self.model = lgb.train(params,
                               train_data,
                               valid_sets=test_data,
                               num_boost_round=50000,
                               early_stopping_rounds=100)

    def predict_proba(self, x_data):
        # calculate predictions
        #x_data = self.scaler.transform(x_data)
        y_pred = self.model.predict(x_data)
        return y_pred


def percentise_diff(x, y, perfect_diff):
    diff = x - y
    if diff >= perfect_diff:
        return 1
    if diff <= -perfect_diff:
        return 1
    return diff / perfect_diff


def region_one_hot(region, seed):
    result = [0, 0, 0, 0, 0]
    if region == 'W':
        result[0] = seed
    elif region == 'X':
        result[1] = seed
    elif region == 'Y':
        result[2] = seed
    elif region == 'Z':
        result[3] = seed
    else:
        result[4] = seed
    return result


class ExtendedStatsLGBMPredictor(AbstractPredictor):
    def __init__(self, load_stats_df_function, clutch_wins_df_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = LightGBM()

        self.features = load_stats_df_function() \
            .groupby(['Season', 'T1TeamID'], as_index=False) \
            .agg({'T1Seed': 'min',  # 0
                  'T1SeedBeat': 'min',  # 1
                  'XScore': 'mean',  # 2
                  'XFGM': 'mean',  # 3
                  'XAst': 'mean',  # 4
                  'XStl': 'mean',  # 5
                  'XTO': 'mean',  # 6
                  'XFGM3': 'mean',  # 7
                  'XFTM': 'mean',  # 8
                  'XDR': 'mean',  # 9
                  'XOR': 'mean',  # 10
                  'XPoss': 'mean',  # 11
                  'XPF': 'mean',  # 12

                  'T1Score': 'mean',  # 13
                  'T1FGM': 'mean',  # 14
                  'T1Ast': 'mean',  # 15
                  'T1Stl': 'mean',  # 16
                  'T1TO': 'mean',  # 17
                  'T1FGM3': 'mean',  # 18
                  'T1FTM': 'mean',  # 19
                  'T1DR': 'mean',  # 20
                  'T1OR': 'mean',  # 21
                  'T1Poss': 'mean',  # 22
                  'T1PF': 'mean',  # 23
                  'T1Region': 'first'  # 24
                  })

        self.results_sequence = results_sequence_map()
        self.massey_df = load_massey_ordinals_df()

    def init_data(self, season):
        # {'AP', 'WOL', 'RTH', 'WLK', 'DOL', 'MOR', 'USA', 'SAG', 'RPI', 'POM', 'COL'}
        self.ratings = self.massey_df[
            (self.massey_df.Season == season) &
            (self.massey_df.SystemName == 'POM')
        ]

        seeds = [x for x in load_tourney_seeds() if x.season == season]
        self.max_seed = max([x.get_seed() for x in seeds])
        self.seeds_map = {}
        for seed in seeds:
            self.seeds_map[seed.team_id] = seed.get_seed()

    def win_streak(self, season_id, team_id, limit):
        data = self.results_sequence[str(season_id)+'-'+str(team_id)][-limit:]
        return (data.count('w')+data.count('W'))/len(data)

    def wins_percentage(self, season_id, team_id):
        data = self.results_sequence[str(season_id)+'-'+str(team_id)]
        return (data.count('W')+data.count('w'))/len(data)

    def get_feature_vector(self, season, team1_id, team2_id):
        team_a_mass = self.ratings[self.ratings.TeamID == team1_id].OrdinalRank.iloc[-1]
        team_b_mass = self.ratings[self.ratings.TeamID == team2_id].OrdinalRank.iloc[-1]

        team1 = None
        team2 = None
        for row in self.features[
            (self.features['T1TeamID'] == team1_id) & (self.features['Season'] == season)].iterrows():
            index, d = row
            team1 = d.tolist()[2:]
            break

        for row in self.features[
            (self.features['T1TeamID'] == team2_id) & (self.features['Season'] == season)].iterrows():
            index, d = row
            team2 = d.tolist()[2:]
            break

        stats = [
            team1[0],
            team1[1],
            team1[2],
            team2[0],
            team2[1],
            team2[2],
            self.wins_percentage(season, team1_id),
            self.wins_percentage(season, team2_id),
            team_a_mass,
            team_b_mass
            #percentise_diff(team2[12], team1[11], 5)
        ]

        return stats

    def train(self, seasons: [int]):
        self.clf = LightGBM()
        train_data = []
        train_results = []
        for season in seasons:
            self.init_data(season)
            for result in self.load_tourney_compact_results_function(season):
                if result.w_team_id < result.l_team_id:
                    train_data.append(self.get_feature_vector(season, result.w_team_id, result.l_team_id))
                    train_results.append(1)
                else:
                    train_data.append(self.get_feature_vector(season, result.l_team_id, result.w_team_id))
                    train_results.append(0)

        print('training')
        # self.scaler.fit(train_data)
        # train_data = self.scaler.transform(train_data)
        self.clf.fit(train_data, train_results)
        print('training is done')

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        self.init_data(season)
        game_predictions = []
        print('starting predictions for season ' + str(season))
        for game in games:
            features = [
                self.get_feature_vector(season, game.team_a_id, game.team_b_id)
            ]

            prob = bound_probability(self.clf.predict_proba(features)[0])
            game_predictions.append(GamePrediction(game, prob))
            # print(str(features) + ' -- ' + str(prob))
        print('predictions done for season ' + str(season))
        return game_predictions


class ExtendedStatsLGBMPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = ExtendedStatsLGBMPredictor(detailed_stats_with_seeds_df, clutch_wins_df,
                                                    load_tourney_compact_results)
        self.active_seasons = range(2003, 2019)  # set([x.season for x in load_detailed_box()])
        self.predictor_description = 'extended_stats_tree_lgbm2'
