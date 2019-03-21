import numpy as np
from catboost import CatBoostClassifier
from catboost import Pool

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_tourney_compact_results, \
    detailed_stats_with_seeds_df, results_sequence_map
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability


class CatBoost:
    def __init__(self) -> None:
        super().__init__()
        self.model = CatBoostClassifier(
            iterations=1600,
            learning_rate=0.003,
            depth=3,
            loss_function='Logloss',
            l2_leaf_reg=3,
            leaf_estimation_iterations=50,
            thread_count=6
        )

    def fit(self, x_data, y_data):
        #self.scaler = MinMaxScaler(feature_range=(0, 1))
        X = np.array(x_data)
        #X = self.scaler.fit_transform(X)
        Y = np.array(y_data)
        # create model
        pool = Pool(X, Y, [])

        print('Starting training...')
        self.model.fit(pool)

    def predict_proba(self, x_data):
        #x_data = self.scaler.transform(x_data)
        return self.model.predict_proba(x_data)[0][1]


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


class ExtendedStatsCatBoostWPredictor(AbstractPredictor):
    def __init__(self, load_stats_df_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = CatBoost()

        self.results_sequence = results_sequence_map()
        self.massey_df = None # load_massey_ordinals_df()

        self.features = load_stats_df_function() \
            .groupby(['Season', 'T1TeamID'], as_index=False) \
            .agg({'T1Seed': 'min',  # 0
                  'T1SeedBeat': 'min',  # 1
                  'XScore': 'mean',  # 2
                  'XFGM': 'mean',  # 3
                  'XAst': 'mean',  # 4 / alone gives random predictions
                  'XStl': 'mean',  # 5 / random results
                  'XTO': 'mean',  # 6 / random results
                  'XFGM3': 'mean',  # 7 / random
                  'XFTM': 'mean',  # 8 / random
                  'XDR': 'mean',  # 9 / bad
                  'XOR': 'mean',  # 10 / random
                  'XPoss': 'mean',  # 11 / random
                  'XPF': 'mean',  # 12 / random

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
                  'T1Region': 'first',  # 24

                  'T2Score': 'mean',  # 25
                  'T2FGM': 'mean',  # 26
                  'T2Ast': 'mean',  # 27
                  'T2Stl': 'mean',  # 28
                  'T2TO': 'mean',  # 29
                  'T2FGM3': 'mean',  # 30
                  'T2FTM': 'mean',  # 31
                  'T2DR': 'mean',  # 32
                  'T2OR': 'mean',  # 33
                  'T2Poss': 'mean',  # 34
                  'T2PF': 'mean'  # 35
                  })

    def win_streak(self, season_id, team_id, limit):
        data = self.results_sequence[str(season_id)+'-'+str(team_id)][-limit:]
        return 100*(data.count('W'))/len(data)

    def wins_percentage(self, season_id, team_id):
        data = self.results_sequence[str(season_id)+'-'+str(team_id)]
        return (data.count('W')+data.count('w'))/len(data)

    def get_feature_vector(self, season, team1_id, team2_id):
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

        team1x = None
        team2x = None
        for row in self.features[
            (self.features['T1TeamID'] == team1_id) & (self.features['Season'] == season-1)].iterrows():
            index, d = row
            team1x = d.tolist()[2:]
            break

        for row in self.features[
            (self.features['T1TeamID'] == team2_id) & (self.features['Season'] == season-1)].iterrows():
            index, d = row
            team2x = d.tolist()[2:]
            break

        power = 13.91
        team_a_prob = (team1[13]**power) / (team1[13]**power + team1[25]**power)
        team_b_prob = (team2[13]**power) / (team2[13]**power + team2[25]**power)

        team1_win = team_a_prob/(team_a_prob + team_b_prob)

        stats = [
            team1[0],
            team2[0],
            team1x[0],
            team2x[0],
            team1[1],
            team2[1],
            team1[2],
            team2[2],
            team1_win,
            #self.wins_percentage(season, team1_id)/(self.wins_percentage(season, team1_id)+self.wins_percentage(season, team2_id))
            #percentise_diff(team2[12], team1[11], 5)
        ]
        return stats

    def train(self, seasons: [int]):
        self.clf = CatBoost()
        train_data = []
        train_results = []
        for season in seasons:
            for result in self.load_tourney_compact_results_function(season):
                if result.w_team_id < result.l_team_id:
                    train_data.append(self.get_feature_vector(season, result.w_team_id, result.l_team_id))
                    train_results.append(1)
                else:
                    train_data.append(self.get_feature_vector(season, result.l_team_id, result.w_team_id))
                    train_results.append(0)

        print('training')
        self.clf.fit(train_data, train_results)
        print('training is done')

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        game_predictions = []
        print('starting predictions for season ' + str(season))
        for game in games:
            features = [
                self.get_feature_vector(season, game.team_a_id, game.team_b_id)
            ]

            prob = bound_probability(self.clf.predict_proba(features))
            game_predictions.append(GamePrediction(game, prob))
            # print(str(features) + ' -- ' + str(prob))
        print('predictions done for season ' + str(season))
        return game_predictions


class ExtendedStatsCatBoostWPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = ExtendedStatsCatBoostWPredictor(detailed_stats_with_seeds_df, load_tourney_compact_results)
        self.active_seasons = range(2011, 2019)  # set([x.season for x in load_detailed_box()])
        self.predictor_description = 'extended_stats_tree_catboost_v1_w'
