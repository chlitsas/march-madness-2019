import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_tourney_compact_results, \
    detailed_stats_with_seeds_df, clutch_wins_df, results_sequence_map
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability


class KerasDeepNeural:
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential()

    def fit(self, x_data, y_data):
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X = np.array(x_data)
        X = self.scaler.fit_transform(X)
        Y = np.array(y_data)
        # create model
        self.model.add(Dense(2, input_dim=6, kernel_initializer='uniform', activation='sigmoid'))
        #self.model.add(Dropout(0.5))
        #self.model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        #self.model.add(Dropout(0.5))
        #self.model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        Adadelta = optimizers.Adadelta(lr=0.8, rho=0.95, epsilon=None, decay=0.9)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
        # Fit the model
        self.model.fit(X, Y, epochs=10000, batch_size=40, verbose=2)

    def predict_proba(self, x_data):
        # calculate predictions
        x_data = self.scaler.transform(x_data)
        return self.model.predict(np.array(x_data))


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


class ExtendedStatsDeepNNPredictor(AbstractPredictor):
    def __init__(self, load_stats_df_function, clutch_wins_df_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = KerasDeepNeural()

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

        self.features_wins = clutch_wins_df_function() \
            .groupby(['Season', 'TeamID'], as_index=False) \
            .agg({
            'ClutchWin': 'sum',
            'ClutchLoose': 'sum',
            'EasyWin': 'sum',
            'EasyLoose': 'sum'
        })
        self.results_sequence = results_sequence_map()

    def win_streak(self, season_id, team_id, limit):
        data = self.results_sequence[str(season_id)+'-'+str(team_id)][-limit:]
        return (data.count('w')+data.count('W'))/len(data)

    def get_feature_vector(self, season, team1_id, team2_id):
        team1 = None
        team2 = None
        team1_wins = None
        team2_wins = None
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

        for row in self.features_wins[
            (self.features_wins['TeamID'] == team1_id) & (self.features_wins['Season'] == season)].iterrows():
            index, d = row
            team1_wins = d.tolist()[2:]
            break

        for row in self.features_wins[
            (self.features_wins['TeamID'] == team2_id) & (self.features_wins['Season'] == season)].iterrows():
            index, d = row
            team2_wins = d.tolist()[2:]
            break

        stats = [
            team1[0],
            team2[0],
            team1[1],
            team2[1],
            team1[2],
            team2[2]
            #percentise_diff(team2[12], team1[11], 5)
        ]
        #wins = [
        #    percentise_diff(team1_wins[0], team2_wins[0], 4),
        #    percentise_diff(team2_wins[1], team1_wins[1], 4),
        #    percentise_diff(team1_wins[2], team2_wins[2], 5),
        #    percentise_diff(team2_wins[3], team1_wins[3], 5)
        #]

        return stats

    def train(self, seasons: [int]):
        self.clf = KerasDeepNeural()
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
        # self.scaler.fit(train_data)
        # train_data = self.scaler.transform(train_data)
        self.clf.fit(train_data, train_results)
        print('training is done')

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
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


class ExtendedStatsDeepNNPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = ExtendedStatsDeepNNPredictor(
            detailed_stats_with_seeds_df, clutch_wins_df, load_tourney_compact_results
        )
        self.active_seasons = range(2003, 2019)  # set([x.season for x in load_detailed_box()])
        self.predictor_description = 'deep_nn'
