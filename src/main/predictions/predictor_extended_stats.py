import numpy as np
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_tourney_compact_results, \
    detailed_stats_with_seeds_df, clutch_wins_df
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
        self.model.add(Dense(8, input_dim=56, kernel_initializer='uniform', activation='sigmoid'))
        self.model.add(Dropout(0.5))
        #self.model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        #self.model.add(Dropout(0.5))
        #self.model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        Adadelta = optimizers.Adadelta(lr=0.8, rho=0.95, epsilon=None, decay=0.9)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
        # Fit the model
        self.model.fit(X, Y, epochs=9000, batch_size=10, verbose=2)

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


class ExtendedStatsSeedsPredictor(AbstractPredictor):
    def __init__(self, load_stats_df_function, clutch_wins_df_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = KerasDeepNeural()

        self.features = load_stats_df_function() \
            .groupby(['Season', 'T1TeamID'], as_index=False) \
            .agg({'T1Seed': 'min',
                  'T1SeedBeat': 'min',
                  'XScore': 'mean',
                  'XFGM': 'mean',
                  'XAst': 'mean',
                  'XStl': 'mean',
                  'XTO': 'mean',
                  'XFGM3': 'mean',
                  'XFTM': 'mean',
                  'XDR': 'mean',
                  'XOR': 'mean',
                  'XPoss': 'mean',
                  'XPF': 'mean',

                  'T1Score': 'mean',
                  'T1FGM': 'mean',
                  'T1Ast': 'mean',
                  'T1Stl': 'mean',
                  'T1TO': 'mean',
                  'T1FGM3': 'mean',
                  'T1FTM': 'mean',
                  'T1DR': 'mean',
                  'T1OR': 'mean',
                  'T1Poss': 'mean',
                  'T1PF': 'mean'
                  })

        self.features_wins = clutch_wins_df_function() \
            .groupby(['Season', 'TeamID'], as_index=False) \
            .agg({
            'ClutchWin': 'sum',
            'ClutchLoose': 'sum',
            'EasyWin': 'sum',
            'EasyLoose': 'sum'
        })

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
            percentise_diff(team2[0], team1[0], 15),
            percentise_diff(team2[0], team1[0], 15),
            #percentise_diff(team1[1], team2[1], 20),
            percentise_diff(team1[2], team2[2], 10),
            #percentise_diff(team1[3], team2[3], 6),
            percentise_diff(team1[4], team2[4], 5),
            percentise_diff(team1[5], team2[5], 6),
            percentise_diff(team1[6], team2[6], 6),
            #percentise_diff(team1[7], team2[7], 7),
            percentise_diff(team1[8], team2[8], 10),
            percentise_diff(team1[9], team2[9], 6),
            #percentise_diff(team1[10], team2[10], 10),
            #percentise_diff(team2[11], team1[11], 5),
            percentise_diff(team2[12], team1[11], 5)
        ]
        wins = [
            percentise_diff(team1_wins[0], team2_wins[0], 4),
            percentise_diff(team2_wins[1], team1_wins[1], 4),
            percentise_diff(team1_wins[2], team2_wins[2], 5),
            percentise_diff(team2_wins[3], team1_wins[3], 5)
        ]
        return team1+team2+team1_wins+team2_wins

    def train(self, seasons: [int]):
        # self.clf = KerasDeepNeural()
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

            prob = bound_probability(self.clf.predict_proba(features)[0][0])
            game_predictions.append(GamePrediction(game, prob))
            # print(str(features) + ' -- ' + str(prob))
        print('predictions done for season ' + str(season))
        return game_predictions


class ExtendedStatsSeedsEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = ExtendedStatsSeedsPredictor(detailed_stats_with_seeds_df, clutch_wins_df,
                                                     load_tourney_compact_results)
        self.active_seasons = range(2012, 2019)  # set([x.season for x in load_detailed_box()])
        self.predictor_description = 'extended_stats'
