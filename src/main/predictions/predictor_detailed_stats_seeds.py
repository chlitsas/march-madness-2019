from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_compact_results, load_detailed_box, load_tourney_compact_results, \
    detailed_stats_with_seeds_df
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class KerasDeepNeural:
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential()

    def fit(self, x_data, y_data):
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        X = np.array(x_data)
        Y = np.array(y_data)
        # create model
        self.model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu'))
        # self.model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
        # self.model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
        # Fit the model
        self.model.fit(X, Y, epochs=2000, batch_size=10, verbose=2)

    def predict_proba(self, x_data):
        # calculate predictions
        return self.model.predict(np.array(x_data))


class DetailedStatsSeedsPredictor(AbstractPredictor):
    def __init__(self, load_stats_df_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = KerasDeepNeural()

        #self.clf = RandomForestClassifier(
        #    n_estimators=500,
        #    max_depth=5,
        #    n_jobs=4
        #)

        self.features = load_stats_df_function() \
            .groupby(['Season', 'T1TeamID'], as_index=False) \
            .agg({'T1Seed': 'min',
                  'T1Score': 'mean',
                  'XScore': 'mean',
                  'T1SeedBeat': 'min'})

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

        seed_idx = (team2[0] - team1[0]) / 15
        team1_pts_idx = (team1[1] + team2[2]) / 2
        team2_pts_idx = (team2[1] + team1[2]) / 2
        pts_idx = 1 if team1_pts_idx - team2_pts_idx > 20 else -1 if team2_pts_idx - team1_pts_idx > 20 \
            else (team1_pts_idx - team2_pts_idx) / 20
        seed_beat_idx = (team2[3] - team1[3]) / 15
        return [seed_idx, pts_idx, seed_beat_idx]

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

            prob = bound_probability(self.clf.predict_proba(features)[0][0])
            game_predictions.append(GamePrediction(game, prob))
            print(str(features) + ' -- ' + str(prob))
        print('predictions done for season ' + str(season))
        return game_predictions


class DetailedStatsSeedsEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = DetailedStatsSeedsPredictor(detailed_stats_with_seeds_df, load_tourney_compact_results)
        self.active_seasons = range(2012, 2019)  # set([x.season for x in load_detailed_box()])
        self.predictor_description = 'detailed_stats_seeds'
