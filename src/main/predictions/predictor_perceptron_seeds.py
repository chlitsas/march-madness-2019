from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_tourney_seeds, load_tourney_compact_results
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
        self.model.add(Dense(2, input_dim=4, init='uniform', activation='sigmoid'))
        self.model.add(Dense(6, init='uniform', activation='relu'))
        self.model.add(Dense(1, init='uniform', activation='sigmoid'))
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
        # Fit the model
        self.model.fit(X, Y, epochs=800, batch_size=10, verbose=2)

    def predict_proba(self, x_data):
        # calculate predictions
        return self.model.predict(np.array(x_data))


class SeedsPerceptronPredictor(AbstractPredictor):
    def __init__(self, load_tourney_seeds_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_tourney_seeds_function = load_tourney_seeds_function
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = KerasDeepNeural()

    class Features:
        def __init__(self, region: str, seed: int) -> None:
            super().__init__()
            self.region = region
            self.seed = seed

        def numerical_region(self):
            if self.region == 'W':
                return 1
            if self.region == 'X':
                return 2
            if self.region == 'Y':
                return 3
            if self.region == 'Z':
                return 4
            raise ValueError('Undefined Region')

    def train(self, seasons: [int]):
        self.clf = KerasDeepNeural()
        train_data = []
        train_results = []
        for season in seasons:
            compact_results = self.load_tourney_compact_results_function(season)
            seeds = [x for x in self.load_tourney_seeds_function() if x.season == season]
            seeds_map = {}
            for seed in seeds:
                seeds_map[seed.team_id] = self.Features(region=seed.get_region(), seed=seed.get_seed())
            for result in compact_results:
                winner = seeds_map[result.w_team_id]
                looser = seeds_map[result.l_team_id]
                if result.w_team_id < result.l_team_id:
                    train_data.append([winner.seed, winner.numerical_region(), looser.seed, looser.numerical_region()])
                    train_results.append(1)
                else:
                    train_data.append([looser.seed, looser.numerical_region(), winner.seed, winner.numerical_region()])
                    train_results.append(0)
        print('training')
        self.clf.fit(train_data, train_results)
        print('training is done')

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        seeds = [x for x in self.load_tourney_seeds_function() if x.season == season]
        seeds_map = {}
        for seed in seeds:
            seeds_map[seed.team_id] = self.Features(region=seed.get_region(), seed=seed.get_seed())
        game_predictions = []
        print('starting predictions for season '+str(season))
        for game in games:
            team_a_seed = seeds_map[game.team_a_id]
            team_b_seed = seeds_map[game.team_b_id]

            features = \
                [team_a_seed.seed, team_a_seed.numerical_region(), team_b_seed.seed, team_b_seed.numerical_region()]

            game_predictions.append(GamePrediction(game, bound_probability(self.clf.predict_proba([features])[0][0])))
        print('predictions done for season '+str(season)+'-- '
              +str([(seeds_map[x.game.team_a_id].seed, seeds_map[x.game.team_b_id].seed, x.prediction) for x in game_predictions])
              )
        return game_predictions


class SeedsPerceptronPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = SeedsPerceptronPredictor(load_tourney_seeds, load_tourney_compact_results)
        self.active_seasons = range(2015, 2019) # set([x.season for x in load_tourney_seeds()])
        self.predictor_description = 'seeds_perceptron'
        # keras()
