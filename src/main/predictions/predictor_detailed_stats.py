from sklearn.neural_network import MLPClassifier

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_compact_results, load_detailed_box, load_tourney_compact_results
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
        self.model.add(Dense(12, input_dim=10, init='uniform', activation='relu'))
        self.model.add(Dense(6, init='uniform', activation='relu'))
        self.model.add(Dense(1, init='uniform', activation='sigmoid'))
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
        # Fit the model
        self.model.fit(X, Y, epochs=400, batch_size=10, verbose=2)

    def predict_proba(self, x_data):
        # calculate predictions
        return self.model.predict(np.array(x_data))


class DetailedStatsPredictor(AbstractPredictor):
    def __init__(self, load_detailed_box_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_detailed_box_function = load_detailed_box_function
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = KerasDeepNeural()

        # self.scaler = StandardScaler()

    class Features:
        def __init__(self, wins: int, points: int, assists: int, rebounds: int, turnovers: int, games: int) -> None:
            super().__init__()
            self.wins = wins
            self.points = points
            self.assists = assists
            self.rebounds = rebounds
            self.turnovers = turnovers
            self.games = games

        def avg_wins(self):
            return self.wins / self.games

        def avg_points(self):
            return self.points / self.games

        def avg_assists(self):
            return self.assists / self.games

        def avg_rebounds(self):
            return self.rebounds / self.games

        def avg_turnovers(self):
            return self.turnovers / self.games

    def train(self, seasons: [int]):
        self.clf = KerasDeepNeural()
        train_data = []
        train_results = []
        for season in seasons:
            detailed_results = [x for x in self.load_detailed_box_function(regular_season=True) if x.season == season]
            teams_map = {}
            for result in detailed_results:
                if result.w_team_id in teams_map:
                    teams_map[result.w_team_id].wins += 5
                    teams_map[result.w_team_id].points += (result.w_score - result.l_score)
                    teams_map[result.w_team_id].assists += (result.w_ast - result.l_ast)
                    teams_map[result.w_team_id].rebounds += (result.w_or + result.w_dr - result.l_dr - result.l_or)
                    teams_map[result.w_team_id].turnovers += (result.w_to - result.l_to)
                    teams_map[result.w_team_id].games += 1
                else:
                    teams_map[result.w_team_id] = self.Features(
                        wins=5, points=(result.w_score - result.l_score), assists=(result.w_ast - result.l_ast),
                        rebounds=(result.w_or + result.w_dr - result.l_dr - result.l_or),
                        turnovers=(result.w_to - result.l_to), games=1
                    )
                if result.l_team_id in teams_map:
                    teams_map[result.l_team_id].wins -= 5
                    teams_map[result.l_team_id].points += (result.l_score - result.w_score)
                    teams_map[result.l_team_id].assists += (result.l_ast - result.w_ast)
                    teams_map[result.l_team_id].rebounds += (result.l_or + result.l_dr - result.w_dr - result.w_or)
                    teams_map[result.l_team_id].turnovers += (result.l_to - result.w_to)
                    teams_map[result.l_team_id].games += 1
                else:
                    teams_map[result.l_team_id] = self.Features(
                        wins=-5, points=(result.l_score - result.w_score), assists=(result.l_ast - result.w_ast),
                        rebounds=(result.l_or + result.l_dr - result.w_dr - result.w_or),
                        turnovers=(result.l_to - result.w_to), games=1
                    )

            for result in self.load_tourney_compact_results_function(season):
                winner = teams_map[result.w_team_id]
                looser = teams_map[result.l_team_id]
                if result.w_team_id < result.l_team_id:
                    train_data.append(
                        [winner.avg_wins(), winner.avg_points(), winner.avg_rebounds(), winner.avg_assists(), winner.avg_turnovers(),
                         looser.avg_wins(), looser.avg_points(), looser.avg_rebounds(), looser.avg_assists(), looser.avg_turnovers()])
                    train_results.append(1)
                else:
                    train_data.append(
                        [looser.avg_wins(), looser.avg_points(), looser.avg_rebounds(), looser.avg_assists(), looser.avg_turnovers(),
                         winner.avg_wins(), winner.avg_points(), winner.avg_rebounds(), winner.avg_assists(), winner.avg_turnovers()])
                    train_results.append(0)

        print('training')
        #self.scaler.fit(train_data)
        #train_data = self.scaler.transform(train_data)
        self.clf.fit(train_data, train_results)
        print('training is done')

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        detailed_results = [x for x in self.load_detailed_box_function(regular_season=True) if x.season == season]
        teams_map = {}
        for result in detailed_results:
            if result.w_team_id in teams_map:
                teams_map[result.w_team_id].wins += 5
                teams_map[result.w_team_id].points += (result.w_score - result.l_score)
                teams_map[result.w_team_id].assists += (result.w_ast - result.l_ast)
                teams_map[result.w_team_id].rebounds += (result.w_or + result.w_dr - result.l_dr - result.l_or)
                teams_map[result.w_team_id].turnovers += (result.w_to - result.l_to)
                teams_map[result.w_team_id].games += 1
            else:
                teams_map[result.w_team_id] = self.Features(
                    wins=5,
                    points=(result.w_score - result.l_score), assists=(result.w_ast - result.l_ast),
                    rebounds=(result.w_or + result.w_dr - result.l_dr - result.l_or),
                    turnovers=(result.w_to - result.l_to), games=1
                )
            if result.l_team_id in teams_map:
                teams_map[result.l_team_id].wins -= 5
                teams_map[result.l_team_id].points += (result.l_score - result.w_score)
                teams_map[result.l_team_id].assists += (result.l_ast - result.w_ast)
                teams_map[result.l_team_id].rebounds += (result.l_or + result.l_dr - result.w_dr - result.w_or)
                teams_map[result.l_team_id].turnovers += (result.l_to - result.w_to)
                teams_map[result.l_team_id].games += 1
            else:
                teams_map[result.l_team_id] = self.Features(
                    wins=-5,
                    points=(result.l_score - result.w_score), assists=(result.l_ast - result.w_ast),
                    rebounds=(result.l_or + result.l_dr - result.w_dr - result.w_or),
                    turnovers=(result.l_to - result.w_to), games=1
                )

        game_predictions = []
        print('starting predictions for season '+str(season))
        for game in games:
            team_a_stats = teams_map[game.team_a_id]
            team_b_stats = teams_map[game.team_b_id]

            features = [
                [team_a_stats.avg_wins(), team_a_stats.avg_points(), team_a_stats.avg_rebounds(),
                 team_a_stats.avg_assists(), team_a_stats.avg_turnovers(),
                 team_b_stats.avg_wins(), team_b_stats.avg_points(), team_b_stats.avg_rebounds(),
                 team_b_stats.avg_assists(), team_b_stats.avg_turnovers()]
            ]

            game_predictions.append(GamePrediction(game, bound_probability(self.clf.predict_proba(features)[0][0])))
        print('predictions done for season '+str(season)+'-- '
              +str([(teams_map[x.game.team_a_id].avg_points(), teams_map[x.game.team_b_id].avg_points(), x.prediction) for x in game_predictions])
              )
        return game_predictions


class DetailedStatsEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = DetailedStatsPredictor(load_detailed_box, load_tourney_compact_results)
        self.active_seasons = range(2015, 2019) # set([x.season for x in load_detailed_box()])
        self.predictor_description = 'detailed_stats'
