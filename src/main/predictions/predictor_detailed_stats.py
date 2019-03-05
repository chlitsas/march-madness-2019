from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_compact_results, load_detailed_box, load_tourney_compact_results
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability
from sklearn.ensemble import RandomForestClassifier


class DetailedStatsPredictor(AbstractPredictor):
    def __init__(self, load_detailed_box_function, load_tourney_compact_results_function) -> None:
        super().__init__()
        self.load_detailed_box_function = load_detailed_box_function
        self.load_tourney_compact_results_function = load_tourney_compact_results_function
        self.clf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=5,
            n_jobs=5
        )

    class Features:
        def __init__(self, points: int, assists: int, rebounds: int, turnovers: int, games: int) -> None:
            super().__init__()
            self.points = points
            self.assists = assists
            self.rebounds = rebounds
            self.turnovers = turnovers
            self.games = games

        def avg_points(self):
            return self.points / self.games

        def avg_assists(self):
            return self.assists / self.games

        def avg_rebounds(self):
            return self.rebounds / self.games

        def avg_turnovers(self):
            return self.turnovers / self.games

    def train(self, seasons: [int]):
        train_data = []
        train_results = []
        for season in seasons:
            detailed_results = [x for x in self.load_detailed_box_function() if x.season == season]
            teams_map = {}
            for result in detailed_results:
                if result.w_team_id in teams_map:
                    teams_map[result.w_team_id].points += (result.w_score - result.l_score)
                    teams_map[result.w_team_id].assists += (result.w_ast - result.l_ast)
                    teams_map[result.w_team_id].rebounds += (result.w_or + result.w_dr - result.l_dr - result.l_or)
                    teams_map[result.w_team_id].turnovers += (result.w_to - result.l_to)
                    teams_map[result.w_team_id].games += 1
                else:
                    teams_map[result.w_team_id] = self.Features(
                        points=(result.w_score - result.l_score), assists=(result.w_ast - result.l_ast),
                        rebounds=(result.w_or + result.w_dr - result.l_dr - result.l_or),
                        turnovers=(result.w_to - result.l_to), games=1
                    )
                if result.l_team_id in teams_map:
                    teams_map[result.l_team_id].points += (result.l_score - result.w_score)
                    teams_map[result.l_team_id].assists += (result.l_ast - result.w_ast)
                    teams_map[result.l_team_id].rebounds += (result.l_or + result.l_dr - result.w_dr - result.w_or)
                    teams_map[result.l_team_id].turnovers += (result.l_to - result.w_to)
                    teams_map[result.l_team_id].games += 1
                else:
                    teams_map[result.l_team_id] = self.Features(
                        points=(result.l_score - result.w_score), assists=(result.l_ast - result.w_ast),
                        rebounds=(result.l_or + result.l_dr - result.w_dr - result.w_or),
                        turnovers=(result.l_to - result.w_to), games=1
                    )

            for result in self.load_tourney_compact_results_function(season):
                winner = teams_map[result.w_team_id]
                looser = teams_map[result.l_team_id]
                if result.w_team_id < result.l_team_id:
                    train_data.append(
                        [winner.avg_points(), winner.avg_assists(), winner.avg_rebounds(), winner.avg_turnovers(),
                         looser.avg_points(), looser.avg_assists(), looser.avg_rebounds(), looser.avg_turnovers()])
                    train_results.append(1)
                else:
                    train_data.append(
                        [looser.avg_points(), looser.avg_assists(), looser.avg_rebounds(), looser.avg_turnovers(),
                         winner.avg_points(), winner.avg_assists(), winner.avg_rebounds(), winner.avg_turnovers()])
                    train_results.append(0)

        print('training')
        self.clf.fit(train_data, train_results)
        print('training is done')

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        detailed_results = [x for x in self.load_detailed_box_function() if x.season == season]
        teams_map = {}
        for result in detailed_results:
            if result.w_team_id in teams_map:
                teams_map[result.w_team_id].points += (result.w_score - result.l_score)
                teams_map[result.w_team_id].assists += (result.w_ast - result.l_ast)
                teams_map[result.w_team_id].rebounds += (result.w_or + result.w_dr - result.l_dr - result.l_or)
                teams_map[result.w_team_id].turnovers += (result.w_to - result.l_to)
                teams_map[result.w_team_id].games += 1
            else:
                teams_map[result.w_team_id] = self.Features(
                    points=(result.w_score - result.l_score), assists=(result.w_ast - result.l_ast),
                    rebounds=(result.w_or + result.w_dr - result.l_dr - result.l_or),
                    turnovers=(result.w_to - result.l_to), games=1
                )
            if result.l_team_id in teams_map:
                teams_map[result.l_team_id].points += (result.l_score - result.w_score)
                teams_map[result.l_team_id].assists += (result.l_ast - result.w_ast)
                teams_map[result.l_team_id].rebounds += (result.l_or + result.l_dr - result.w_dr - result.w_or)
                teams_map[result.l_team_id].turnovers += (result.l_to - result.w_to)
                teams_map[result.l_team_id].games += 1
            else:
                teams_map[result.l_team_id] = self.Features(
                    points=(result.l_score - result.w_score), assists=(result.l_ast - result.w_ast),
                    rebounds=(result.l_or + result.l_dr - result.w_dr - result.w_or),
                    turnovers=(result.l_to - result.w_to), games=1
                )

        game_predictions = []
        print('starting predictions for season '+str(season))
        for game in games:
            team_a_stats = teams_map[game.team_a_id]
            team_b_stats = teams_map[game.team_b_id]

            features = \
                [team_a_stats.avg_points(), team_a_stats.avg_assists(), team_a_stats.avg_rebounds(), team_a_stats.avg_turnovers(),
                 team_b_stats.avg_points(), team_b_stats.avg_assists(), team_b_stats.avg_rebounds(), team_b_stats.avg_turnovers()]

            game_predictions.append(GamePrediction(game, bound_probability(self.clf.predict_proba([features])[0][1])))
        print('predictions done for season '+str(season)+'-- '
              +str([(teams_map[x.game.team_a_id].avg_points(), teams_map[x.game.team_b_id].avg_points(), x.prediction) for x in game_predictions])
              )
        return game_predictions


class DetailedStatsEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = DetailedStatsPredictor(load_detailed_box, load_tourney_compact_results)
        self.active_seasons = set([x.season for x in load_detailed_box()])
        self.predictor_description = 'detailed_stats'
