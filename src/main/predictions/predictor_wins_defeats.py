from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_parsers import parse_tourney_seeds, parse_compact_results
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability
from sklearn.ensemble import RandomForestClassifier


class WinsDefeatsPredictor(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            n_jobs=2
        )

    class Features:
        def __init__(self, wins: int, defeats: int) -> None:
            super().__init__()
            self.wins = wins
            self.defeats = defeats

        def get_ratio(self):
            return self.wins / (self.wins + self.defeats)

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        results = [x for x in parse_compact_results(regular_season=True) if x.season == season]
        teams_map = {}
        for result in results:
            if result.w_team_id in teams_map:
                teams_map[result.w_team_id].wins += 1
            else:
                teams_map[result.w_team_id] = self.Features(wins=1, defeats=0)
            if result.l_team_id in teams_map:
                teams_map[result.l_team_id].defeats += 1
            else:
                teams_map[result.l_team_id] = self.Features(wins=0, defeats=1)

        max_win_ratio = max([x.get_ratio() for x in teams_map.values()])
        min_win_ratio = min([x.get_ratio() for x in teams_map.values()])

        print(max_win_ratio)
        print(min_win_ratio)
        print('------')

        game_predictions = []
        for game in games:
            team_a_ratio = teams_map[game.team_a_id].get_ratio()
            team_b_ratio = teams_map[game.team_b_id].get_ratio()

            team_a_advantage = (team_a_ratio - team_b_ratio) / (max_win_ratio - min_win_ratio)

            game_predictions.append(GamePrediction(game, bound_probability(0.5 + team_a_advantage)))
        return game_predictions


class WinsDefeatsEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = WinsDefeatsPredictor()
        self.active_seasons = set([x.season for x in parse_compact_results()])
        self.predictor_description = 'wins_defeats'
