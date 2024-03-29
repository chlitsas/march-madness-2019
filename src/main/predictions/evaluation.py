from math import log

from src.main.domain.CompactResult import CompactResult
from src.main.domain.GamePrediction import GamePrediction
from src.main.predictions.predictors import AbstractPredictor


def log_loss(win_prob, team_a_wins) -> float:
    return log(win_prob) if team_a_wins else log(1 - win_prob)


def evaluate_predictions(season: int, game_predictions: [GamePrediction], compact_results: [CompactResult]) -> float:
    games_map = {}
    for compact_result in compact_results:
        if compact_result.season == season:
            if compact_result.w_team_id < compact_result.l_team_id:
                games_map[str(compact_result.w_team_id) + '-' + str(compact_result.l_team_id)] = 1
            else:
                games_map[str(compact_result.l_team_id) + '-' + str(compact_result.w_team_id)] = 0

    sum_loss = 0
    n = 0
    for game_prediction in game_predictions:
        game_id = str(game_prediction.game.team_a_id) + '-' + str(game_prediction.game.team_b_id)
        team_a_wins = games_map.get(game_id)
        sum_loss += log_loss(game_prediction.prediction, team_a_wins)
        n += 1
    return -sum_loss / n


class PredictorEvaluationTemplate:
    predictor: AbstractPredictor
    active_seasons: [int]
    predictor_description: str
    # create data for predictor
    # instantiate predictor
    # train predictor
    # create data for target season
    # create predictions
    # evaluate predictions

    def evaluate(
            self,
            games_loader,
            compact_results_loader
    ) -> [[]]:
        result = []
        for season in self.active_seasons:
            training = [x for x in self.active_seasons if x != season]
            self.predictor.train(seasons=training)
            games = games_loader(season)
            compact_results = compact_results_loader(season)
            evaluation = evaluate_predictions(
                season=season,
                compact_results=compact_results,
                game_predictions=self.predictor.get_predictions(season=season, games=games)
            )
            print([self.predictor_description, season, evaluation, len(games)])
            result.append([self.predictor_description, season, evaluation, len(games)])
        return result
