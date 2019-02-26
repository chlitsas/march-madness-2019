from math import log

from src.main.domain import CompactResult
from src.main.domain.GamePrediction import Game, GamePrediction


def log_loss(win_prob, team_a_wins):
    return log(win_prob) if team_a_wins else log(1 - win_prob)


class PredictionGenerator:
    def __init__(self, future_games: [Game]) -> None:
        super().__init__()
        self.future_games = future_games

    def get_predictions(self):
        game_predictions = []
        for game in self.future_games:
            game_predictions.append(GamePrediction(game, 0.5))
        return game_predictions


def evaluate_predictions(season: int, game_predictions: [GamePrediction], compact_results: [CompactResult]):
    games_map = {}
    for compact_result in compact_results:
        if compact_result.season == season:
            if compact_result.w_team_id < compact_result.l_team_id:
                games_map[str(compact_result.w_team_id)+'-'+str(compact_result.l_team_id)] = 1
            else:
                games_map[str(compact_result.l_team_id)+'-'+str(compact_result.w_team_id)] = 0

    sum_loss = 0
    n = 0
    for game_prediction in game_predictions:
        game_id = str(game_prediction.game.team_a_id)+'-'+str(game_prediction.game.team_b_id)
        team_a_wins = games_map.get(game_id)
        sum_loss += log_loss(game_prediction.prediction, team_a_wins)
        n += 1
    return -sum_loss / n
