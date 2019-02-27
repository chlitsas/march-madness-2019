import random
from src.main.domain.GamePrediction import Game, GamePrediction


class AbstractPredictor:
    def get_predictions(self, future_games: [Game]) -> [GamePrediction]:
        pass


class FiftyFiftyPredictor(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()

    def get_predictions(self, future_games: [Game]) -> [GamePrediction]:
        game_predictions = []
        for game in future_games:
            game_predictions.append(GamePrediction(game, 0.5))
        return game_predictions


class RandomPredictor(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()

    def get_predictions(self, future_games: [Game]) -> [GamePrediction]:
        game_predictions = []
        for game in future_games:
            game_predictions.append(GamePrediction(game, random.uniform(0, 1)))
        return game_predictions
