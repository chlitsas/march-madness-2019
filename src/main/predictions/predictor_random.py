import random

from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_parsers import parse_tourney_seeds
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor


class RandomPredictor(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        game_predictions = []
        for game in games:
            game_predictions.append(GamePrediction(game, random.uniform(0, 1)))
        return game_predictions


class RandomPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = RandomPredictor()
        self.active_seasons = set([x.season for x in parse_tourney_seeds()])
        self.predictor_description = 'random'


