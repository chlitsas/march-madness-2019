from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_parsers import parse_tourney_seeds
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor


class FiftyFiftyPredictor(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        game_predictions = []
        for game in games:
            game_predictions.append(GamePrediction(game, 0.5))
        return game_predictions


class FiftyFiftyPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = FiftyFiftyPredictor()
        self.active_seasons = set([x.season for x in parse_tourney_seeds()])
        self.predictor_description = 'test'

    def train(self, training_seasons: [int]):
        super().train(training_seasons)

