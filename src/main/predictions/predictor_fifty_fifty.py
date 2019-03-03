from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor


class FiftyFiftyPredictor(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()

    def get_predictions(self, future_games: [Game]) -> [GamePrediction]:
        game_predictions = []
        for game in future_games:
            game_predictions.append(GamePrediction(game, 0.5))
        return game_predictions


class FiftyFiftyPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = FiftyFiftyPredictor()
        self.active_seasons = range(2010, 2018)
        self.predictor_description = 'test'

    def train(self, training_seasons: [int]):
        super().train(training_seasons)

