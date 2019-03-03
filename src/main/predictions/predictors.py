from src.main.domain.GamePrediction import Game, GamePrediction


class AbstractPredictor:
    def train(self, seasons: [int]) -> None:
        pass

    def get_predictions(self, season: int, games) -> [GamePrediction]:
        pass


def bound_probability(prob):
    return 0.01 + prob * 0.98
