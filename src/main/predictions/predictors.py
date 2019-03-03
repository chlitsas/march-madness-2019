from src.main.domain.GamePrediction import Game, GamePrediction


class AbstractPredictor:
    def train(self) -> None:
        pass

    def get_predictions(self, future_games: [Game]) -> [GamePrediction]:
        pass


