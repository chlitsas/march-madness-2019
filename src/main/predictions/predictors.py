from src.main.domain.GamePrediction import GamePrediction


class AbstractPredictor:
    def train(self, seasons: [int]) -> None:
        pass

    def get_predictions(self, season: int, games) -> [GamePrediction]:
        pass


def bound_probability(prob):
    if prob > 1:
        print('wrong probability with value '+str(prob))
        return 0.99
    if prob < 0:
        print('wrong probability with value '+str(prob))
        return 0.01
    return 0.01 + prob * 0.98
