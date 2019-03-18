from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_tourney_seeds, load_massey_ordinals_df
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability


class CustomAlgoPredictor(AbstractPredictor):
    def __init__(self, load_tourney_seeds_function) -> None:
        super().__init__()
        self.load_tourney_seeds_function = load_tourney_seeds_function

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        # {'AP', 'WOL', 'RTH', 'WLK', 'DOL', 'MOR', 'USA', 'SAG', 'RPI', 'POM', 'COL'}
        massey_df = load_massey_ordinals_df()
        data = massey_df[(massey_df.Season == season) & (massey_df.RankingDayNum >= 133) & (massey_df.SystemName == 'POM')]
        diff = data.OrdinalRank.max()-data.OrdinalRank.min()
        game_predictions = []
        for game in games:
            team_a = data[data.TeamID == game.team_a_id].OrdinalRank.iloc[0]
            team_b = data[data.TeamID == game.team_b_id].OrdinalRank.iloc[0]

            team_a_advantage = (team_b - team_a) / (2 * diff)

            game_predictions.append(GamePrediction(game, bound_probability(0.5 + team_a_advantage)))
        return game_predictions


class CustomAlgoPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = CustomAlgoPredictor(load_tourney_seeds)
        self.active_seasons = range(2003, 2019)
        self.predictor_description = 'custom_algo_pom'
