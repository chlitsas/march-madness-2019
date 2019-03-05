from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_loaders import load_tourney_seeds
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability


class SeedsPredictor(AbstractPredictor):
    def __init__(self, load_tourney_seeds_function) -> None:
        super().__init__()
        self.load_tourney_seeds_function = load_tourney_seeds_function

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        seeds = [x for x in self.load_tourney_seeds_function() if x.season == season]
        max_seed = max([x.get_seed() for x in seeds])
        seeds_map = {}
        for seed in seeds:
            seeds_map[seed.team_id] = seed.get_seed()
        game_predictions = []
        for game in games:
            team_a_seed = seeds_map[game.team_a_id]
            team_b_seed = seeds_map[game.team_b_id]

            team_a_advantage = (team_b_seed - team_a_seed) / (2 * max_seed)

            game_predictions.append(GamePrediction(game, bound_probability(0.5 + team_a_advantage)))
        return game_predictions


class SeedsPredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = SeedsPredictor(load_tourney_seeds)
        self.active_seasons = set([x.season for x in load_tourney_seeds()])
        self.predictor_description = 'seeds'
