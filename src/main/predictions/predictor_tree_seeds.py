from src.main.domain.GamePrediction import Game, GamePrediction
from src.main.domain.data_parsers import parse_tourney_seeds, parse_tourney_compact_results
from src.main.predictions.evaluation import PredictorEvaluationTemplate
from src.main.predictions.predictors import AbstractPredictor, bound_probability
from sklearn.ensemble import RandomForestClassifier


class SeedsTreePredictor(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            n_jobs=2
        )

    class Features:
        def __init__(self, region: str, seed: int) -> None:
            super().__init__()
            self.region = region
            self.seed = seed

        def numerical_region(self):
            if self.region == 'W':
                return 1
            if self.region == 'X':
                return 2
            if self.region == 'Y':
                return 3
            if self.region == 'Z':
                return 4
            raise ValueError('Undefined Region')

    def train(self, seasons: [int]):
        train_data = []
        train_results = []
        for season in seasons:
            compact_results = parse_tourney_compact_results(season)
            seeds = [x for x in parse_tourney_seeds() if x.season == season]
            seeds_map = {}
            for seed in seeds:
                seeds_map[seed.team_id] = self.Features(region=seed.get_region(), seed=seed.get_seed())
            for result in compact_results:
                winner = seeds_map[result.w_team_id]
                looser = seeds_map[result.l_team_id]
                if result.w_team_id < result.l_team_id:
                    train_data.append([winner.seed, winner.numerical_region(), looser.seed, looser.numerical_region()])
                    train_results.append(1)
                else:
                    train_data.append([looser.seed, looser.numerical_region(), winner.seed, winner.numerical_region()])
                    train_results.append(0)
        print('training')
        self.clf.fit(train_data, train_results)
        print('training is done')

    def get_predictions(self, season: int, games: [Game]) -> [GamePrediction]:
        seeds = [x for x in parse_tourney_seeds() if x.season == season]
        seeds_map = {}
        for seed in seeds:
            seeds_map[seed.team_id] = self.Features(region=seed.get_region(), seed=seed.get_seed())
        game_predictions = []
        print('starting predictions for season '+str(season))
        for game in games:
            team_a_seed = seeds_map[game.team_a_id]
            team_b_seed = seeds_map[game.team_b_id]

            features = \
                [team_a_seed.seed, team_a_seed.numerical_region(), team_b_seed.seed, team_b_seed.numerical_region()]

            game_predictions.append(GamePrediction(game, bound_probability(self.clf.predict_proba([features])[0][1])))
        print('predictions done for season '+str(season)+'-- '
              +str([(seeds_map[x.game.team_a_id].seed, seeds_map[x.game.team_b_id].seed, x.prediction) for x in game_predictions])
              )
        return game_predictions


class SeedsTreePredictorEvaluator(PredictorEvaluationTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = SeedsTreePredictor()
        self.active_seasons = set([x.season for x in parse_tourney_seeds()])
        self.predictor_description = 'seeds_tree'
