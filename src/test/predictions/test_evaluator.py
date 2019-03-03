import unittest

from src.main.domain.CompactResult import CompactResult
from src.main.domain.GamePrediction import GamePrediction, Game
from src.main.predictions.evaluation import evaluate_predictions, PredictorEvaluationTemplate
from src.main.predictions.predictor_fifty_fifty import FiftyFiftyPredictor


def create_compact_result(season, w_team_id, l_team_id):
    return CompactResult(
        Season=season,
        WTeamID=w_team_id,
        LTeamID=l_team_id,
        DayNum=1,
        WScore=80,
        LScore=70,
        WLoc=4,
        NumOT=0
    )


class TestPredictionEvaluator(unittest.TestCase):

    def test_evaluator(self):
        # create some dummy predictions
        game_predictions = [
            GamePrediction(Game(team_a_id=1, team_b_id=2), 0.3),
            GamePrediction(Game(team_a_id=2, team_b_id=4), 0.7)
        ]

        # create an array of compact_results
        compact_results = [
            create_compact_result(season=2018, w_team_id=1, l_team_id=2),
            create_compact_result(season=2018, w_team_id=4, l_team_id=2)
        ]

        # evaluate the prediction accuracy based on the given results (compact_results)
        score = evaluate_predictions(2018, game_predictions=game_predictions, compact_results=compact_results)
        self.assertAlmostEqual(1.20397280432, score, places=10)

    def test_predictor_evaluation_template(self):
        def games_loader(x):
            return [Game(team_a_id=1, team_b_id=2), Game(team_a_id=2, team_b_id=4)]

        def compact_results_loader(x):
            return [
                create_compact_result(season=2018, w_team_id=1, l_team_id=2),
                create_compact_result(season=2018, w_team_id=4, l_team_id=2)
            ]

        class Test(PredictorEvaluationTemplate):
            def __init__(self) -> None:
                super().__init__()
                self.predictor = FiftyFiftyPredictor()
                self.active_seasons = [2018]
                self.predictor_description = 'test'

        data = Test().evaluate(
            games_loader=games_loader,
            compact_results_loader=compact_results_loader
        )
        self.assertEqual(1, len(data))
        self.assertEqual(3, len(data[0]))

        self.assertEqual('test', data[0][0])
        self.assertEqual(2018, data[0][1])
        self.assertAlmostEqual(0.69314718056, data[0][2], places=10)

