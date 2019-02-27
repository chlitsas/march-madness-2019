import unittest

from src.main.domain.CompactResult import CompactResult
from src.main.domain.GamePrediction import GamePrediction, Game
from src.main.predictions.evaluation import evaluate_predictions, evaluate_predictor
from src.main.predictions.predictors import FiftyFiftyPredictor


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
        game_predictions = [GamePrediction(Game(1, 2), 0.3), GamePrediction(Game(2, 4), 0.7)]
        compact_results = [
            create_compact_result(season=2018, w_team_id=1, l_team_id=2),
            create_compact_result(season=2018, w_team_id=4, l_team_id=2)
        ]
        score = evaluate_predictions(2018, game_predictions=game_predictions, compact_results=compact_results)
        self.assertAlmostEqual(1.20397280432, score, places=10)

    def test_predictor_evaluator(self):
        games = [Game(1, 2), Game(2, 4)]
        compact_results = [
            create_compact_result(season=2018, w_team_id=1, l_team_id=2),
            create_compact_result(season=2018, w_team_id=4, l_team_id=2)
        ]
        score = evaluate_predictor(
            season=2018,
            games=games,
            compact_results=compact_results,
            predictor=FiftyFiftyPredictor()
        )
        self.assertAlmostEqual(0.69314718056, score, places=10)
