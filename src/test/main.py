import unittest

from src.main.domain.GamePrediction import GamePrediction, Game
from src.main.domain.data_loaders import load_tourney_seeds
from src.main.main import write_predictions


class TestSomeMethods(unittest.TestCase):

    def test_upper(self):
        game_predictions = [GamePrediction(Game(123, 345), 0.234), GamePrediction(Game(2, 4), 0.432)]
        write_predictions('test.csv', game_predictions)
        seeds = [x for x in load_tourney_seeds() if x.season == 2018]
        sorted_seeds = sorted(seeds, key=lambda x: x.team_id, reverse=False)
        games = []
        for i in range(len(sorted_seeds) - 1):
            for j in range(i + 1, len(sorted_seeds) - 1):
                games.append((sorted_seeds[i].team_id, sorted_seeds[j].team_id))
        print(games)
        print([(1, 2), (3, 4)][0][0])
        self.assertEqual('foo'.upper(), 'FOO')
