import csv

from src.main.domain.GamePrediction import GamePrediction, Game
from src.main.domain.data_parsers import parse_tourney_seeds, parse_compact_results, parse_tourney_games, \
    parse_tourney_compact_results
from src.main.predictions.predictor_fifty_fifty import FiftyFiftyPredictorEvaluator
from src.main.predictions.predictor_random import RandomPredictor, RandomPredictorEvaluator
from src.main.predictions.predictor_seeds import SeedsPredictorEvaluator


def write_predictions(filename, game_predictions: [GamePrediction]):
    file = open(filename, 'w')
    file.write('ID,Pred\n')
    for game_prediction in game_predictions:
        file.write(
            str(game_prediction.game.team_a_id) + '_' + str(game_prediction.game.team_b_id) + ',' + str(
                game_prediction.prediction) + '\n'
        )
    file.close()


if __name__ == '__main__':
    evaluator = SeedsPredictorEvaluator()
    data = evaluator.evaluate(
        games_loader=parse_tourney_games,
        compact_results_loader=parse_tourney_compact_results
    )

    with open('./reports/'+evaluator.predictor_description+'.csv', 'w') as myfile:
        for row in data:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator='\n')
            print(row)
            wr.writerow(row)
