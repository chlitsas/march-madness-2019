import csv

from src.main.domain.GamePrediction import GamePrediction
from src.main.domain.data_loaders import load_tourney_games, \
    load_tourney_compact_results
from src.main.predictions.predictor_extended_stats_catboost import ExtendedStatsCatBoostPredictorEvaluator


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
    evaluator = ExtendedStatsCatBoostPredictorEvaluator()
    data = evaluator.evaluate(
        games_loader=load_tourney_games,
        compact_results_loader=load_tourney_compact_results
    )

    with open('./reports/'+evaluator.predictor_description+'.csv', 'w') as myfile:
        for row in data:
            wr = csv.writer(myfile, lineterminator='\n')
            print(row)
            wr.writerow(row)
