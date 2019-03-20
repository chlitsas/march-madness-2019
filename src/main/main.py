import csv

from src.main.domain.GamePrediction import GamePrediction
from src.main.domain.data_loaders import load_tourney_games, \
    load_tourney_compact_results, detailed_stats_with_seeds_df, load_games_to_submit
from src.main.predictions.predictor_extended_stats_catboost import ExtendedStatsCatBoostPredictorEvaluator, \
    ExtendedStatsCatBoostPredictor


def write_predictions(filename, game_predictions: [GamePrediction]):
    file = open(filename, 'w')
    file.write('ID,Pred\n')
    for game_prediction in game_predictions:
        file.write(
            '2019_'+str(game_prediction.game.team_a_id) + '_' + str(game_prediction.game.team_b_id) + ',' + str(
                game_prediction.prediction) + '\n'
        )
    file.close()


if __name__ == '__main__':
    predictions = False
    if predictions:
        predictor = ExtendedStatsCatBoostPredictor(detailed_stats_with_seeds_df, load_tourney_compact_results)
        predictor.train(seasons=range(2003, 2019))
        games = load_games_to_submit()
        write_predictions('first.csv', predictor.get_predictions(2019, games))
    else:
        evaluator = ExtendedStatsCatBoostPredictorEvaluator()
        data = evaluator.evaluate(
            games_loader=load_tourney_games,
            compact_results_loader=load_tourney_compact_results
        )

        with open('./reports/'+evaluator.predictor_description+'.csv', 'w') as myfile:
            _sum = 0
            _cnt = 0
            for row in data:
                wr = csv.writer(myfile, lineterminator='\n')
                print(row)
                _sum += row[-2]
                _cnt += 1
                print('Accumulated average: '+str(_sum/_cnt))
                wr.writerow(row)
