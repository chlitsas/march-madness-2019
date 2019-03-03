from src.main.domain.GamePrediction import GamePrediction, Game
from src.main.domain.data_parsers import parse_tourney_seeds, parse_compact_results, parse_tourney_games, \
    parse_tourney_compact_results
from src.main.predictions.predictor_fifty_fifty import FiftyFiftyPredictorEvaluator
from src.main.predictions.predictor_random import RandomPredictor, RandomPredictorEvaluator


def write_predictions(filename, game_predictions: [GamePrediction]):
    file = open(filename, 'w')
    file.write('ID,Pred\n')
    for game_prediction in game_predictions:
        file.write(
            str(game_prediction.game.team_a_id) + '_' + str(game_prediction.game.team_b_id) + ',' + str(
                game_prediction.prediction) + '\n'
        )
    file.close()


def bound_probability(prob):
    return 0.01 + prob * 0.98


if __name__ == '__main__':
    seeds = [x for x in parse_tourney_seeds() if x.season == 2018]
    sorted_seeds = sorted(seeds, key=lambda x: x.team_id, reverse=False)
    games = []
    for i in range(len(sorted_seeds) - 1):
        for j in range(i + 1, len(sorted_seeds)):
            games.append(Game(sorted_seeds[i].team_id, sorted_seeds[j].team_id))
    print(games)

    generator = RandomPredictor()

    write_predictions('tests.csv', generator.get_predictions(future_games=games))
    compact_results = [x for x in parse_compact_results(regular_season=False) if x.season == 2018]

    print(compact_results)

    eval = FiftyFiftyPredictorEvaluator()
    data = eval.evaluate(
        games_loader=parse_tourney_games,
        compact_results_loader=parse_tourney_compact_results
    )
    print(data)

    eval = RandomPredictorEvaluator()
    data = eval.evaluate(
        games_loader=parse_tourney_games,
        compact_results_loader=parse_tourney_compact_results
    )
    print(data)
