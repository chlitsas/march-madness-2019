class Game:
    def __init__(
            self,
            team_a_id: int,
            team_b_id: int
    ) -> None:
        super().__init__()
        if team_b_id < team_a_id:
            raise ValueError('team_a_id should be less than team_b_id')
        self.team_a_id = team_a_id
        self.team_b_id = team_b_id


class GamePrediction:
    def __init__(
            self,
            game,
            prediction
    ) -> None:
        super().__init__()
        self.game = game
        self.prediction = prediction
