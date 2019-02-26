class PlayByPlayEvent:
    def __init__(
            self,
            winner_points,
            looser_points,
            time,
            player_id,
            team_id,
            event_type
    ) -> None:
        super().__init__()
        self._winner_points = winner_points
        self._looser_points = looser_points
        self._time = time
        self._player_id = player_id
        self._team_id = team_id
        self._event_type = event_type

    @property
    def winner_points(self):
        return self._winner_points

    @property
    def looser_points(self):
        return self._looser_points

    @property
    def time(self):
        return self._time

    @property
    def player_id(self):
        return self._player_id

    @property
    def possession_team_id(self):
        return self._possession_team_id

    @property
    def defensive_team_id(self):
        return self._defensive_team_id

    @property
    def team_id(self):
        return self._team_id

    @property
    def event_type(self):
        return self._event_type
