from src.main.domain.pbp_event import PlayByPlayEvent


class GameLogs:
    def __init__(
            self,
            game_id,
            winning_team_id,
            loosing_team_id,
            home_team_id,
            road_team_id,
            events: [PlayByPlayEvent]
    ) -> None:
        super().__init__()
        self._game_id = game_id
        self._winning_team_id = winning_team_id
        self._loosing_team_id = loosing_team_id
        self._home_team_id = home_team_id
        self._road_team_id = road_team_id
        self._events = events

    @property
    def game_id(self):
        return self._game_id

    @property
    def winning_team_id(self):
        return self._winning_team_id

    @property
    def loosing_team_id(self):
        return self._loosing_team_id

    @property
    def home_team_id(self):
        return self._home_team_id

    @property
    def road_team_id(self):
        return self._road_team_id

    @property
    def events(self) -> [PlayByPlayEvent]:
        return self._events
