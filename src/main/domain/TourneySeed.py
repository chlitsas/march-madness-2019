class TourneySeed:
    def __init__(
            self,
            Season,
            Seed,
            TeamID
    ) -> None:
        super().__init__()
        self.season = Season
        self.seed = Seed
        self.team_id = TeamID

