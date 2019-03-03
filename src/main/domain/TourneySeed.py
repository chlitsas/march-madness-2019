class TourneySeed:
    def __init__(
            self,
            Season,
            Seed,
            TeamID
    ) -> None:
        super().__init__()
        self.season = int(Season)
        self.seed = Seed
        self.team_id = int(TeamID)

    def get_region(self):
        return self.seed[0]

    def get_seed(self):
        return int(self.seed[1:3])

