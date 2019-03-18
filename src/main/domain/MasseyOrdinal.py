class MasseyOrdinal:
    def __init__(
            self,
            Season,
            RankingDayNum,
            SystemName,
            TeamID,
            OrdinalRank
    ) -> None:
        super().__init__()
        self.season = int(Season)
        self.ranking_day_num = int(RankingDayNum)
        self.system_name = SystemName
        self.ordinal_rank = int(OrdinalRank)
        self.team_id = int(TeamID)

