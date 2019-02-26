class CompactResult:
    def __init__(
            self,
            Season,
            DayNum,
            WTeamID,
            WScore,
            LTeamID,
            LScore,
            WLoc,
            NumOT
    ) -> None:
        super().__init__()
        self.season = Season
        self.day_num = DayNum
        self.w_team_id = WTeamID
        self.w_score = WScore
        self.l_team_id = LTeamID
        self.l_score = LScore
        self.w_loc = WLoc
        self.num_ot = NumOT

    def winner_scored_more_threes(self):
        return self.w_fgm3 > self.l_fgm3

    def winner_scored_more_twos(self):
        return self.w_fga - self.w_fgm3 > self.l_fga - self.l_fgm3

    def winner_had_better_ball_handling(self):
        multiplier = 3.1
        return multiplier*self.w_ast - self.w_to > multiplier*self.l_ast - self.l_to

    def winner_scored_more_points(self):
        return self.w_score > self.l_score

    def __str__(self) -> str:
        return 'season = '+str(self.season)+', day_num = '+str(self.day_num)+\
               ' ,winner = ' + str(self.w_team_id)+', looser = ' + str(self.l_team_id)+\
               ', score = ' + str(self.w_score)+'-' + str(self.l_score)
