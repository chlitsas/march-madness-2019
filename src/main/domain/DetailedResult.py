class DetailedResult:
    def __init__(
            self,
            Season,
            DayNum,
            WTeamID,
            WScore,
            LTeamID,
            LScore,
            WLoc,
            NumOT,
            WFGM,
            WFGA,
            WFGM3,
            WFGA3,
            WFTM,
            WFTA,
            WOR,
            WDR,
            WAst,
            WTO,
            WStl,
            WBlk,
            WPF,
            LFGM,
            LFGA,
            LFGM3,
            LFGA3,
            LFTM,
            LFTA,
            LOR,
            LDR,
            LAst,
            LTO,
            LStl,
            LBlk,
            LPF

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
        self.w_fgm = WFGM
        self.w_fga = WFGA
        self.w_fgm3 = WFGM3
        self.w_fga3 = WFGA3
        self.w_ftm = WFTM
        self.w_fta = WFTA
        self.w_or = WOR
        self.w_dr = WDR
        self.w_ast = WAst
        self.w_to = WTO
        self.w_stl = WStl
        self.w_blk = WBlk
        self.w_pf = WPF
        self.l_fgm = LFGM
        self.l_fga = LFGA
        self.l_fgm3 = LFGM3
        self.l_fga3 = LFGA3
        self.l_ftm = LFTM
        self.l_fta = LFTA
        self.l_or = LOR
        self.l_dr = LDR
        self.l_ast = LAst
        self.l_to = LTO
        self.l_stl = LStl
        self.l_blc = LBlk
        self.l_pf = LPF

    def winner_scored_more_threes(self):
        return self.w_fgm3 > self.l_fgm3

    def winner_scored_more_twos(self):
        return self.w_fga - self.w_fgm3 > self.l_fga - self.l_fgm3

    def winner_had_better_ball_handling(self):
        multiplier = 3.1
        return multiplier*self.w_ast - self.w_to > multiplier*self.l_ast - self.l_to

    def winner_scored_more_points(self):
        return self.w_score > self.l_score
