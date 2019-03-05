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
        self.season = int(Season)
        self.day_num = DayNum
        self.w_team_id = int(WTeamID)
        self.w_score = int(WScore)
        self.l_team_id = int(LTeamID)
        self.l_score = int(LScore)
        self.w_loc = WLoc
        self.num_ot = int(NumOT)
        self.w_fgm = int(WFGM)
        self.w_fga = int(WFGA)
        self.w_fgm3 = int(WFGM3)
        self.w_fga3 = int(WFGA3)
        self.w_ftm = int(WFTM)
        self.w_fta = int(WFTA)
        self.w_or = int(WOR)
        self.w_dr = int(WDR)
        self.w_ast = int(WAst)
        self.w_to = int(WTO)
        self.w_stl = int(WStl)
        self.w_blk = int(WBlk)
        self.w_pf = int(WPF)
        self.l_fgm = int(LFGM)
        self.l_fga = int(LFGA)
        self.l_fgm3 = int(LFGM3)
        self.l_fga3 = int(LFGA3)
        self.l_ftm = int(LFTM)
        self.l_fta = int(LFTA)
        self.l_or = int(LOR)
        self.l_dr = int(LDR)
        self.l_ast = int(LAst)
        self.l_to = int(LTO)
        self.l_stl = int(LStl)
        self.l_blc = int(LBlk)
        self.l_pf = int(LPF)

    def winner_scored_more_threes(self):
        return self.w_fgm3 > self.l_fgm3

    def winner_scored_more_twos(self):
        return self.w_fga - self.w_fgm3 > self.l_fga - self.l_fgm3

    def winner_had_better_ball_handling(self):
        multiplier = 3.1
        return multiplier*self.w_ast - self.w_to > multiplier*self.l_ast - self.l_to

    def winner_scored_more_points(self):
        return self.w_score > self.l_score
