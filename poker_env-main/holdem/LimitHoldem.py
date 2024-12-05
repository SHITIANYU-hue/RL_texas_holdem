import rlcard


class LimitHoldemEnv:
    def __init__(self, num_players=2, imperfect_info=False):
        self.num_players = num_players
        self.env = None
        self.cur_state = None
        self.cur_player = None
        self.imperfect = imperfect_info
        self.agents = {}
        self.legal_actions = []  # a list for each player
        self.trajectories = []
        self.action_space = None

    def ResetEnv(self, agents=None):
        """重置游戏环境，包括重新发牌和初始化玩家状态。
        """
        self.env = rlcard.make('limit-holdem', {'game_num_players': self.num_players})
        self.legal_actions.clear()
        self.trajectories.clear()
        for i in range(self.env.num_players):
            self.legal_actions.append(self.env.actions[:])
            self.trajectories.append([])
        if agents:
            self.agents.update(agents)
        assert len(self.agents) == self.env.num_players
        self.env.set_agents(self.agents)
        self.cur_state, self.cur_player = self.env.reset()

    def GetGameState(self):
        """获取当前游戏状态，包括：
          - 玩家手牌。
          - 公共牌（Flop, Turn, River）。
          - 当前的游戏轮次（Pre-flop, Flop, Turn, River）。
          - 当前的投注池大小和玩家筹码。
          """
        if not self.imperfect:
            per = self.env.get_perfect_information()
            self.cur_state['all_hand_cards'] = per['hand_cards']
        self.cur_state['cur_player'] = self.cur_player
        return self.cur_state

    def GetLegalActions(self, player_id):
        """获取指定玩家可执行的合法操作（如 Check, Call, Raise, Fold）"""
        assert 0 <= player_id < self.num_players
        return self.legal_actions[player_id]

    def PlayerAction(self, player_id, action, amount=None):
        """
          - player_id: 当前玩家 ID。
          - action: 玩家选择的操作，如 Check, Call, Raise, Fold。
          - amount: 可选，适用于 Raise 或 All-in 的下注金额。
          - 返回值：执行结果（0 表示成功，其他表示失败及错误码）。
        """
        if player_id == self.cur_player:
            a = {'action': action, 'amount': amount}
            self.env.step(a)

    def GetPlayerState(self, player_id):
        """返回当前玩家的状态信息，如手牌、筹码量、当前下注金额等。"""
        return self.env.get_state(player_id)

    def GetPublicState(self):
        """返回当前的公共牌状态（Flop, Turn, River）"""
        return self.cur_state

    def GetReward(self, player_id):
        """获取指定玩家在当前局结束时的奖励（筹码变化情况）。"""
        if not self.env.is_over():
            return 0
        return self.env.get_payoffs()[player_id]

    def CheckGameOver(self):
        """检查游戏是否结束"""
        return self.env.is_over()

    def GetGameHistory(self):
        """返回从游戏开始到当前的所有动作历史，便于训练过程中回顾。"""
        return self.trajectories

    def Step(self, action):
        """ RL 环境的标准接口，执行一个动作，返回：
        - 当前状态 state
        - 动作结果 result
        - 下一状态 next_state
        - 奖励 reward
        - 是否结束 done
        """
        old_state = self.cur_state
        player_id = self.cur_player
        next_state, next_player_id = self.env.step(action)
        self.cur_state = next_state
        self.cur_player = next_player_id
        r = self.GetReward(player_id)
        done = self.CheckGameOver()
        return old_state, action, r, next_state, done

    def SaveObservation(self, player_id, state, action, reward, next_state, done):
        """存储每一步的观测数据，用于后续训练"""
        self.trajectories[player_id].append((state, action, reward, next_state, done))

    def EnableImperfectInformation(self):
        """启用不完全信息处理（如隐藏对手手牌）。"""
        self.imperfect = True

    def EnableMultiplayerCommunication(self):
        """如果涉及多智能体协作，可提供通信机制接口。"""
        # 不需要
        pass
