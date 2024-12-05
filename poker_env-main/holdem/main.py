import random

import rlcard


def policy_rand(s):
    return random.choice(list(s['legal_actions'].keys()))


def minimal_demo():
    brains = {
        0: policy_rand,
        1: policy_rand,
        2: policy_rand,
    }

    env = rlcard.make('limit-holdem', {'game_num_players': 8})
    print(env.actions)
    print(env.num_players)

    state, player_id = env.reset()
    print(state)

    while not env.is_over():
        action = brains[player_id](state)  # 当前玩家做出决策
        state, player_id = env.step(action)  # 执行一步，更新下一个状态和下一个玩家

    rewards = env.get_payoffs()  # 对应每个玩家的收益，需要在结束后才有
    print(rewards)


class DummyAgent:
    def __init__(self, player_id, env):
        self._player_id = player_id
        self.env = env
        self.use_raw = False

    def step(self, state):
        print(state)
        return random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        return self.step(state)


def wrapped_env_demo():
    from LimitHoldem import LimitHoldemEnv

    num_players = 3
    env = LimitHoldemEnv(num_players=num_players, imperfect_info=False)
    agents = {
        0: DummyAgent(0, env),
        1: DummyAgent(1, env),
        2: DummyAgent(2, env)
    }
    env.ResetEnv(agents)
    # env.EnableImperfectInformation()

    while not env.CheckGameOver():
        state = env.GetGameState()
        player_id = state['cur_player']
        # print(state)
        # print(env.GetLegalActions(player_id))
        action = agents[player_id].step(state)
        s, a, r, s_, done = env.Step(action)
        env.SaveObservation(player_id, s, a, r, s_, done)

    hist = env.GetGameHistory()
    print(hist)


if __name__ == '__main__':
    # minimal_demo()
    wrapped_env_demo()
