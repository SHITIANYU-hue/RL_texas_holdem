# Poker Env

这是一个德州扑克的RL环境

因为 [rlcard](https://github.com/datamllab/rlcard) 已有德州的环境(`limit-holdem`)，我们可以直接拿来用，但它有一些限制：

* 当前的 small blind 和 big blind 是固定的1和2
* 没有`all_in`操作
* 没有复杂的 main pot 和 side pots 收益计算
* `raise_amount` 初始等于big blind(即2)，但在`TURN`和 `RIVER`阶段会翻倍(即4)
* 当前玩家是随机指定的

`no-limit-holdem`环境没有这些限制，但它的加注数量也不是自由的，并且实现有bug。



## Demo

```python
import random
import rlcard

def policy_rand(s):
    return random.choice(list(s['legal_actions'].keys()))
def policy_rule(s):
    return logic(s)
def policy_rl(s):
    return rl_model.predict(s)
def policy_cfr(s):
    return cfr.infer(s)
def policy_llm(s):
    return llm.infer(s)

# 为每个玩家设置策略函数
brains = {
    0: policy_rand,          
    1: policy_rule,
    2: policy_rl,
    3: policy_cfr,
    4: policy_llm,
    5: policy_rand
}

env = rlcard.make('limit-holdem', {'game_num_players': 6})
print(env.actions)  # action space
print(env.num_players)  # 玩家数

state, player_id = env.reset()
print(state)
# state 是一个dict，包含下列信息:
#   legal_actions: 合法动作，一个 OrderedDict
#   obs: 编码后的观察，一个np.array，其shape是(72,)
#   action_record: 动作记录
#   raw_legal_actions: 原始的合法动作，即未编码的
#   raw_obs: 原始观察，dict，它包含：
#       hand 手牌、
#       public_cards 公共牌、
#       all_chips 每人下的注
#       my_chips 当前玩家下的注
#       legal_actions 合法行为
#       raise_nums 加的注

while not env.is_over():
    action = brains[player_id](state)  # 当前玩家做出决策
    state, player_id = env.step(action)  # 执行一步，更新下一个状态和下一个玩家

# 结束后获得奖励
rewards = env.get_payoffs()  # 对应每个玩家的收益，需要在结束后才有

```

## 封装的环境

也可以使用文档上推荐的接口

```python
# 实现自己的agent
class DummyAgent:
    def __init__(self, player_id, env):
        pass

    def step(self, state):
        """根据状态返回动作，兼容rlcard"""
        pass


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
    action = agents[player_id].step(state)
    s, a, r, s_, done = env.Step(action)
    env.SaveObservation(player_id, s, a, r, s_, done)

history = env.GetGameHistory()
# save the history

```

