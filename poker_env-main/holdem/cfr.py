import os
import argparse

import rlcard
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)

def train(args):
    env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': args.seed,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': args.seed,
        }
    )

    set_seed(args.seed)

    agent = CFRAgent(
    env,
    os.path.join(
        args.log_dir,
        'cfr_model',
    )
)

    agent.load()  

    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    # 开始训练
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.train()

            if episode % args.evaluate_every == 0 and episode > 0:
                agent.save() # Save model
                reward = tournament(
                    eval_env,
                    args.num_eval_games
                )[0]
                logger.log_performance(
                    episode,
                    reward
                )
                print(f"Episode: {episode}, Reward: {reward}")

        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'cfr')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=100000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/path/storage',
    )

    args = parser.parse_args()

    train(args)
