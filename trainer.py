from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from agent import AgentDDPG, AgentPPO, AgentSAC, AgentTD3
from random_generator_battery import ESSEnv
from tools import ReplayBuffer, get_episode_return, optimization_base_result, test_one_episode

DEFAULT_SEEDS = (1234, 2234, 3234, 4234, 5234)
EVAL_COLUMNS = [
    'time_step',
    'price',
    'netload',
    'action',
    'real_action',
    'soc',
    'battery',
    'gen1',
    'gen2',
    'gen3',
    'unbalance',
    'operation_cost',
]


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str
    agent_factory: Callable[[], object]
    output_name: str
    num_episode: int = 2000
    gamma: float = 0.995
    learning_rate: float = 2 ** -14
    soft_update_tau: float = 2 ** -8
    net_dim: int = 256
    batch_size: int = 4096
    repeat_times: int = 2 ** 5
    target_step: int = 4096
    max_memo: int = 500_000
    if_per_or_gae: bool = False
    random_seed_list: tuple[int, ...] = DEFAULT_SEEDS
    num_threads: int = 8
    visible_gpu: Optional[str] = None
    prefill_steps: int = 10_000
    collect_interval: int = 10
    train: bool = True
    save_network: bool = True
    test_network: bool = True
    save_test_data: bool = True
    compare_with_pyomo: bool = True
    plot_on: bool = True
    plot_feature_change: str = ''
    output_root: Path = Path('.')

    def metadata(self) -> dict:
        return {
            'algorithm': self.algorithm,
            'agent_factory': self.agent_factory.__name__,
            'output_name': self.output_name,
            'num_episode': self.num_episode,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'soft_update_tau': self.soft_update_tau,
            'net_dim': self.net_dim,
            'batch_size': self.batch_size,
            'repeat_times': self.repeat_times,
            'target_step': self.target_step,
            'max_memo': self.max_memo,
            'if_per_or_gae': self.if_per_or_gae,
            'random_seed_list': list(self.random_seed_list),
            'num_threads': self.num_threads,
            'visible_gpu': self.visible_gpu,
            'prefill_steps': self.prefill_steps,
            'collect_interval': self.collect_interval,
            'train': self.train,
            'save_network': self.save_network,
            'test_network': self.test_network,
            'save_test_data': self.save_test_data,
            'compare_with_pyomo': self.compare_with_pyomo,
            'plot_on': self.plot_on,
            'plot_feature_change': self.plot_feature_change,
            'output_root': str(self.output_root),
        }


DEFAULT_CONFIGS = {
    'ddpg': TrainingConfig(
        algorithm='ddpg',
        agent_factory=AgentDDPG,
        output_name='AgentDDPG',
    ),
    'td3': TrainingConfig(
        algorithm='td3',
        agent_factory=AgentTD3,
        output_name='AgentTD3',
    ),
    'sac': TrainingConfig(
        algorithm='sac',
        agent_factory=AgentSAC,
        output_name='AgentSAC',
    ),
    'ppo': TrainingConfig(
        algorithm='ppo',
        agent_factory=AgentPPO,
        output_name='AgentPPO',
        learning_rate=2e-4,
        repeat_times=2 ** 3,
        max_memo=4096,
        prefill_steps=0,
    ),
}


def build_experiment_config(algorithm: str) -> TrainingConfig:
    key = algorithm.lower()
    if key not in DEFAULT_CONFIGS:
        supported = ', '.join(sorted(DEFAULT_CONFIGS))
        raise ValueError(f'Unsupported algorithm "{algorithm}". Supported values: {supported}.')
    return DEFAULT_CONFIGS[key]


def configure_runtime(config: TrainingConfig, seed: int) -> None:
    if config.visible_gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.visible_gpu)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.num_threads)
    torch.set_default_dtype(torch.float32)


def make_output_dir(config: TrainingConfig, seed: int) -> Path:
    seed_dir = Path(config.output_root) / config.output_name / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def save_pickle(path: Path, payload) -> None:
    with path.open('wb') as file:
        pickle.dump(payload, file)


def save_json(path: Path, payload: dict) -> None:
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)


def create_loss_record() -> dict:
    return {'episode': [], 'steps': [], 'critic_loss': [], 'actor_loss': [], 'entropy_loss': []}


def create_reward_record() -> dict:
    return {'episode': [], 'steps': [], 'mean_episode_reward': [], 'unbalance': []}


def off_policy_update_buffer(buffer: ReplayBuffer, trajectory, gamma: float) -> tuple[int, float]:
    ten_state = torch.as_tensor([item[0] for item in trajectory], dtype=torch.float32, device=buffer.device)
    ary_other = torch.as_tensor([item[1] for item in trajectory], dtype=torch.float32, device=buffer.device)
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma
    buffer.extend_buffer(ten_state, ary_other)
    buffer.update_now_len()
    return int(ten_state.shape[0]), float(ary_other[:, 0].mean().item())


def on_policy_update_buffer(trajectory, gamma: float):
    if not trajectory:
        return None, 0, 0.0

    stacked_trajectory = list(map(list, zip(*trajectory)))
    ten_state = torch.as_tensor(stacked_trajectory[0], dtype=torch.float32)
    ten_reward = torch.as_tensor(stacked_trajectory[1], dtype=torch.float32)
    ten_mask = (1.0 - torch.as_tensor(stacked_trajectory[2], dtype=torch.float32)) * gamma
    ten_action = torch.as_tensor(stacked_trajectory[3], dtype=torch.float32)
    ten_noise = torch.as_tensor(stacked_trajectory[4], dtype=torch.float32)
    buffer = [ten_state, ten_action, ten_noise, ten_reward, ten_mask]
    return buffer, int(ten_reward.shape[0]), float(ten_reward.mean().item())


def unpack_losses(losses) -> tuple[float, float, float]:
    if not isinstance(losses, tuple):
        losses = (losses,)

    critic_loss = float(losses[0]) if len(losses) > 0 else float('nan')
    actor_loss = float(losses[1]) if len(losses) > 1 else float('nan')
    entropy_loss = float(losses[2]) if len(losses) > 2 else float('nan')
    return critic_loss, actor_loss, entropy_loss


def record_metrics(
    episode: int,
    steps: int,
    losses,
    episode_reward: float,
    episode_unbalance: float,
    loss_record: dict,
    reward_record: dict,
) -> None:
    critic_loss, actor_loss, entropy_loss = unpack_losses(losses)

    loss_record['episode'].append(episode)
    loss_record['steps'].append(steps)
    loss_record['critic_loss'].append(critic_loss)
    loss_record['actor_loss'].append(actor_loss)
    loss_record['entropy_loss'].append(entropy_loss)

    reward_record['episode'].append(episode)
    reward_record['steps'].append(steps)
    reward_record['mean_episode_reward'].append(float(episode_reward))
    reward_record['unbalance'].append(float(episode_unbalance))


def evaluate_episode(agent, env) -> tuple[float, float]:
    with torch.no_grad():
        return get_episode_return(env, agent.act, agent.device)


def maybe_plot_results(seed_dir: Path, test_data_path: Path, base_result, feature_change: str) -> None:
    try:
        from plotDRL import PlotArgs, make_dir, plot_evaluation_information, plot_optimization_result
    except Exception as exc:
        print(f'Plotting skipped because plotting dependencies are unavailable: {exc}')
        return

    try:
        plot_args = PlotArgs()
        plot_args.feature_change = feature_change
        plot_dir = Path(make_dir(str(seed_dir), plot_args.feature_change))
        if base_result is not None:
            plot_optimization_result(base_result, str(plot_dir))
        if test_data_path.exists():
            plot_evaluation_information(str(test_data_path), str(plot_dir))
    except Exception as exc:
        print(f'Plotting skipped because figure generation failed: {exc}')


def run_single_seed(config: TrainingConfig, seed: int) -> dict:
    configure_runtime(config, seed)
    seed_dir = make_output_dir(config, seed)

    env = ESSEnv()
    env.seed(seed)

    agent = config.agent_factory()
    agent.init(
        config.net_dim,
        env.state_space.shape[0],
        env.action_space.shape[0],
        config.learning_rate,
        config.if_per_or_gae,
    )
    agent.state = env.reset()

    loss_record = create_loss_record()
    reward_record = create_reward_record()
    total_steps = 0
    actor_path = seed_dir / 'actor.pth'

    if getattr(agent, 'if_off_policy', True) is False:
        if config.train:
            for episode in range(config.num_episode):
                with torch.no_grad():
                    trajectory = agent.explore_env(env, config.target_step)
                buffer, steps, _ = on_policy_update_buffer(trajectory, config.gamma)
                if buffer is None:
                    continue

                total_steps += steps
                losses = agent.update_net(
                    buffer,
                    config.batch_size,
                    config.repeat_times,
                    config.soft_update_tau,
                )
                episode_reward, episode_unbalance = evaluate_episode(agent, env)
                record_metrics(
                    episode,
                    total_steps,
                    losses,
                    episode_reward,
                    episode_unbalance,
                    loss_record,
                    reward_record,
                )
                print(
                    f'[{config.output_name}][seed {seed}] episode {episode} '
                    f'reward={episode_reward:.4f} unbalance={episode_unbalance:.4f}'
                )
    else:
        buffer = ReplayBuffer(
            max_len=config.max_memo,
            state_dim=env.state_space.shape[0],
            action_dim=env.action_space.shape[0],
        )
        if config.train:
            while buffer.now_len < min(config.prefill_steps, config.max_memo):
                with torch.no_grad():
                    trajectory = agent.explore_env(env, config.target_step)
                steps, _ = off_policy_update_buffer(buffer, trajectory, config.gamma)
                total_steps += steps
                print(f'[{config.output_name}][seed {seed}] prefill buffer={buffer.now_len}')

            for episode in range(config.num_episode):
                losses = agent.update_net(
                    buffer,
                    config.batch_size,
                    config.repeat_times,
                    config.soft_update_tau,
                )
                if episode % config.collect_interval == 0:
                    with torch.no_grad():
                        trajectory = agent.explore_env(env, config.target_step)
                    steps, _ = off_policy_update_buffer(buffer, trajectory, config.gamma)
                    total_steps += steps

                episode_reward, episode_unbalance = evaluate_episode(agent, env)
                record_metrics(
                    episode,
                    total_steps,
                    losses,
                    episode_reward,
                    episode_unbalance,
                    loss_record,
                    reward_record,
                )
                print(
                    f'[{config.output_name}][seed {seed}] episode {episode} '
                    f'reward={episode_reward:.4f} unbalance={episode_unbalance:.4f} '
                    f'buffer={buffer.now_len}'
                )

    if loss_record['episode']:
        save_pickle(seed_dir / 'loss_data.pkl', loss_record)
        save_pickle(seed_dir / 'reward_data.pkl', reward_record)

    if config.save_network:
        torch.save(agent.act.state_dict(), actor_path)

    summary = {'seed': seed}
    if reward_record['mean_episode_reward']:
        summary['final_mean_episode_reward'] = reward_record['mean_episode_reward'][-1]
        summary['final_unbalance'] = reward_record['unbalance'][-1]

    needs_evaluation = any(
        [
            config.test_network,
            config.save_test_data,
            config.compare_with_pyomo,
            config.plot_on,
        ]
    )
    if not needs_evaluation:
        save_json(seed_dir / 'run_config.json', config.metadata())
        save_json(seed_dir / 'summary.json', summary)
        return summary

    if actor_path.exists():
        agent.act.load_state_dict(torch.load(actor_path, map_location=agent.device))

    with torch.no_grad():
        record = test_one_episode(env, agent.act, agent.device)

    test_data_path = seed_dir / 'test_data.pkl'
    if config.save_test_data or config.plot_on:
        save_pickle(test_data_path, record)

    eval_data = pd.DataFrame(record['information'], columns=EVAL_COLUMNS)
    summary['test_operation_cost'] = float(eval_data['operation_cost'].sum())

    base_result = None
    if config.compare_with_pyomo:
        month = record['init_info'][0][0]
        day = record['init_info'][0][1]
        initial_soc = record['init_info'][0][3]
        try:
            base_result = optimization_base_result(env, month, day, initial_soc)
            summary['pyomo_operation_cost'] = float(base_result['step_cost'].sum())
            summary['cost_ratio'] = summary['test_operation_cost'] / summary['pyomo_operation_cost']
        except Exception as exc:
            summary['pyomo_error'] = str(exc)
            print(f'[{config.output_name}][seed {seed}] Pyomo/Gurobi comparison skipped: {exc}')

    if config.plot_on:
        maybe_plot_results(seed_dir, test_data_path, base_result, config.plot_feature_change)

    save_json(seed_dir / 'run_config.json', config.metadata())
    save_json(seed_dir / 'summary.json', summary)
    return summary


def run_experiment(config: TrainingConfig) -> pd.DataFrame:
    summaries = []
    output_dir = Path(config.output_root) / config.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in config.random_seed_list:
        summaries.append(run_single_seed(config, seed))

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / 'summary.csv', index=False)
    return summary_df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run DRL experiments for energy system scheduling.')
    parser.add_argument('algorithm', choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument('--episodes', type=int, help='Override the number of training episodes.')
    parser.add_argument('--seeds', type=int, nargs='+', help='Override the default random seed list.')
    parser.add_argument('--output-root', type=Path, help='Directory used to store run artifacts.')
    parser.add_argument('--visible-gpu', help='Value for CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--prefill-steps', type=int, help='Override off-policy replay warmup.')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting.')
    parser.add_argument('--no-pyomo', action='store_true', help='Disable the Pyomo/Gurobi baseline.')
    parser.add_argument('--no-save-network', action='store_true', help='Skip saving actor weights.')
    parser.add_argument('--feature-tag', default=None, help='Optional suffix for plot directories.')
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = build_experiment_config(args.algorithm)

    updates = {}
    if args.episodes is not None:
        updates['num_episode'] = args.episodes
    if args.seeds:
        updates['random_seed_list'] = tuple(args.seeds)
    if args.output_root is not None:
        updates['output_root'] = args.output_root
    if args.visible_gpu is not None:
        updates['visible_gpu'] = args.visible_gpu
    if args.prefill_steps is not None:
        updates['prefill_steps'] = args.prefill_steps
    if args.no_plot:
        updates['plot_on'] = False
    if args.no_pyomo:
        updates['compare_with_pyomo'] = False
    if args.no_save_network:
        updates['save_network'] = False
    if args.feature_tag is not None:
        updates['plot_feature_change'] = args.feature_tag

    return replace(config, **updates)


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    args = parse_args(argv)
    config = config_from_args(args)
    return run_experiment(config)


if __name__ == '__main__':
    main()
