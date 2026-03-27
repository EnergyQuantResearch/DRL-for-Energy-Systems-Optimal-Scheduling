
# Performance Comparison of Deep RL Algorithms for Energy Systems Optimal Scheduling

* This code accompanies the paper <i>Performance Comparison of Deep RL Algorithms for Energy Systems Optimal Scheduling</i>, published at IEEE PES ISGT EUROPE 2022.
# Abstract 
* Taking advantage of their data-driven and model-free features, Deep Reinforcement Learning (DRL) algorithms have the potential to deal with the increasing level of uncertainty due to the introduction of renewable-based generation. To deal simultaneously with the energy systems' operational cost and technical constraints (e.g, generation-demand power balance) DRL algorithms must consider a trade-off when designing the reward function. This trade-off introduces extra hyperparameters that impact the DRL algorithms' performance and capability of providing feasible solutions. In this paper, a performance comparison of different DRL algorithms, including DDPG, TD3, SAC, and PPO, are presented. We aim to provide a fair comparison of these DRL algorithms for energy systems optimal scheduling problems. Results show DRL algorithms' capability of providing in real-time good-quality solutions, even in unseen operational scenarios, when compared with a mathematical programming model of the energy system optimal scheduling problem. Nevertheless, in the case of large peak consumption, these algorithms failed to provide feasible solutions, which can impede their practical implementation.
# Organization
* Folder `data/` -- Historical and processed data.
* `random_generator_battery.py` -- The energy system environment.
* `agent.py` and `net.py` -- Shared agent and network implementations.
* `trainer.py` -- Shared training / evaluation pipeline for DDPG, TD3, SAC, and PPO.
* `DDPG.py`, `TD3.py`, `SAC.py`, `PPO.py` -- Thin wrappers around the shared training pipeline.
* `run_experiment.py` -- Unified CLI entry point.

# Dependencies
Install the core dependencies with:

```bash
pip install -r requirements.txt
```

`gurobipy` is optional and only needed when comparing against the optimization baseline in `tools.py`.

Plotting no longer requires LaTeX as a hard dependency. If a local `latex` executable is available, figures will use it automatically; otherwise Matplotlib falls back to its default text rendering.

# Quick Start
Run a single algorithm with the legacy wrapper scripts:

```bash
python DDPG.py
python PPO.py
```

Or use the shared CLI:

```bash
python run_experiment.py ppo --episodes 2000 --seeds 1234 2234 3234
```

Artifacts are stored under `AgentDDPG/`, `AgentTD3/`, `AgentSAC/`, and `AgentPPO/`, with one subdirectory per seed.

# Data Provenance
The CSV files under `data/` are prepared experiment inputs used inside the project/lab to reproduce the scheduling setup in the paper. They should be treated as processed inputs for this repository, not as a raw public dataset release with full provenance metadata.

# PPO Status
Earlier versions of this repository had a separate PPO path that drifted from the shared training code, which led to confusion in the issue tracker. The current repository routes PPO through the shared implementation in `agent.py` and the shared training pipeline in `trainer.py`.

If you are reproducing older results or comparing against older forks, please note that older issue discussions about PPO may refer to the pre-cleanup implementation rather than the current code path.
# Recommended citation
A preprint is available, and you can check this paper for more details  [Link of the paper](https://ieeexplore.ieee.org/document/9960642).
* Paper authors: Hou Shengren, Edgar Mauricio Salazar, Pedro P. Vergara, Peter Palensky
* Accepted for publication at IEEE PES ISGT 2022
* If you use (parts of) this code, please cite the preprint or published paper
## Additional Information
* The repository now uses one shared training pipeline across all algorithms to avoid drift between duplicated entry scripts.
* PPO previously had a separate historical implementation. The current version routes PPO through the shared implementation in `agent.py`, and fixes the rollout / buffer wiring so the environment outputs are unpacked consistently.
