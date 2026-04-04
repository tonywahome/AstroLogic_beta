# AstroLogic - Reinforcement Learning for Astrobiological Exploration

AstroLogic trains a spacecraft agent to navigate a simulated solar system and detect/transmit biosignatures from Mars, Europa, and Enceladus. The project compares **DQN**, **PPO**, and **REINFORCE** under 10 hyperparameter variants each (30 total runs).

## Project Structure

```
AstroLogic_beta/
├── environment/           # Gymnasium env registration + AstroExplorationEnv
├── agents/                # Random agent + reusable REINFORCE policy module
├── training/              # dqn_training.py and pg_training.py (PPO/REINFORCE)
├── evaluation/            # evaluate_agent.py, compare_models.py, diagrams
├── visualization/         # Pygame renderer and UI overlay
├── models/
│   ├── dqn/               # 10 DQN runs (models + logs + tb)
│   └── pg/                # 10 PPO + 10 REINFORCE runs
├── results/
│   ├── final_summary.csv
│   ├── diagrams/
│   └── plots/
├── run_with_render.py     # Foreground renderer for trained checkpoints
└── training_process.ipynb # End-to-end training/comparison notebook
```

## Environment: AstroExploration-v0

### MDP Specification

| Component | Current Definition |
| --- | --- |
| **Observation** | 26-d continuous vector (position, velocity, target distances/headings, fuel, battery, SNR, mission progress, yaw, sun distance, active instrument) |
| **Action** | MultiDiscrete `[5, 3, 3, 4, 2]`: thrust, pitch, yaw, instrument, communication |
| **DQN Wrapper** | `FlattenMultiDiscreteToDiscrete` maps action space to `Discrete(360)` |
| **Start State** | Earth orbit with initialized orbital velocity, fuel=1.0, battery=1.0 |
| **Success** | Transmit 3 unique biosignatures |
| **Terminal** | Success, collision, out-of-bounds, resource depletion, or max episode steps |

### Rewards (Current Shaping)

`RewardCalculator` combines sparse mission rewards with dense shaping:

| Event / Signal | Reward |
| --- | --- |
| Detect liquid water | +500 |
| Detect ice | +300 |
| Detect organic compounds | +750 |
| Detect signs of intelligence | +5000 |
| New transmission (per biosignature) | +200 |
| Orbital insertion bonus | +100 |
| Approach delta shaping | `max(0, approach_delta) * 5.0` |
| Heading alignment shaping | `max(0, heading_alignment) * 1.5` |
| Fuel penalty | `-0.001 * fuel_used` |
| Time penalty | `-0.0001` per step |
| Collision / out-of-bounds | -100 |

### Target Bodies & Biosignatures

- **Mars**: ice, organic_compounds
- **Europa**: liquid_water, organic_compounds
- **Enceladus**: liquid_water, ice, signs_of_intelligence

## Setup

```bash
pip install -r requirements.txt

# Verify environment registration
python -c "import environment, gymnasium as gym; env = gym.make('AstroExploration-v0'); print(env.observation_space.shape, env.action_space.nvec); env.close()"
```

## Usage

### Train Single Configurations

```bash
# DQN config index 0-9
python training/dqn_training.py --run 0 --seed 42

# PPO config index 0-9
python training/pg_training.py --algo ppo --run 0 --seed 42

# REINFORCE config index 0-9
python training/pg_training.py --algo reinforce --run 0 --seed 42
```

### Evaluate / Render Trained Models

```bash
# Headless evaluation
python evaluation/evaluate_agent.py --model models/dqn/dqn_baseline/final_model.zip --algorithm DQN --episodes 10

# Rendered run
python evaluation/evaluate_agent.py --model models/pg/ppo_baseline/final_model.zip --algorithm PPO --episodes 1 --render

# Render with compatibility handling for older REINFORCE checkpoints
python evaluation/evaluate_agent.py --model models/pg/reinforce_deep_net/policy.pt --algorithm REINFORCE --episodes 1 --render
```

### Run Foreground Renderer Helper

```bash
python run_with_render.py --model models/pg/ppo_baseline/final_model.zip --algorithm PPO --episodes 3
```

## Model Comparison (Current Repo State)

All three algorithms have 10 configured runs. The current training/evaluation code emphasizes:

- **DQN**: value-based learning with flattened discrete action space and prioritized replay when `sb3-contrib` is available.
- **PPO**: stable policy optimization over native MultiDiscrete control.
- **REINFORCE**: custom multi-head policy gradient with baseline variants (`none`, `mean`, `running`).

Based on `results/final_summary.csv` in this repo snapshot:

- Top run overall: **reinforce_deep_net** (`final_mean_reward` 1319.85)
- Strong REINFORCE variants: `reinforce_large_net`, `reinforce_very_low_gamma`, `reinforce_running_baseline`
- Best PPO variants: `ppo_tight_clip`, `ppo_wide_clip`, `ppo_baseline`
- DQN runs in this snapshot trend lower on `final_mean_reward` than PPO/REINFORCE

## Results

Current outputs are organized as:

- `models/dqn/<run_name>/`:
	- `final_model.zip`, `best_model.zip`, `logs/`, `tb/`
- `models/pg/<run_name>/`:
	- PPO runs: `final_model.zip`, `best_model.zip`, `logs/`, `tb/`
	- REINFORCE runs: `policy.pt`, `final_model.pt`, `rewards.csv`
- `results/final_summary.csv`: consolidated cross-algorithm summary
- `results/diagrams/` and `results/plots/`: exported visuals

Note: some legacy scripts in `evaluation/compare_models.py` still reference `results/logs` and `results/models`; the current trained artifacts are under `models/dqn` and `models/pg`.
