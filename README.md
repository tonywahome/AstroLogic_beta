# AstroLogic Beta - Reinforcement Learning for Astrobiological Exploration

A reinforcement learning framework where a spacecraft navigates a simulated solar system to detect biosignatures on Mars, Europa, and Enceladus. Compares RL algorithms: **DQN**, **REINFORCE**, and **PPO**.

## Project Structure

```
AstroLogic_beta/
├── astro_env/              # Custom Gymnasium environment (Python)
├── agents/                 # Random agent demo + custom REINFORCE
├── training/               # SB3 training scripts + hyperparameter configs
├── evaluation/             # Comparison plots, evaluation, environment diagram
├── visualization/          # Pygame rendering layer
└── results/                # Training outputs (logs, models, plots, diagrams)
```

## Environment: AstroExploration-v0

### MDP Specification

| Component       | Details                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------- |
| **Observation** | 23-dim continuous vector: position, velocity, target headings/distances, fuel, battery, SNR, mission progress |
| **Action**      | MultiDiscrete [5,3,3,3,4,2]: thrust, pitch, yaw, roll, instrument, communication                              |
| **Start State** | Earth orbit (1 AU), fuel=100%, battery=100%                                                                   |
| **Success**     | Detect and transmit 3 distinct biosignatures                                                                  |
| **Terminal**    | Success, resource depletion, collision, out-of-bounds (>50 AU), 100K steps                                    |

### Rewards

| Event                 | Reward                   |
| --------------------- | ------------------------ |
| Liquid Water          | +500                     |
| Ice                   | +300                     |
| Organic Compounds     | +750                     |
| Signs of Intelligence | +5000                    |
| Step Penalty (dense)  | -(fuel_cost + time_cost) |
| Collision/Failure     | -1000                    |

### Target Bodies & Biosignatures

- **Mars**: ice, organic compounds
- **Europa** (moon of Jupiter): liquid water, organic compounds
- **Enceladus** (moon of Saturn): liquid water, ice, signs of intelligence

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment
python -c "import astro_env; import gymnasium; env = gymnasium.make('AstroExploration-v0'); print(env.reset()[0].shape)"
```

## Usage

### 1. Random Agent Demo (no training)

```bash
# Pygame visualization
python -m agents.random_agent
```

### 2. Train Individual Algorithms

```bash
# Train a single config (index 0-9)
python training/train_ppo.py --run 0
python training/train_dqn.py --run 0
python training/train_reinforce.py --run 0
```

### 3. Run All 30 Experiments

```bash
# Run everything (30 experiments total)
python training/run_all_experiments.py

# Or filter by algorithm
python training/run_all_experiments.py --algorithm ppo
python training/run_all_experiments.py --algorithm dqn --start 0 --end 5
```

### 4. Evaluate Trained Models

```bash
python evaluation/evaluate_agent.py --model results/models/ppo_baseline/final_model --algorithm PPO --episodes 10
python evaluation/evaluate_agent.py --model results/models/reinforce_baseline/policy.pt --algorithm REINFORCE
```

### 5. Generate Comparison Plots

```bash
python evaluation/compare_models.py
```

### 6. Generate Environment Diagram

```bash
python evaluation/generate_diagram.py
```

## Algorithm Comparison

| Algorithm     | Type                 | Key Behavior                                                                   |
| ------------- | -------------------- | ------------------------------------------------------------------------------ |
| **DQN**       | Value-Based          | Uses Discrete(1080) wrapper; good at discrete decisions (instrument selection) |
| **REINFORCE** | Policy Gradient      | Custom PyTorch implementation; high variance but learns full trajectories      |
| **PPO**       | Proximal Policy Opt. | Most stable for continuous 3D navigation; handles MultiDiscrete natively       |

### Hyperparameter Tuning

Each algorithm has 10 configurations varying:

- Learning rate, network architecture, discount factor
- Algorithm-specific: buffer size (DQN), clip range (PPO), baseline method (REINFORCE)

Total: **30 experiment runs** with results logged for comparison.

## Results

After running experiments, outputs are in:

- `results/logs/` - Training logs (Monitor CSV + TensorBoard)
- `results/models/` - Saved model checkpoints
- `results/plots/` - Comparison visualizations
- `results/diagrams/` - Environment interaction diagram
- `results/comparison_summary.csv` - Full results table
