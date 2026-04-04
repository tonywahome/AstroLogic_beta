# AstroLogic — Video Demo Script
## Machine Learning Techniques II | Summative Assignment
**Target duration:** ~3 minutes | Camera ON | Full screen shared

---

## PRE-RECORDING CHECKLIST
- [ ] Open **two side-by-side terminal windows** in the project root
  - Left terminal: for running the simulation command
  - Right terminal: for showing project structure / results
- [ ] Open `results/diagrams/environment_diagram.png` in an image viewer (minimised, ready to pop up)
- [ ] Open `results/final_summary.csv` in a text editor / spreadsheet (minimised)
- [ ] Camera is on, mic tested, screen resolution set to 1920×1080
- [ ] `cd C:\Users\LENOVO\Desktop\summative_rl\AstroLogic_beta` in both terminals

---

## SEGMENT 1 — INTRODUCTION & PROBLEM STATEMENT [0:00 – 0:30]

**On screen:** Project root directory in terminal, camera in corner

> *"Hi — in this demo I'll walk you through AstroLogic, a custom reinforcement
> learning environment I built for astrobiological exploration."*

> *"The problem is this: how do we train an autonomous spacecraft agent to navigate
> a scaled solar system, identify planets and moons with conditions favourable
> for life, deploy the right scientific instruments, and transmit biosignature
> data back to Earth — all while managing limited fuel and battery?"*

> *"This is a hard RL problem because the rewards are sparse, the action space
> has 360 combinations, and the agent must plan across thousands of timesteps."*

**Action:** While speaking, quickly show the environment diagram —
```
# In terminal, open the diagram
start results\diagrams\environment_diagram.png
```
*Let the diagram display for 5 seconds, pointing to the RL loop arrows,
the observation box, and the reward table at the bottom.*

---

## SEGMENT 2 — AGENT BEHAVIOUR & OBSERVATION SPACE [0:30 – 1:00]

**On screen:** environment_diagram.png still visible OR switch to terminal

> *"The agent is a spacecraft probe. At every timestep it receives a
> 26-dimensional observation — its position and velocity in AU, normalised
> distances and heading vectors to Mars, Europa, and Enceladus, fuel level,
> battery level, a biosignature SNR signal, and its current instrument state."*

> *"Based on this, it chooses a joint action across five dimensions
> simultaneously: thrust level, pitch, yaw, which instrument to deploy,
> and whether to transmit data. That gives 360 unique action combinations
> per timestep."*

> *"I benchmarked three algorithms: DQN — which flattens those 360 actions
> into a single discrete space — PPO and REINFORCE, which work natively
> with the joint action structure."*

**Action:** In the right terminal, show the final summary:
```
type results\final_summary.csv
```
*Point to the three sections — REINFORCE on top, PPO in the middle, DQN negative.*

---

## SEGMENT 3 — REWARD STRUCTURE & OBJECTIVE [1:00 – 1:30]

**On screen:** Back to environment_diagram.png reward table section

> *"The reward function has three layers. Dense shaping rewards guide the
> agent every step — approach delta scaled by 5.0 when it's getting closer
> to a target, and heading alignment scaled by 1.5 to reward pointing toward
> a target body. These prevent the agent from drifting aimlessly."*

> *"Sparse rewards are the real objectives: plus 300 for detecting ice,
> plus 500 for liquid water, plus 750 for organic compounds, and a massive
> plus 5,000 for signs of intelligence — all only triggered when the correct
> instrument is active inside a detection zone."*

> *"Each biosignature detected must then be transmitted — that's another
> plus 200 per transmission. The agent is penalised minus 100 for collision
> or leaving the 50 AU boundary, and there are small per-step fuel and time
> penalties to encourage efficiency."*

> *"The objective: detect and transmit at least 3 biosignatures before
> running out of fuel, battery, or exceeding 10,000 steps."*

---

## SEGMENT 4 — RUNNING THE SIMULATION [1:30 – 2:30]

**On screen:** Left terminal in focus

> *"Now let's see the best-performing agent in action. Across all 30
> experiments, the winner is REINFORCE with a deep [256, 128, 64] policy
> network — it achieved a final mean reward of positive 1,319 over
> 1,000 training episodes, compared to PPO's best of 364 and DQN which
> never left negative territory."*

> *"I'm loading the saved policy weights and running 3 episodes
> with the pygame renderer enabled."*

**Action:** Type and run this command LIVE on camera:
```
python run_with_render.py --model models/pg/reinforce_deep_net/policy.pt --algorithm REINFORCE --episodes 3
```

> *(As the pygame window opens — point to it)*

> *"The white triangle is the spacecraft. The yellow circle is the Sun.
> The coloured circles are Mars, Jupiter, Saturn and their moons Europa
> and Enceladus. The green shaded zones are biosignature detection areas —
> the agent must enter these zones with the right instrument active."*

> *"The white dotted trail shows the trajectory the agent has flown.
> Up in the HUD you can see fuel, battery, SNR signal strength,
> and which instrument is currently deployed."*

> *(Watch the terminal output as episodes complete — read them out)*

> *"Episode 1 complete — reward of [read value], [N] steps,
> found [N] biosignatures, transmitted [N]."*

> *"You can see the agent isn't flying randomly — it's heading toward
> target bodies, adjusting thrust and attitude to align its nose with
> the detection zones, and switching instruments as it approaches."*

---

## SEGMENT 5 — AGENT PERFORMANCE EXPLANATION [2:30 – 3:00]

**On screen:** Terminal with all 3 episode results visible

> *"Let's talk about what we're seeing performance-wise."*

> *"REINFORCE converged because its Monte Carlo return estimator, running
> over 1,000 full episodes, accumulated enough gradient signal to learn the
> compositional structure of the task — approach, detect, transmit — as a
> coherent sequence."*

> *"PPO got to positive rewards faster, within about 110 episodes, but
> plateaued at 364 because 100,000 timesteps wasn't enough to fully
> exploit discovered trajectories."*

> *"DQN never left negative territory. Flattening 360 joint actions
> into a single unordered space destroyed the sub-action semantics —
> the agent couldn't learn that thrust and instrument are independent
> decisions."*

> *"The key takeaway is that algorithm–action-space alignment matters as
> much as hyperparameter tuning. Policy gradient methods that respect the
> MultiDiscrete structure outperform value-based methods that treat it
> as a flat lookup table."*

> *"Thank you for watching — the full report, training notebooks, and all
> 30 experiment logs are in the linked repository."*

---

## QUICK REFERENCE — KEY NUMBERS TO MENTION

| Metric | Value |
|--------|-------|
| Best agent | REINFORCE — `reinforce_deep_net` |
| Best final mean reward | **+1,319.85** |
| PPO best | +364.73 (`ppo_tight_clip`) |
| DQN best | −50.29 (`dqn_large_buffer`) |
| Observation space | Box(26,) float32 |
| Action space | MultiDiscrete[5,3,3,4,2] = 360 actions |
| Total experiments | 30 (10 DQN + 10 PPO + 10 REINFORCE) |
| Max episode steps | 10,000 |
| Biosigs needed for success | 3 transmitted |
| REINFORCE training budget | 1,000 episodes |
| PPO training budget | 100,000 timesteps (~110–138 eps) |
| DQN training budget | 500,000 timesteps (~550 eps) |

---

## RUN COMMAND (copy-paste ready)
```bash
cd C:\Users\LENOVO\Desktop\summative_rl\AstroLogic_beta
python run_with_render.py --model models/pg/reinforce_deep_net/policy.pt --algorithm REINFORCE --episodes 3
```

---

## TIMING GUIDE

| Segment | Duration | Key action |
|---------|----------|-----------|
| 1 — Problem statement | 0:00–0:30 | Show environment diagram |
| 2 — Agent & observation | 0:30–1:00 | Show final_summary.csv |
| 3 — Reward & objective | 1:00–1:30 | Point to reward table in diagram |
| 4 — Live simulation | 1:30–2:30 | Run command, narrate GUI + terminal |
| 5 — Performance analysis | 2:30–3:00 | Read terminal results, compare algos |
