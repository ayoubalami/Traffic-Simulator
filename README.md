# Uncertainty-Aware Neuroevolutionary Traffic Signal Control

Code and experiment artifacts for **"Uncertainty-Aware Neuroevolutionary Traffic Signal Control in a Simulation-Based Digital Twin with Camera-Inspired Observations"**

The project simulates a four-approach intersection and trains a neural traffic-signal controller with neuroevolution. The paper studies how the controller changes when its observation is limited to a 50 m camera region of interest (ROI), with and without distance-dependent measurement uncertainty.

This release is intentionally vehicle-only. Pedestrians are outside the scope of the reported experiment and have been removed from the public implementation.

## Method summary

- Four approaches with two incoming and two outgoing lanes each.
- Twelve independently scored vehicle movements: through, protected left, and right for each approach.
- Neural policy: 59 inputs, one hidden layer with 10 `tanh` units, and 12 sigmoid outputs (732 parameters).
- Diagonal evolution strategy with staged screening, promotion, and anchor evaluation.
- Exact-camera and uncertain-camera observations use the same 50 m ROI and 1 s sampling interval.
- Fitness combines throughput, vehicle and emergency waiting time, stops, braking, clearance loss, wasted green, intersection blocking, turn delay, and worst-approach waiting time.
- Fixed-time and categorical phase controllers are included as secondary baselines.

## Installation

Python 3.10 or newer is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the supplied controllers

Exact 50 m camera ROI:

```powershell
python main_movement_policy.py --model models/vehicle_movement_policy_exact.json --observation-mode exact-camera
```

Distance-dependent uncertain camera ROI:

```powershell
python main_movement_policy.py --model models/vehicle_movement_policy_uncertain.json --observation-mode uncertain-camera
```

Full simulator state:

```powershell
python main_movement_policy.py --model models/vehicle_movement_policy_full_state_v1.json --observation-mode full-state
```

Secondary baselines:

```powershell
python main_fixed_time.py
python main_six_phase.py
```

During an interactive run, keys `1`-`4` select an approach, `5`-`7` select turn/emergency probabilities, and the arrow or `+`/`-` keys change the selected value. Press `0` to set it to zero and `R` to reset all arrival settings.

## Train a movement policy

The three observation systems must be trained separately but with identical optimization and traffic settings. Change only `--observation-mode` and `--output` between runs.

```powershell
python train_movement_policy.py `
  --observation-mode exact-camera `
  --optimizer diagonal-es `
  --population 32 `
  --generations 40 `
  --seeds 1,2,3 `
  --evaluation-duration 180 `
  --validation-seeds 101,102,103 `
  --validation-duration 300 `
  --minimum-green 10 `
  --maximum-green 40 `
  --random-seed 42 `
  --screen-duration 30 `
  --promotion-duration 90 `
  --promotion-scenarios 2 `
  --anchor-interval 5 `
  --anchor-candidates 2 `
  --robustness-penalty 0.25 `
  --output models/vehicle_movement_policy.json
```

Use `--skip-validation` for exploratory runs. Validation does not update the network; it estimates generalization on unseen random seeds. Training and validation can be computationally expensive because every candidate is simulated under multiple traffic profiles and seeds. `--workers` controls parallel candidate evaluation.

## Reproduce the comparison

The paper comparison evaluates four traffic profiles and six seeds, for 24 matched scenarios per controller:

```powershell
python compare_policies.py `
  --fixed-plan models/fixed_time_policy_v1.json `
  --categorical-model models/six_phase_policy_v8.json `
  --movement-model models/vehicle_movement_policy_exact.json `
  --uncertain-movement-model models/vehicle_movement_policy_uncertain.json `
  --full-state-movement-model models/vehicle_movement_policy_full_state_v1.json `
  --observation-ablation `
  --seeds 101,102,103,1001,1002,1003 `
  --evaluation-duration 300 `
  --json-output results/comparison_observation_ablation.json
```

This full command performs many long, fixed-timestep simulations and can take hours. For a smoke test, use one seed and a short duration, for example `--seeds 1001 --evaluation-duration 10`.

The supplied result artifact reports:

| Observation system | Fitness | Throughput | Mean vehicle wait (s) | Emergency completion rate |
|---|---:|---:|---:|---:|
| Full state | -11844.15 | 450.25 | 73.65 | 0.37 |
| Exact 50 m ROI | -7032.86 | 425.00 | 75.45 | 0.55 |
| Uncertain 50 m ROI | -10460.91 | 430.38 | 76.87 | 0.35 |

These are end-to-end results from separately trained policies. They measure the realized systems, not the isolated causal value of additional state information.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The test suite covers camera visibility and uncertainty, arrival processes, movement compatibility, emergency priority, gridlock termination, fixed-time control, policy schemas, optimizer reproducibility, and safety/efficiency metrics.

## Repository layout

- `config.py` - experiment, geometry, traffic, camera, and fitness settings.
- `simulation/` - vehicle dynamics, signal controllers, metrics, evaluation, and neuroevolution.
- `renderer/` - Pygame visualization and live demand controls.
- `models/` - the five controller artifacts used by the paper.
- `results/` - the final matched observation-system comparison.
- `tests/` - vehicle-only regression and reproducibility tests.

Random seeds control stochastic arrivals, routes, vehicle properties, and camera measurements. Comparisons reuse the same traffic profiles and seeds across controllers so that demand scenarios are matched.
