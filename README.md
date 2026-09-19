# MC-PILOT simulation reproducibility package

This ROS 1 package contains the code required to reproduce the simulation experiments reported for MC-PILOT.

The package provides:

- the Gazebo throwing environment and the sphere model used in simulation;
- MC-PILOT model learning, release-uncertainty calibration, and policy optimization;
- the analytical, supervised neural-network, and Soft Actor-Critic (SAC) baselines;
- the MC-PILOT ablations on Gaussian-process propagation, particle count, effective release-time uncertainty, and rotational data augmentation;
- fixed experiment runners using the seeds and interaction budgets described below.

## 1. Reference software stack

The experiments were tested on:

- Ubuntu 20.04;
- ROS 1 Noetic;
- Gazebo 11;
- Python 3.8;
- CUDA 12.1 / PyTorch 2.2.2+cu121 for GPU execution.

The supplied `environment.yaml` is distilled from the reference Python virtual environment. ROS/Gazebo packages are intentionally installed through ROS/apt rather than duplicated inside the Conda environment.

## 2. ROS workspace and dependencies

Create a clean catkin workspace and place this package in `src`:

```bash
mkdir -p ~/mcpilot_ws/src
cd ~/mcpilot_ws/src
# Copy or clone the mcpilot package here so that this directory exists:
# ~/mcpilot_ws/src/mcpilot
```

Install ROS Noetic and initialize `rosdep` using the standard ROS Noetic instructions. The package also requires the Franka Gazebo simulation packages, MoveIt, `panda_moveit_config`, and `gazebo_ros_link_attacher`.

For the two source dependencies used by the reference setup:

```bash
cd ~/mcpilot_ws/src

git clone https://github.com/pal-robotics/gazebo_ros_link_attacher.git
cd gazebo_ros_link_attacher
git checkout 1d170c44c2f12dad4e8c23e299d46f3f655e7def
git apply ../mcpilot/dependencies/gazebo_ros_link_attacher_scheduled_detach.patch
cd ..

git clone https://github.com/ros-planning/panda_moveit_config.git
cd panda_moveit_config
git checkout a86da56ab1c756a851d8ee2a06dd04266d1653d6
cd ../..
```

The patch adds the scheduled-detach service used to reproduce release-time uncertainty. It is expected to apply cleanly at the pinned commit above.

Install the remaining ROS dependencies and build the workspace with the system ROS Python installation:

```bash
source /opt/ros/noetic/setup.bash
cd ~/mcpilot_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

`catkin build` can be used instead of `catkin_make` if `catkin_tools` is installed.

## 3. Python environment

Create the Python environment after building the ROS workspace:

```bash
cd ~/mcpilot_ws
conda env create -f src/mcpilot/environment.yaml
conda activate mcpilot-review
source /opt/ros/noetic/setup.bash
source ~/mcpilot_ws/devel/setup.bash
```

The reference environment includes NumPy 1.24.4, SciPy 1.10.1, pandas 2.0.3, scikit-learn 1.3.2, `bayesian-optimization` 1.4.3, PyTorch 2.2.2+cu121, Gymnasium 0.29.1, and Stable-Baselines3 2.4.1.

GPU execution is the reference configuration. The MC-PILOT and neural-network runners accept the `DEVICE` environment variable in installations where CPU execution is preferred, although CPU execution is substantially slower.

## 4. Simulation model and release-time convention

The physical simulated opening delay is

```text
U(120, 130) ms.
```

For complete MC-PILOT, exploration begins with a deterministic 100 ms command anticipation and therefore exposes the calibration routine to the equivalent residual uncertainty

```text
U(20, 30) ms.
```

After calibration, if the estimated residual distribution is `U(a_hat, a_hat + b_hat)`, complete MC-PILOT adds `a_hat` to the physical command anticipation. Policy optimization then propagates only the remaining uncertainty `U(0, b_hat)`.

The fixed-anticipation and no-anticipation MC-PILOT configurations do not apply this estimated additional anticipation.


## 5. Starting the simulation

Every terminal below must use the same ROS workspace and Python environment:

```bash
conda activate mcpilot-review
source /opt/ros/noetic/setup.bash
source ~/mcpilot_ws/devel/setup.bash
```

### 5.1 MC-PILOT, analytical baseline, and neural-network baseline

Terminal 1 — Gazebo/MoveIt:

```bash
roslaunch mcpilot test_friction.launch \
  gazebo_gui:=false \
  delay_distribution:=uniform \
  delay_min_steps:=120 \
  delay_max_steps:=130 \
  synthetic_ground_truth:=true \
  synthetic_delay_distribution:=uniform \
  synthetic_delay_min_ms:=20.0 \
  synthetic_delay_max_ms:=30.0 \
  synthetic_delay_seed:=0
```

Terminal 2 — projectile-state tracker:

```bash
rosrun mcpilot read_bullet_state.py
```

Terminal 3 — throwing service:

```bash
rosrun mcpilot tossing_experiment_script.py
```

For MC-PILOT only, Terminal 4 — policy-learning proxy:

```bash
cd $(rospack find mcpilot)/experiments/gazebo_ros/robot_tossing
python3 -u MC_PILCO_ros_tossing_proxy.py
```

Wait until Gazebo, MoveIt, and the throwing services are ready before starting an experiment runner.

### 5.2 SAC baselines

Use a fresh ROS/Gazebo session. Terminal 1:

```bash
roslaunch mcpilot sac_sim.launch \
  gazebo_gui:=false \
  delay_min_steps:=120 \
  delay_max_steps:=130 \
  synthetic_ground_truth:=true \
  synthetic_delay_min_ms:=120.0 \
  synthetic_delay_max_ms:=130.0
```

The SAC launcher starts the projectile-state tracker. Start only the throwing service in Terminal 2:

```bash
rosrun mcpilot tossing_experiment_script.py
```

The synthetic SAC path uses the full `U(120,130) ms` release-time uncertainty. The fixed-anticipation agent uses 125 ms, whereas the learned-anticipation agent chooses anticipation in `[100,150] ms` as part of its action.

## 6. Experiment runners

All runners are in:

```bash
cd $(rospack find mcpilot)/experiments/gazebo_ros/robot_tossing
```

By default results are written under `$(rospack find mcpilot)/results`. A different output root can be selected before running a script:

```bash
export RESULTS_ROOT=/path/to/results
```

The default independent experiment seeds are `100 101 102 103 104`. They can be overridden, for example for a smoke test, with `SEEDS="100"`.

### 6.1 Analytical baseline

With the simulation and throwing service from Section 5.1 running:

```bash
./run_analytical.sh
```

The analytical baseline uses 100 ms deterministic anticipation and 40 evaluation targets for each of the five seeds.

### 6.2 Complete MC-PILOT and release-handling variants

With all four processes from Section 5.1 running:

```bash
./run_mcpilot.sh
```

This executes, for each seed:

1. complete MC-PILOT with release-distribution calibration;
2. MC-PILOT with fixed 100 ms anticipation and no residual-uncertainty model;
3. MC-PILOT without command anticipation and without a release-uncertainty model.

The main MC-PILOT setting uses 5 exploration throws, 400 policy-optimization particles, 1500 policy updates, and 40 held-out evaluation targets per seed.

Complete MC-PILOT first calibrates the residual release-time distribution from the five exploration trajectories. The estimated lower bound is then added to the command anticipation before policy optimization and evaluation.

### 6.3 MC-PILOT ablations

Run the complete MC-PILOT experiment first because the ablations reuse the paired exploration trajectories and fitted delay configuration:

```bash
./run_mcpilot.sh
./run_ablations.sh
```

Each ablation configuration is evaluated on 30 targets for each of the five seeds (150 evaluation throws per configuration).

The script reproduces the following comparisons:

- **GP propagation:** predictive-posterior sampling (the main complete-MC-PILOT setting) versus predictive mean;
- **Monte Carlo particles:** 50, 200, and 400 particles, with 400 supplied by the complete-MC-PILOT reference run;
- **assumed full effective release-time distributions:**
  - `U(60,70) ms`,
  - `U(90,100) ms`,
  - `U(110,120) ms`,
  - `U(120,130) ms`,
  - `U(130,140) ms`,
  - `U(150,160) ms`;
- **data augmentation:** two measured trajectories with no rotated copies versus two measured trajectories plus two rotations of each trajectory, yielding six GP-training trajectories from the same two simulator interactions.

For every effective-delay sensitivity case `U(L,U)`, the physical simulator remains at `U(120,130) ms`, the opening command is anticipated by `L`, and policy optimization assumes the remaining width `U(0,U-L)`.

For the data-augmentation experiment, Gaussian noise with standard deviation `0.01 m` is injected into measured positions. Positions are filtered/resampled and velocities used for GP training are obtained by numerical differentiation of those position measurements. No independent velocity measurement noise is injected. Rotated copies are generated from the same noisy measured trajectory, so augmentation does not create additional simulator interactions.

### 6.4 Supervised inverse-model neural-network baseline

With the simulation and throwing service from Section 5.1 running:

```bash
./run_nn.sh
```

Only the configuration reported in the paper is run: three hidden layers, 200 ReLU units per layer, Adam learning rate `1e-2`, 1000 optimization steps, and mini-batches of 25 samples. The inverse model uses interaction budgets

```text
5, 15, 25, 35, 45, 55, 65, 75 throws
```

and is evaluated on the same 40 held-out targets at every budget. Network optimization is offline and therefore does not add simulator interactions.

### 6.5 Soft Actor-Critic baselines

With the SAC session from Section 5.2 running:

```bash
./run_sac.sh
```

The runner evaluates both paper configurations:

- SAC with fixed 125 ms anticipation;
- SAC with learned anticipation in `[100,150] ms`.

Both use two hidden layers of 128 units, learning rate `3e-4`, replay-buffer size `100000`, batch size `64`, 20 gradient steps per interaction, learning start after 5 throws, `tau=0.005`, `gamma=0`, and automatic entropy tuning initialized at `0.1`.

The interaction budgets are

```text
5, 15, 25, 35, 45, 55, 65, 75 throws
```

with 40 held-out evaluation targets per seed and budget. Evaluation throws are never used for training.

## 7. Output layout

The top-level output structure is:

```text
results/
├── analytical/
├── main/
│   ├── mcpilot_complete/
│   ├── mcpilot_fixed/
│   └── mcpilot_no_anticipation/
├── ablations/
│   ├── particles/
│   ├── gp_sampling/
│   ├── effective_delay/
│   └── data_augmentation/
├── nn_nh3/
└── sac/
```

MC-PILOT runs save the learned policy, GP/policy logs, cached exploration trajectories, evaluation configuration, release-delay calibration information, and test-throw results in the corresponding seed directory. NN and SAC runners save machine-readable CSV/JSON summaries in addition to per-throw records.

The cached exploration trajectories are intentionally reused only where the ablation requires paired training data. Evaluation throws are kept separate from training data.

## 8. Reproducibility details

- Target radial distance: `0.75–2.40 m`.
- Target yaw: `[-pi/6, pi/6]`.
- Target altitude in simulation: `0.10 m`.
- Success radius: `0.05 m`.
- Maximum MC-PILOT release velocity: `3.5 m/s`.
- Main MC-PILOT exploration interactions: 5.
- Main MC-PILOT Monte Carlo particles: 400.
- Main MC-PILOT policy optimization steps: 1500.
- Data-augmentation ablation position-noise standard deviation: `0.01 m`.
- In the data-augmentation ablation, GP-training velocities are differentiated from the filtered noisy position measurements.
- Default release-calibration objective: trajectory energy score.

Random seeds are explicitly set by the runners for training, target generation, release-time sampling, data augmentation/noise, and paired policy evaluation where applicable.

## 9. Licensing and anonymous-review note

MC-PILOT is derived from MC-PILCO. The public MC-PILCO repository currently distributes its main implementation under `AGPL-3.0-or-later`; the `gpr_lib` source files used here are marked MIT upstream. Required third-party copyright notices are retained in those files and documented in `THIRD_PARTY_NOTICES`.

All project-specific names, affiliations, e-mail addresses, private repository references, and absolute development-machine paths have been removed from this package. A name that remains solely inside a required third-party copyright notice is licensing metadata and is not an authorship declaration for the anonymous submission.

The external ROS dependencies keep their own licenses and are not vendored into this package.
