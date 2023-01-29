# EnergyShield (ICCPS '23, Submission 251)

EnergyShield is a framework for provably-safe offloading of neural network controllers at the edge. 
<!-- It is described in the enclosed submission ICCPS_2023.pdf, hereafter referred to as [EnergyShield-ICCPS2023] -->

This README describes how to replicate the results in [EnergyShield-ICCPS2023] using the code supplied in this virtual machine. This includes generating new experimental data from new Carla simulation instances and creating plots for three numerical experiments:

	(Experiment 1) Energy Efficiency and Safety Evaluation of EnergyShield through Carla Simulation runs (Section 5.2)
	(Experiment 2) Performance Gains from EnergyShield given wireless channel variation (Section 5.3)
	(Experiment 3) Generality to other neural network controllers (Section 5.4) 

## Contents

1. System Requirements
2. Setup
3. Experiment 1 - Energy Efficiency and Safety Evaluation
4. Experiment 2 - Performance under wireless channel variation
5. Experiment 3 - Generality to other controllers

## 1. System Requirements

## 2. Setup

### Model checkpoints and Excel files
- To reuse our pretrained models in generating new results, download from [Google Drive](https://drive.google.com/file/d/1ryR7FuCEwSy5KiOBlVQQ5OpPMQEEZdXn/view?usp=sharing) and unzip in the main EnergyShield to have the models data under directory `./EnergyShield/models/` 

Our main experimental model and results is named "casc_agent_1". We created another instance of it "casc_agent_1_new" to allow users to generate their own test runs using the same model checkpoint. The rest of the models in the directory are the ones used to construct Table 1. Each model folder (e.g., `./EnergyShield/models/casc_agent_1/`) should contain a subfolder for the checkpoint and another for its experimental data. 

## 3. Experiment 1 - Energy Efficiency and Safety Evaluation
As per the description in 5.2, we provide the script to initiate new Carla simulation runs that compare EnergyShield to the conventional local execution mode in terms of energy efficiency and safety. 
If users want to reuse our own generated data in the paper, they can skip this step and always set `model` argument to "casc_agent_1". Otherwise, they can run their own as follows:
```Bash
# Experiment 1 Carla simulation (skip if reusing our generated data)
sh ./scripts/run_exp1.sh `model` `num_eps`
```
`model` is the name of the RL agent (default: "casc_agent_1_new"). This default `model` setting initiates new simulation runs using our trained RL model. `num_eps` is an optional argument number of episodes per experimental configuration (default: 3). Note `num_eps` in our paper was 35, but we kept it here at 3 to speed up the simulation runs for the users. For running the script with `num_eps` set to 3 takes around 1 hour of computation time on a workstation with 32 GB RAM and NVIDA GPU 2070 RTX super. 

Once finished, the script generates raw .csv files under `./EnergyShield/models/$model$/experiments/` for each experimental configuration to be used later for the plots. 

To generate Figures 5, 6, and 7, run the following script:
```Bash
# Experiment 1 Figures' Generation
sh ./scripts/exp1_generate_results.sh `model`
```
The generated figures can be found in .pdf as follows:

./results/Fig5_Energy.pdf
./results/Fig5_Safety.pdf
./results/Fig6_traj_noise_False.pdf
./results/Fig6_traj_noise_True.pdf
./results/Fig7_Ergy_v_dist.pdf

The 'distance per 1 m' statistics used to implement Figure 7 were generated for the selected `model` under `./results/distance/`. Note also that if users are regenerating their own results, Figure 6 trajectories may not exhibit the same driving patterns as ours for it depends on the RL agent's driving decisions within the episode.

## 4. Experiment 2 - Performance under wireless channel variation
In 5.3, Carla simulations are rerun to evaluate EnergyShield's resilience under variations of wireless network conditions (represented by the parameters of channel throughput and queuing delays). To run such variations, users can run the following script:
```Bash
# Experiment 2 Carla simulations (skip if reusing our generated data)
sh ./scripts/run_exp2.sh `model` `num_eps`
```
where `model` and `num_eps` are the same arguments from experiment 1. This script will instantiate additional 5 experiments of varying wireless conditions. 

Figures 8 and 9 can be regenerated through the following script:
```Bash
# Experiment 2 Figures' Generation
sh ./scripts/exp2_generate_results.sh `model`
```

This will generate the box and whisker plots under varying network connectivity conditions:

./results/Fig8_windows.pdf
./results/Fig9_energy.pdf

## 5. Experiment 3 - Generality to other controllers

In this experiment,we evaluate how EnergyShield generalizes to DRL controllers trained with different policies. The results for our 4 controllers are in Table 1, and users can call the following script to recompute performance statistics for a model as follows:
```Bash
# Experiment 3 display model statistics
sh ./scripts/exp3_generate_results.sh `model`
```
The statistics will be displayed directly for each configuration in the user's terminal. `model` can be set to one of {"casc_agent_1", "casc_agent_2", "casc_agent_3", "casc_agent_4"} to genreate our exact numbers.