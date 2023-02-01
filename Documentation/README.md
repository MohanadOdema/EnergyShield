# EnergyShield (ICCPS '23, Submission 251)

EnergyShield is a framework for provably-safe offloading of neural network controllers at the edge. It is described in the paper:

>_EnergyShield: Provably-Safe Offloading of Neural Network Controllers for Energy Efficiency._  
>Mohanad Odema, James Ferlez, Goli Vaisi, Yasser Shoukry and Mohammad Abdullah Al Faruque. ICCPS 2023: 14th ACM/IEEE International Conference on Cyber-Physical Systems.

also enclosed as `ICCPS23_251.pdf`, and hereafter referred to as [EnergyShield-ICCPS23].

This README describes how to replicate the results in [EnergyShield-ICCPS2023] using code packaged in a provided Docker image. This includes generating new experimental data from new Carla simulation instances and creating plots for three numerical experiments:

	(Experiment 1) Energy Efficiency and Safety Evaluation of EnergyShield through Carla Simulation runs (Section 5.2)
	(Experiment 2) Performance Gains from EnergyShield given wireless channel variation (Section 5.3)
	(Experiment 3) Generality to other neural network controllers (Section 5.4) 


## Contents

0. Terminology
1. System Requirements
2. Setup
3. Experiment 1 - Energy Efficiency and Safety Evaluation
4. Experiment 2 - Performance under wireless channel variation
5. Experiment 3 - Performance Statistics
6. Appendix

## 0. Terminology

This repeatability artifact uses [Docker](https://docker.com). We will use the following terminology throughout:

* The **HOST** will refer to the system running Docker (e.g. your laptop).
* The **CONTAINER** will refer to the "virtualized" system created by Docker (this is where the code from the artifact is run).

Commands meant to be executed inside the host or container will be prefixed with one of the respective comments:

```Bash
# <<< HOST COMMANDS >>>
# <<< CONTAINER COMMANDS >>>
```

## 1. System Requirements

**HARDWARE (HOST):**

1. An x86-64 CPU
2. 32GB of RAM
3. An NVIDIA GPU with 8GB of VRAM **(Geforce 20xx series or later; GTX 2080Ti and V100 cards were tested)**; headless GPUs will work (e.g. servers and Amazon EC2/Microsoft Azure instances)
4. At least 100GB of free disk space on the filesystem where Docker stores images (`/var/lib/docker` [by default](https://docs.docker.com/config/daemon/#daemon-data-directory))

**SOFTWARE (HOST):**

1. Un-virtualized Linux operating system (tested on Ubuntu 20.04 but any distribution that meets the remaining requirements should work; headless installs will work)
2. Official [Linux NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) **(version >= 515.76 is \*\*_REQUIRED_\*\*)**
3. A recent version of [Docker Engine](https://docs.docker.com/engine/) **(version >= 19.03)**; also known as Docker Server but **not** [Docker Desktop](https://docker.com)
4. A recent version of `git` on the path
5. The `bash` shell installed in `/bin/bash`
6. A user account that can run Docker containers in priviledged mode (i.e. with the [`--priviledged` switch](https://docs.docker.com/engine/reference/run/#runtime-privilege-and-linux-capabilities))

> **WARNING:** NVIDIA driver version >=515.76 is a **STRICT REQUIREMENT**. This repeatability artifact **WILL NOT WORK** unless the host has official NVIDIA drivers version 515.76 or higher installed.

## 2. Setup

### _(i) Host Paths_

Choose an install location on the host:

```Bash
# <<< HOST COMMANDS >>>
HOST_LOCATION=/path/to/some/place/convenient
cd "$HOST_LOCATION"
```

> **NOTE:** All subsequent **HOST** paths in this readme are assumed to be relative to `$HOST_LOCATION`.

### _(ii) Starting the Docker Container_

To start the EnergyShield Docker container, execute the following in a Bash shell on the host (from `$HOST_LOCATION`):

```Bash
# <<< HOST COMMANDS >>>
git clone --recursive https://github.com/MohanadOdema/EnergyShield
cd EnergyShield
./dockerbuild.sh # WARNING: downloads ~30GB of data, and may take > 1 hour even after download!
./dockerrun.sh --interactive --start-carla
```
This should place you at a Bash shell inside a container with EnergyShield installed. The container's Bash shell will have a prompt that looks like:

```Bash
carla@ece2ade62bc5:~$ 
```
where `ece2ade62bc5` is a unique container id (i.e. yours will be different).

> **WARNING:** if you exit the container's Bash shell, then the **container and all experiments will stop**. You may restart the container with the **host** command:
> ```Bash
> # <<< HOST COMMANDS >>>
> ./dockerrun.sh --interactive --start-carla
> ```
> **NOTE:** There is no need to rerun `dockerbuild.sh` when restarting the container in this fashion.

### _(iii) Testing the Container_

From the **container**'s Bash shell execute:

```Bash
# <<< CONTAINER COMMANDS >>>
ps | grep Carla
```
You should see output listing two processes related to Carla:

```Bash
     45 pts/0    00:00:00 CarlaUE4.sh
     52 pts/0    00:01:00 CarlaUE4-Linux-
```
> **WARNING:** if the above command produces no output, then Carla is not running, and this repeatability artifact **WILL NOT WORK**.
>
> **FIX:** Double check that you have installed the correct NVIDIA drivers on the host (see Section 1). Then restart the container using the following sequence of commands:
> ```Bash
> # <<< CONTAINER COMMANDS >>>
> # Exit from the container if it's still running:
> exit
> ```
> ```Bash
> # <<< HOST COMMANDS >>>
> ./dockerrun.sh --remove # Remove any existsing container
> ./dockerrun.sh --interactive --start-carla
> ```

## 3. Experiment 1 - Energy Efficiency and Safety Evaluation
As per the description in 5.2, we provide the script to initiate new Carla simulation runs that compare the performance of EnergyShield (with both its eager and uniform modes) to the conventional local execution mode with regards to energy efficiency and safety. 

Users can generate their own test results using our pretrained model using the following script:
```Bash
# <<< CONTAINER COMMANDS >>>
cd /home/carla/EnergyShield
# Run Experiment 1 Carla simulation
./scripts/run_exp1.sh NUM_EPS
```
<!-- `model` is the name of the pretrained model directory(default: "casc_agent_1_new"). Users do not need to alter this argument for it indicates new simulation experiments using our pretrained RL model. -->
`NUM_EPS` is an optional argument describing the number of episodes per experimental configuration (default: 3). Note that although `NUM_EPS` in our paper was 35, we kept it here at 3 to speed up the simulation runs for the evaluators. From our experience, running the script with `NUM_EPS` set to 3 takes around 1 hour of computation time on a workstation with 32 GB RAM and NVIDIA GPU 2070 RTX super. 

Once the simulations terminate, raw data generated by carla simulations are copied to `../results/raw_data/`. The raw data provides information and statistics on the agent's performance based on its driving and offloading decisions. For convenience, we provide a description of the file hierarchy and what each data field represents in this readme's appendix. 

Afterwards, figures 5, 6, and 7 can be generated in .pdf format based on the raw data in the below paths. We remark the following:
- Figure 6 trajectories are generated based on 3 random episodes from the user's generated files. The figure may not possess the same driving patterns as ours for it depends on the RL agent's driving decisions (For instance, in an extreme corner cases, vehicle's trajectory can be a point indicating the RL agent remained stationary)
- Figure 7 (Energy vs. distance) is generated based on statistical representation of raw data in `../results/distance/*.csv` - also described in the appendix. 

The figures can be found on the **HOST** at the following paths:
```Bash
$HOST_LOCATION/container_results/Fig5_Energy.pdf
$HOST_LOCATION/container_results/Fig5_Safety.pdf
$HOST_LOCATION/container_results/Fig6_traj_noise_False.pdf
$HOST_LOCATION/container_results/Fig6_traj_noise_True.pdf
$HOST_LOCATION/container_results/Fig7_Ergy_v_dist.pdf
```

## 4. Experiment 2 - Performance under wireless channel variation
In 5.3, Additional Carla simulations are conducted to evaluate EnergyShield's resilience under variations of wireless network conditions, which are represented in this paper by the parameters of channel throughput and queuing delays. To run these additional simulations, users can run the following script:
```Bash
# <<< CONTAINER COMMANDS >>>
cd /home/carla/EnergyShield
# Run Experiment 2 Carla simulations:
./scripts/run_exp2.sh NUM_EPS
```
`NUM_EPS` is the same argument from experiment 1 (default:3). This script will instantiate 5 additional experiments of varying wireless conditions. Once the simulations are terminated, their corresponding raw data files are copied to `../results/raw_data`, which are then used to generate the box and whisker plots of Figures 8 and 9 describing additional offloading windows (%) and the normalized energy consumption (w.r.t local execution) under different wireless conditions: 
<!-- ```Bash
# Experiment 2 Figures' Generation
sh ./scripts/exp2_generate_results.sh
``` -->
```Bash
$HOST_LOCATION/container_results/Fig8_windows.pdf
$HOST_LOCATION/container_results/Fig9_energy.pdf
```

## 5. Experiment 3 - Performance Statistics

This last script generates a `.csv` file describing the performance statistics of the user's model in accordance with the metrics in Table 1. These performance metrics are the average center deviance (CD), Track Completion Rate (TCR), and average energy consumption (E) based on the generated results from the users' experiments as follows: 
```Bash
# <<< CONTAINER COMMANDS >>>
cd /home/carla/EnergyShield
# Run Experiment 3 generate model statistics
./scripts/exp3_generate_results.sh
```
The results will be displayed in the terminal and saved in `.csv` format under the path `$HOST_LOCATION/container_results/stats_for_casc_agent_1_new.csv`. 

<!-- can be set to one of {"casc_agent_1", "casc_agent_2", "casc_agent_3", "casc_agent_4"} to generate our exact numbers. -->

## 6. Appendix

Raw Data

Under `$HOST_LOCATION/container_results/raw_data/`, every directory describes the evaluation results for a set of episodes. The name suffix of each directory describes the experimental configuration setting these evaluations belong to. For instance: 

- Directory `Town04_OPT_ResNet152_Shield2_early`: experiment1 evaluations for EnergyShield early mode at default wireless settings
- Directory `Town04_OPT_ResNet152_Shield2_belay_10Mbps`: experiment2 evaluations for EnergyShield uniform mode when varying wireless channel throughput to 10Mbps

Within each of these directories, four subdirectories exist to describe whether the experiments were conducted with/without gaussian noise and with/without safety filter activated. In each one of these subdirectories, we can find the experimental data in `*.csv` format as follows: 
-	`plots/*.csv` files containing the individual episode's data incurred by the carla simulator when running each episode for the corresponding experimental setting.
-	`valid_data.csv` is the file describing the final statistics for every episode within the corresponding experimental configuration

Information in the former include simulation time, position, steering angle, relative obstacle position, wireless conditions, safety time window, offloading actions, and performance evaluations in terms of latency and energy - all calculated instantaneously for every simulation tick. 

Information in the latter include statistics for each episode about total number of ticks, reward, distance traveled, speed, average latency, average energy per inference, average center deviance, and whether the agent has hit an obstacle/curb.

Distance data

For figure 7 (normalized energy variation vs distance (meters)), raw data is used to construct the tables in `$HOST_LOCATION/container_results/distance/*.csv` which aggregate average normalized energy consumption and #occurrences across 1 m increments of 'distance from obstacle' parameter from all episodes within the subdirectory. Each of these tables is then used to construct a plot from the figure describing how this relation varies under different experimental settings.