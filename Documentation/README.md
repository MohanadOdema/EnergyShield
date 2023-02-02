# EnergyShield (ICCPS '23, Submission 251)

EnergyShield is an Autonomous Driving System framework designed to save energy on-vehicle by wirelessly offloading NN computations to edge computers, while at the same time maintaining a formal guarantee of safety. In particular, EnergyShield uses a barrier function/controller shield to provide provably safe edge response times for NN offloading requests; on vehicle evaluation provides a safety fallback.

EnergyShield is described in the paper:

>_EnergyShield: Provably-Safe Offloading of Neural Network Controllers for Energy Efficiency._  
>Mohanad Odema, James Ferlez, Goli Vaisi, Yasser Shoukry and Mohammad Abdullah Al Faruque. ICCPS 2023: 14th ACM/IEEE International Conference on Cyber-Physical Systems.

also enclosed as `ICCPS23_251.pdf`, and hereafter referred to as [EnergyShield-ICCPS23].

[EnergyShield-ICCPS2023] contains a number of experiments showing the efficacy of EnergyShield in the [Carla](https://carla.org) simulation environment. This README describes how to replicate those results using code packaged in a [Docker](https://docs.docker.com/engine/) image. In particular, this artifact reruns from scratch the following experiments from [EnergyShield-ICCPS2023]:

	(Experiment 1) Energy Efficiency and Safety Evaluation of EnergyShield through Carla Simulation runs (Section 5.2)
	(Experiment 2) Performance Gains from EnergyShield given wireless channel variation (Section 5.3)
	(Experiment 3) Generality to other neural network controllers (Section 5.4) 

For each of these experiments, this artifact re-generates both new raw data and the analogous plots shown in [EnergyShield-ICCPS2023].

## Contents

0. Terminology
1. System Requirements
2. Setup
3. Experiment 1 - Energy Efficiency and Safety Evaluation
4. Experiment 2 - Performance under wireless channel variation
5. Experiment 3 - Performance Statistics
6. Appendix

## 0. Terminology

This repeatability artifact uses [Docker](https://docs.docker.com/engine/). We will use the following terminology throughout:

* The **HOST** will refer to the system running Docker (e.g. your laptop).
* The **CONTAINER** will refer to the "virtualized" system created by Docker (this is where the code from the artifact is run).

Commands meant to be executed from a shell on the host or the container will be prefixed with one of the following comments, respectively:

```Bash
# <<< HOST COMMANDS >>>
# <<< CONTAINER COMMANDS >>>
```

## 1. System Requirements

**HARDWARE (HOST):**

1. An x86-64 CPU
2. 32GB of RAM
3. An NVIDIA GPU with 8GB of VRAM **(Geforce 20xx series or later; GTX 2080Ti and V100 cards were tested)**; headless GPUs will work (e.g. servers and Amazon EC2/Microsoft Azure instances)
4. At least 100GB of free disk space on the filesystem where Docker stores images (the filesystem containing `/var/lib/docker` [by default](https://docs.docker.com/config/daemon/#daemon-data-directory))

**SOFTWARE (HOST):**

1. Un-virtualized Linux operating system; headless and cloud (e.g. Amazon EC2/Microsoft Azure) installs will work (tested on Ubuntu 20.04, but any distribution that meets the remaining requirements should work)
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
You should see output listing two processes related to [Carla](https://carla.org):

```Bash
     45 pts/0    00:00:00 CarlaUE4.sh
     52 pts/0    00:01:00 CarlaUE4-Linux-
```
> **WARNING:** if the above command produces no output, then Carla is not running, and this repeatability artifact **WILL NOT WORK**.
>
> **FIX:** Double check that your host system meets the requirements in Section 1, especially the NVIDIA driver requirements (incorrect NVIDIA drivers are usually what prevents Carla from starting). If your system meets these requirements, then try restarting the container using the following sequence of commands:
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

In this experiment, we compared EnergyShield with purely on-vehicle NN controller evaluation, both in terms of energy consumption and safety; see [EnergyShield-ICCPS2023], Section 5.2. This comparison was made using a single RL-trained NN controller driving a fixed track; safety entails avoiding randomly spawned stationary obstacles along this track. This artifact reuses the same track and NN controller from our experiment, but the obstacle locations are randomized.

To rerun this experiment (with randomized obstacles), execute the following commands in the **container**:
```Bash
# <<< CONTAINER COMMANDS >>>
cd /home/carla/EnergyShield
# Run Experiment 1 Carla simulation
./scripts/run_exp1.sh NUM_EPS
```
<!-- `model` is the name of the pretrained model directory(default: "casc_agent_1_new"). Users do not need to alter this argument for it indicates new simulation experiments using our pretrained RL model. -->
`NUM_EPS` is an optional argument specifying the number of episodes to run (default: `NUM_EPS=3`); an episode is defined as one run of the fixed track until either the vehicle completes the track or hits an obstacle. For [EnergyShield-ICCPS2023], we ran this experiment for 35 episodes.

> **NOTE:**  Running this script with the default `NUM_EPS=3`  takes around 1 hour on a workstation with 32 GB RAM and NVIDIA GPU 2070 RTX super.

<!-- `NUM_EPS` is an optional argument describing the number of episodes per experimental configuration (default: 3). Note that although `NUM_EPS` in our paper was 35, we kept it here at 3 to speed up the simulation runs for the evaluators. From our experience, running the script with `NUM_EPS` set to 3 takes around 1 hour of computation time on a workstation with 32 GB RAM and NVIDIA GPU 2070 RTX super. 

Once the simulations terminate, raw data generated by carla simulations are copied to `../results/raw_data/`. The raw data provides information and statistics on the agent's performance based on its driving and offloading decisions. For convenience, we provide a description of the file hierarchy and what each data field represents in this readme's appendix.  -->

The primary output of this script are figures that summarize the energy savings and safety properties of EnergyShield (across the number of episodes specified by `NUM_EPS`). These figures can be found on the **HOST** at the following paths:
```Bash
$HOST_LOCATION/container_results/Fig5_Energy.pdf
$HOST_LOCATION/container_results/Fig5_Safety.pdf
$HOST_LOCATION/container_results/Fig6_traj_noise_False.pdf
$HOST_LOCATION/container_results/Fig6_traj_noise_True.pdf
$HOST_LOCATION/container_results/Fig7_Ergy_v_dist.pdf
```
Their filenames match them to the figures that appear in [EnergyShield-ICCPS2023]. For reference, the figures from [EnergyShield-ICCPS2023] are also available with the same filenames in `$HOST_LOCATION/paper_results`.

This script also outputs the raw Carla simulation data of each episode (i.e. the simulation time-stamped positions, velocities, etc. of the vehicle); this data is placed in `$HOST_LOCATION/container_results/raw_data`; the analogous raw data from our simulations can be found in `$HOST_LOCATION/paper_results/raw_data` for comparison. The format structure of this data is described in the subsequent section **6. Appendix**.

Finally, Figure 7 (Energy vs. distance) is derived from some summary statistics of raw data noted above; these summaries are produced as several `.CSV` files output to `$HOST_LOCATION/container_results/distance/*.csv`.

<!-- Afterwards, figures 5, 6, and 7 can be generated in .pdf format based on the raw data in the below paths. We remark the following:
- Figure 6 trajectories are generated based on 3 random episodes from the user's generated files. The figure may not possess the same driving patterns as ours for it depends on the RL agent's driving decisions (For instance, in an extreme corner cases, vehicle's trajectory can be a point indicating the RL agent remained stationary)
- Figure 7 (Energy vs. distance) is generated based on statistical representation of raw data in `../results/distance/*.csv` - also described in the appendix.  -->



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