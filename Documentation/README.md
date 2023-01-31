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

## 0. Terminology

This repeatability artifact uses [Docker](https://docker.com). We will use the following terminology throughout:

* The **HOST** will refer to the system running Docker (e.g. your laptop).
* The **CONTAINER** will refer to the "virtualized" system created by Docker (this is where the code from the artifact is run).

Commands meant to be executed inside the host or container will be prefixed with one of the respective comments:

```Bash
# HOST COMMANDS
# CONTAINER COMMANDS
```

## 1. System Requirements

**HARDWARE (HOST):**

1. An x86-64 CPU
2. 32GB of RAM
3. An NVIDIA GPU with 8GB of VRAM **(Geforce 20xx series or later; GTX 2080Ti and V100 cards were tested)**; headless GPUs will work (e.g. servers and Amazon EC2/Microsoft Azure instances)
4. At least 100GB of free disk space

**SOFTWARE (HOST):**

1. Un-virtualized Linux operating system (tested on Ubuntu 20.04 but any distribution that meets the remaining requirements should work; headless installs will work)
2. Official [Linux NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) **(version >= 515.76 is \*\*_REQUIRED_\*\*)**
3. A recent version of [Docker Engine](https://docs.docker.com/engine/) **(version >= 19.03)**;  also known as `containerd` or Docker Server but **not** [Docker Desktop](https://docker.com)
4. A recent version of `git` on the path
5. The `bash` shell installed in `/bin/bash`
6. A user account that can run Docker containers in priviledged mode (i.e. with the [`--priviledged` switch](https://docs.docker.com/engine/reference/run/#runtime-privilege-and-linux-capabilities))

> **WARNING:** NVIDIA driver version >=515.76 is a **STRICT REQUIREMENT**. This repeatability artifact **WILL NOT WORK** unless the host has official NVIDIA drivers version 515.76 or higher installed.

## 2. Setup

### _(i) Host Paths_

Choose an install location on the host:

```Bash
# HOST COMMANDS
HOST_LOCATION=/path/to/someplace/convenient
cd "$HOST_LOCATION"
```

> **NOTE:** All subsequent **HOST** paths in this readme are assumed to be relative to `$HOST_LOCATION`.

### _(ii) Starting the Docker Container_

To start the EnergyShield Docker container, exectute the following in a Bash shell on the host (from `$HOST_LOCATION`):

```Bash
# HOST COMMANDS
git clone --recursive https://github.com/MohanadOdema/EnergyShield
cd EnergyShield
./dockerbuild.sh # WARNING: downloads ~30GB of data, and may take > 1 hour!
./dockerrun.sh --interactive --start-carla
```
This should place you at a Bash shell inside a container with EnergyShield installed.

> **WARNING:** if you exit the container's Bash shell, then the **container and all experiments will stop**. You may restart the container with the **host** command:
> ```Bash
> # HOST COMMANDS
> ./dockerrun.sh --interactive --start-carla
> ```

### _(iii) Container Paths_

The container's Bash shell will have a prompt that looks like:

```Bash
carla@ece2ade62bc5:~$ 
```
where `ece2ade62bc5` is a unique container id (i.e. yours will be different).

Change to the EnergyShield directory inside the container:
```Bash
# CONTAINER COMMANDS
cd /home/carla/EnergyShield
```
> **NOTE:** All subsequent **CONTAINER** paths in this readme are assumed to be relative to `/home/carla/EnergyShield`, unless otherwise noted.

### _(iv) Results Paths_

The output produced by subsequent experiment scripts will accessible from the **HOST** path:

```Bash
$HOST_LOCATION/EnergyShield/container_results
```

You may access this directory and its contents from the **host** using whichever file manager/terminal is convenient (i.e. you can "double-click" on this folder from the **host** file manager). 

(This path is [bind-mounted](https://docs.docker.com/storage/bind-mounts/) to `/home/carla/results` in the **CONTAINER**: i.e. the contents of one will mirror the contents of the other.)