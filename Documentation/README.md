# EnergyShield (ICCPS '23, Submission 56)

FastBATLLNN is a fast verifier of box-like (hyper-rectangle) output properties for Two-Level Lattice (TLL) Neural Networks. It is described in the enclosed submission HSCC_56.pdf, hereafter referred to as [EnergyShield-ICCPS23]

This README describes how to replicate the results in [EnergyShield-ICCPS23] using the code supplied in this virtual machine. This includes generating raw verification data and creating plots for three numerical experiments:

	(Experiment 1) Scalability of FastBATLLNN as a function of the input dimension of the TLL to be verified (Section 6.2)
	(Experiment 2) Scalability of FastBATLLNN as a function of TLL NN size, viz. number of local linear functions (Section 6.3)
	(Experiment 3) Comparison of FastBATLLNN with generic NN verifiers (Section 6.4)


## Contents

0. Terminology
1. System Requirements
2. Setup

## 0. Terminology

This repeatability artifact uses [Docker](https://docker.com). We will use the following terminology throughout:

* The **HOST** will refer to the system running [Docker](https://docker.com) (e.g. your laptop).
* The **CONTAINER** will refer to the "virtualized" system created by [Docker](https://docker.com) (this is where the code from the artifact is run).

Commands meant to be executed inside the host or container will be prefixed with one of the respective comments:

```Bash
# HOST COMMANDS
# CONTAINER COMMANDS
```

## 1. System Requirements

**HARDWARE (HOST):**

1. An x86-64 CPU
2. 32GB of RAM
3. An NVIDIA GPU with 8GB of VRAM **(Geforce 20xx series or later; GTX 2080Ti and V100 cards were tested)**
4. At least 100GB of free disk space

**SOFTWARE (HOST):**

1. Un-virtualized Linux operating system (any distribution that meets the remaining requirements will likely work)
2. Official Linux NVIDIA drivers **(version >= 515.76 is _required_)**
3. A recent version of [Docker](https://docker.com) **(version >= 19.03)**
4. A recent version of `git` on the path
5. The `bash` shell installed in `/bin/bash`
6. A user account that can run [Docker](https://docker.com) containers in priviledged mode (i.e. with the [`--priviledged` switch](https://docs.docker.com/engine/reference/run/#runtime-privilege-and-linux-capabilities))


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
git clone --recursive https://github.com/jferlez/EnergyShield
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

The conainer's Bash shell will have a prompt that looks like:

```Bash
carla@abedb3 ~$ 
```
where `abedb3` is a (unique) container id.

Change to the EnergyShield directory inside the container:
```Bash
# CONTAINER COMMANDS
cd /home/carla/EnergyShield
```
> **NOTE:** All subsequent **CONTAINER** paths in this readme are assumed to be relative to `/home/carla/EnergyShield`.