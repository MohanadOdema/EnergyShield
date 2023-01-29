# EnergyShield (ICCPS '23, Submission 251)

EnergyShield is a framework for provably-safe offloading of neural network controllers at the edge. 
<!-- It is described in the enclosed submission ICCPS_2023.pdf, hereafter referred to as [EnergyShield-ICCPS2023] -->

This README describes how to replicate the results in [EnergyShield-ICCPS2023] using the code supplied in this virtual machine. This includes generating raw verification data and creating plots for three numerical experiments:

	(Experiment 1) Energy Efficiency and Safety Evaluation of EnergyShield through Carla Simulation runs (Section 5.2)
	(Experiment 2) Performance Gains from EnergyShield given wireless channel variation (Section 5.3)
	(Experiment 3) Generality to other neural network controllers (Section 5.4) 

<!-- the last one is compute_stats.py -->
In addition, we have provided a facility for the reviewers to generate results for new Carla Simulation runs 

## Contents

1. System Requirements
2. Setup
3. Experiment 1
4. Experiment 2
5. Experiment 3
6. Generating and Verifying New TLLs/Properties

## 1. System Requirements

## 2. Setup

## 3. Experiment 1 - EnergyEfficiency 
<!-- Need to explain S, N, and this will run 12 experiments -->
Using the experimental setup provided in 5.2, We provide scripts for regenerating carla simulation that yields energy efficiency barplots (Fig. 5), Trajectory (Fig. 6), and Energy v. Distance (Fig. 7). Our pretrained RL agent here is titled 'casc_agent_1'. If reviewrs want to regenerate results based on our collected test runs, skip this command. Otherwise, reviewers can generate their own set of evaluation results as follows:
<!-- Put script to either regenerate results or do new ones.  -->
```Bash
# Existing Agent Experiment:
sh ./scripts/run_exp1.sh `model` `num_eps`
```
where `model` is the name of the RL agent (default: "casc_agent_1_new"), whereas `num_eps` is an optional number of episodes per experimental configuration (default here is 2). Note in the paper, `num_eps` was set to 35 which amounted for 1 day computation time on a workstation with 32 GB RAM and NVIDA GPU 2070 RTX super.

This script generate .csv files for each configuration as will be needed for the plots. Though not required, the newly generated directories with all detailed information and .csv files can be found under `/EnergyShield/models/$model$/experiments/`

To generate the first barplot in Figure 5, run the following script:
```Bash
sh ./scripts/exp1_generate_results.sh `model`
```

The source networks and properties are stored as files in pickled format in `FastBATLLNN/experiments/experiment1/`.
Figure 4 as shown in [FastBATLLNN-HSCC22], and the data used to create it, can be found in `results_paper` with the same file names used above.


## 4. Experiment 2 - Network Size Scalability

Using the methodology described in Section 6.1.1, we generated TLL networks with an increasing number of local linear functions, N, but with with a fixed n=15 (and M=N). Then we generated random properties for each network according to the methodology described in Section 6.1.2. The following scripts repeat this experiment (or a subset thereof) by running FastBATLLNN on each network and property that we generated:
```Bash
# Full Experiment (16GB RAM VMs):
./scripts/run_exp2.sh TIMEOUT

# All sizes *except* N=M=512 (8GB RAM VMs):
./scripts/run_exp2_8GB_RAM.sh TIMEOUT
```

where `TIMEOUT` is an _optional_ maximum runtime for each problem in seconds (default value is 300 seconds).

The results of the script replicate the raw verification data and recreate Figure 5. After the script completes, these results will appear in:

	results/experiment2_4.txt
	results/experiment_2.pdf

The source networks and properties are stored as files in pickled format in `FastBATLLNN/experiments/experiment2/`.
Figure 5 as shown in [FastBATLLNN-HSCC22], and the data used to create it, can be found in `results_paper` with the same file names used above.


## 5. Experiment 3 - Comparison with Generic NN Verifiers

In this experiment, we used externally obtained TLL networks of n=2 and N=M=8 through N=M=64. Then we generated random properties for each network according to the methodology described in Section 6.1.2. 

We compared FastBATLLNN with three other tools: PeregriNN, nnenum, Alpha-Beta-Crown. Each of these tools, including FastBATLLNN, can be run on the Experiment 3 test suite (or a subset) using its own command (respectively, low-memory alternative command):
```Bash
# WARNING: Total runtime for other tools can exceed 5 hours each with 300 seconds timeout per instance

# Full experiment (8GB RAM VMs for FastBATLLNN; 64GB RAM VMs for other tools):
./scripts/experiment3/run_fast.sh TIMEOUT
./scripts/experiment3/run_peregrinn.sh TIMEOUT
./scripts/experiment3/run_nnenum.sh TIMEOUT
./scripts/experiment3/run_alpha_beta_crown.sh TIMEOUT

# Sizes N=M=8 through N=M=48 (8GB RAM VMs; 16GB RAM VMs for alpha-beta-crown):
./scripts/experiment3/run_peregrinn_8GB_RAM.sh TIMEOUT
./scripts/experiment3/run_nnenum_8GB_RAM.sh TIMEOUT
./scripts/experiment3/run_alpha_beta_crown_16GB_RAM.sh TIMEOUT
```

where `TIMEOUT` is an _optional_ maximum runtime for each problem in seconds (default value is 300 seconds).

The raw data from running each tool is stored in its own file as follows:

	results/results_fast_4.txt
	results/results_peregrinn_4.txt
	results/results_nnenum_4.txt
	results/results_alpha_beta_crown_4.txt

The following command will recreate the cactus plot shown in Figure 6 (after running any subset of the tools above):
```Bash
python ./scripts/experiment3/cactus.py MAX_TIME
```

where the `MAX_TIME` sets the maximum value on the y-axis on the cactus plot. It should be set to the same timeout used to run the tools (the default value is 300 seconds).

The result will be a file `results/cactus_4.pdf` that replicates Figure 6 using whichever data files are available in `results` directory (from running the tools via their scripts listed above).

The source networks and properties for this experiment are stored as follows: as files in pickled format in `FastBATLLNN/sizeVsTime_n2_input.p` (for FastBATLLNN); as files in ONNX format in `models/` (for nnenum and Alpha-Beta-Crown), and as files in keras format in `/models_keras/` (for PeregriNN).
Figure 6 as shown in [FastBATLLNN-HSCC22], and the data used to create it, can be found in `results_paper` with the same file names used above


## 6. Generating and Verifying New TLLs

Additionally, we provide the following scripts to generate and verify scalar-output TLL NNs not considered in the paper.

The script `generate_random_tll.py` in the `scripts/` folder can be called as follows to generate a random TLL NN by the method described in Section 6.1.2:

```Bash
./scripts/generate_random_tll.py random_tll.p n=2 N=10 M=10
```

where
<dl>
	<dt><tt>random_tll.p</tt></dt>
    <dd>is a (user-chosen) file name in which to store the network</dd>
	<dt><tt>n</tt></dt>
    <dd>is the input dimension (integer <tt>1 <= n <= 30</tt>)</dd>
	<dt><tt>N</tt></dt>
    <dd>is the number of local linear functions (integer <tt>2 <= N <= 64</tt>)</dd>
	<dt><tt>M</tt></dt>
    <dd>is the number of selector matrices (integer <tt>1 <= M <= 64</tt>)</dd>
</dl>

and there should be no spaces around the numerical arguments, i.e. `n=2` instead of `n = 2` or `n= 2`. _NOTE: the restrictions on `n`, `N` and `M` are only to ensure that TensorFlow models of the TLLs fit in memory; this is needed only for calculating samples, it not a FastBATLLNN requirement._

A network stored in the file `random_tll.p` can then be verified using the following command (*quotes must be as shown, and there are no spaces in the property*):

```Bash
./scripts/FastBATLLNN.sh random_tll.p '>=-7.2'
./scripts/FastBATLLNN.sh random_tll.p '<=2.2'	
```

where *quotes must be as shown* and:

<dl>
	<dt><tt>random_tll.p</tt></dt>
    <dd>is a file containing a TLL (generated by <tt>generate_random_tll.py</tt> as above)</dd>
	<dt><tt>'>=-7.2' or '<=2.2'</tt></dt>
    <dd>verifies that the TLL is always >= -7.2 or <= 2.2, respectively on the input set P_X = [-2,2]^n; the property value can be any parseable floating point string; there should be no spaces bewteen it and the inequality</dd>
</dl>

Both `FastBATLLNN.sh` and `generate_random_tll.py` print out the maximum and minimum of 10000 output samples taken from inputs randomly generated in P_X = [-2,2]^n. This is to help identify interesting properties to verify. **NOTE: as n increases, this number of samples is less effective at getting close to the actual minimum and maximum on P_X.**










