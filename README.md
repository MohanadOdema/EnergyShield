# EnergyShield: Provably-Safe Offloading of Neural Network Controllers for Energy Efficiency
This is the repository for the EneryShield framework: 
<!-- ***"Supervised Compression for Resource-Constrained Edge Computing Systems"***. -->

## Description
EnergyShield provides safety interventions and formal state-based quantification of the tolerable edge response times before vehicle safety is compromised.

<!-- ## Citation
[[Paper](https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html)] [[Preprint](https://arxiv.org/abs/2108.11898)]
```bibtex
@inproceedings{
}
``` -->

## Requirements
- Installation sequence and needed packages in `install_shell_script.sh`

## Model checkpoints and Excel files
- Download from [Google Drive](https://drive.google.com/file/d/1JYnrtAYDDPcGgPm5Q6P0DFq5KkhzEx2w/view?usp=sharing). Unzip in the main EnergyShield directory
- The 4 models used for evaluation are arranged in the order provided in Table 1. Each model folder contains a subfolder for the checkpoint and another for the excel files. 

## Running Carla main experiments 
This is for the main experiments involving a vehicle traversing a track with 4 obstacles. There are also files to launch experiments using docker containers that will be detailed later. 
- Launch Carla Server: `./CarlaUE4.sh -opengl`
- Launch experimental test run: 
`
python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT
`
- Policies used in the experiments are `local` and `Shield2`
- `-gaussian` and `-safety_filter` arguments are the N and S experimental parameters.
- `--off_belay` argument is the flag to set EnergyShield to uniform mode. Without it EnergyShield is eager.
- Additional Examples of this run with different parameters are provided in `test_script.sh` and `test_offload.sh`

## Regenerating Results Statistics
This needs to have the Excel files placed in the correct directory path as described above. The summary from the excel files for each experimental configuration
can be generated as follows:
`
python compute_stats.py --model_name casc_agent_1 --deadline 100 --offload_policy Shield2 --off_belay --len_route short --map Town04_OPT -gaussian
`

## Regenerating Artifiacts
This is for rendering the plots/figures used in the results section
- Trajectories (switch between the two subfigures through `-gaussian` flag)
`
python plot_trajectory.py -gaussian
`
- Energy barplot (Numbers were obtained through `compute_stats.py` above)
`
python plot_energy.py 
`
- TCR and reward barplot
`
python plot_safety.py 
`
- Energy vs. distance plot
`
python plot_distance.py --plot_all
`
- Box and Whisker plots
`
python plot_wireless_boxplot.py -mode energy 
python plot_wireless_boxplot.py -mode windows 
`