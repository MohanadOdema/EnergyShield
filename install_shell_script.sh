# Install conda
# ./Anaconda3-2020.07-Linux-x86_64.sh

# saved environment
conda create --name carla_0.9.11 python=3.7 -y
conda activate carla_0.9.11
pip3 install numpy tensorflow matplotlib pygame onnx onnx_tf tensorflow_probability opencv-python scipy tensorflow_hub
pip3 install gym networkx pandas

# install carla client
# cd carla-0.9.11-py3.7-linux-x86_64
pip3 install -e carla-0.9.11-py3.7-linux-x86_64

pwd

#  install object detector
cd TensorFlow/models
pip3 install -e research

# args.viz_utils set to False to avoid fixing typo in installed package TensorFlow/models/research/object_detection/utils/visualization_utils.py

# e.g. train
# python3 run.py --model_name casc_agent_D -obstacle --offload_policy local --len_obs 4 --len_route medium --obs_start_idx 40 --viz_utils False
# --num_episodes 6000 -penalize_dist_obstacle --reward reward_speed_centering_angle_add --min_speed 35 --max_speed 45 -display_off --port 4000 --carla_map Town04

# e.g. test
# python3 run.py --model_name casc_agent_D -obstacle --offload_policy local --len_obs 4 --len_route medium --obs_start_idx 40 --viz_utils False -test
# --num_episodes 100 -penalize_dist_obstacle --reward reward_speed_centering_angle_add --min_speed 35 --max_speed 45 -display_off --port 4000 --carla_map Town04

# e.g. docker run --name carla3 -d -e SDL_VIDEODRIVER=offscreen -p 4000-4002:4000-4002 -it --runtime=nvidia --gpus 'device=2' carlasim/carla:0.9.11 ./CarlaUE4.sh -carla-port=4000 -opengl -nosound


