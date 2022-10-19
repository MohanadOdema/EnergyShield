# Install conda
./Anaconda3-2020.07-Linux-x86_64.sh

# saved environment
conda env create -f carla.yml

conda activate carla

pip install tensorflow matplotlib pygame onnx onnx_tf tensorflow_probability opencv-python scipy tensorflow_hub

# install carla client
cd carla-0.9.11-py3.7-linux-x86_64
pip install -e carla-0.9.11-py3.7-linux-x86_64

#  install object detector
cd TensorFlow/models/research
pip install -e research




