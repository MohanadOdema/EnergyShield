import os
import time
from multiprocessing import Process

os.system('gnome-terminal -- /opt/carla-simulator/CarlaUE4.sh;')
os.system('echo "$!')



time.sleep(5)

os.system("cd /opt/carla-simulator/PythonAPI/util;")
os.system("python config.py -m Town04 --no-rendering;")
