import carla
import random
import time
import math
import numpy as np
from matplotlib.pyplot import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# If needed, we can average and benchmark these ratings to a mean and standard deviation we can

# Local Execution Overheads using TensorRT
full_local_latency = {"Radiate": {"PX2": {"ResNet18": 18.41, "ResNet50": 41.24, "DenseNet169": 91.51, "ViT": None, "ResNet18_mimic": 12.66, "ResNet50_mimic": 26.19, "DenseNet169_mimic": 47.27},
                                "TX2": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Orin": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Xavier": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Nano": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None}},
                        "480p": {"PX2": {"ResNet18": 29.35, "ResNet50": 67.26, "DenseNet169": 143.52, "ViT": None, "ResNet18_mimic": 20, "ResNet50_mimic": 45.88, "DenseNet169_mimic": 73.49},
                                "TX2": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Orin": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Xavier": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Nano": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None}},
                        "720p": {"PX2": {"ResNet18": 61.84, "ResNet50": 142.19, "DenseNet169": 309.32, "ViT": None, "ResNet18_mimic": 41.65, "ResNet50_mimic": 89.53, "DenseNet169_mimic": 157.37},
                                "TX2": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Orin": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Xavier": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Nano": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None}},
                        "1080p": {"PX2": {"ResNet18": 135.49, "ResNet50": 314.92, "DenseNet169": 684.22, "ViT": None, "ResNet18_mimic": 93.22, "ResNet50_mimic": 197.88, "DenseNet169_mimic": 344.61},
                                "TX2": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Orin": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Xavier": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None},
                                "Nano": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None}}
                        }

# Measurements on the PX2 are obtained for the network only, 1-3 mseconds can be added for the initial preprocessing

# ResNet50_mimic (Head Network Distillation) PX2 readings: {Radiate: 29.85, 480p: 50.41, 720p: 103.12, 1080p: 229.25}
# ResNet50_mimic (Sage/Testudo) PX2 readings: {Radiate: 26.19, 480p: 45.88, 720p: 89.53, 1080p: 197.88}

# DenseNet169_mimic (Head Network Distillation) PX2 readings: {Radiate: 54.39, 480p: 85.02, 720p: 182.06, 1080p: 402.37}
# DenseNet169_mimic (Sage/Testudo) PX2 readings: {Radiate: 47.27, 480p: 73.49, 720p: 157.37, 1080p: 344.61}


# Bottleneck Execution Overheads using TensorRT
bottleneck_local_latency =    {"Radiate": {"PX2": {"ResNet18_mimic": 3.54, "ResNet50_mimic": 3.54, "DenseNet169_mimic": 3.54, "ViT": None},
                                        "TX2": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Orin": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Xavier": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Nano": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None}},
                                "480p": {"PX2": {"ResNet18_mimic": 5.52, "ResNet50_mimic": 5.52, "DenseNet169_mimic": 5.52, "ViT": None},
                                        "TX2": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Orin": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Xavier": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Nano": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None}},
                                "720p": {"PX2": {"ResNet18_mimic": 12.27, "ResNet50_mimic": 12.27, "DenseNet169_mimic": 12.27, "ViT": None},
                                        "TX2": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Orin": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Xavier": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Nano": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None}},
                                "1080p": {"PX2": {"ResNet18_mimic": 27.34, "ResNet50_mimic": 27.34, "DenseNet169_mimic": 27.34, "ViT": None},
                                        "TX2": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Orin": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Xavier": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None},
                                        "Nano": {"ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None, "ViT": None}}
                        }

local_execution_power = {"PX2": 7,      # W
                        "TX2": None,
                        "Orin": None,
                        "Xavier": None,
                        "Nano": 2}      # W

# Should I add communication establishment overhead?

alphaUP = {"WiFi": 283.17, "LTE": 438.39, "3G": 868.98}             # mW/Mbps
alphaDOWN = {"WiFi": 137.01, "LTE": 51.97, "3G": 122.12}            # mW/Mbps
beta = {"WiFi": 132.86, "LTE": 1288.04, "3G": 817.88}               # mW/Mbps


class EnergyMonitor():
    def __init__(self, params):
        self.arch                   = params["arch"]
        self.offload_policy         = params["offload_policy"]
        self.HW                     = params["HW"]
        self.deadline               = params["deadline"]
        self.noise_addition         = params["noise_addition"]
        self.img_resolution         = params["img_resolution"]
        self.comm_tech              = params["comm_tech"]
        self.conn_overhead          = params["conn_overhead"]
        self.bottleneck_ch          = params["bottleneck_ch"]
        self.bottleneck_quant       = params["bottleneck_quant"]

        self.input_size             = self.compute_input_size()
        self.bottleneck_size        = self.compute_bottleneck_size()
        self.full_local_latency     = full_local_latency[self.img_resolution][self.HW][self.arch]
        if self.offload_policy == 'bottleneck':
            self.bottleneck_latency     = bottleneck_local_latency[self.img_resolution][self.HW][self.arch]
        self.local_power            = local_execution_power[self.HW]
        self.ret_size               = None
        self.local_energy           = self.local_power * self.full_local_latency


    def estimate_energy(self, tu, td=None):            # Estimate energy based on probed Upload and Download throughput in Mbps        
        if self.offload_policy == 'local':
            self.total_energy = self.local_energy
        elif self.offload_policy == 'direct':
            local_energy = 0
            upload_latency = self.estimate_comm_latency(self.input_size, tu)
            upload_power = self.compute_upload_data_transfer_power(tu, self.comm_tech)
            if self.ret_size is not None and td is not None:
                download_latency = self.estimate_comm_latency(self.ret_size, td)
                download_power = self.estimate_download_data_transfer_power(td, self.comm_tech)
            else:
                download_latency = 0
                download_power = 0
            self.total_energy = local_energy + (upload_latency*upload_power + download_latency*download_power) / 1000       # mJ
        elif self.offload_policy.startswith('bottleneck'):
            local_energy = self.local_power * self.bottleneck_latency
            upload_latency = self.estimate_comm_latency(self.bottleneck_size, tu)
            upload_power = self.compute_upload_data_transfer_power(tu, self.comm_tech)
            if self.ret_size is not None and td is not None:
                download_latency = self.estimate_comm_latency(self.ret_size, td)
                download_power = self.estimate_download_data_transfer_power(td, self.comm_tech)
            else:
                download_latency = 0
                download_power = 0            
            self.total_energy = local_energy + (upload_latency*upload_power + download_latency*download_power) / 1000       # mJ


    def select_best_energy_action(self, tu, td=None):   # Select the operational mode based on the energy estimates without violating latency
        # Need to account for latency constraints --> can have it based on the local execution latency
        self.estimate_energy(tu[0])
        if self.offload_policy == "local":
            return (0, self.local_energy)                    
        else:
            if self.total_energy < self.local_energy:
                return(1, self.total_energy)
            else: 
                return(0, self.local_energy)

    def compute_input_size(self):                       # Two aspect ratios are applicable -> Classic: 4:3 and Widescreen: 16:9
        if self.img_resolution == '480p':
            return (852*480*3)*8 / (1024**2)            # (w*l*ch)*bits in Mbits
        elif self.img_resolution == '720p':
            return (1280*720*3)*8  / (1024**2) 
        elif self.img_resolution == '1080p':
            return (1920*1080*3)*8  / (1024**2) 
        elif self.img_resolution == 'nuScences':
            return (1600*900*3)*8 / (1024**2)   
        elif self.img_resolution == 'Radiate':
            return (672*376*3)*8 / (1024**2)          
        elif self.img_resolution == 'TeslaFSD':
            return (1280*960*3)*8 / (1024**2)
        elif self.img_resolution == 'Waymo':
            return (1920*1280*3)*8 / (1024**2) 
        else:
            raise ValueError("Resolution not supported!")

    def compute_bottleneck_size(self):                  # Currently assuming all share the same encoder structure we had before
        if self.img_resolution == '480p':
            return ((852/8)*(480/8)*self.bottleneck_ch) * self.bottleneck_quant / (1024**2)            # (w*l*ch)*bits in Mbits
        elif self.img_resolution == '720p':
            return ((1280/8)*(720/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)         
        elif self.img_resolution == '1080p':
            return ((1920/8)*(1080/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)
        elif self.img_resolution == 'nuScences':
            return ((1600/8)*(900/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)        
        elif self.img_resolution == 'Radiate':
            return ((672/8)*(376/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)        
        elif self.img_resolution == 'TeslaFSD':
            return ((1280/8)*(960/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)   
        elif self.img_resolution == 'Waymo':
            return ((1920/8)*(1280/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)            
        else:
            raise ValueError("Resolution not supported!")

    def estimate_comm_latency(self, tx_size, throughput):                   # verify against the bottleneck calculator in google sheets
        return (tx_size / throughput) * 1000    # in ms

    def compute_upload_data_transfer_power(self, throughput, comm_tech):
        return alphaUP[comm_tech] * throughput + beta[comm_tech]        # mW

    def compute_download_data_transfer_power(self, throughput, comm_tech):
        return alphaDOWN[comm_tech] * throughput + beta[comm_tech]      # mW


class UploadThroughputSampler():
    def __init__(self, params):
        self.rayleigh_sigma = params["rayleigh_sigma"]
        self.noise_addition = params["noise_addition"]
        self.gaussian_var = params["gaussian_var"]

    def sample(self, no_of_samples=1, rounding=False):              # Explore the possibility of having correlated communication models
        tu_list = np.random.rayleigh(self.rayleigh_sigma, no_of_samples)    
        if rounding:
            raise NotImplementedError
        if self.noise_addition == 'gaussian':
            tu_list = [x + np.random.normal(0,self.guassian_var,1)[0] for x in tu_list]
        elif self.noise_addition == 'markov':
            raise NotImplementedError
        else: 
            pass      
        return tu_list






        
