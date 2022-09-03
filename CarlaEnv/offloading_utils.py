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
                        "480p": {"PX2": {"ResNet18": 29.35, "ResNet50": 67.26, "DenseNet169": 143.52, "ViT": None, "ResNet18_mimic": 20, "ResNet50_mimic": 45.88, "DenseNet169_mimic": 73.49, "ResNet50FasterRCNN": 67.26, "ResNet50FasterRCNN_mimic": None},
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
                                "Nano": {"ResNet18": None, "ResNet50": None, "DenseNet169": None, "ViT": None, "ResNet18_mimic": None, "ResNet50_mimic": None, "DenseNet169_mimic": None}},
                        "80p"  : {"PX2": {"ResNet50": 6.16, "ResNet152": 16.211}}
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

alphaUP = {"WiFi": 283.17, "LTE": 438.39, "3G": 868.98}             # mW/Mbps
alphaDOWN = {"WiFi": 137.01, "LTE": 51.97, "3G": 122.12}            # mW/Mbps
beta = {"WiFi": 132.86, "LTE": 1288.04, "3G": 817.88}               # mW/Mbps

#===============================================================================
# Offloading Manager
#===============================================================================

class OffloadingManager():
    def __init__(self, params):
        self.arch                   = params["arch"]
        self.offload_position       = params["offload_position"]
        self.offload_policy         = params["offload_policy"]
        self.HW                     = params["HW"]
        self.deadline               = params["deadline"]
        self.img_resolution         = params["img_resolution"]
        self.comm_tech              = params["comm_tech"]
        self.conn_overhead          = params["conn_overhead"]
        self.bottleneck_ch          = params["bottleneck_ch"]
        self.bottleneck_quant       = params["bottleneck_quant"]
        self.scale                  = params["noise_scale"]
        self.input_size             = self.compute_input_size()
        self.bottleneck_size        = self.compute_bottleneck_size()
        self.full_local_latency     = full_local_latency[self.img_resolution][self.HW][self.arch]
        if 'bottleneck' in self.offload_position:
            self.head_latency       = bottleneck_local_latency[self.img_resolution][self.HW][self.arch]
        elif 'direct' in self.offload_position:
            self.head_latency       = 0
        self.local_power            = local_execution_power[self.HW]
        self.ret_size               = None      # return size from the control outputs
        self.full_local_energy      = self.local_power * self.full_local_latency
        self.probe_off_latency      = None      # estimate
        self.probe_off_energy       = None      
        self.true_off_latency       = None      # actual
        self.true_off_energy        = None

        self.reset()

    def reset(self):
        self.missed_deadline_flag   = False
        self.missed_deadlines       = 0         # Total missed_deadlines or fail-safe - calculated on a one-window granularity        
        self.succ_interrupts        = 0
        self.max_succ_interrupts    = 0 
        self.missed_offloads        = 0         # When local is chosen and good offloading opportunity is missed
        self.misguided_energy       = 0

        # Shield related Parameters        
        self.delta_T                = 1         # Final dealdine for receiving outputs, defined as multiples of time windows (always 1 if not shield)
        self.recover_flag           = False     # If delta_T predicted by the shield was not met during operation
        self.transit_flag           = False     # For when the shield has already initiated transmission for multiple time windows
        self.transit_window         = 0         # Current order of operational time window relative to delta_T
        self.rem_size               = 0         # If tx size carries to the following window
        self.rem_val                = 0         # For when a latency componenet initiated but will finish in a successive window
        self.current_transition     = 0         # What offloading processes finished {0: Tx, 1: 0.5*RTT, 2: L_q, 3: 0.5*RTT} - no need to define the last explicitly as condition cause if it happens, we move to the next window
        self.current_tx_latency     = 0         # For breaking down the violation causes
        self.current_tx_energy      = 0
        self.current_rtt1_latency   = 0
        self.current_que_latency    = 0
        self.current_rtt2_latency   = 0

    def determine_offloading_decision(self, channel_params, delta_T):
        self.delta_T = delta_T
        if self.transit_flag:        # TODO: I'll have to record a colum for self.in_transit_flag and self.transit_window in the excel as I'm recording based on remainders now
            self.transit_window += 1 # continuing from last window
            self.true_off_latency, self.true_off_energy = self.update(channel_params['phi_true'], channel_params['rtt_true'], channel_params['que_true'])
            self.record_violations(channel_params)
            return      # as I do not have to make offloading decisions
        else:
            self.transit_window = 0 
            self.true_off_latency, self.true_off_energy = self.evaluate(channel_params['phi_true'], channel_params['rtt_true'], channel_params['que_true'])     # true offloading estimates for decision making
            # TODO: UPDATE REMAINDER SIZE, OFFLOADING OVERHEADS, AND SET IN_TRANSIT FLAG IF SHIELD operation and transmission not completed
        # This will have to be for the window based policies 
        self.correct_action = self.select_offloading_action('ideal')                                   # ideally should offload or not
        if self.offload_policy == "local":
            self.selected_action = 0
        elif 'offload' in self.offload_policy:              # static offloading policies
            self.probe_off_latency, self.probe_off_energy = self.evaluate(channel_params['phi_est'], channel_params['rtt_est'], channel_params['que_est'])  # estimate based on probed throughput
            self.selected_action = 1 
        else:                                               # adaptive offloading policies
            self.probe_off_latency, self.probe_off_energy = self.evaluate(channel_params['phi_est'], channel_params['rtt_est'], channel_params['que_est'])     
            self.selected_action = self.select_offloading_action('est')
        self.record_violations(channel_params)

    def update(self, phi, rtt, que):        # TODO: check the parameters defined in Shield related parameters to make sure all are accounted for
        rem_off_energy = 0                  # unless otherwise stated
        if self.current_transition == 0:
            self.current_tx_latency, self.current_tx_energy = self.offload_overheads(phi, self.rem_size)
            rem_off_latency = self.current_tx_latency + rtt + que        # rtt and queue changed for new window
            rem_off_energy = self.current_tx_energy
        elif self.current_transition == 1:
            self.current_tx_latency = 0     
            # fn should calculate the remainder of complicit latency component given. I can reshape the deadlines met function in the shield conditions to go over each latency component separately and update the transition state 
            rem_rtt1 = self.rem_val             # remeber to divide by 2 in the argument or return
            rem_off_latency = rem_rtt1 + que + 0.5*rtt
        elif self.current_transition == 2:
            self.current_tx_latency, self.current_rtt1 = 0, 0    
            rem_que = self.rem_val
            rem_off_latency = rem_que + 0.5*rtt
        elif self.current_transition == 3:
            self.current_tx_latency, self.current_rtt1, self.current_que = 0, 0, 0
            rem_rtt2 = self.rem_val # remeber to divide by 2 in the argument or return
            rem_off_latency = rem_rtt2
        return rem_off_latency, rem_off_energy

    def evaluate(self, phi, rtt, que):   
        self.current_tx_latency, self.current_tx_energy = self.offload_overheads(phi, self.input_size)
        off_latency = self.head_latency + self.current_tx_latency + rtt + que
        off_energy = self.head_latency*self.local_power + self.current_tx_energy
        return off_latency, off_energy

    def select_offloading_action(self, estimate):
        if estimate.startswith('est'): 
            latency, energy = self.est_off_latency, self.est_off_energy
        else:
            latency, energy = self.true_off_latency, self.true_off_energy
        if self.offload_policy == 'strictShield':   # strictShield is based on comparing the values for one window (i.e., restrictive - want to offload every time window)
            # base decisions on a per-window basis
            if self.delta_T  == 1:              # specific for delta_T=1 as it is a single window for behavioral remedy
                if (energy < self.full_local_energy) and (latency < (self.deadline - (self.full_local_latency - self.head_latency))): # I need to think about including the fail-safe in this calculation
                    return 1
            elif self.delta_T > 1:              # slightly more relaxed latency constraint cause of the subsequent time windows, but still aiming to get the result every window 
                if (energy < self.full_local_energy) and (latency < self.deadline):   
                    return 1 
        elif self.offload_policy == 'energyShield':
            # energy on a window basis but latency on delta_T windows
            if (energy < self.full_local_energy) and (latency < ((self.delta_T * self.deadline) - (self.full_local_latency - self.head_latency))):
                return 1
        elif self.offload_policy == 'looseShield':
            # base decisions on delta T windows
            if (energy < (self.full_local_energy * self.delta_T)) and (latency < ((self.delta_T * self.deadline) - (self.full_local_latency - self.head_latency))):
                return 1
        elif 'failsafe' not in self.offload_policy:
            if (energy < self.full_local_energy) and (latency < self.deadline):
                return 1 
        else:
            if (energy < self.full_local_energy) and (latency < (self.deadline - (self.full_local_latency - self.head_latency))):     # Leaving room for tail latency
                return 1
        return 0

    def record_violations(self, channel_params):     
        if self.selected_action > self.correct_action:          # Wrong offload decision
            if self.deadline_missed():                          # is violation due to latency?
                self.missed_deadline_flag = True
                self.missed_deadlines += 1
                if 'Shield' in self.offload_policy:
                    assert self.transit_window <= (self.delta_T - 1)
                    violation_breakdown(channel_params) 
                    if self.transit_window < (self.delta_t - 1):
                        self.transit_flag = True
                    elif self.transit_window == (self.delta_t - 1):
                        self.transit_flag = False                      # Next window is a new window
                        self.recover_flag = True 
                    else:
                        raise RuntimeError("Incorrent transit window #!")
                self.succ_interrupts += 1
            else:
                if not self.transit_flag:                     # non-transit window with violation
                    self.misguided_energy += 1                # is violation due to energy?
                self.missed_deadline_flag = False   
                self.transit_flag = False                                            
                self.succ_interrupts = 0    
        else:                                                 # Currently shield is for selected offload actions
            self.missed_deadline_flag = False
            self.succ_interrupts = 0
            self.transit_flag = False
            if self.selected_action < self.correct_action:      # if better energy from offload (this is without conteextual safety)
                self.missed_offloads += 1
        self.exp_total_latency, self.exp_total_energy = self.remedy(channel_params)        # Changes in TX overhead or failsafe invocation
        self.max_succ_interrupts = max(self.max_succ_interrupts, self.succ_interrupts)

    def deadline_missed(self):
        if 'Shield' in self.offload_policy:
            if self.transit_flag is False and (self.head_latency + self.true_off_latency > self.deadline):
                return True
            elif self.transit_flag is True and (self.true_off_latency > self.deadline):
                return True 
        elif 'failsafe' in self.offload_policy:
            if self.full_local_latency + self.true_off_latency > self.deadline:
                return True
        else: 
            if self.head_latency + self.true_off_latency > self.deadline:
                return True 
        return False

    def violation_breakdown(channel_params):
        if self.current_tx_latency > self.deadline:
            self.current_transition = 0
            self.rem_size = self.input_size - ((channel_params['phi_true']*self.deadline)/1000)     # Mbit - (Mbps*ms/1000)
        elif self.current_tx_latency + self.current_rtt1_latency > self.deadline:
            self.current_transition = 1
            self.rem_val = self.current_tx_latency + self.current_rtt1_latency - self.deadline       # The rolling-over latency
        elif self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency > self.deadline:
            self.current_transition = 2
            self.rem_val = self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency - self.deadline 
        elif self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency + self.current_rtt2_latency > self.deadline:
            self.current_transition = 3
            self.rem_val = self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency + self.current_rtt2_latency - self.deadline 
        # remedy: total_energy = self.full_local_latency * self.local_power + (min((self.deadline * self.delta_T - self.full_local_latency), tx_latency)*upload_power) / 1000

    def remedy(self, channel_params):
        upload_power = self.compute_upload_data_transfer_power(channel_params['phi_true'], self.comm_tech)
        if self.transit_flag is True:   # for shield
            total_latency = self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency + self.current_rtt2_latency
            if total_latency > self.deadline:
                total_latency = self.deadline 
            total_energy = self.current_tx_latency*upload_power
            return total_latency, total_energy
        tx_latency = self.compute_comm_latency(channel_params['phi_true']) 
        assert self.current_tx_latency == tx_latency
        upload_latency = tx_latency + channel_params['rtt_true'] + channel_params['que_true']       
        if self.selected_action == 0:
            total_latency = self.full_local_latency
            total_energy = self.full_local_energy            
        elif 'failsafe' in self.offload_policy and (self.head_latency + upload_latency) > (self.deadline - (self.full_local_latency - self.head_latency)):                   # fail-safe invoked
            total_latency = self.deadline               # cap
            total_energy = self.full_local_latency * self.local_power + ((self.deadline - self.full_local_latency)*upload_power) / 1000                     # The min is to reflect whether transmission concluded or not before the window expired
        else:
            total_latency = self.head_latency + upload_latency
            if total_latency > self.deadline:
                upload_latency = self.deadline - self.head_latency       
                total_latency = self.deadline           # cap
            total_energy = self.head_latency*self.local_power + (upload_latency*upload_power) / 1000
        return total_latency, total_energy

    def offload_overheads(self, phi_u, upload_size, phi_d=None):
        # Upload overheads
        upload_latency = self.compute_comm_latency(phi_u, upload_size)
        upload_power = self.compute_upload_data_transfer_power(phi_u, self.comm_tech)
        # Download overheads
        if self.ret_size is not None and phi_d is not None:
            download_latency = self.estimate_comm_latency(phi_d, self.ret_size)
            download_power = self.compute_download_data_transfer_power(phi_d, self.comm_tech)
        else:
            download_latency = 0
            download_power = 0
        # Total overheads
        tx_latency = upload_latency + download_latency
        tx_energy = (upload_latency*upload_power + download_latency*download_power) / 1000       # mJ
        return tx_latency, tx_energy

    def compute_comm_latency(self, phi):
        if self.offload_position == 'direct':
            upload_latency = self.estimate_comm_latency(phi, self.input_size)
        elif self.offload_position == '0.5_direct':
            upload_latency = self.estimate_comm_latency(phi, self.input_size*0.5)
        elif self.offload_position == '0.25_direct':
            upload_latency = self.estimate_comm_latency(phi, self.input_size*0.25)
        elif self.offload_position == '0.11_direct':    # l/3 and w/3
            upload_latency = self.estimate_comm_latency(phi, self.input_size*0.11)
        elif self.offload_position == 'bottleneck':
            upload_latency = self.estimate_comm_latency(phi, self.bottleneck_size)
        else:
            raise ValueError("Unsupporeted offloading position")
        return upload_latency

    def compute_input_size(self):                       # Two aspect ratios are applicable -> Classic: 4:3 and Widescreen: 16:9
        if self.img_resolution == '80p':
            return (160*80*3)*8 /(1024**2)
        elif self.img_resolution == '480p':
            return (852*480*3)*8 / (1024**2)            # (w*l*ch)*bits in Mbits
        elif self.img_resolution == '720p':
            return (1280*720*3)*8  / (1024**2) 
        elif self.img_resolution == '1080p':
            return (1920*1080*3)*8  / (1024**2) 
        elif self.img_resolution == 'Radiate':
            return (672*376*3)*8 / (1024**2) 
        # elif self.img_resolution == 'nuScences':
        #     return (1600*900*3)*8 / (1024**2)   
        # elif self.img_resolution == 'TeslaFSD':
        #     return (1280*960*3)*8 / (1024**2)
        # elif self.img_resolution == 'Waymo':
        #     return (1920*1280*3)*8 / (1024**2)
        # elif self.img_resolution == '360p':
        #     return (640*360*3)*8 / (1024**2)  
        else:
            raise ValueError("Resolution {} not supported!".format(self.img_resolution))

    def compute_bottleneck_size(self):                  # Currently assuming all share the same encoder structure we had before
        if self.img_resolution == '80p':
            return ((160/8)*(80/8)*self.bottleneck_ch) * self.bottleneck_quant /(1024**2)
        elif self.img_resolution == '480p':
            return ((852/8)*(480/8)*self.bottleneck_ch) * self.bottleneck_quant / (1024**2)            # (w*l*ch)*bits in Mbits
        elif self.img_resolution == '720p':
            return ((1280/8)*(720/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)         
        elif self.img_resolution == '1080p':
            return ((1920/8)*(1080/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)
        elif self.img_resolution == 'Radiate':
            return ((672/8)*(376/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)  
        # elif self.img_resolution == 'nuScences':
        #     return ((1600/8)*(900/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)              
        # elif self.img_resolution == 'TeslaFSD':
        #     return ((1280/8)*(960/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)   
        # elif self.img_resolution == 'Waymo':
        #     return ((1920/8)*(1280/8)*self.bottleneck_ch) * self.bottleneck_quant  / (1024**2)  
        # elif self.img_resolution == '360p':
        #     return ((640/8)*(360/8)*self.bottleneck_ch) * self.bottleneck_quant / (1024**2)          
        else:
            raise ValueError("Resolution not supported!")

    def estimate_comm_latency(self, phi, tx_size):                   # verify against the bottleneck calculator in google sheets
        return (tx_size / phi) * 1000    # in ms

    def compute_upload_data_transfer_power(self, phi, comm_tech):
        return alphaUP[comm_tech] * phi + beta[comm_tech]        # mW

    def compute_download_data_transfer_power(self, phi, comm_tech):
        return alphaDOWN[comm_tech] * phi + beta[comm_tech]      # mW

    def certify_deadline(self):
        if self.deadline < self.full_local_latency: 
            self.deadline = self.full_local_latency
            print("Deadline modified to the local execution latency: ", self.full_local_latency)

    def verify_combinations(self):
        return self.full_local_latency <= self.deadline

#===============================================================================
# Original Throughput Sampler Implementation
#===============================================================================

class UploadThroughputSampler():
    # Origianl Implementation
    def __init__(self, params):
        self.rayleigh_sigma = params["rayleigh_sigma"]
        self.noise_scale = params["noise_scale"]

    def sample(self, no_of_samples=1, rounding=False):   
        # Sample throughput estimates -- we assume a rayleigh distribution as if decisions are made based on prior througphut estimates, and then add a gaussian variance for the instantenous variations          
        probe_tu_list = np.random.rayleigh(self.rayleigh_sigma, no_of_samples)
        # gaussian
        delta_tu_list = np.random.normal(0, self.noise_scale, no_of_samples) 
        # make sure no negative throughputs   
        i = 0
        while(i<len(probe_tu_list)):
            if probe_tu_list[i] + delta_tu_list[i] < 0:
                delta_tu_list[i] = -1*probe_tu_list[i] + 1e-8
            i = i+1
        return probe_tu_list, delta_tu_list

#===============================================================================
# True Value Samplers
#===============================================================================

class TrueSamples():
    def __init__(self):
        pass

    def average(self, _list, data_type=None, stop_index=-1):
        # average of prior true values
        return sum(_list[:stop_index]) / len(_list[:stop_index]) 

    def worst(self, _list, data_type='delay', stop_index=-1):
        # recent worst true values
        if data_type == 'delay':
            return max(_list[:stop_index])
        elif data_type == 'datarate':
            return min(_list[:stop_index])
        else:
            raise ValueError("Unknown list type")

class RayleighSampler(TrueSamples):
    # Data rates
    def __init__(self, scale, shift=0):
        self.rayleigh_sigma = scale
        self.rayleigh_shift = shift
        super().__init__()

    def sample(self, no_of_samples=1):
        return np.random.rayleigh(self.rayleigh_sigma, no_of_samples) + self.rayleigh_shift

class ShiftedGammaSampler(TrueSamples):
    #  RTT delays
    def __init__(self, shape, scale, shift=0):
        self.gamma_shape = shape
        self.gamma_scale = scale
        self.gamma_shift = shift
        super().__init__()

    def sample(self, no_of_samples=1):
        assert self.gamma_shift >= 0
        return np.random.gamma(self.gamma.shape, self.gamma_scale, no_of_samples) + self.gamma_shift

class NetworkQueueModel(TrueSamples):
    # Queuing Delays
    def __init__(self, qsize, arate, srate):
        self.load = arate/srate
        self.xk = np.arange(qsize)
        self.pk = [((1 - self.load) * self.load**step) / (1-self.load**(qsize+1)) for step in self.xk]
        self.pk_sum = sum(self.pk)
        self.pk_norm = tuple(p / self.pk_sum for p in self.pk)
        self.distribution = rv_discrete(name='Queuing', values=(self.xk, self.pk_norm))
        super().__init__()

    def sample(self, no_of_samples=1):
        # assuming each task takes 1 ms
        occupancy = self.server_delays.distribution.rvs(size=no_of_samples)
        wait_time = occupancy/srate
        return wait_time