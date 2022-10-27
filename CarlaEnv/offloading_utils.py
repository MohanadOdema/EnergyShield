import random
import time
import math
import numpy as np
from matplotlib.pyplot import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import rv_discrete

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
        self.time_window            = params["time_window"]
        self.deadline               = params["deadline"]
        self.img_resolution         = params["img_resolution"]
        self.comm_tech              = params["comm_tech"]
        self.bottleneck_ch          = params["bottleneck_ch"]
        self.bottleneck_quant       = params["bottleneck_quant"]
        self.scale                  = params["noise_scale"]
        self.disc_deadline          = self.discretize_deadline(self.deadline)
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
        self.est_off_latency        = None      # estimate
        self.est_off_energy         = None      
        self.true_off_latency       = None      # actual
        self.true_off_energy        = None
        self.off_belay              = params["off_belay"]          
        self.local_belay            = params["local_belay"]
        self.local_early            = params["local_early"]
        self.debug                  = params["debug"]

        assert (self.local_belay and self.local_early) is False
        self.reset()

    def reset(self):
        self.end_tx_flag            = False     # The overall deadline
        self.missed_window_flag     = False
        self.missed_deadlines       = 0
        self.missed_windows         = 0         # Total missed_windows or fail-safe - calculated on a one-window granularity        
        self.succ_interrupts        = 0
        self.max_succ_interrupts    = 0 
        self.missed_offloads        = 0         # When local is chosen and good offloading opportunity is missed
        self.misguided_energy       = 0

        # Time window parameters        
        self.recover_flag           = False     # If delta_T predicted by the shield was not met during operation
        self.transit_flag           = False     # For when the shield has already initiated transmission for multiple time windows
        self.rx_flag                = False     # For when the offloaded result is returned - only needed for shield belaying
        self.belaying               = False
        self.process_current        = False     # To override processing the old input image with one from corresponding time step
        self.transit_window         = 1         # starts from 1 to be comparable to delta_T (i.e., time windows are identified by their ending)
        self.rem_size               = 0         # If tx size carries to the following window
        self.rem_val                = 0         # For when a latency componenet initiated but will finish in a successive window
        self.current_transition     = 0         # What offloading processes finished {0: Tx, 1: 0.5*RTT, 2: L_q, 3: 0.5*RTT} 
        self.current_tx_latency     = 0         # For breaking down the violation causes
        self.current_tx_energy      = 0
        self.current_rtt1_latency   = 0         # These following components have no effect on energy
        self.current_que_latency    = 0
        self.current_rtt2_latency   = 0

    def determine_offloading_decision(self, channel_params, delta_T_ms=None, initialize=False):
        self.initialize = initialize
        self.delta_T = self.discretize_deadline(delta_T=delta_T_ms) if delta_T_ms is not None else self.disc_deadline
        self.rx_flag = False                                   # reset every time window
        self.process_current = False                               # This is needed for holding the detection images
        if self.transit_flag and not self.end_tx_flag:             # This is only for the offloading actions
            self.transit_window += 1 
            self.true_off_latency, self.true_off_energy = self.update(channel_params['phi_true'], channel_params['rtt_true'], channel_params['que_true'])
        elif self.recover_flag:             
            self.transit_window += 1 
            assert self.transit_window == self.delta_T
            self.true_off_latency, self.true_off_energy = self.update(channel_params['phi_true'], channel_params['rtt_true'], channel_params['que_true']) # just for comparison
            self.recover_flag = False
            self.rx_flag = True
            self.sample_new = True
            self.selected_action = 0
        elif self.belaying:             
            self.transit_window += 1
            self.exp_off_latency, self.exp_off_energy = 0, 0 
            if self.transit_window == self.delta_T:
                if self.selected_action == 0 and self.local_belay:                       # Executing locally on the last time window within the deadline horizon
                    self.rx_flag = True
                    self.process_current = True
                    self.exp_off_latency, self.exp_off_energy = self.remedy(channel_params)         
                self.belaying = False                      # Next time window will be a new offloading decision
                self.sample_new = True
            return
        else:
            self.missed_window_flag = False
            self.end_tx_flag = False
            self.transit_window = 1 
            self.rem_size = 0
            self.true_off_latency, self.true_off_energy = self.evaluate(channel_params['phi_true'], channel_params['rtt_true'], channel_params['que_true'])     # true offloading estimates for decision making
            self.correct_action = self.select_offloading_action('ideal')                                   # ideally should offload or not
            if 'local' in self.offload_policy or initialize == True or (self.delta_T < 2 and ('Shield' in self.offload_policy or 'failsafe' in self.offload_policy)):     # Last condition is to ensure that no offloading is done if delta_T is minuscule (not more than one window)
                assert self.transit_flag == self.recover_flag == self.belaying == False     
                self.selected_action = 0
            elif 'offload' in self.offload_policy:              # static offloading policies
                self.est_off_latency, self.est_off_energy = self.evaluate(channel_params['phi_est'], channel_params['rtt_est'], channel_params['que_est'], probe=True)      # estimate based on corresponding estimates of network conditions
                self.selected_action = 1 
            else:                                               # adaptive offloading policies
                self.est_off_latency, self.est_off_energy = self.evaluate(channel_params['phi_est'], channel_params['rtt_est'], channel_params['que_est'], probe=True)     
                self.selected_action = self.select_offloading_action('est')
        self.manage_timings(channel_params)

    def update(self, phi, rtt, que):        
        rem_off_energy = 0                                              # unless current_transition == 0
        self.current_tx_latency, self.current_tx_energy = self.offload_overheads(phi, continuing_upload=True)  
        self.current_rtt1_latency, self.current_rtt2_latency, self.current_que_latency = 0.5*rtt, 0.5*rtt, que
        if self.current_transition == 0:
            rem_off_latency = self.current_tx_latency + rtt + que        # networks parameters changed for new window
            rem_off_energy = self.current_tx_energy
        elif self.current_transition == 1:
            self.current_tx_latency = 0     
            rem_rtt1 = self.rem_val             
            rem_off_latency = rem_rtt1 + que + 0.5*rtt
        elif self.current_transition == 2:
            self.current_tx_latency, self.current_rtt1_latency = 0, 0    
            rem_que = self.rem_val
            rem_off_latency = rem_que + 0.5*rtt
        elif self.current_transition == 3:
            self.current_tx_latency, self.current_rtt1_latency, self.current_que_latency = 0, 0, 0
            rem_rtt2 = self.rem_val # remeber to divide by 2 in the argument or return
            rem_off_latency = rem_rtt2
        if self.debug:
            print(f"transit_window: {self.transit_window}, offloading_energy: {round(rem_off_energy,2)} (local: 0)")
        return rem_off_latency, rem_off_energy

    def evaluate(self, phi, rtt, que, probe=False):   
        tx_latency, tx_energy = self.offload_overheads(phi)
        if not probe:
            self.current_tx_latency, self.current_tx_energy = tx_latency, tx_energy     # store true values for later analysis
            self.current_rtt1_latency, self.current_rtt2_latency, self.current_que_latency = 0.5*rtt, 0.5*rtt, que 
        off_latency = self.head_latency + tx_latency + rtt + que
        off_energy = self.head_latency*self.local_power + tx_energy
        return off_latency, off_energy

    def select_offloading_action(self, estimate):
        if estimate.startswith('est'): 
            latency, energy = self.est_off_latency, self.est_off_energy
        else:
            latency, energy = self.true_off_latency, self.true_off_energy
            if self.debug:
                print(f"transit_window: {self.transit_window}, offloading_energy: {round(energy,2)} (local: 113.5)")
        if 'Shield' in self.offload_policy or 'failsafe' in self.offload_policy:            
            if (energy < self.full_local_energy) and ((latency < (self.delta_T - 1) * self.time_window)):    # recovery window enforced
                return 1 
        else:
            if (energy < self.full_local_energy) and (latency < (self.delta_T * self.time_window)):    
                return 1
        return 0

    def manage_timings(self, channel_params):
        if self.selected_action == 1:                               # Offloading
            if self.window_elapsed():
                self.missed_window_flag = True
                self.missed_windows += 1
                # For all offloading actions
                self.violation_breakdown(channel_params)               
                if self.transit_window == (self.delta_T - 1) and ('Shield' in self.offload_policy or 'failsafe' in self.offload_policy):  # policy with recovery window at last acceptable window        
                    self.transit_flag = False
                    self.recover_flag = True                         
                elif self.transit_window < self.delta_T:
                    self.transit_flag = True
                elif self.transit_window == self.delta_T:             # full deadline passed and still no response
                    self.transit_flag = True
                    self.end_tx_flag = True
                    self.sample_new = True
                else:
                    raise RuntimeError("Incorrect transit window #")
                self.succ_interrupts += 1
            else:
                self.rx_flag = True                                                                
                if self.off_belay and (self.transit_window < self.delta_T):     # stay idle after rx till delta_T expires
                    self.belaying = True
                else:
                    self.sample_new = True
                if self.transit_window == 1 and (self.full_local_energy < self.true_off_energy):        # Wrong energy decision / condition on transit_window so as not to recount wrong decisions
                    self.misguided_energy += 1
                self.missed_window_flag = False
                self.transit_flag = False
                self.succ_interrupts = 0
        else:
            self.transit_flag = False
            self.succ_interrupts = 0
            if self.transit_window == 1 and self.correct_action == 1 and self.initialize == False:     # if better energy from offload (this is without contextual safety)
                self.missed_offloads += 1 
            # Execution after recovery
            elif self.rx_flag:          # This is a recovery window - otherwise manage_timings is only accessed in the first transit window for local_execution
                self.process_current = True                
                assert self.transit_window == self.delta_T              
            # Local Execution Behavior - either in explicit local policies or adaptive
            if self.local_early and not self.rx_flag and not self.initialize:    # not rx_flag to ensure executing when no recovery cond.               
                self.process_current = True
                self.rx_flag = True
                self.belaying = True
                assert self.transit_window == 1
            elif self.local_belay and not self.rx_flag and not self.initialize:
                self.rx_flag = False
                self.belaying = True 
                self.exp_off_latency, self.exp_off_energy = 0, 0   
                assert self.transit_window == 1
                return                              # no execution at initial time window
            else:
                self.rx_flag = True                 # This executes every window
            self.sample_new = True
        self.exp_off_latency, self.exp_off_energy = self.remedy(channel_params)        # Obtain the actual transmission overheads in this window
        self.max_succ_interrupts = max(self.max_succ_interrupts, self.succ_interrupts)
        # count missed dealdines
        if (self.transit_window == self.delta_T) and (not self.rx_flag) and (not self.belaying):
            self.missed_deadlines += 1

    def window_elapsed(self):
        if (self.true_off_latency > self.time_window):      # self.true_off_latency takes into consideration both self.head_latency if needed (evaluate() and update() distinction)
            return True 
        return False

    def violation_breakdown(self, channel_params):
        if self.current_tx_latency > self.time_window:
            current_tx_size = self.input_size if self.transit_window == 1 else self.rem_size
            self.current_transition = 0
            init_latency = self.head_latency if self.transit_window == 1 else 0
            self.rem_size = current_tx_size - ((channel_params['phi_true']*(self.time_window-init_latency))/1000)     # Mbit: (Mbps*ms/1000)
        elif self.current_tx_latency + self.current_rtt1_latency > self.time_window:
            self.current_transition = 1
            self.rem_size = 0
            self.rem_val = self.current_tx_latency + self.current_rtt1_latency - self.time_window       # The rolling-over latency
        elif self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency > self.time_window:
            self.current_transition = 2
            self.rem_size = 0
            self.rem_val = self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency - self.time_window 
        elif self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency + self.current_rtt2_latency > self.time_window:
            self.current_transition = 3
            self.rem_size = 0
            self.rem_val = self.current_tx_latency + self.current_rtt1_latency + self.current_que_latency + self.current_rtt2_latency - self.time_window 
        else:
            raise RuntimeError("There was no deadline violation as far as the breakdown is concerned!")

    def remedy(self, channel_params):          
        upload_power = self.compute_upload_data_transfer_power(channel_params['phi_true'], self.comm_tech)
        init_latency = self.head_latency if self.transit_window == 1 else 0           
        if self.transit_flag is True:   
            assert self.time_window < self.true_off_latency       # sanity check as tranist flag is only set when violation occurs, total_latency should always be time_window
            total_latency = min(self.true_off_latency, self.time_window)        
            total_energy = min(self.current_tx_latency, self.time_window)*upload_power / 1000      # To account for cases when transmission exceeds one window
        elif self.selected_action == 0:        
            total_latency = self.full_local_latency
            total_energy = self.full_local_energy     
        elif self.missed_window_flag:                                                           # per-window condition
            total_latency = self.time_window
            total_energy = init_latency*self.local_power + min(self.current_tx_latency, self.time_window)*upload_power / 1000 
        else:
            total_latency = self.true_off_latency
            total_energy = self.true_off_energy
        return total_latency, total_energy

    def offload_overheads(self, phi_u, continuing_upload=False, phi_d=None):
        # Upload overheads
        upload_latency = self.compute_comm_latency(phi_u, continuing_upload)
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

    def compute_comm_latency(self, phi, continuing_upload=False):
        if continuing_upload:          # remainder transmission overhead from a prior time window
            upload_latency = self.estimate_comm_latency(phi, self.rem_size)
        elif self.offload_position == 'direct':
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
            raise RuntimeError("Not able to compute transmission latency!")
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

    def discretize_deadline(self, delta_T):
        if delta_T % self.time_window != 0:     # Deadline multiple of time windows of 20 ms
            print(f"Deadline modified: {delta_T} --> {(delta_T//self.time_window)*self.time_window}")
        delta_T = max(self.time_window, (delta_T//self.time_window)*self.time_window)
        # if delta_T < self.time_window:
        #     raise ValueError("Not enough time windows to meet the deadline!")
        return delta_T // self.time_window

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
    def __init__(self, estimation_fn):
        self.estimation_fn = estimation_fn

    def avg(self, _list, data_type=None):
        # average of prior true values
        return sum(_list) / len(_list) 

    def worst(self, _list, data_type='delay'):
        # recent worst true values
        if data_type == 'delay':
            return max(_list)
        elif data_type == 'datarate':
            return min(_list)
        else:
            raise ValueError("Unknown list type")

    def estimate(self, fn='avg', _list=None, data_type='delay'):
        _list = list(_list)
        if self.estimation_fn == 'avg':
            return self.avg(_list, data_type)
        elif self.estimation_fn == 'worst':
            return self.worst(_list, data_type)

class RayleighSampler(TrueSamples):
    # Data rates
    def __init__(self, estimation_fn, scale, shift=0):
        self.rayleigh_sigma = scale
        self.rayleigh_shift = shift
        super().__init__(estimation_fn)

    def sample(self, no_of_samples=1):
        return np.random.rayleigh(self.rayleigh_sigma, no_of_samples) + self.rayleigh_shift

class ShiftedGammaSampler(TrueSamples):
    #  RTT delays
    def __init__(self, estimation_fn, shape, scale, shift=0):
        self.gamma_shape = shape
        self.gamma_scale = scale
        self.gamma_shift = shift
        super().__init__(estimation_fn)

    def sample(self, no_of_samples=1):
        assert self.gamma_shift >= 0
        return np.random.gamma(self.gamma_shape, self.gamma_scale, no_of_samples) + self.gamma_shift

class NetworkQueueModel(TrueSamples):
    # Queuing Delays
    def __init__(self, estimation_fn, qsize, arate, srate, queue_state):
        self.srate = srate
        self.load = arate/self.srate
        self.xk = np.arange(qsize)
        self.pk = [((1 - self.load) * self.load**step) / (1-self.load**(qsize+1)) for step in self.xk]
        self.pk_sum = sum(self.pk)
        self.pk_norm = tuple(p / self.pk_sum for p in self.pk)
        self.distribution = rv_discrete(name='Queuing', values=(self.xk, self.pk_norm))
        self.queue_state = queue_state
        super().__init__(estimation_fn)

    def sample(self, no_of_samples=1):
        # assuming each task takes 1 ms
        if self.queue_state == None:
            occupancy = self.distribution.rvs(size=no_of_samples)
        else:
            occupancy = np.array([(self.queue_state + 1) * 1000])
        wait_time = occupancy/self.srate
        return wait_time