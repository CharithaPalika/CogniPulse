import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))

if project_root not in sys.path:
    sys.path.append(project_root)

from stn_gpe import *
from .BGNetwork import BGNetwork
import torch
import numpy as np
from tqdm.autonotebook import tqdm
import yaml
from matplotlib import pyplot as plt

def update_epsilon(ep_old,TD_error,alpha_ep, eta_ep, baseline_val = 0.02):
    ep = ep_old + alpha_ep * (1- torch.exp((-TD_error**2)/eta_ep) - ep_old) + baseline_val
    return ep.item()

def run_STN_GPe_system(yaml_path):
    arguments = load_yaml(yaml_path)
    arguments['time'] = int(50000) #int(120000)
    save_yaml(arguments, 'temp.yaml')
    results = STN_GPe_loop('temp.yaml')
    spikes_data = np.array(results['spike_stn'])[40000:50000]#[20000:120000]
    Analysis_= Analysis(spikes_data)
    rate_data = Analysis_.spike_rate(binsize = 100)
    STN_4_processed = rate_data['processed_stn']
    ts = torch.tensor(STN_4_processed).T      # shape (time, 4)
    print(torch.mean(torch.std(ts, dim = 1)))
    
    mu = ts.mean(axis=0)           # mean of each column
    ts_shifted = ts + (1 - mu)     # shift each column
    print(torch.mean(ts_shifted), torch.mean(torch.std(ts_shifted, dim = 1)))
    stn_out = torch.randn((1000,4)) * torch.mean(torch.std(ts_shifted, dim = 1)) + 1
    print(torch.mean(stn_out), torch.mean(torch.std(stn_out, dim = 1)))
    # rewrite ts_shifted as same shape torc htensor but with random number with ,eam amd std
    # ts_shifted = torch.randn_like(ts_shifted) * torch.std(ts_shifted, dim = 0) + 1
    return stn_out #ts_shifted

def train(env, trails, epochs, bins, lr , 
          d1_amp = 1, d2_amp = 5, gpi_threshold = 3, max_gpi_iters = 250,
          STN_data = None, del_lim = None, del_med = None, train_IP = False,
          printing = False, gpi_mean = 1,
          ep_0 = 0.5,
          alpha_ep = 0.0, #0.25,
          eta_ep = 0.1, ep_lim = None, 
          baseline_ep = 0.0, #0.02, 
          track_arms = False):
    

    num_arms = env.num_arms
    picks_per_bin = int(trails//bins)
    arm_chosen_monitor = torch.zeros(epochs,trails)
    arm_tracker_full = []

    reward_monitor = torch.zeros(epochs,trails)
    avg_counts = {i: torch.zeros(epochs,bins,1) for i in np.arange(num_arms)}
    ip_monitor = {i: torch.zeros(epochs,trails,1) for i in np.arange(num_arms)}
    dp_monitor = {i: torch.zeros(epochs,trails,1) for i in np.arange(num_arms)}
    ep_monitor = torch.zeros(epochs, trails)

    if STN_data is not None:
        print(f'Running STN-GPe system')
        stn_out_= run_STN_GPe_system(yaml_path = STN_data)

    for epoch in tqdm(range(epochs)):
        # print(f'**************************{epoch}*************************************')

        env.reset()
        arm_tracker = []
        if track_arms:
            arm_tracker.append(env.arms)
        ep = ep_0
        if ep_lim is not None:
            ep = torch.clamp(torch.tensor(ep), max = ep_lim).item()
        bg_network = BGNetwork(max_gpi_iters = max_gpi_iters, 
                               d1_amp = d1_amp, 
                               d2_amp = d2_amp, 
                               gpi_threshold = gpi_threshold,
                               seed = epoch,
                               num_arms=env.num_arms)
        
        optimizer = torch.optim.Adam(params = bg_network.parameters(), lr = lr)
        
        for trail in range(trails):
            ep_monitor[epoch, trail] = ep
            bin_num = int(trail//picks_per_bin)
            rand_num = np.random.choice(900)#np.random.choice(50)#(900)
            if STN_data is None:
                stn_output = torch.randn((1,max_gpi_iters,num_arms), requires_grad= False) * ep + gpi_mean 
                # print(stn_output.shape)
            else:
                stn_out = stn_out_[rand_num: rand_num + max_gpi_iters].unsqueeze(0)
                stn_output = stn_out[:,:, 0: num_arms]
                # print(stn_output.shape)

            gpi_out, gpi_iters, dp_output, ip_output = bg_network(stn_output)
            arm_chosen = torch.argmax(gpi_out)

            avg_counts[arm_chosen.item()][epoch,bin_num] = avg_counts[arm_chosen.item()][epoch,bin_num] + 1
            
            for arm in range(num_arms):
                ip_monitor[arm][epoch,trail] = ip_output[0,arm]
                dp_monitor[arm][epoch,trail] = dp_output[0,arm]

            reward = env.step(arm_chosen.item())


            if track_arms:
                arm_tracker.append(env.arms)

            TD_error =  reward - dp_output[:, arm_chosen] 
            if train_IP == False:
                for param in bg_network.D2_pathway.parameters():
                    param.requires_grad = False  
            
            if del_lim is not None:             
                TD_error = torch.clamp(TD_error, max=del_lim)
            
            if del_med is not None:
                TD_error = TD_error + del_med

            loss = TD_error**2    
            
            # setting gradients to zero
            optimizer.zero_grad()
            
            # Computing gradient
            loss.backward()
            
            # Updating weights
            optimizer.step()

            #network weights to clamped to only positive
            with torch.no_grad():
                for param in bg_network.parameters():
                    param.clamp_(min=0)  

            arm_chosen_monitor[epoch, trail] = arm_chosen.item()
            reward_monitor[epoch, trail] = reward

            ep = update_epsilon(ep_old=ep, TD_error=TD_error.detach(), alpha_ep=alpha_ep, eta_ep=eta_ep, baseline_val = baseline_ep)
            if ep_lim is not None:
                ep = torch.clamp(torch.tensor(ep), max = ep_lim).item()

            if printing:
                print(f'{trail}: dp: {dp_output}, arm_chosen:{arm_chosen}, TD error: {TD_error}, epsilon: {ep}')#dp_output, ip_output, gpi_out, gpi_iters,  arm_chosen, reward, TD_error)
    
        if track_arms:
            arm_tracker_full.append(arm_tracker)
    return reward_monitor, arm_chosen_monitor,avg_counts,ip_monitor, dp_monitor, ep_monitor, arm_tracker_full


