import torch
import torch.nn as nn
import sys
import os


class BGNetwork(nn.Module):
    def __init__(self, max_gpi_iters, d1_amp, d2_amp, gpi_threshold, seed = None,num_arms = 4,STN_neurons = 256):
        super(BGNetwork, self).__init__()
        # env 
        # Number of IGT arms
        self.num_arms = num_arms
        # Race model params
        self.d1_amp = d1_amp #5
        self.d2_amp = d2_amp #1
        self.tau_gpi = 0.1 # 100ms
        self.dt = 0.01 
        self.gpi_threshold = gpi_threshold #3
        self.max_gpi_iters = max_gpi_iters
        self.STN_neurons = STN_neurons
        self.seed = seed
        self.input = torch.ones(1,self.num_arms)

        self.str_d1 = nn.Linear(in_features=self.num_arms,out_features=self.num_arms)
        
        self.snc = nn.Linear(in_features=self.num_arms,out_features=1)
        
        self.d1_gpi = nn.Linear(in_features=self.num_arms,out_features=self.num_arms)
        self.d2_gpi = nn.Linear(in_features=self.STN_neurons,out_features=self.num_arms)
        self.activation = nn.Sigmoid()
        self.D1_pathway = nn.Sequential(self.str_d1,
                                        self.activation,
                                        self.d1_gpi,
                                        self.activation)
        self.D2_pathway = nn.Sequential(self.d2_gpi,
                                        self.activation)

        self.snc_value = nn.Sequential(self.snc,
                                       self.activation)
        

        self._initialize_weights(self.seed)

    def _initialize_weights(self, seed):
        if seed is not None:
            torch.manual_seed(seed)  # Set the seed for reproducibility

         # Initialize weights and biases for each Linear layer
        for layer in self.children():  # Iterate over all layers in the model
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        # Initialize weights using uniform distribution
                        # nn.init.uniform_(sublayer.weight, a=0.0, b = 0.001) #0.01)
                        torch.nn.init.constant_(sublayer.weight, 0.001)
                        if sublayer.bias is not None:
                            # nn.init.uniform_(sublayer.bias, a=0.0, b= 0.001) #0.01)
                            torch.nn.init.constant_(sublayer.bias, 0.001)

    def forward(self,stn_input):
        '''
        Args:
        stn_input(torch.tensor): shape (1, time_points, num_neurons = 4)
        '''
        time_points = self.max_gpi_iters 
        assert time_points >= self.max_gpi_iters, "Number of timepoints is smaller than max iters"

        v_gpi = torch.zeros((1,self.num_arms))
        D1_output = self.D1_pathway(self.input)
        value = self.snc(D1_output)
        t = 0
        while t < time_points:
            D2_output = stn_input[:,t,:] 
            # running race model: 
            v_gpi = v_gpi + (self.dt/self.tau_gpi) * (-v_gpi - self.d1_amp *D1_output + torch.round(self.d2_amp *D2_output, decimals = 2))
            v_gpi_out = -v_gpi
            max, _ = torch.max(v_gpi_out,1)
            t += 1
            
            if max > self.gpi_threshold:
                break

        dp_output = self.d1_amp * D1_output
        ip_output =  torch.round(self.d2_amp *D2_output, decimals = 2)

        return v_gpi_out, t, dp_output, ip_output