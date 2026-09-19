# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import torch
import numpy as np
from scipy import signal




class Policy(torch.nn.Module):
    """
    Superclass of policy objects
    """
    def __init__(self, state_dim, input_dim,
                 flg_squash=False, u_max=1, 
                 dtype=torch.float64, device=torch.device('cpu')):
        super(Policy, self).__init__()
        # model parameters
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dtype = dtype
        self.device = device
        # set squashing function
        if flg_squash:
            self.f_squash = lambda x : self.squashing(x, u_max)
        else:
            # assign the identity function
            self.f_squash = lambda x : x

    
    def forward(self, states, t=None, p_dropout=0.0):
        raise NotImplementedError()


    def forward_np(self, state, t=None):
        """
        Numpy implementation of the policy
        """
        input_tc = self(states=torch.tensor(state,
                                            dtype=self.dtype,
                                            device=self.device),
                        t=t)
        return input_tc.detach().cpu().numpy()


    def to(self, device):
        """
        Move the model parameters to 'device'
        """
        super(Policy, self).to(device)
        self.device = device


    def squashing(self, u, u_max):
        """
        Squash the inputs inside (-u_max, +u_max)
        """
        if np.isscalar(u_max):
            return u_max*torch.tanh(u/u_max) 
        else:
            u_max = torch.tensor(u_max,dtype=self.dtype,device=self.device)
            return u_max*torch.tanh(u/u_max)


    def get_np_policy(self):
        """
        Returns a function handle to the numpy version of the policy
        """
        f = lambda state, t: self.forward_np(state, t) 

        return f

    def reinit(self, scaling=1):
        raise NotImplementedError()




class Random_exploration(Policy):
    """
    Random control action, uniform dist. (-u_max, +u_max)
    """
    def __init__(self, state_dim, input_dim,
                 flg_squash=True, u_max=1., 
                 dtype=torch.float64, device=torch.device('cpu')):

        super(Random_exploration, self).__init__(state_dim=state_dim,
                                                 input_dim=input_dim,
                                                 flg_squash=flg_squash, 
                                                 u_max=u_max, 
                                                 dtype=dtype,
                                                 device=device)
        self.u_max = u_max

    def forward(self, states, t):
        # returns random control action
        rand_u = self.u_max*(2*np.random.rand(self.input_dim)-1).reshape([-1,self.input_dim])
        return torch.tensor(rand_u, dtype=self.dtype, device=self.device)




class Filtered_Gaussian_noise_exploration(Policy):
    """
    Random control action, obtained filtering Gaussian noise
    """
    def __init__(self, state_dim, input_dim,
                 T_sampling, f_cut, std_coefficients, 
                 filt_order=1, flg_squash=True, u_max=1., 
                 dtype=torch.float64, device=torch.device('cpu')):

        super(Filtered_Gaussian_noise_exploration, self).__init__(state_dim=state_dim,
                                                                  input_dim=input_dim,
                                                                  flg_squash=flg_squash, 
                                                                  u_max=u_max, 
                                                                  dtype=dtype,
                                                                  device=device)
        self.u_max = u_max
        # set the filter parameters
        self.filt_order = filt_order
        b_list = []
        a_list = []
        for input_index in range(input_dim):
            f_sampling = 1/T_sampling
            fc = f_cut[input_index]/(f_sampling/2)
            b, a = signal.butter(filt_order, fc)
            b_list.append(b)
            a_list.append(a)
        print('b_list', b_list)
        print('a_list', a_list)
        self.b_tensor = torch.tensor(b_list, dtype=dtype, device=device).reshape([input_dim, filt_order+1])
        self.a_tensor = torch.tensor(a_list, dtype=dtype, device=device).reshape([input_dim, filt_order+1])
        print('self.b_tensor', self.b_tensor)
        print('self.a_tensor', self.a_tensor)
        self.std_coefficients = torch.tensor(std_coefficients, dtype=dtype, device=device).reshape([input_dim,1])


    def forward(self, states, t):
        print('\nFiltered_Gaussian_noise_exploration: t =', t)
        if t == 0.:
            # initi the filter state
            self.past_noise = torch.randn(self.input_dim, self.filt_order, dtype=self.dtype, device=self.device)
            self.past_input = torch.zeros([self.input_dim, self.filt_order], dtype=self.dtype, device=self.device)
        # returns random control action
        new_noise = torch.randn(self.input_dim, 1, dtype=self.dtype, device=self.device)*self.std_coefficients
        noise = torch.cat([new_noise, self.past_noise],1)
        rand_u = ((torch.sum(self.b_tensor*noise,1)-torch.sum(self.a_tensor[:,1:]*self.past_input,1))/self.a_tensor[:,0]).reshape([self.input_dim,1])
        print('Input =', self.f_squash(rand_u))
        self.past_noise = noise[:,:-1]
        self.past_input = torch.cat([rand_u,self.past_input[:,:-1]],1)
        return self.f_squash(rand_u)




class Sum_of_sinusoids(Policy):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """
    def __init__(self, state_dim, input_dim, num_sin, 
                 omega_min, omega_max,
                 amplitude_min, amplitude_max,
                 flg_squash=False, u_max=1,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(Sum_of_sinusoids, self).__init__(state_dim=state_dim,
                                               input_dim=input_dim,
                                               flg_squash=flg_squash, 
                                               u_max=u_max, 
                                               dtype=dtype,
                                               device=device)
        self.num_sin = num_sin
        amplitude_min = np.array(amplitude_min)
        amplitude_max = np.array(amplitude_max)
        # generate random parameters
        self.amplitudes = torch.nn.Parameter(torch.tensor(amplitude_min+(amplitude_max-amplitude_min)*np.random.rand(num_sin,input_dim),
                                                          dtype=self.dtype,
                                                          device=self.device),
                                             requires_grad=False)
        self.omega = torch.nn.Parameter(torch.tensor(np.random.choice([-1,1], [num_sin,input_dim])*(omega_min+(omega_max-omega_min)*np.random.rand(num_sin,input_dim)),
                                                     dtype=self.dtype,
                                                     device=self.device),
                                        requires_grad=False)
        self.phases = torch.nn.Parameter(torch.tensor(np.random.choice([-1,1], [num_sin,input_dim])*(np.pi*(np.random.rand(num_sin,input_dim)-0.5)),
                                                      dtype=self.dtype,
                                                      device=self.device),
                                         requires_grad=False)


    def forward(self, states, t):
        # returns the sinusoid values at time t
        return self.f_squash(torch.sum(self.amplitudes*(torch.sin(self.omega*t+self.phases)),dim=0).reshape([-1,self.input_dim]))




class Sum_of_gaussians(Policy):
    """
    Control policy: sum of 'num_basis' gaussians
    """
    def __init__(self, state_dim, input_dim, num_basis,
                 flg_train_lengthscales = True, lengthscales_init = None, 
                 flg_train_centers=True, centers_init = None, centers_init_min = -1, centers_init_max = 1,
                 weight_init = None, flg_train_weight = True,
                 flg_bias = False, bias_init=None, flg_train_bias = False,
                 flg_squash = False, u_max=1, scale_factor = None, active_dims=None,
                 flg_drop = True, dtype=torch.float64, device=torch.device('cpu')):
        super(Sum_of_gaussians, self).__init__(state_dim=state_dim,
                                               input_dim=input_dim,
                                               flg_squash=flg_squash, 
                                               u_max=u_max, 
                                               dtype=dtype,
                                               device=device)
        # set number of gaussian basis functions
        self.num_basis = num_basis
        # set active_dims
        if active_dims is None:
            active_dims = list(range(state_dim))
        self.active_dims = active_dims
        self.policy_dim = len(self.active_dims)
        # get initial log lengthscales
        if lengthscales_init is None:
            lengthscales_init = np.ones(self.policy_dim)
        self.log_lengthscales = torch.nn.Parameter(torch.tensor(np.log(lengthscales_init),
                                                                dtype=self.dtype,
                                                                device=self.device).reshape([1,-1]),
                                                   requires_grad=flg_train_lengthscales)
        # get initial centers
        if centers_init is None:
            centers_init = centers_init_min*np.ones([num_basis,self.policy_dim]) + (centers_init_max-centers_init_min)*np.random.rand(num_basis,self.policy_dim)
        self.centers = torch.nn.Parameter(torch.tensor(centers_init,
                                                       dtype=self.dtype,
                                                       device=self.device),
                                          requires_grad=flg_train_centers)
        # initilize the linear ouput layer
        self.f_linear = torch.nn.Linear(in_features=num_basis,
                                        out_features=input_dim,
                                        bias=flg_bias)
        # check weight initialization
        if not (weight_init is None):
            self.f_linear.weight.data = torch.tensor(weight_init, dtype=dtype, device=device)
        else:
            self.f_linear.weight.data = torch.tensor(np.ones([input_dim,num_basis]), dtype=dtype, device=device)
                
        self.f_linear.weight.requires_grad = flg_train_weight
        # check bias initialization
        if flg_bias:
            self.f_linear.bias.requires_grad = flg_train_bias
            if not (bias_init is None):
                self.f_linear.bias.data = torch.tensor(bias_init)
        # set type and device
        self.f_linear.type(self.dtype)
        self.f_linear.to(self.device)

        if scale_factor is None:
            scale_factor = np.ones(self.policy_dim)
        self.scale_factor = torch.tensor(scale_factor,dtype=self.dtype,device=self.device).reshape([1,-1])

        # set dropout
        if flg_drop == True:
            self.f_drop = torch.nn.functional.dropout
        else: 
            self.f_drop = lambda x, p: x

    def reinit(self, lenghtscales_par, centers_par, weight_par):
        self.log_lengthscales.data = torch.tensor(np.log(lenghtscales_par),dtype=self.dtype,device=self.device).reshape([1,-1])
        self.centers.data = torch.tensor(centers_par, dtype=self.dtype, device= self.device)*2*(torch.rand(self.num_basis,self.policy_dim,dtype=self.dtype, device=self.device)-0.5)
        self.f_linear.weight.data = weight_par*(torch.rand(self.input_dim,self.num_basis,dtype=self.dtype, device=self.device)-0.5)

    def forward(self, states, t=None, p_dropout=0.0):
        """ 
        Returns a linear combination of gaussian functions
        with input given by the the distances between that state
        and the vector of centers of the gaussian functions
        """
        # get the lengthscales from log
        lengthscales = torch.exp(self.log_lengthscales)
        # unsqueeze states
        states = states.reshape([-1, self.state_dim]).unsqueeze(1)[:,:,self.active_dims]
        states = states/self.scale_factor 
        # normalize states and centers
        norm_states = states/lengthscales
        norm_centers = self.centers/lengthscales
        # get the square distance
        dist = torch.sum(norm_states**2, dim=2, keepdim=True)
        dist =  dist + torch.sum(norm_centers**2, dim=1, keepdim=True).transpose(0,1)
        dist -= 2*torch.matmul(norm_states,norm_centers.transpose(dim0=0, dim1=1))
        # apply exp and get output
        exp_dist_dropped = self.f_drop(torch.exp(-dist), p_dropout)
        inputs = self.f_linear(exp_dist_dropped).reshape([-1,self.input_dim])
        # returns the constrained control action
        return self.f_squash(inputs)




class Sum_of_gaussians_with_angles(Sum_of_gaussians):
    """
    Extends sum of gaussians policy. Angle indices are mapped in cos and sin before computing the policy 
    """
    def __init__(self, state_dim, input_dim, num_basis,
                 angle_indices, non_angle_indices,
                 flg_train_lengthscales=True, lengthscales_init=None, 
                 flg_train_centers=True, centers_init=None, centers_init_min=-1, centers_init_max=1,
                 weight_init=None, flg_train_weight=True,
                 flg_bias=False, bias_init=None, flg_train_bias=False,
                 flg_squash=False, u_max=1,
                 flg_drop=True,
                 active_dims=None,
                 dtype=torch.float64, device=torch.device('cpu')):
        self.angle_indices = angle_indices
        self.non_angle_indices = non_angle_indices
        self.num_angle_indices = angle_indices.size
        self.num_non_angle_indices = non_angle_indices.size
        super(Sum_of_gaussians_with_angles, self).__init__(state_dim=self.num_non_angle_indices+2*self.num_angle_indices, input_dim=input_dim, num_basis=num_basis,
                                                           flg_train_lengthscales=flg_train_lengthscales, lengthscales_init=lengthscales_init, 
                                                           flg_train_centers=flg_train_centers, centers_init=centers_init,
                                                           centers_init_min=centers_init_min, centers_init_max=centers_init_max,
                                                           weight_init=weight_init, flg_train_weight=flg_train_weight,
                                                           flg_bias=flg_bias, bias_init=bias_init, flg_train_bias=flg_train_bias,
                                                           flg_squash=flg_squash, u_max=u_max,
                                                           flg_drop=flg_drop,
                                                           active_dims=active_dims,
                                                           dtype=dtype, device=device)
        self.state_dim_ = state_dim

    def forward(self, states, t=None, p_dropout=0.0):
        # build a state with non angle features and cos,sin of angle features
        states = states.reshape([-1,self.state_dim_])
        #print(states)
        new_state = torch.cat([states[:,self.non_angle_indices],
                               torch.cos(states[:,self.angle_indices]),
                               torch.sin(states[:,self.angle_indices])], 1)
        # call the forward method of the superclass
        return super().forward(new_state, t=t, p_dropout=p_dropout)




class Sum_of_gaussians_with_target_trajectory(Sum_of_gaussians):
    """ 
    Use sum of gaussians policy with a reference target trajectory. It considers and extended state: [state, target-state]
    """
    def __init__(self, state_dim, input_dim, num_basis, target_traj,
                 flg_train_lengthscales=True, lengthscales_init=None, 
                 flg_train_centers=True, centers_init=None, centers_init_min=-1, centers_init_max=1,
                 weight_init=None, flg_train_weight=True,
                 flg_bias=False, bias_init=None, flg_train_bias=False,
                 flg_squash=False, u_max=1, flg_drop=True,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(Sum_of_gaussians_with_target_trajectory, self).__init__(state_dim=state_dim, input_dim=input_dim, num_basis=num_basis,
                                                       flg_train_lengthscales=flg_train_lengthscales, lengthscales_init=lengthscales_init, 
                                                       flg_train_centers=flg_train_centers, centers_init=centers_init,
                                                       centers_init_min=centers_init_min, centers_init_max=centers_init_max,
                                                       weight_init=weight_init, flg_train_weight=flg_train_weight,
                                                       flg_bias=flg_bias, bias_init=bias_init, flg_train_bias=flg_train_bias,
                                                       flg_squash=flg_squash, u_max=u_max, flg_drop=flg_drop,
                                                       dtype=dtype, device=device)
        self.target_traj = torch.tensor(target_traj,dtype=self.dtype,device=self.device)

    def forward(self, states, t=None, p_dropout=0.0):
        if states.dim() == 1:
            # single state
            target = self.target_traj[t,:]
            policy_in = torch.cat((states,target-states),0)
            # policy_in = target-states
        elif states.dim() == 2:
            # particles batch
            target = self.target_traj[t,:]
            particle_targets = target.repeat(1,states.shape[0]).view(states.shape)
            policy_in = torch.cat((states,particle_targets-states),1)
            # policy_in = particle_targets-states
        u = super().forward(policy_in, t=t, p_dropout=p_dropout)

        return u




class PD_controller(Policy):
    """
    PD controller (N.B. it takes error as input)
    """
    def __init__(self, state_dim, input_dim,  sqrt_Kp_gains, sqrt_Kd_gains, target_traj=None,
                 flg_squash=True, u_max=1., flg_trainable=False,
                 dtype=torch.float64, device=torch.device('cpu')):

        super(PD_controller, self).__init__(state_dim=state_dim,
                                            input_dim=input_dim,
                                            flg_squash=flg_squash, 
                                            u_max=u_max, 
                                            dtype=dtype,
                                            device=device)

        self.target_traj = target_traj
        self.sqrt_Kp_gains = torch.nn.Parameter(torch.tensor(sqrt_Kp_gains,
                                                        dtype=self.dtype,
                                                        device=self.device),
                                           requires_grad=flg_trainable)
        self.sqrt_Kd_gains = torch.nn.Parameter(torch.tensor(sqrt_Kd_gains,
                                                        dtype=self.dtype,
                                                        device=self.device),
                                           requires_grad=flg_trainable)
        

    def forward(self, states, t, p_dropout=0.):
        """
        in states we pass actually the error: target-states
        """
        states = states.reshape([-1, self.state_dim])
        target = self.target_traj[t,:]

        particle_targets = target.repeat(1,states.shape[0]).view(states.shape)
        err = particle_targets-states
        Kp_gains = self.sqrt_Kp_gains**2
        Kd_gains = self.sqrt_Kd_gains**2
        inputs = Kp_gains*err[:,0:int(self.state_dim/2)] + Kd_gains*err[:,int(self.state_dim/2):]
        return self.f_squash(inputs)


class PointPD_controller(Policy):
    """
    PD controller (w.r.t. point)
    """

    def __init__(self, state_dim, input_dim, sqrt_Kp_gains, sqrt_Kd_gains, target_state=None,
                 flg_squash=True, u_max=1., flg_trainable=False,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(PointPD_controller, self).__init__(state_dim=state_dim,
                                            input_dim=input_dim,
                                            flg_squash=flg_squash,
                                            u_max=u_max,
                                            dtype=dtype,
                                            device=device)

        self.target_state = target_state
        self.sqrt_Kp_gains = torch.nn.Parameter(torch.tensor(sqrt_Kp_gains,
                                                             dtype=self.dtype,
                                                             device=self.device),
                                                requires_grad=flg_trainable)
        self.sqrt_Kd_gains = torch.nn.Parameter(torch.tensor(sqrt_Kd_gains,
                                                             dtype=self.dtype,
                                                             device=self.device),
                                                requires_grad=flg_trainable)

    def forward(self, states, t, p_dropout=0.):
        """
        in states we pass actually the error: target-states
        """
        states = states.reshape([-1, self.state_dim])
        target = copy.deepcopy(self.target_state)

        particle_targets = target.repeat(1, states.shape[0]).view(states.shape)
        err = particle_targets - states
        Kp_gains = self.sqrt_Kp_gains ** 2
        Kd_gains = self.sqrt_Kd_gains ** 2
        inputs = Kp_gains * err[:, 0:int(self.state_dim / 2)] + Kd_gains * err[:, int(self.state_dim / 2):]
        return self.f_squash(inputs)


class NN_policy(Policy):
    """
    Control policy: NN policy with two hidden layers
    """
    def __init__(self, state_dim, input_dim,
                 hidden_dim_1, hidden_dim_2, flg_bias,
                 active_dims = None, flg_squash = False, u_max=1,
                 flg_drop = True, dtype=torch.float64, device=torch.device('cpu')):
        super(NN_policy, self).__init__(state_dim=state_dim,
                                               input_dim=input_dim,
                                               flg_squash=flg_squash, 
                                               u_max=u_max, 
                                               dtype=dtype,
                                               device=device)
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.flg_bias = flg_bias
        if active_dims is None:
            active_dims = list(range(state_dim))
        self.active_dims = active_dims

        # initilize the linear functions
        self.f_linear_1 = torch.nn.Linear(in_features=len(self.active_dims),
                                          out_features=hidden_dim_1,
                                          bias=flg_bias)
        self.f_linear_2 = torch.nn.Linear(in_features=hidden_dim_1,
                                          out_features=hidden_dim_2,
                                          bias=flg_bias)
        self.f_linear_3 = torch.nn.Linear(in_features=hidden_dim_2,
                                          out_features=input_dim,
                                          bias=flg_bias)
        # set type and device
        self.f_linear_1.type(self.dtype)
        self.f_linear_1.to(self.device)
        self.f_linear_2.type(self.dtype)
        self.f_linear_2.to(self.device)
        self.f_linear_3.type(self.dtype)
        self.f_linear_3.to(self.device)
        # set dropout
        if flg_drop == True:
            self.f_drop = torch.nn.functional.dropout
        else: 
            self.f_drop = lambda x, p: x


    def reinit (self):
        self.f_linear_1 = torch.nn.Linear(in_features=len(self.active_dims),
                                          out_features=self.hidden_dim_1,
                                          bias=self.flg_bias,
                                          device=self.device, 
                                          dtype=self.dtype)
        self.f_linear_2 = torch.nn.Linear(in_features=self.hidden_dim_1,
                                          out_features=self.hidden_dim_2,
                                          bias=self.flg_bias,
                                          device=self.device, 
                                          dtype=self.dtype)
        self.f_linear_3 = torch.nn.Linear(in_features=self.hidden_dim_2,
                                          out_features=self.input_dim,
                                          bias=self.flg_bias,
                                          device=self.device, 
                                          dtype=self.dtype)


    def forward(self, states, t=None, p_dropout=0.0):
        """ 
        Returns the output of the NN
        """
        
        states = states.reshape([-1, self.state_dim])#.unsqueeze(1)
        h_layer_1 = torch.relu(self.f_linear_1(self.f_drop(states[:,self.active_dims], p_dropout)))
        h_layer_2 = torch.relu(self.f_linear_2(self.f_drop(h_layer_1, p_dropout)))
        inputs = self.f_linear_3(self.f_drop(h_layer_2, p_dropout)).reshape([-1,self.input_dim])
        
        # returns the constrained control action
        return self.f_squash(inputs)
    
    
    def get_np_policy(self, ret_dict=False):
        """
        Returns a function handle to the numpy version of the policy
        """
        f = lambda state, t: self.forward_np(state, t) 
        if ret_dict:
            return f, self.get_params_dict()
        return f
    
    def get_params_dict(self):
        d = {"f_linear_1_weight": self.f_linear_1.weight.cpu().detach().numpy(),
             "f_linear_1_bias": self.f_linear_1.bias.cpu().detach().numpy(),
             "f_linear_2_weight": self.f_linear_2.weight.cpu().detach().numpy(),
             "f_linear_2_bias": self.f_linear_2.bias.cpu().detach().numpy(),
             "f_linear_3_weight": self.f_linear_3.weight.cpu().detach().numpy(),
             "f_linear_3_bias": self.f_linear_3.bias.cpu().detach().numpy()}
        
        return d

class GuidedExplorationPolicy(Policy):
    def __init__(self, control_policy, exploration_policy, exploration_fun, state_dim, input_dim, u_max=1,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(GuidedExplorationPolicy, self).__init__(state_dim=state_dim, input_dim=input_dim,
                                                u_max=u_max, dtype=dtype, device=device)
        self.control_policy = control_policy
        self.exploration_policy = exploration_policy
        self.exploration_fun = exploration_fun # f: (x,u) -> [0,1]

    def forward(self, states, t=None, p_dropout=0.0):
        u = self.control_policy(states, t, p_dropout)
        alpha = self.exploration_fun(states.reshape([-1, self.state_dim]), u.reshape([-1, self.input_dim]))

        return self.f_squash((1-alpha) * u + alpha * self.exploration_policy(states, t))
