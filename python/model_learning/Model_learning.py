# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.utils.data
import sys 
import gpr_lib.GP_prior.GP_prior as GP
import gpr_lib.GP_prior.Stationary_GP as SGP
import gpr_lib.GP_prior.Sparse_GP as Sparse_GP
from torch.distributions.normal import Normal
import gpr_lib.Utils.Parameters_covariance_functions as cov
import gpr_lib.Utils.Scaling_functions as F_scaling
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
from torch.distributions.normal import Normal
# from model_learning.helper_functions import generate_grid

# from sklearn.cluster import KMeans,DBSCAN
from scipy.spatial import distance
from torch.nn import Module

class Model_learning(torch.nn.Module):
    """
    Model learning Class
    """
    def __init__(self, num_gp, init_dict_list,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'),
                 flg_norm = False, flg_norm_mean = False,
                 flg_parametric_model = False, evolution_indices=None,
                 cnst_indices = None, gp_adaptive_init=False):
        super(Model_learning, self).__init__()
        # Set model info
        self.num_samples = 0
        self.dtype = dtype
        self.device = device
        self.flg_parametric_model = flg_parametric_model
        self.init_dict_list = init_dict_list
        # Get the gp list
        self.num_gp = num_gp
        self.alpha_list = [None]*num_gp
        self.m_X_list = [None]*num_gp 
        self.K_X_inv_list = [None]*num_gp
        self.gp_inputs_tr_list = [None]*num_gp
        self.num_gp_inputs_list = [0]*num_gp
        # check approximation
        self.approximation_mode = approximation_mode
        if approximation_mode is None:
            print('EXACT GP INFERENCE SELECTED')
            self.get_gp_estimate = self.get_exact_gp_estimate
        else:
            self.approximation_dict = approximation_dict
            print('GP APPROXIMATION SELECTED')
            print('APPROXIMATION MODE: ', approximation_mode)
            print('APPROXIMATION OPTIONS: ', approximation_dict)
            if approximation_mode == 'SOR':
                self.get_gp_estimate = self.get_SOR_gp_estimate
                self.Sigma_SOR_list = [None] * num_gp
                self.reg_indices_SOR_list = [None] * num_gp
                self.SOR_threshold_mode = approximation_dict['SOR_threshold_mode']
                self.SOR_threshold = approximation_dict['SOR_threshold']
                if flg_parametric_model:
                    self.sample_model_parameters = self.sample_SOR_model_parameters
                    self.get_gp_estimate_parametric_model = self.get_SOR_gp_estimate_parametric_model

            elif approximation_mode == 'SOR2':
                self.get_gp_estimate = self.get_SOR2_gp_estimate
                self.Sigma_SOR2_list = [None] * num_gp
                self.grid_exact_all_range = [None] * num_gp
                self.inducing_input_data_list = [None] * num_gp
                self.scale_ranges = approximation_dict['scale_ranges']
                self.state_ranges = approximation_dict['state_ranges']
                self.grid_step = approximation_dict['grid_step']
                self.active_dims_SOR = approximation_dict['active_dims']
                if flg_parametric_model:
                    self.sample_model_parameters = self.sample_SOR2_model_parameters
                    self.get_gp_estimate_parametric_model = self.get_SOR2_gp_estimate_parametric_model

            elif approximation_mode == 'SOD':
                self.SOD_indices = [None]*num_gp
                self.get_gp_estimate = self.get_SOD_gp_estimate
                self.SOD_threshold_mode = approximation_dict['SOD_threshold_mode']
                self.SOD_threshold = approximation_dict['SOD_threshold']
                self.flg_SOD_permutation = approximation_dict['flg_SOD_permutation']
                if 'criterion' in approximation_dict:
                    self.SOD_criterion = approximation_dict['criterion']
                else:
                    self.SOD_criterion = 'std'

            elif approximation_mode == 'SOD_proj':
                self.SOD_indices = [None]*num_gp
                self.get_gp_estimate = self.get_SOD_gp_estimate
                self.SOD_threshold_mode = approximation_dict['SOD_threshold_mode']
                self.SOD_threshold = approximation_dict['SOD_threshold']
                self.flg_SOD_permutation = approximation_dict['flg_SOD_permutation']
                self.SOD_num_pts = approximation_dict['num_pts']


            elif approximation_mode == 'nystrom':
                self.get_gp_estimate = self.get_nystrom_gp_estimate
                if flg_parametric_model:
                    self.sample_model_parameters = self.sample_nystrom_model_parameters
                    self.get_gp_estimate_parametric_model = self.get_nystrom_gp_estimate_parametric_model
                #self.nystrom_max_rank = approximation_dict['nystrom_max_rank']
                self.R_perc = approximation_dict['R_perc']
                #self.nystrom_init_dict = approximation_dict['nystrom_init_dict']
                self.w_nystrom = [None]*num_gp
                self.Sigma_w_nystrom = [None]*num_gp
                self.gp_list_nystrom = [None]*num_gp
                self.gp_inputs_tr_list_nystrom = [None]*num_gp
        # init the GP models
        self.init_gp_models()
        # set normalization parameters
        self.flg_norm = flg_norm
        self.flg_norm_mean = flg_norm_mean
        self.norm_scale_coef_list = [1.]*self.num_gp
        self.norm_mean_coef_list = [0.]*self.num_gp
        if cnst_indices is None:
            cnst_indices = []
        self.cnst_indices = cnst_indices
        self.evolution_indices = evolution_indices
        self.gp_adaptive_init = gp_adaptive_init

    def init_gp_models(self):
        """
        Init GP models
        """
        self.gp_list = torch.nn.ModuleList([self.get_gp(gp_index=gp_index,
                                                        init_dict=self.init_dict_list[gp_index])
                                            for gp_index in range(0, self.num_gp)])
        if (self.approximation_mode == 'SOR') or (self.approximation_mode == 'SOR2'):
            self.gp_list = torch.nn.ModuleList([Sparse_GP.get_SOR_GP(gp) for gp in self.gp_list])


    def set_eval_mode(self):
        """
        Set all the gp in eval mode
        """ 
        for gp in self.gp_list:
            gp.set_eval_mode()


    def set_training_mode(self):
        """
        Set all the gp in traing mode
        """
        for gp in self.gp_list:
            gp.set_training_mode()


    def add_data(self, new_state_samples, new_input_samples):
        """
        Transform state_sample and input samples in input-output of the gp and store them
        """
        if self.num_samples == 0:
            if self.evolution_indices is None:
                self.evolution_indices = list(set(range(new_state_samples.shape[1])) - set(self.cnst_indices))
            self.dim_state = new_state_samples.shape[1] - len(self.evolution_indices)
            self.gp_inputs, self.gp_output_list = self.data_to_gp_IO(torch.tensor(new_state_samples,
                                                                                  dtype = self.dtype,
                                                                                  device = self.device), 
                                                                     torch.tensor(new_input_samples,
                                                                                  dtype = self.dtype,
                                                                                  device = self.device))          
            self.num_samples = new_state_samples.shape[0]
            _, self.dim_input = new_input_samples.shape
        else:
            
            new_gp_inputs, new_gp_output_list = self.data_to_gp_IO(torch.tensor(new_state_samples,
                                                                                dtype = self.dtype,
                                                                                device = self.device), 
                                                                   torch.tensor(new_input_samples,
                                                                                dtype = self.dtype,
                                                                                device = self.device))      
            # update samples set
            self.gp_inputs = torch.cat([self.gp_inputs, (new_gp_inputs)])
            self.gp_output_list = [torch.cat([self.gp_output_list[gp_index], 
                                              new_gp_output_list[gp_index]], 0)
                                   for gp_index in range(0, self.num_gp)]
            self.num_samples, _ = self.gp_inputs.shape

        if self.gp_adaptive_init:
            init_dict_list = [self.init_dict_list[gp_index] for gp_index in range(0, self.num_gp)]
            for gp_index in range(0, self.num_gp):
                # print(self.gp_inputs.size())
                # print(torch.std(self.gp_inputs, dim=0).size())
                lenscales = list(torch.std(self.gp_inputs[:, init_dict_list[gp_index]['active_dims']], dim=0).detach().cpu().numpy())
                lenscales = torch.tensor(lenscales, dtype=self.gp_inputs.dtype, device=self.gp_inputs.device)
                print(lenscales)
                init_dict_list[gp_index]['lengthscales_init'] = lenscales.detach().cpu().numpy()
                init_dict_list[gp_index]['scale_init'] = torch.std(self.gp_output_list[gp_index], dim=0).detach().cpu().numpy()
                init_dict_list[gp_index]['sigma_n_init'] = copy.deepcopy(init_dict_list[gp_index]['scale_init'])


    def reinforce_model(self, optimization_opt_list = None):
        """
        Optimize GP model
        optimization_opt_list is a list collecting dictionaries with the optimization options
        """
        #initialize the GP models
        self.init_gp_models()
        # train each gp
        for gp_index in range(0,self.num_gp):
            self.train_gp(gp_index = gp_index,
                          optimization_opt_dict = optimization_opt_list[gp_index])
            print('\nNORMALIZATION COEFFICIENTS:')
            print('-norm_mean_coef_list:',self.norm_mean_coef_list)
            print('-norm_scale_coef_list:',self.norm_scale_coef_list)
            # pretrain each gp (compute alpha, m_X and K_X_inv)
            with torch.no_grad():
                self.pretrain_gp(gp_index = gp_index)


    def pretrain_gp(self, gp_index):
        """
        Compute traininng estimates and returns alpha and K_X_inv
        """
        # make computations
        if self.approximation_mode is None:
            Y_hat, var, alpha, m_X, K_X_inv = self.gp_list[gp_index].get_estimate(X = self.gp_inputs,
                                                                                  Y = (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                                  X_test = self.gp_inputs,
                                                                                  flg_return_K_X_inv = True)
            self.K_X_inv_list[gp_index] = K_X_inv
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs

        elif self.approximation_mode == 'SOD':
            # get the threshold
            if self.SOD_threshold_mode == 'relative':
                threshold = self.SOD_threshold[gp_index] * torch.sqrt(self.gp_list[gp_index].get_sigma_n_2())
            elif self.SOD_threshold_mode == 'absolute':
                threshold = self.SOD_threshold[gp_index]
            # get the SOD
            self.SOD_indices[gp_index] = self.gp_list[gp_index].get_SOD(X = self.gp_inputs,
                                                                        Y = (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                        threshold = threshold,
                                                                        flg_permutation = self.flg_SOD_permutation,
                                                                        criterion = self.SOD_criterion)
            #compute the posterior
            Y_hat, var, alpha, m_X, K_X_inv = self.gp_list[gp_index].get_estimate(X = self.gp_inputs[self.SOD_indices[gp_index],:],
                                                                                  Y = (self.gp_output_list[gp_index][self.SOD_indices[gp_index],:]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                                  X_test = self.gp_inputs,
                                                                                  flg_return_K_X_inv = True)
            self.K_X_inv_list[gp_index] = K_X_inv
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs[self.SOD_indices[gp_index],:]

        elif self.approximation_mode == 'SOD_proj':
            # get the threshold
            if self.SOD_threshold_mode == 'relative':
                threshold = self.SOD_threshold[gp_index] * torch.sqrt(self.gp_list[gp_index].get_sigma_n_2()).cpu().detach().numpy()
            elif self.SOD_threshold_mode == 'absolute':
                threshold = self.SOD_threshold[gp_index]
            
            # get the SOD with input selection
            self.SOD_indices[gp_index], alpha, K_X_inv = self.gp_list[gp_index].get_SOD_projection_error(X = self.gp_inputs,
                                                                                         Y = (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                                         threshold = threshold,
                                                                                         num_pts=self.SOD_num_pts,
                                                                                         flg_permutation = self.flg_SOD_permutation,
                                                                                         return_alpha=True)

            self.K_X_inv_list[gp_index] = K_X_inv
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = self.gp_list[gp_index].get_mean(self.gp_inputs[self.SOD_indices[gp_index],:])
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs[self.SOD_indices[gp_index],:]
            Y_hat, _, _ = self.gp_list[gp_index].get_estimate_from_alpha(self.gp_inputs[self.SOD_indices[gp_index],:], self.gp_inputs, alpha, K_X_inv=K_X_inv)

        elif self.approximation_mode == 'SOR':
            Y_hat, var, alpha, m_X, Sigma = self.gp_list[gp_index].get_SOR_estimate(X = self.gp_inputs,
                                                                                    Y = (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                                    X_test = self.gp_inputs,
                                                                                    flg_return_Sigma = True)
            self.Sigma_SOR_list[gp_index] = Sigma
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs

        elif self.approximation_mode == 'SOR2':
            # result = None
            # step_scale = 3
            # while result is None:
            #     try:
            #         print('Set grid of GP'+str(gp_index)+': stepscale = '+str(step_scale))
            #         # check state ranges
            #         if self.state_ranges is not None:
            #             ranges = self.state_ranges
            #         else:
            #             # set ranges from data
            #             ranges = [(torch.min(self.gp_inputs[:, i]).item() * self.scale_ranges, torch.max(self.gp_inputs[:, i]).item() * self.scale_ranges) for i in range(self.gp_inputs.shape[1])]
            #         # check step
            #         if self.grid_step is not None:
            #             grid_step = self.grid_step
            #             grid_step = [grid_step[active_dims_index] for active_dims_index in self.active_dims_SOR]
            #         else:
            #             grid_step = np.exp(self.gp_list[gp_index].log_lengthscales_par.data.numpy())/step_scale
            #         # select active dims
            #         ranges = [ranges[active_dims_index] for active_dims_index in self.active_dims_SOR]
            #         # generate the grid
            #         grid = self.generate_grid(ranges, grid_step)
            #         grid_flat = tuple([grid[i].ravel() for i in range(len(grid))])
            #         inducing_input_data_ = np.column_stack(grid_flat)
            #         num_grid_points = inducing_input_data_.shape[0]
            #         print('Number of inducing points:',num_grid_points)
            #         # check discarded features
            #         if len(self.active_dims_SOR) < self.gp_inputs.shape[1]:
            #             inducing_input_data = np.zeros([num_grid_points,self.gp_inputs.shape[1]])
            #             inducing_input_data[:, self.active_dims_SOR] = inducing_input_data_
            #         else:
            #             inducing_input_data = inducing_input_data_
            #         # print('inducing_input_data:')
            #         # np.set_printoptions(threshold=np.inf)
            #         # print(inducing_input_data_)
            #         self.grid_exact_all_range[gp_index] = torch.tensor(inducing_input_data.reshape([-1, self.gp_inputs.shape[1]]), dtype=self.dtype, device=self.device)
            #         self.gp_list[gp_index].init_inducing_inputs(self.grid_exact_all_range[gp_index], flg_train_inducing_inputs=False)
            Y_hat, var, alpha, m_X, Sigma = self.gp_list[gp_index].get_SOR_estimate(X=self.gp_inputs,
                                                                                    Y=(self.gp_output_list[gp_index] - self.norm_mean_coef_list[gp_index]) / self.norm_scale_coef_list[gp_index],
                                                                                    X_test=self.gp_inputs,
                                                                                    flg_return_Sigma=True)
            self.Sigma_SOR2_list[gp_index] = Sigma
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.inducing_input_data_list[gp_index] = self.grid_exact_all_range[gp_index]
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs


        elif self.approximation_mode == 'nystrom':
            # train the full GP
            Y_hat_full, var_full, alpha, m_X, K_X_inv = self.gp_list[gp_index].get_estimate(X = self.gp_inputs,
                                                                                            Y = (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                                            X_test = self.gp_inputs,
                                                                                            flg_return_K_X_inv = True)
            self.K_X_inv_list[gp_index] = K_X_inv
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs
            print('MSE full gp '+str(gp_index)+': ', torch.mean((self.gp_output_list[gp_index]-(Y_hat_full*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index]))**2))
            # train the nystrom GP
            w, Sigma_w = self.gp_list_nystrom[gp_index].get_parameters_inv_lemma(self.gp_inputs, (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index])
            self.w_nystrom[gp_index] = w
            self.Sigma_w_nystrom[gp_index] = Sigma_w
            Y_hat = torch.matmul(self.gp_list_nystrom[gp_index].get_phi(self.gp_inputs), self.w_nystrom[gp_index]) + self.gp_list_nystrom[gp_index].get_mean(self.gp_inputs)

        self.num_gp_inputs_list[gp_index] = self.gp_inputs_tr_list[gp_index].shape[0]
        MSE = torch.mean((self.gp_output_list[gp_index]-(Y_hat*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index]))**2)
        print('MSE gp '+str(gp_index)+': ', MSE.detach().cpu().numpy())


    def get_next_state(self, current_state, current_input, particle_pred = True):
        """
        Predict the next state given the the current state-input (batches supported).
        Method returns next state samples, together with mean and variance of the gp prediction
        """
        # Get the gp estimate
        _, _, gp_output_mean_list, gp_output_var_list = self.get_one_step_gp_out(states = current_state,
                                                                                 inputs = current_input)

        # Get the next state form gp IO and return
        return self.get_next_state_from_gp_output(current_state = current_state,
                                                  current_input = current_input,
                                                  gp_output_mean_list = gp_output_mean_list,
                                                  gp_output_var_list = gp_output_var_list,
                                                  particle_pred = particle_pred)


    def get_one_step_gp_out(self, states, inputs):
        """
        Compute input-output of the gp and performs estimation:
        The function returns (gp_inputs, gp_outputs, gp_mean_hat, gp_var_hat)
        """
        gp_inputs = self.data_to_gp_input(states = states, inputs = inputs)
        gp_outputs_list = None
        # get gp estimates
        gp_output_mean_list, gp_output_var_list = self.get_gp_estimate(gp_inputs = gp_inputs,
                                                                       gp_index_list = range(0,self.num_gp))
        return gp_inputs, gp_outputs_list, gp_output_mean_list, gp_output_var_list


    def get_next_state_parametric_model(self, current_state, current_input, W_list):
        """
        Predict the next state given the the current state-input (batches supported).
        Method returns next state samples, together with mean and variance of the gp prediction
        W = realization of the model parameters sampled from parameters distribution
        """
        # Get the gp estimate
        _, _, gp_output_mean_list = self.get_one_step_gp_out_parametric_model(states = current_state,
                                                                              inputs = current_input,
                                                                              W_list = W_list)

        # Get the next state form gp IO and return
        return self.get_next_state_from_gp_output(current_state = current_state,
                                                  current_input = current_input,
                                                  gp_output_mean_list = gp_output_mean_list,
                                                  gp_output_var_list = None,
                                                  particle_pred = False)


    def get_one_step_gp_out_parametric_model(self, states, inputs, W_list):
        """
        Compute input-output of the gp and performs estimation:
        The function returns (gp_inputs, gp_outputs, gp_mean_hat, gp_var_hat)
        """
        gp_inputs = self.data_to_gp_input(states = states, inputs = inputs)
        gp_outputs_list = None
        # get gp estimates
        gp_output_mean_list = self.get_gp_estimate_parametric_model(gp_inputs = gp_inputs,
                                                                    W_list = W_list,
                                                                    gp_index_list = range(0,self.num_gp))
        return gp_inputs, gp_outputs_list, gp_output_mean_list


    def get_gp_estimate_from_data(self, states, inputs, flg_pretrain = False, gp_index_list = None, flg_onestep = False):
        """
        Compute input-output of the gp and performs estimation:
        The function returns (gp_inputs, gp_outputs, gp_mean_hat, gp_var_hat)
        """
        # check index list
        if gp_index_list is None:
            gp_index_list = range(0,self.num_gp)
        if flg_onestep: # get one-step gp estimates
            gp_inputs = self.data_to_gp_input(states = states, inputs = inputs)
            gp_outputs_list = None
        else: # get the input-output of the gp
            gp_inputs, gp_outputs_list = self.data_to_gp_IO(states = states,
                                                            inputs = inputs)
        # pretrain gp
        if flg_pretrain:
            for gp_index in gp_index_list:
                self.pretrain_gp(gp_index = gp_index)
        # get gp estimates
        gp_output_mean_list, gp_output_var_list = self.get_gp_estimate(gp_inputs = gp_inputs,
                                                                       gp_index_list = gp_index_list)
        return gp_inputs, gp_outputs_list, gp_output_mean_list, gp_output_var_list


    def get_exact_gp_estimate(self, gp_inputs, gp_index_list = None):
        """
        Return the gp ouput (mean and variance)
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0,self.num_gp)

        est_list = [self.gp_list[i].get_estimate_from_alpha(X = self.gp_inputs_tr_list[i], 
                                                            X_test = gp_inputs,
                                                            alpha = self.alpha_list[i],
                                                            K_X_inv = self.K_X_inv_list[i]) for i in gp_index_list]
        gp_output_mean_list = [gp_out[0]*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index] for gp_index, gp_out in enumerate(est_list)]
        gp_output_var_list = [gp_out[1].reshape([-1,1])*(self.norm_scale_coef_list[gp_index]**2) for gp_index, gp_out in enumerate(est_list)]

        return gp_output_mean_list, gp_output_var_list


    def get_nystrom_gp_estimate(self, gp_inputs, gp_index_list = None):
        """
        Return the gp ouput (mean and variance) computed with the nystrom approximation
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0,self.num_gp)

        phi_list = [self.gp_list_nystrom[gp_index].get_phi(gp_inputs) for gp_index in gp_index_list]
        gp_output_mean_list = [(torch.matmul(phi_list[gp_index], self.w_nystrom[gp_index]) +
                               self.gp_list_nystrom[gp_index].get_mean(gp_inputs))*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index]
                               for gp_index in gp_index_list]
        gp_output_var_list =  [torch.sum(torch.matmul(phi_list[gp_index], self.Sigma_w_nystrom[gp_index])*(phi_list[gp_index]), dim=1).reshape(-1,1)*(self.norm_scale_coef_list[gp_index]**2)
                               for gp_index in gp_index_list]
        return gp_output_mean_list, gp_output_var_list


    def sample_nystrom_model_parameters(self, num_samples):
        """
        Sample num_samples realization of the nystrom parameters from the current posterior distribution
        """
        L_sigma_w_list  = [torch.linalg.cholesky(self.Sigma_w_nystrom[gp_index]) for gp_index in range(self.num_gp)]
        return [(self.w_nystrom[gp_index] + 
                torch.matmul(L_sigma_w_list[gp_index],
                             torch.randn([num_samples, self.w_nystrom[gp_index].shape[0], 1],
                                         dtype=self.dtype,
                                         device=self.device))).squeeze()
                for gp_index in range(self.num_gp)]


    def get_nystrom_gp_estimate_parametric_model(self, gp_inputs, W_list, gp_index_list = None):
        """
        Compute the gp estimates based on the nystrom parametric model and the parameters sampled in W_list
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0,self.num_gp)
        # initilize the output lists
        phi_list = [self.gp_list_nystrom[gp_index].get_phi(gp_inputs) for gp_index in gp_index_list]
        gp_output_mean_list = [(torch.sum(phi_list[gp_index]*W_list[gp_index], dim=1, keepdim=True) +
                               self.gp_list_nystrom[gp_index].get_mean(gp_inputs))*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index]
                               for gp_index in gp_index_list]
        return gp_output_mean_list



    def get_SOR_gp_estimate(self, gp_inputs, gp_index_list = None):
        """
        Return the gp ouput (mean and variance)
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0,self.num_gp)
        # get estimates
        est_list = [self.gp_list[gp_index].get_SOR_estimate_from_alpha(X_test = gp_inputs,
                                                                       SOR_alpha = self.alpha_list[gp_index],
                                                                       Sigma = self.Sigma_SOR_list[gp_index])
                    for gp_index in gp_index_list]
        # normalize signals
        gp_output_mean_list = [gp_out[0]*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index] for gp_index, gp_out in enumerate(est_list)]
        gp_output_var_list = [gp_out[1].reshape([-1,1])*(self.norm_scale_coef_list[gp_index]**2) for gp_index, gp_out in enumerate(est_list)]
        return gp_output_mean_list, gp_output_var_list


    def sample_SOR_model_parameters(self, num_samples):
        """
        Sample num_samples realization of the SOR parameters from the current posterior distribution
        self.alpha_list is the list of the posterior means
        self.Sigma_SOR_list is the list of the posterior variances
        """
        L_sigma_w_list  = [torch.linalg.cholesky(self.Sigma_SOR_list[gp_index]) for gp_index in range(self.num_gp)]
        return [(self.alpha_list[gp_index] + 
                torch.matmul(L_sigma_w_list[gp_index],
                             torch.randn([num_samples, self.alpha_list[gp_index].shape[0], 1],
                                         dtype=self.dtype,
                                         device=self.device))).squeeze()
                for gp_index in range(self.num_gp)]


    def get_SOR_gp_estimate_parametric_model(self, gp_inputs, W_list, gp_index_list = None):
        """
        Compute the gp estimates based on the nystrom parametric model and the parameters sampled in W_list
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0,self.num_gp)
        
        gp_output_mean_list = [(torch.sum(gp.get_covariance(gp_inputs, gp.U)*W, dim=1, keepdim=True) + gp.get_mean(gp_inputs))*norm_s+norm_m
                               for W, gp, norm_m, norm_s in zip(W_list,
                                                                self.gp_list,
                                                                self.norm_mean_coef_list,
                                                                self.norm_scale_coef_list)]

        return gp_output_mean_list


    def get_SOD_gp_estimate(self, gp_inputs, gp_index_list):
        """
        Return the gp ouput (mean and variance)
        """
        # get mean and variance estimation
        est_list = [self.gp_list[i].get_estimate_from_alpha(X = self.gp_inputs_tr_list[i], 
                                                            X_test = gp_inputs,
                                                            alpha = self.alpha_list[i],
                                                            K_X_inv = self.K_X_inv_list[i]) for i in gp_index_list]
        gp_output_mean_list = [gp_out[0]*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index] for gp_index, gp_out in enumerate(est_list)]
        gp_output_var_list = [gp_out[1].reshape([-1,1])*(self.norm_scale_coef_list[gp_index]**2) for gp_index, gp_out in enumerate(est_list)]
        return gp_output_mean_list, gp_output_var_list


    def get_SOD_proj_gp_estimate(self, gp_inputs, gp_index_list):
        """
        Return the gp ouput (mean and variance)
        """
        # get mean and variance estimation
        est_list = [self.gp_list[i].get_estimate_from_alpha(X = self.gp_inputs_tr_list[i], 
                                                            X_test = gp_inputs,
                                                            alpha = self.alpha_list[i],
                                                            K_X_inv = self.K_X_inv_list[i]) for i in gp_index_list]
        gp_output_mean_list = [gp_out[0]*self.norm_scale_coef_list[gp_index]+self.norm_mean_coef_list[gp_index] for gp_index, gp_out in enumerate(est_list)]
        gp_output_var_list = [gp_out[1].reshape([-1,1])*(self.norm_scale_coef_list[gp_index]**2) for gp_index, gp_out in enumerate(est_list)]
        return gp_output_mean_list, gp_output_var_list



    def get_SOR2_gp_estimate(self, gp_inputs, gp_index_list=None):
        """
        Return the GP output (mean and variance) using the SOR2 approximation
        """
        if gp_index_list is None:
            gp_index_list = range(0, self.num_gp)
        est_list = [self.gp_list[gp_index].get_SOR_estimate_from_alpha(X_test=gp_inputs,
                                                                       SOR_alpha=self.alpha_list[gp_index],
                                                                       Sigma=self.Sigma_SOR2_list[gp_index])
                    for gp_index in gp_index_list]
        gp_output_mean_list = [gp_out[0] * self.norm_scale_coef_list[gp_index] + self.norm_mean_coef_list[gp_index] for gp_index, gp_out in enumerate(est_list)]
        gp_output_var_list = [gp_out[1].reshape([-1, 1]) * (self.norm_scale_coef_list[gp_index] ** 2) for gp_index, gp_out in enumerate(est_list)]
        return gp_output_mean_list, gp_output_var_list

    def sample_SOR2_model_parameters(self, num_samples):
        """
        Sample num_samples realization of the SOR2 parameters from the current posterior distribution
        """
        L_sigma_w_list = [torch.linalg.cholesky(self.Sigma_SOR2_list[gp_index]) for gp_index in range(self.num_gp)]
        return [(self.alpha_list[gp_index] +
                 torch.matmul(L_sigma_w_list[gp_index],
                              torch.randn([num_samples, self.alpha_list[gp_index].shape[0], 1],
                                          dtype=self.dtype,
                                          device=self.device))).squeeze()
                for gp_index in range(self.num_gp)]

    def get_SOR2_gp_estimate_parametric_model(self, gp_inputs, W_list, gp_index_list=None):
        """
        Compute the GP estimates based on the SOR2 parametric model and the parameters sampled in W_list
        """
        if gp_index_list is None:
            gp_index_list = range(0, self.num_gp)
        gp_output_mean_list = [(torch.sum(gp.get_covariance(gp_inputs, gp.U) * W, dim=1, keepdim=True) + gp.get_mean(gp_inputs)) * norm_s + norm_m
                               for W, gp, norm_m, norm_s in zip(W_list,
                                                                self.gp_list,
                                                                self.norm_mean_coef_list,
                                                                self.norm_scale_coef_list)]
        return gp_output_mean_list

    def to(self, device):
        """
        Move the model parameters to 'device'
        """
        super(Model_learning, self).to(device)
        self.device = device
        for gp in self.gp_list:
            gp.to(device)


    def print_model(self):
        """
        Print the model
        """
        for gp_index, gp in enumerate(self.gp_list):
            print('GP '+str(gp_index+1)+':')
            gp.print_model()


    def train_gp(self, gp_index, optimization_opt_dict):
        """
        Call train_gp_likelihood
        """
        self.train_gp_likelihood(gp_index, optimization_opt_dict)
        if self.approximation_mode == 'SOR':
            print('\nSelect the SOR regressors...')
            # get the threshold
            if self.SOR_threshold_mode == 'relative':
                threshold = self.SOR_threshold* torch.sqrt(self.gp_list[gp_index].get_sigma_n_2())
            elif self.SOR_threshold_mode == 'absolute':
                threshold = self.SOR_threshold[gp_index]
            print('threshold', threshold)
            # get selection criterion
            if 'criterion' in self.approximation_dict:
                criterion = self.approximation_dict['criterion']
            else:
                criterion = 'std'
            result = None
            while result is None:
                with torch.no_grad():
                    permutation_indices = np.arange(0,self.gp_inputs.shape[0])
                    self.reg_indices_SOR_list[gp_index] = self.gp_list[gp_index].set_inducing_inputs_from_data(X = self.gp_inputs[permutation_indices,:],
                                                                                                                Y = (self.gp_output_list[gp_index][permutation_indices,:]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                                                                threshold = threshold,
                                                                                                                flg_trainable = self.approximation_dict['flg_regressors_trainable'],
                                                                                                                flg_permutation = self.approximation_dict['flg_SOR_permutation'],
                                                                                                                criterion = criterion,
                                                                                                                num_pts = self.approximation_dict['num_pts'] if 'num_pts' in self.approximation_dict.keys() else None)
                    try:
                        Y_hat, var, alpha, m_X, Sigma = self.gp_list[gp_index].get_SOR_estimate(X=self.gp_inputs,
                                                                                            Y=(self.gp_output_list[gp_index] - self.norm_mean_coef_list[gp_index]) / self.norm_scale_coef_list[gp_index],
                                                                                            X_test=self.gp_inputs,
                                                                                            flg_return_Sigma=True)
                        self.Sigma_SOR_list[gp_index] = Sigma
                    
                        L_sigma_w_list = torch.linalg.cholesky(self.Sigma_SOR_list[gp_index])
                        result = True
                    except Exception as e:
                        print(e)
                        threshold = threshold*1.1
                        print('threshold', threshold)
                        pass

            if self.approximation_dict['flg_regressors_trainable']:
                self.train_SOR_gp_likelihood(gp_index, optimization_opt_dict)
        elif self.approximation_mode == 'nystrom':
            print('Train exact GP likelihood')
            self.train_gp_likelihood(gp_index, optimization_opt_dict)
            print('Get the nystrom approximation of GP'+str(gp_index))
            nystrom_init_dict = {}
            if self.approximation_dict['retrain_likelihood']:
                nystrom_init_dict['sigma_n_init'] = np.ones(1)
            else:
                nystrom_init_dict['sigma_n_init'] = np.sqrt(self.gp_list[gp_index].get_sigma_n_2().detach().cpu().numpy())
            nystrom_init_dict['flg_train_sigma_n'] = True
            nystrom_init_dict['norm_mean_coef'] = self.gp_list[gp_index].norm_mean_coef
            nystrom_init_dict['norm_scale_coef'] = self.gp_list[gp_index].norm_scale_coef
            if self.gp_list[gp_index].flg_mean:
                nystrom_init_dict['f_mean'] = self.gp_list[gp_index].f_mean
                nystrom_init_dict['f_mean_add_par_dict'] = self.gp_list[gp_index].f_mean_add_par_dict
                if self.gp_list[gp_index].flg_pos_par_mean:
                    nystrom_init_dict['pos_par_mean_init'] = torch.exp(self.gp_list[gp_index].pos_par_mean_log).detach().clone().cpu().numpy()
                    nystrom_init_dict['flg_train_pos_par_mean'] = self.gp_list[gp_index].flg_train_pos_par_mean
                if self.gp_list[gp_index].flg_free_par_mean:
                    nystrom_init_dict['free_par_mean_init'] = self.gp_list[gp_index].free_par_mean.detach().clone().cpu().numpy()
                    nystrom_init_dict['flg_train_free_par_mean'] = self.gp_list[gp_index].flg_train_free_par_mean
            nystrom_init_dict['flg_offset'] = False
            nystrom_init_dict['scale_init'] = np.ones(1)
            nystrom_init_dict['flg_train_scale'] = False
            nystrom_init_dict['sigma_n_num'] = self.gp_list[gp_index].sigma_n_num.detach().cpu().numpy()
            if self.approximation_dict['nystrom_input_reduction_mode']=='downsampling':
                self.gp_inputs_tr_list_nystrom[gp_index] = range(0,self.gp_inputs.shape[0],self.approximation_dict['nystrom_input_downsampling_rate'])
            elif self.approximation_dict['nystrom_input_reduction_mode']=='threshold':
                # set threshold
                if self.approximation_dict['nystrom_threshold_mode'] == 'relative':
                    threshold = self.approximation_dict['nystrom_input_threshold']* torch.sqrt(self.gp_list[gp_index].get_sigma_n_2())
                elif self.approximation_dict['nystrom_threshold_mode'] == 'absolute':
                    threshold = self.approximation_dict['nystrom_input_threshold'][gp_index]
                # set reduction criterion
                if 'criterion' in self.approximation_dict:
                    criterion = self.approximation_dict['criterion']
                else:
                    criterion = 'std'
                self.gp_inputs_tr_list_nystrom[gp_index] = self.gp_list[gp_index].get_SOD(X = self.gp_inputs,
                                                                                          Y = (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index],
                                                                                          threshold = threshold,
                                                                                          flg_permutation = self.approximation_dict['flg_permutation'],
                                                                                          criterion=criterion)
            self.gp_list_nystrom[gp_index] = Sparse_GP.get_nystrom_GP(exact_GP_object=self.gp_list[gp_index],
                                                                      X_nystrom=self.gp_inputs[self.gp_inputs_tr_list_nystrom[gp_index],:],
                                                                      # X_nystrom=self.gp_inputs_tr_list_nystrom[gp_index] ,
                                                                      R_max=self.gp_inputs.shape[0],#self.nystrom_max_rank,
                                                                      init_dict=nystrom_init_dict,
                                                                      R_perc=self.R_perc)
            if self.approximation_dict['retrain_likelihood']:
                print('Retrain the likelihood...')
                # check the batch size
                batch_size = self.gp_inputs.shape[0]
                # get the dataloader
                dataset = torch.utils.data.TensorDataset(self.gp_inputs, (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index])
                trainloader = torch.utils.data.DataLoader(dataset, 
                                                          batch_size=batch_size,
                                                          shuffle=False)
                # fit the model
                f_optim = eval(optimization_opt_dict['f_optimizer'])
                self.gp_list_nystrom[gp_index].fit_model(trainloader = trainloader, 
                                                         optimizer = f_optim(self.gp_list_nystrom[gp_index].parameters()),
                                                         criterion = optimization_opt_dict['criterion'](),
                                                         N_epoch = optimization_opt_dict['N_epoch'],
                                                         N_epoch_print = optimization_opt_dict['N_epoch_print'])
        elif self.approximation_mode == 'SOR2':
            result = None
            step_scale = 1.5

            while result is None:
                try:
                    print('Set grid of GP' + str(gp_index) + ': stepscale = ' + str(step_scale))

                    # check state ranges
                    if self.state_ranges is not None:
                        ranges = self.state_ranges
                    else:
                        # set ranges from data
                        ranges = [(torch.min(self.gp_inputs[:, i]).item() * self.scale_ranges, torch.max(self.gp_inputs[:, i]).item() * self.scale_ranges) for i in range(self.gp_inputs.shape[1])]

                    # select active dims
                    ranges = [ranges[active_dims_index] for active_dims_index in self.active_dims_SOR]

                    # check step
                    if self.grid_step is not None:
                        grid_step = self.grid_step
                        grid_step = [grid_step[active_dims_index] for active_dims_index in self.active_dims_SOR]
                    else:
                        grid_step = np.exp(self.gp_list[gp_index].log_lengthscales_par.data.detach().cpu().numpy()) / step_scale

                    print('########\n')
                    print(f'grid_steps list: {grid_step}')
                    print('########\n')

                    # generate the grid (try to try differeent options from where to start the grid (min,max,center))
                    grid = self.generate_grid(ranges, grid_step, option='center')
                    grid_flat = tuple([grid[i].ravel() for i in range(len(grid))])
                    inducing_input_data_ = np.column_stack(grid_flat)
                    num_grid_points = inducing_input_data_.shape[0]
                    print('Number of inducing points:', num_grid_points)

                    # check discarded features
                    if len(self.active_dims_SOR) < self.gp_inputs.shape[1]:
                        inducing_input_data = np.zeros([num_grid_points, self.gp_inputs.shape[1]])
                        inducing_input_data[:, self.active_dims_SOR] = inducing_input_data_
                    else:
                        inducing_input_data = inducing_input_data_

                    self.grid_exact_all_range[gp_index] = torch.tensor(inducing_input_data.reshape([-1, self.gp_inputs.shape[1]]), dtype=self.dtype, device=self.device)
                    self.gp_list[gp_index].init_inducing_inputs(self.grid_exact_all_range[gp_index], flg_train_inducing_inputs=False)
                    Y_hat, var, alpha, m_X, Sigma = self.gp_list[gp_index].get_SOR_estimate(X=self.gp_inputs,
                                                                                            Y=(self.gp_output_list[gp_index] - self.norm_mean_coef_list[gp_index]) / self.norm_scale_coef_list[gp_index],
                                                                                            X_test=self.gp_inputs,
                                                                                            flg_return_Sigma=True)
                    self.Sigma_SOR2_list[gp_index] = Sigma
                    L_sigma_w_list = torch.linalg.cholesky(self.Sigma_SOR2_list[gp_index])
                    result = True
                except:
                    step_scale = step_scale * 0.9 #0.9 #0.95
                    pass



    def train_gp_likelihood(self, gp_index, optimization_opt_dict):
        """
        Train the gp with index gp_index optimizing the likelihood
        """
        print('\n\nTrain GP model '+str(gp_index))
        # check the batch size
        num_samples = self.gp_inputs.shape[0]
        batch_size = num_samples
        drop_last = False
        flg_shuffle = False
        if 'batch_size' in optimization_opt_dict: 
            if optimization_opt_dict['batch_size'] <= num_samples:
                batch_size = optimization_opt_dict['batch_size']
                flg_shuffle = True
                drop_last =  num_samples%batch_size > 0
        # check number of epochs
        if 'N_steps' in optimization_opt_dict:
            N_epoch = 1 + int(optimization_opt_dict['N_steps']/int(num_samples/batch_size))
            N_epoch_print = int(N_epoch/3)
        else:
            N_epoch = optimization_opt_dict['N_epoch']
            N_epoch_print =optimization_opt_dict['N_epoch_print']
        print('num_samples', num_samples)
        print('N_epoch', N_epoch)
        print('N_epoch_print', N_epoch_print)
            
        # get the dataloader
        if self.flg_norm:
            with torch.no_grad():
                m_X = self.gp_list[gp_index].get_mean(self.gp_inputs)*self.gp_list[gp_index].norm_scale_coef
            if self.flg_norm_mean:
                self.norm_mean_coef_list[gp_index] = torch.mean(self.gp_output_list[gp_index]-m_X)
            self.norm_scale_coef_list[gp_index] = torch.std(self.gp_output_list[gp_index]-m_X)
            self.gp_list[gp_index].norm_mean_coef = self.norm_mean_coef_list[gp_index]
            self.gp_list[gp_index].norm_scale_coef = self.norm_scale_coef_list[gp_index]
        dataset = torch.utils.data.TensorDataset(self.gp_inputs, (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index])
        trainloader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=batch_size,
                                                  shuffle=flg_shuffle,
                                                  drop_last=drop_last)
        # fit the model
        f_optim = eval(optimization_opt_dict['f_optimizer'])
        self.gp_list[gp_index].fit_model(trainloader = trainloader,
                                         optimizer = f_optim(self.gp_list[gp_index].parameters()),
                                         criterion = optimization_opt_dict['criterion'](),
                                         N_epoch = N_epoch,
                                         N_epoch_print = N_epoch_print)


    def train_SOR_gp_likelihood(self, gp_index, optimization_opt_dict):
        """
        Train the gp with index gp_index optimizing the likelihood
        """
        # check the batch size
        batch_size = self.gp_inputs.shape[0]
            
        # get the dataloader
        dataset = torch.utils.data.TensorDataset(self.gp_inputs, (self.gp_output_list[gp_index]-self.norm_mean_coef_list[gp_index])/self.norm_scale_coef_list[gp_index])
        trainloader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=batch_size,
                                                  shuffle=False)
        # fit the model
        f_optim = eval(optimization_opt_dict['f_optimizer'])
        self.gp_list[gp_index].fit_SOR_model(trainloader = trainloader, 
                                             optimizer = f_optim(self.gp_list[gp_index].parameters()),
                                             criterion = optimization_opt_dict['criterion'](),
                                             N_epoch = optimization_opt_dict['N_epoch']*2,
                                             N_epoch_print = optimization_opt_dict['N_epoch_print'])

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        raise NotImplementedError()
        return None


    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        In this basic implementation gp inputs are the concatenation
        of states and inputs
        """
        return torch.cat([states,inputs],1)


    def data_to_gp_output(self, states):
        """
        Returns a list with the gp ouputs given the states.
        In this basic implementation gp output are the delta in each state dimension
        """
        return [(states[1:,i]-states[:-1,i]).reshape([-1,1]) for i in self.evolution_indices]


    def data_to_gp_IO(self, states, inputs):
        """
        Returns the GP dataset given states and inputs
        """
        return self.data_to_gp_input(states,inputs)[:-1,:], self.data_to_gp_output(states)


    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list, particle_pred = True):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        """
        next_states = torch.zeros(current_state.shape, dtype=self.dtype, device=self.device)
        if particle_pred == True:
            #  mean and variance of delta distribution
            delta_mean = torch.cat(gp_output_mean_list, 1)
            delta_var = torch.cat(gp_output_var_list, 1)
            delta_var[delta_var <= 0] = torch.finfo(self.dtype).eps  # should avoid negative values problem
            # sample delta from distribution
            delta_distribution = Normal(delta_mean, torch.sqrt(delta_var))
            delta_sample = delta_distribution.rsample()
            # delta_sample = delta_mean + torch.sqrt(delta_var)*torch.randn(delta_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_mean = torch.cat(gp_output_mean_list, 1)
            delta_var = None
            delta_sample = delta_mean
        # get the next state
        next_states[:, self.evolution_indices] = current_state[:, self.evolution_indices] + delta_sample
        # These entries remain equal through the whole process
        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]
        # return the next state and the delta distribution
        return next_states, delta_mean, delta_var

    # def generate_grid(self, ranges, length_scales):
    #     """
    #     Generate a grid of inducing points from the given ranges and length scales.
    #     """
    #     grids = [np.arange(r[0], r[1], ls) for r, ls in zip(ranges, length_scales)]
    #     return np.meshgrid(*grids)


    def generate_grid(self, ranges, length_scales, option='center'):
        """
        Generate a grid of inducing points from the given ranges and length scales.

        Parameters:
        ranges: list of tuples
            Each tuple represents the (min, max) range for a dimension.
        length_scales: list of floats
            The length scales that define the step size for each dimension.
        option: str
            - 'min': start from the minimum value of the range.
            - 'max': start from the maximum value of the range.
            - 'center': start from the center of the range.

        Returns:
        A meshgrid generated based on the specified option, stopping at the first point that exceeds the min or max.
        """
        grids = []

        for r, ls in zip(ranges, length_scales):
            min_val, max_val = r
            if option == 'min':
                grid = [min_val]
                while True:
                    next_val = grid[-1] + ls
                    if next_val > max_val:
                        grid.append(next_val)  # Include the first point that exceeds the max
                        break
                    grid.append(next_val)
                grid = np.array(grid)
            elif option == 'max':
                grid = [max_val]
                while True:
                    next_val = grid[-1] - ls
                    if next_val < min_val:
                        grid.append(next_val)  # Include the first point that exceeds the min
                        break
                    grid.append(next_val)
                grid = np.array(grid[::-1])  # Reverse to maintain increasing order
            elif option == 'center':
                center = (min_val + max_val) / 2
                positive_side = [center]
                while True:
                    next_val = positive_side[-1] + ls
                    if next_val > max_val:
                        positive_side.append(next_val)  # Include the first point that exceeds the max
                        break
                    positive_side.append(next_val)

                negative_side = [center]
                while True:
                    next_val = negative_side[-1] - ls
                    if next_val < min_val:
                        negative_side.append(next_val)  # Include the first point that exceeds the min
                        break
                    negative_side.append(next_val)

                negative_side = negative_side[::-1]  # Reverse to maintain consistency
                grid = np.concatenate((negative_side[:-1], positive_side))  # Avoid duplicating center
                print('r', r)
                print('ls', ls)
                print('grid', grid.shape)
            elif option == 'center2':
                center = (min_val + max_val) / 2
                positive_side = [center+ls/2]
                while True:
                    next_val = positive_side[-1] + ls
                    if next_val > max_val:
                        #positive_side.append(next_val)  # Include the first point that exceeds the max
                        break
                    positive_side.append(next_val)

                negative_side = [center-ls/2]
                while True:
                    next_val = negative_side[-1] - ls
                    if next_val < min_val:
                        #negative_side.append(next_val)  # Include the first point that exceeds the min
                        break
                    negative_side.append(next_val)

                negative_side = negative_side[::-1]  # Reverse to maintain consistency
                grid = np.concatenate((negative_side, positive_side))
                print('r', r)
                print('ls', ls)
                print('grid', grid.shape)
            else:
                raise ValueError(f"Invalid option '{option}'. Choose from 'min', 'max', 'center', or 'center2'.")

            grids.append(grid)

        return np.meshgrid(*grids)

    # def generate_grid(self, ranges, length_scales, option='center'):
    #     """
    #     Generate a grid of inducing points from the given ranges and length scales.

    #     Parameters:
    #     ranges: list of tuples
    #         Each tuple represents the (min, max) range for a dimension.
    #     length_scales: list of floats
    #         The length scales that define the step size for each dimension.
    #     option: str
    #         - 'min': start from the minimum value of the range.
    #         - 'max': start from the maximum value of the range.
    #         - 'center': start from the center of the range.

    #     Returns:
    #     A meshgrid generated based on the specified option, ensuring min and max are included.
    #     """
    #     grids = []

    #     for r, ls in zip(ranges, length_scales):
    #         min_val, max_val = r
    #         if option == 'min':
    #             grid = np.arange(min_val, max_val, ls)
    #             if grid[-1] < max_val:
    #                 grid = np.append(grid, max_val)  # Ensure max_val is included
    #         elif option == 'max':
    #             grid = np.arange(max_val, min_val, -ls)
    #             if grid[-1] > min_val:
    #                 grid = np.append(grid, min_val)  # Ensure min_val is included
    #         elif option == 'center':
    #             center = (min_val + max_val) / 2
    #             positive_side = np.arange(center, max_val, ls)
    #             negative_side = np.arange(center, min_val, -ls)[::-1]  # Reverse for consistency
    #             if positive_side[-1] < max_val:
    #                 positive_side = np.append(positive_side, max_val)  # Ensure max_val is included
    #             if negative_side[0] > min_val:
    #                 negative_side = np.insert(negative_side, 0, min_val)  # Ensure min_val is included
    #             grid = np.concatenate((negative_side, positive_side[1:]))  # Avoid duplicating center
    #         else:
    #             raise ValueError(f"Invalid option '{option}'. Choose from 'min', 'max', or 'center'.")

    #         grids.append(grid)

    #     return np.meshgrid(*grids)

class Model_learning_RBF(Model_learning):
    """
    Model learning class with all the gp given by RBF kernel
    """
    def __init__(self, num_gp, init_dict_list,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'), flg_norm = False, flg_norm_mean = False,
                 flg_parametric_model = False, evolution_indices=None, cnst_indices = None):
        super(Model_learning_RBF, self).__init__(num_gp = num_gp,
                                                 init_dict_list = init_dict_list,
                                                 approximation_mode = approximation_mode,
                                                 approximation_dict = approximation_dict,
                                                 dtype = dtype,
                                                 device = device,
                                                 flg_norm = flg_norm, 
                                                 flg_norm_mean = flg_norm_mean,
                                                 flg_parametric_model = flg_parametric_model,
                                                 evolution_indices=evolution_indices,
                                                 cnst_indices=cnst_indices)

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        return SGP.RBF(**init_dict)


class Model_learning_RBF_angle_state(Model_learning):
    """
    Model learning class with all the gp given by RBF kernel and
    the possibility to use sin-cos angle representation in GP inputs
    """
    def __init__(self, num_gp, init_dict_list, angle_indeces, not_angle_indeces,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, evolution_indices=None, cnst_indices=None,):
        super(Model_learning_RBF_angle_state, self).__init__(num_gp = num_gp,
                                                             init_dict_list = init_dict_list,
                                                             approximation_mode = approximation_mode,
                                                             approximation_dict = approximation_dict,
                                                             dtype = dtype,
                                                             device = device,
                                                             flg_norm = flg_norm,
                                                             flg_norm_mean = flg_norm_mean,
                                                             flg_parametric_model = flg_parametric_model,
                                                             evolution_indices=evolution_indices,
                                                             cnst_indices=cnst_indices)
        self.angle_indeces = angle_indeces
        self.not_angle_indeces = not_angle_indeces

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        return SGP.RBF(**init_dict)

    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        Extended state is considered: [x, x_dot, theta_dot, sin(theta), cos(theta)]
        where state = [x, theta], with theta angular components of the state and x the other states
        """        
        extended_states = torch.cat([states[:,self.not_angle_indeces],
                                     torch.sin(states[:,self.angle_indeces]),
                                     torch.cos(states[:,self.angle_indeces])],1)

        return torch.cat([extended_states, inputs],1)



class Model_learning_RBF_angle_state_u_delay(Model_learning_RBF_angle_state):
    """
    Model learning class with all the gp given by RBF kernel and
    the possibility to use sin-cos angle representation in GP inputs
    """
    def __init__(self, num_gp, init_dict_list, angle_indeces, not_angle_indeces, u_delay_indices,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, evolution_indices=None, cnst_indices=None):
        super(Model_learning_RBF_angle_state_u_delay, self).__init__(num_gp=num_gp,
                                                                     init_dict_list=init_dict_list,
                                                                     angle_indeces=angle_indeces,
                                                                     not_angle_indeces=not_angle_indeces,
                                                                     approximation_mode = approximation_mode,
                                                                     approximation_dict = approximation_dict,
                                                                     dtype = dtype,
                                                                     device = device,
                                                                     flg_norm=flg_norm,
                                                                     flg_norm_mean=flg_norm_mean,
                                                                     flg_parametric_model = flg_parametric_model,
                                                                     evolution_indices=evolution_indices,
                                                                     cnst_indices=cnst_indices)
        self.u_delay_indices = u_delay_indices


    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list, particle_pred = True):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        """
        next_states = torch.zeros(current_state.shape, dtype=self.dtype, device=self.device)
        if particle_pred == True:
            #  mean and variance of delta distribution
            delta_mean = torch.cat(gp_output_mean_list, 1)
            delta_var = torch.cat(gp_output_var_list, 1)
            delta_var[delta_var <= 0] = torch.finfo(self.dtype).eps  # should avoid negative values problem
            # sample delta from distribution
            delta_distribution = Normal(delta_mean, torch.sqrt(delta_var))
            delta_sample = delta_distribution.rsample()
            # delta_sample = delta_mean + torch.sqrt(delta_var)*torch.randn(delta_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_mean = torch.cat(gp_output_mean_list, 1)
            delta_var = None
            delta_sample = delta_mean
        # get the next state
        next_states[:, self.evolution_indices] = current_state[:, self.evolution_indices] + delta_sample
        # These entries remain equal through the whole process
        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]
        # update past inputs
        next_states[:, self.u_delay_indices[0]] = current_input
        for m in range(len(self.u_delay_indices)-1):
            next_states[:,self.u_delay_indices[m+1]] = current_state[:,self.u_delay_indices[m]].clone()
        # next_states[:, self.u_delay_indices[0:1]] = current_input
        # next_states[:, self.u_delay_indices[1:]] = current_state[:, self.u_delay_indices[:-1]]
        # return the next state and the delta distribution
        return next_states, delta_mean, delta_var


class Model_learning_RBF_MPK_angle_state_u_delay(Model_learning_RBF_angle_state_u_delay):
    """docstring for ClassName"""
    def __init__(self, num_gp, init_dict_list, angle_indeces, not_angle_indeces, u_delay_indices,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, evolution_indices=None, cnst_indices=None):
        super(Model_learning_RBF_MPK_angle_state_u_delay, self).__init__(num_gp=num_gp,
                                                                         init_dict_list=init_dict_list,
                                                                         angle_indeces=angle_indeces,
                                                                         not_angle_indeces=not_angle_indeces,
                                                                         u_delay_indices=u_delay_indices,
                                                                         approximation_mode = approximation_mode,
                                                                         approximation_dict = approximation_dict,
                                                                         dtype = dtype,
                                                                         device = device,
                                                                         flg_norm=flg_norm,
                                                                         flg_norm_mean=flg_norm_mean,
                                                                         flg_parametric_model = flg_parametric_model,
                                                                         evolution_indices=evolution_indices,
                                                                         cnst_indices=cnst_indices)

    def get_gp(self, gp_index, init_dict):
      gp_list = []
      gp_list.append(SGP.RBF(**init_dict[0]))
      gp_list.append(Sparse_GP.get_Volterra_MPK_GP(**init_dict[1]))
      return GP.Sum_Independent_GP(*gp_list)




class Model_learning_RBF_MPK_angle_state(Model_learning_RBF_angle_state):
    """
    Model learning class with all the gp given by a combination of RBF and MP kernel and
    the possibility to use sin-cos angle representation in GP inputs
    """
    def __init__(self, num_gp, init_dict_list,
                 angle_indeces, not_angle_indeces,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'),flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False):
      super(Model_learning_RBF_MPK_angle_state, self).__init__(num_gp = num_gp, 
                                                               init_dict_list = init_dict_list, 
                                                               angle_indeces = angle_indeces,
                                                               not_angle_indeces = not_angle_indeces,
                                                               approximation_mode = approximation_mode,
                                                               approximation_dict = approximation_dict,
                                                               dtype = dtype, 
                                                               device = device,
                                                               flg_norm = flg_norm,
                                                               flg_norm_mean = flg_norm_mean,
                                                               flg_parametric_model = flg_parametric_model)

    def get_gp(self, gp_index, init_dict):
        gp_list = []
        gp_list.append(SGP.RBF(**init_dict[0]))
        gp_list.append(Sparse_GP.get_Volterra_MPK_GP(**init_dict[1]))
        return GP.Sum_Independent_GP(*gp_list)  


class Speed_Model_learning_RBF_angle_state(Model_learning):
    """
    Speed model learning class with all the gp given by RBF kernel.
    The GP model predicts speed changes. vel_indeces and not_vel_indeces are related:
    1st-index-state in vel_indeces is the derivative of 1st-index-state in not_vel_indeces.
    """
    def __init__(self, num_gp, init_dict_list, T_sampling,
                 angle_indeces, not_angle_indeces, vel_indeces, not_vel_indeces,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'),flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, cnst_indices=None, evolution_indices=None,
                 gp_adaptive_init = False):
        super(Speed_Model_learning_RBF_angle_state, self).__init__(num_gp = num_gp,
                                                                   init_dict_list = init_dict_list,
                                                                   approximation_mode = approximation_mode,
                                                                   approximation_dict = approximation_dict,
                                                                   dtype = dtype,
                                                                   device = device,
                                                                   flg_norm = flg_norm,
                                                                   flg_norm_mean = flg_norm_mean,
                                                                   flg_parametric_model = flg_parametric_model,
                                                                   cnst_indices=cnst_indices,
                                                                   evolution_indices=evolution_indices,
                                                                   gp_adaptive_init=gp_adaptive_init)
        self.vel_indeces = vel_indeces
        self.not_vel_indeces = not_vel_indeces
        self.angle_indeces = angle_indeces
        self.not_angle_indeces = not_angle_indeces
        self.T_sampling = T_sampling


    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        return SGP.RBF(**init_dict)
    
    
    def data_to_gp_output(self, states):
        """
        Returns a list with the gp ouputs given the states.
        GP outputs are the speed changes.
        """
        
        return [(states[1:,i]-states[:-1,i]).reshape([-1,1]) for i in self.vel_indeces]


    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        sin-cos extended state is the gp input vector.
        """        
        extended_states = torch.cat([states[:,self.not_angle_indeces],
                                     torch.sin(states[:,self.angle_indeces]),
                                     torch.cos(states[:,self.angle_indeces])],1)
        return torch.cat([extended_states,inputs],1)
  

    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list, particle_pred=True):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        by integrating the speed changes (GP outputs)
        """

        # preallocate variables
        next_states = torch.zeros(current_state.shape,dtype = self.dtype, device = self.device)

        if particle_pred==True:
            #  mean and variance of delta speed distribution
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            # delta_vel_mean = torch.nan_to_num(torch.cat(gp_output_mean_list, 1))
            delta_vel_var = torch.cat(gp_output_var_list,1)
            # delta_vel_var = torch.nan_to_num(torch.cat(gp_output_var_list, 1))
            # sample delta speed from distribution

            # delta_vel_var = torch.nan_to_num(delta_vel_var, nan=0.0) # TODO: fix here
            delta_vel_std = torch.sqrt(delta_vel_var)
            delta_vel_std = torch.max(delta_vel_std, torch.ones_like(delta_vel_var)*10**(-15))  # TODO: fix negative values problem
            # delta_vel_std = torch.nan_to_num(delta_vel_std, nan=10**(-15)) # TODO: fix here

            # delta_vel_mean = torch.nan_to_num(delta_vel_mean, 0.0)

            delta_speed_distribution = Normal(delta_vel_mean, delta_vel_std)
            delta_speed_sample = delta_speed_distribution.rsample()
            # delta_speed_sample = delta_vel_mean + torch.sqrt(delta_vel_var)*torch.randn(delta_vel_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            delta_vel_var = None
            delta_speed_sample = delta_vel_mean

        # compute next states
        next_states[:,self.vel_indeces] = current_state[:,self.vel_indeces] + delta_speed_sample
        next_states[:,self.not_vel_indeces] = current_state[:,self.not_vel_indeces]+\
                                               self.T_sampling*current_state[:,self.vel_indeces]+\
                                               self.T_sampling/2*delta_speed_sample
        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]
        return next_states, delta_vel_mean, delta_vel_var




class Speed_Model_learning_RBF_angle_state_u_delay(Speed_Model_learning_RBF_angle_state):
    """Speed_Model_learning_RBF_angle_state with input delay"""
    def __init__(self, num_gp, init_dict_list, T_sampling, u_delay_indices,
                 angle_indeces, not_angle_indeces, vel_indeces, pos_indices,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'),flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
        super(Speed_Model_learning_RBF_angle_state_u_delay, self).__init__(num_gp=num_gp,
                                                                           init_dict_list=init_dict_list,
                                                                           T_sampling=T_sampling,
                                                                           angle_indeces=angle_indeces,
                                                                           not_angle_indeces=not_angle_indeces,
                                                                           vel_indeces=vel_indeces,
                                                                           not_vel_indeces=pos_indices,
                                                                           cnst_indices=cnst_indices,
                                                                           approximation_mode = approximation_mode,
                                                                           approximation_dict = approximation_dict,
                                                                           dtype = dtype,
                                                                           device = device,
                                                                           flg_norm = flg_norm,
                                                                           flg_norm_mean = flg_norm_mean,
                                                                           flg_parametric_model = flg_parametric_model,
                                                                           evolution_indices=evolution_indices)
        self.u_delay_indices = u_delay_indices


    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list, particle_pred=True):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        by integrating the speed changes (GP outputs)
        """

        # preallocate variables
        next_states = torch.zeros(current_state.shape,dtype = self.dtype, device = self.device)

        if particle_pred==True:
            #  mean and variance of delta speed distribution
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            # delta_vel_mean = torch.nan_to_num(torch.cat(gp_output_mean_list, 1))
            delta_vel_var = torch.cat(gp_output_var_list,1)
            delta_vel_var[delta_vel_var <= 0] = torch.finfo(self.dtype).eps
            # sample delta speed from distribution
            delta_speed_distribution = Normal(delta_vel_mean, torch.sqrt(delta_vel_var))
            delta_speed_sample = delta_speed_distribution.rsample()
            # delta_speed_sample = delta_vel_mean + torch.sqrt(delta_vel_var)*torch.randn(delta_vel_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            delta_vel_var = None
            delta_speed_sample = delta_vel_mean

        # compute next states
        next_states[:,self.vel_indeces] = current_state[:,self.vel_indeces] + delta_speed_sample
        next_states[:,self.not_vel_indeces] = current_state[:,self.not_vel_indeces]+\
                                               self.T_sampling*current_state[:,self.vel_indeces]+\
                                               self.T_sampling/2*delta_speed_sample
        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]

        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]
        if type(self.u_delay_indices[0]) == int:
            # print(next_states[:, self.u_delay_indices[0]].size())
            # print(current_input.size())
            next_states[:, self.u_delay_indices[0]] = current_input[:, 0]
        else:
            next_states[:, self.u_delay_indices[0]] = current_input
        for m in range(len(self.u_delay_indices)-1):
            next_states[:,self.u_delay_indices[m+1]] = current_state[:,self.u_delay_indices[m]].clone()
        # next_states[:, self.u_delay_indices[0:1]] = current_input
        # next_states[:, self.u_delay_indices[1:]] = current_state[:, self.u_delay_indices[:-1]]
        # next_states[:, self.u_delay_indices[0:1]] = current_input
        # next_states[:, self.u_delay_indices[1:]] = current_state[:, self.u_delay_indices[:-1]]
        return next_states, delta_vel_mean, delta_vel_var



class Speed_Model_learning_RBF_angle_state_delta_u(Speed_Model_learning_RBF_angle_state):
    """Speed_Model_learning_RBF_angle_state with input delay"""
    def __init__(self, num_gp, init_dict_list, T_sampling, last_u_indices,
                 angle_indeces, not_angle_indeces, vel_indeces, pos_indices,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'),flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
        super(Speed_Model_learning_RBF_angle_state_delta_u, self).__init__(num_gp=num_gp,
                                                                           init_dict_list=init_dict_list,
                                                                           T_sampling=T_sampling,
                                                                           angle_indeces=angle_indeces,
                                                                           not_angle_indeces=not_angle_indeces,
                                                                           vel_indeces=vel_indeces,
                                                                           not_vel_indeces=pos_indices,
                                                                           cnst_indices=cnst_indices,
                                                                           approximation_mode = approximation_mode,
                                                                           approximation_dict = approximation_dict,
                                                                           dtype = dtype,
                                                                           device = device,
                                                                           flg_norm = flg_norm,
                                                                           flg_norm_mean = flg_norm_mean,
                                                                           flg_parametric_model = flg_parametric_model,
                                                                           evolution_indices=evolution_indices)
        self.last_u_indices = last_u_indices

    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        sin-cos extended state is the gp input vector.
        """
        extended_states = torch.cat([states[:,self.not_angle_indeces],
                                     torch.sin(states[:,self.angle_indeces]),
                                     torch.cos(states[:,self.angle_indeces])],1)
        # print(inputs.shape)
        # print(states.shape)
        # print(states[:, self.last_u_indices].shape)
        actual_input = inputs+states[:, self.last_u_indices]
        gp_in = torch.cat([extended_states, actual_input], 1)
        # print(gp_in.shape)
        return gp_in

    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list, particle_pred=True):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        by integrating the speed changes (GP outputs)
        """

        # preallocate variables
        next_states = torch.zeros(current_state.shape,dtype = self.dtype, device = self.device)

        if particle_pred==True:
            #  mean and variance of delta speed distribution
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            # delta_vel_mean = torch.nan_to_num(torch.cat(gp_output_mean_list, 1))
            delta_vel_var = torch.cat(gp_output_var_list,1)
            delta_vel_var[delta_vel_var <= 0] = torch.finfo(self.dtype).eps
            # sample delta speed from distribution
            delta_speed_distribution = Normal(delta_vel_mean, torch.sqrt(delta_vel_var))
            delta_speed_sample = delta_speed_distribution.rsample()
            # delta_speed_sample = delta_vel_mean + torch.sqrt(delta_vel_var)*torch.randn(delta_vel_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            delta_vel_var = None
            delta_speed_sample = delta_vel_mean

        # compute next states
        next_states[:,self.vel_indeces] = current_state[:,self.vel_indeces] + delta_speed_sample
        next_states[:,self.not_vel_indeces] = current_state[:,self.not_vel_indeces]+\
                                               self.T_sampling*current_state[:,self.vel_indeces]+\
                                               self.T_sampling/2*delta_speed_sample
        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]
        next_states[:, self.last_u_indices] = current_state[:, self.last_u_indices] + current_input

        return next_states, delta_vel_mean, delta_vel_var


class Speed_Model_learning_RBF_angle_state_u_delay_friction(Speed_Model_learning_RBF_angle_state_u_delay):
    """docstring for Speed_Model_learning_RBF_angle_state_u_delay_friction"""
    def __init__(self, num_gp, init_dict_list, T_sampling, u_delay_indices,
                 angle_indeces, not_angle_indeces, vel_indeces, pos_indices,
                 extented_state_vel_indices_list,
                 approximation_mode = None, approximation_dict = None,
                 dtype = torch.float64, device = torch.device('cpu'),flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
        self.extented_state_vel_indices_list = extented_state_vel_indices_list
        super(Speed_Model_learning_RBF_angle_state_u_delay_friction, self).__init__(num_gp=num_gp,
                                                                                    init_dict_list=init_dict_list,
                                                                                    T_sampling=T_sampling,
                                                                                    u_delay_indices=u_delay_indices,
                                                                                    angle_indeces=angle_indeces,
                                                                                    not_angle_indeces=not_angle_indeces,
                                                                                    vel_indeces=vel_indeces,
                                                                                    pos_indices=pos_indices,
                                                                                    approximation_mode = approximation_mode,
                                                                                    approximation_dict = approximation_dict,
                                                                                    dtype = dtype,
                                                                                    device = device,
                                                                                    flg_norm=flg_norm,
                                                                                    flg_norm_mean=flg_norm_mean,
                                                                                    flg_parametric_model = flg_parametric_model,
                                                                                    cnst_indices=cnst_indices,
                                                                                    evolution_indices=evolution_indices)


    def get_gp(self, gp_index, init_dict):
        # return SGP.RBF(**init_dict)
        gp_list = []
        gp_list.append(SGP.RBF(**init_dict))
        f_cov = cov.diagonal_covariance_ARD
        gp_list.append(Sparse_GP.Linear_GP(active_dims = self.extented_state_vel_indices_list[gp_index],
                                           f_transform=lambda x:x,
                                           f_add_par_list=[],
                                           sigma_n_init=None,
                                           flg_train_sigma_n=False,
                                           f_mean=None,
                                           f_mean_add_par_dict={},
                                           pos_par_mean_init=None,
                                           flg_train_pos_par_mean=False,
                                           free_par_mean_init=None,
                                           flg_train_free_par_mean=False,
                                           norm_mean_coef=0.,
                                           norm_scale_coef=1.,
                                           Sigma_function=f_cov,
                                           Sigma_f_additional_par_list=[],
                                           Sigma_pos_par_init=np.ones(2),
                                           flg_train_Sigma_pos_par=True,
                                           Sigma_free_par_init=None,
                                           flg_train_Sigma_free_par=False,
                                           flg_offset=False,
                                           scale_init=np.ones(1),
                                           flg_train_scale=False,
                                           name='Linear friction',
                                           dtype=torch.float64,
                                           sigma_n_num=None,
                                           device=self.device))
        return GP.Sum_Independent_GP(*gp_list)


    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        sin-cos extended state is the gp input vector.
        """
        extended_states = torch.cat([states[:,self.not_angle_indeces],
                                     # torch.atan(100*states[:,self.vel_indeces]),
                                     torch.sign(states[:,self.vel_indeces]),
                                     torch.sin(states[:,self.angle_indeces]),
                                     torch.cos(states[:,self.angle_indeces])],1)
        return torch.cat([extended_states,inputs],1)




class Speed_Model_learning_RBF_MPK_angle_state_u_delay(Speed_Model_learning_RBF_angle_state_u_delay):
    """
    Models the velocity of each delta of velocity with a RBF+MPK
    """
    def __init__(self, num_gp, init_dict_list, T_sampling, u_delay_indices,
                   angle_indeces, not_angle_indeces, vel_indeces, pos_indices,
                   approximation_mode = None, approximation_dict = None,
                   dtype=torch.float64, device=torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                   flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
      super(Speed_Model_learning_RBF_MPK_angle_state_u_delay, self).__init__(num_gp = num_gp,
                                                                             init_dict_list = init_dict_list,
                                                                             T_sampling = T_sampling,
                                                                             u_delay_indices=u_delay_indices,
                                                                             angle_indeces = angle_indeces,
                                                                             not_angle_indeces = not_angle_indeces,
                                                                             vel_indeces = vel_indeces,
                                                                             pos_indices = pos_indices,
                                                                             approximation_mode = approximation_mode,
                                                                             approximation_dict = approximation_dict,
                                                                             dtype = dtype,
                                                                             device = device,
                                                                             flg_norm = flg_norm,
                                                                             flg_norm_mean = flg_norm_mean,
                                                                             flg_parametric_model = flg_parametric_model,
                                                                             cnst_indices=cnst_indices,
                                                                             evolution_indices=evolution_indices)

    def get_gp(self, gp_index, init_dict):
      gp_list = []
      gp_list.append(SGP.RBF(**init_dict[0]))
      gp_list.append(Sparse_GP.get_Volterra_MPK_GP(**init_dict[1]))
      return GP.Sum_Independent_GP(*gp_list)



class Speed_Model_learning_RBF(Model_learning):
    """
    Speed model learning class with all the gp given by RBF kernel.
    The GP model predicts speed changes. vel_indeces and not_vel_indeces are related:
    1st-index-state in vel_indeces is the derivative of 1st-index-state in not_vel_indeces.
    """

    def __init__(self, num_gp, init_dict_list, T_sampling, vel_indeces, not_vel_indeces, cnst_indices=None,
                 approximation_mode=None, approximation_dict=None,
                 dtype=torch.float64, device=torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model=False, integration_cond=None, gp_adaptive_init=False):
        super(Speed_Model_learning_RBF, self).__init__(num_gp=num_gp,
                                                       init_dict_list=init_dict_list,
                                                       approximation_mode=approximation_mode,
                                                       approximation_dict=approximation_dict,
                                                       dtype=dtype, cnst_indices=cnst_indices,
                                                       device=device,
                                                       flg_norm=flg_norm,
                                                       flg_norm_mean=flg_norm_mean,
                                                       flg_parametric_model=flg_parametric_model,
                                                       gp_adaptive_init=gp_adaptive_init)

        self.vel_indeces = vel_indeces
        self.not_vel_indeces = not_vel_indeces
        self.T_sampling = T_sampling
        self.integration_cond = integration_cond

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        return SGP.RBF(**init_dict)

    def data_to_gp_output(self, states):
        """
        Returns a list with the gp ouputs given the states.
        GP outputs are the speed changes.
        """

        return [(states[1:, i] - states[:-1, i]).reshape([-1, 1]) for i in self.vel_indeces]

    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list, particle_pred=True):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        by integrating the speed changes (GP outputs)
        """

        # preallocate variables
        next_states = torch.zeros(current_state.shape, dtype=self.dtype, device=self.device)
        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]
        # These entries remain equal through the whole process

        if particle_pred == True:
            #  mean and variance of delta speed distribution
            delta_vel_mean = torch.cat(gp_output_mean_list, 1)
            delta_vel_var = torch.cat(gp_output_var_list, 1)
            # sample delta speed from distribution
            delta_speed_distribution = Normal(delta_vel_mean, torch.sqrt(delta_vel_var))
            delta_speed_sample = delta_speed_distribution.rsample()
            # delta_speed_sample = delta_vel_mean + torch.sqrt(delta_vel_var)*torch.randn(delta_vel_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_vel_mean = torch.cat(gp_output_mean_list, 1)
            delta_vel_var = None
            delta_speed_sample = delta_vel_mean

        # compute next states
        next_states[:, self.vel_indeces] = current_state[:, self.vel_indeces] + delta_speed_sample
        next_states[:, self.not_vel_indeces] = current_state[:, self.not_vel_indeces]

        delta_pos_sample = self.T_sampling * current_state[:, self.vel_indeces] + self.T_sampling / 2 * delta_speed_sample

        # print('\n\nPos samp: {}'.format(current_state[:, self.not_vel_indeces]))
        # print('Delta pos samp: {}'.format(delta_pos_sample))
        # print('Delta speed samp: {}'.format(delta_speed_sample))#

        if self.integration_cond is None:
            next_states[:, self.not_vel_indeces] += delta_pos_sample
        else:
            for k in range(len(self.not_vel_indeces)):
                next_states[:, self.not_vel_indeces[k]] += delta_pos_sample[:, k] * self.integration_cond(
                    current_state).reshape(list(next_states[:, self.not_vel_indeces[k]].size()))


        return next_states, delta_vel_mean, delta_vel_var


class Speed_Model_learning_RBF_MPK(Speed_Model_learning_RBF):
    """
        Speed model learning class with all the gp given by RBF + MPK kernel.
        The GP model predicts speed changes. vel_indeces and not_vel_indeces are related:
        1st-index-state in vel_indeces is the derivative of 1st-index-state in not_vel_indeces.
    """

    def __init__(self, num_gp, init_dict_list, T_sampling, vel_indeces, not_vel_indeces, cnst_indices=None,
                 approximation_mode=None, approximation_dict=None,
                 dtype=torch.float64, device=torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model=False, integration_cond=None):
        super(Speed_Model_learning_RBF_MPK, self).__init__(num_gp=num_gp, init_dict_list=init_dict_list,
                                                           T_sampling=T_sampling, vel_indeces=vel_indeces,
                                                           not_vel_indeces=not_vel_indeces, cnst_indices=cnst_indices,
                                                           integration_cond=integration_cond,
                                                           approximation_mode=approximation_mode,
                                                           approximation_dict=approximation_dict,
                                                           dtype=dtype, device=device, flg_norm=flg_norm,
                                                           flg_norm_mean=flg_norm_mean,
                                                           flg_parametric_model=flg_parametric_model)

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp, sum of RBF + MPK
        """
        gp_list = []
        gp_list.append(SGP.RBF(**init_dict[0]))
        gp_list.append(Sparse_GP.MPK_GP(**init_dict[1]))
        return GP.Sum_Independent_GP(*gp_list)





class Speed_Model_learning_RBF_MPK_angle_state(Speed_Model_learning_RBF_angle_state):
    """
    Models the velocity of each delta of velocity with a RBF+MPK
    """
    def __init__(self, num_gp, init_dict_list, T_sampling,
                   angle_indeces, not_angle_indeces, vel_indeces, not_vel_indeces,
                   approximation_mode = None, approximation_dict = None,
                   dtype=torch.float64, device=torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                   flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
      super(Speed_Model_learning_RBF_MPK_angle_state, self).__init__(num_gp = num_gp, 
                                                                     init_dict_list = init_dict_list, 
                                                                     T_sampling = T_sampling,
                                                                     angle_indeces = angle_indeces,
                                                                     not_angle_indeces = not_angle_indeces,
                                                                     vel_indeces = vel_indeces,
                                                                     not_vel_indeces = not_vel_indeces,
                                                                     approximation_mode = approximation_mode,
                                                                     approximation_dict = approximation_dict,
                                                                     dtype = dtype, 
                                                                     device = device,
                                                                     flg_norm = flg_norm,
                                                                     flg_norm_mean = flg_norm_mean,
                                                                     flg_parametric_model = flg_parametric_model,
                                                                     cnst_indices=cnst_indices,
                                                                     evolution_indices=evolution_indices)

    def get_gp(self, gp_index, init_dict):
      gp_list = []
      gp_list.append(SGP.RBF(**init_dict[0]))
      gp_list.append(Sparse_GP.get_Volterra_MPK_GP(**init_dict[1]))
      return GP.Sum_Independent_GP(*gp_list)  



class Speed_Model_learning_RBF_angle_state_with_friction(Speed_Model_learning_RBF_angle_state):
    """
    Models the velocity of each delta of velocity with a RBF+friction compensations
    """
    def __init__(self, num_gp, init_dict_list, T_sampling,
                   angle_indeces, not_angle_indeces,
                   vel_indeces, not_vel_indeces,
                   extented_state_vel_indices_list,
                   approximation_mode = None, approximation_dict = None,
                   dtype=torch.float64, device=torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                   flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
        self.extented_state_vel_indices_list = extented_state_vel_indices_list
        super(Speed_Model_learning_RBF_angle_state_with_friction, self).__init__(num_gp = num_gp,
                                                                               init_dict_list = init_dict_list,
                                                                               T_sampling = T_sampling,
                                                                               angle_indeces = angle_indeces,
                                                                               not_angle_indeces = not_angle_indeces,
                                                                               vel_indeces = vel_indeces,
                                                                               not_vel_indeces = not_vel_indeces,
                                                                               approximation_mode = approximation_mode,
                                                                               approximation_dict = approximation_dict,
                                                                               dtype = dtype,
                                                                               device = device,
                                                                               flg_norm = flg_norm,
                                                                               flg_norm_mean = flg_norm_mean,
                                                                               flg_parametric_model = flg_parametric_model,
                                                                               cnst_indices=cnst_indices,
                                                                               evolution_indices=evolution_indices)


    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        sin-cos extended state is the gp input vector.
        """
        # extended_states = torch.cat([states[:,self.vel_indeces],
        #                              torch.sign(states[:,self.vel_indeces]),
        #                              states[:,self.not_angle_indeces],
        #                              torch.sin(states[:,self.angle_indeces]),
        #                              torch.cos(states[:,self.angle_indeces])],1)
        extended_states = torch.cat([torch.sign(states[:,self.vel_indeces]),
                                     states[:,self.not_angle_indeces],
                                     torch.sin(states[:,self.angle_indeces]),
                                     torch.cos(states[:,self.angle_indeces])],1)
        return torch.cat([extended_states,inputs],1)


    def get_gp(self, gp_index, init_dict):
        # return SGP.RBF(**init_dict)
        gp_list = []
        gp_list.append(SGP.RBF(**init_dict))
        # gp_list.append(SGP.RBF(active_dims=self.extented_state_vel_indices_list[gp_index],
        #                        lengthscales_init=np.ones(len(self.extented_state_vel_indices_list[gp_index])), flg_train_lengthscales=True,
        #                        sigma_n_init=None, flg_train_sigma_n=False,
        #                        f_mean=None, f_mean_add_par_dict={},
        #                        pos_par_mean_init=None, flg_train_pos_par_mean=False,
        #                        free_par_mean_init=None, flg_train_free_par_mean=False,
        #                        norm_mean_coef=0., norm_scale_coef=1.,
        #                        scale_init=np.ones(1), flg_train_scale=True,
        #                        norm_coef_input=None,
        #                        name='Friction GP', dtype=torch.float64, sigma_n_num=None, device=None))
        GP_friction_prior_par_dict = {}
        GP_friction_prior_par_dict['active_dims'] = self.extented_state_vel_indices_list[gp_index]
        GP_friction_prior_par_dict['lengthscales_init'] = np.ones(len(self.extented_state_vel_indices_list[gp_index]))
        GP_friction_prior_par_dict['flg_train_lengthscales'] = True
        GP_friction_prior_par_dict['sigma_n_init']=None
        GP_friction_prior_par_dict['flg_train_sigma_n']=False
        GP_friction_prior_par_dict['scale_init']=np.ones(1)
        GP_friction_prior_par_dict['flg_train_scale']=True
        GP_friction_prior_par_dict['name']='Friction GP'
        GP_friction_prior_par_dict['dtype']=self.dtype
        GP_friction_prior_par_dict['device'] = self.device
        GP_friction = GP.Scale_GP_prior(GP_prior_class=SGP.RBF,
                                        GP_prior_par_dict=GP_friction_prior_par_dict,
                                        f_scale=F_scaling.f_get_sign_abs,
                                        active_dims_f_scale=[self.extented_state_vel_indices_list[gp_index][1]],
                                        pos_par_f_init=0.1,
                                        flg_train_pos_par_f=True,
                                        free_par_f_init=None,
                                        flg_train_free_par_f=False,
                                        additional_par_f_list=[False])
        gp_list.append(GP_friction)
        return GP.Sum_Independent_GP(*gp_list)



class Speed_Model_learning_RBF_angle_state_with_friction(Speed_Model_learning_RBF_angle_state):
    """
    Models the velocity of each delta of velocity with a RBF+friction compensations
    """
    def __init__(self, num_gp, init_dict_list, T_sampling,
                   angle_indeces, not_angle_indeces,
                   vel_indeces, not_vel_indeces,
                   approximation_mode = None, approximation_dict = None,
                   dtype=torch.float64, device=torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                   flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
        super(Speed_Model_learning_RBF_angle_state_with_friction, self).__init__(num_gp = num_gp,
                                                                               init_dict_list = init_dict_list,
                                                                               T_sampling = T_sampling,
                                                                               angle_indeces = angle_indeces,
                                                                               not_angle_indeces = not_angle_indeces,
                                                                               vel_indeces = vel_indeces,
                                                                               not_vel_indeces = not_vel_indeces,
                                                                               approximation_mode = approximation_mode,
                                                                               approximation_dict = approximation_dict,
                                                                               dtype = dtype,
                                                                               device = device,
                                                                               flg_norm = flg_norm,
                                                                               flg_norm_mean = flg_norm_mean,
                                                                               flg_parametric_model = flg_parametric_model,
                                                                               cnst_indices=cnst_indices,
                                                                               evolution_indices=evolution_indices)


    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        sin-cos extended state is the gp input vector.
        """
        extended_states = torch.cat([states[:,self.not_angle_indeces],
                                     torch.sign(states[:,self.vel_indeces]),
                                     torch.sin(states[:,self.angle_indeces]),
                                     torch.cos(states[:,self.angle_indeces])],1)
        return torch.cat([extended_states,inputs],1)


    def get_gp(self, gp_index, init_dict):
        return SGP.RBF(**init_dict)
        # gp_list = []
        # gp_list.append(SGP.RBF(**init_dict))
        # f_cov = cov.diagonal_covariance_ARD
        # gp_list.append(Sparse_GP.Linear_GP(active_dims = init_dict['active_dims'],
        #                                    f_transform=lambda x:x,
        #                                    f_add_par_list=[],
        #                                    sigma_n_init=None,
        #                                    flg_train_sigma_n=False,
        #                                    f_mean=None,
        #                                    f_mean_add_par_dict={},
        #                                    pos_par_mean_init=None,
        #                                    flg_train_pos_par_mean=False,
        #                                    free_par_mean_init=None,
        #                                    flg_train_free_par_mean=False,
        #                                    norm_mean_coef=0.,
        #                                    norm_scale_coef=1.,
        #                                    Sigma_function=f_cov,
        #                                    Sigma_f_additional_par_list=[],
        #                                    Sigma_pos_par_init=np.ones(init_dict['active_dims'].size),
        #                                    flg_train_Sigma_pos_par=True,
        #                                    Sigma_free_par_init=None,
        #                                    flg_train_Sigma_free_par=False,
        #                                    flg_offset=False,
        #                                    scale_init=np.ones(1),
        #                                    flg_train_scale=False,
        #                                    name='Linear model',
        #                                    dtype=torch.float64,
        #                                    sigma_n_num=None,
        #                                    device=self.device))
        # return GP.Sum_Independent_GP(*gp_list)


def get_vel_features(v):
    return torch.cat([v, torch.sign(v)], 1)




class SP_Speed_Model_learning_Furuta(Model_learning):
    """
    Speed model learning class for the FP with semiparametric kernel.
    The GP model predicts speed changes. vel_indeces and not_vel_indeces are related:
    1st-index-state in vel_indeces is the derivative of 1st-index-state in not_vel_indeces.
    The state is assumed to be:  [theta_hor, theta_ver, theta_hor_dot, theta_ver_dot]
    """
    def __init__(self, num_gp, init_dict_list, T_sampling,
                 vel_indeces, not_vel_indeces,
                 approximation_mode = None, approximation_dict = None,
                 dtype=torch.float64, device=torch.device('cpu'), flg_norm=False, flg_norm_mean=False,
                 flg_parametric_model = False, cnst_indices=None, evolution_indices=None):
        super(SP_Speed_Model_learning_Furuta, self).__init__(num_gp = num_gp,
                                                             init_dict_list = init_dict_list,
                                                             approximation_mode = approximation_mode,
                                                             approximation_dict = approximation_dict,
                                                             dtype = dtype,
                                                             device = device,
                                                             flg_norm = flg_norm,
                                                             flg_norm_mean = flg_norm_mean,
                                                             flg_parametric_model = flg_parametric_model,
                                                             evolution_indices=evolution_indices,
                                                             cnst_indices=cnst_indices)
        self.vel_indeces = vel_indeces
        self.not_vel_indeces = not_vel_indeces
        self.T_sampling = T_sampling


    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        gp_list = []
        # get the RBF GP
        gp_list.append(SGP.RBF(**init_dict[0]))
        # get the model-based GP
        gp_list.append(Sparse_GP.Linear_GP(**init_dict[1]))
        # return the SP GP
        return GP.Sum_Independent_GP(*gp_list)  
    
    
    def data_to_gp_output(self, states):
        """
        Returns a list with the gp ouputs given the states.
        GP outputs are the speed changes.
        """
        
        return [(states[1:,i]-states[:-1,i]).reshape([-1,1]) for i in self.vel_indeces]


    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        extended state is the gp input vector, which accounts for the FP state and 
        some features suggested by the forward dynamics model.
        """        
        extended_states = torch.cat([states,
                                     inputs,
                                     torch.sin(states[:,1:2])*states[:,3:4]**2,
                                     states[:,2:3]*states[:,3:4]*torch.sin(2*states[:,1:2]),
                                     states[:,2:3],
                                     states[:,2:3]**2*torch.sin(2*states[:,1:2]),
                                     states[:,3:4],
                                     torch.sin(states[:,1:2]),
                                     inputs*torch.cos(states[:,1:2])], 1)
        return extended_states
  

    def get_next_state_from_gp_output(self, current_state, current_input,
                                      gp_output_mean_list, gp_output_var_list, particle_pred = True):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        by integrating the speed changes (GP outputs)
        """

        # preallocate variables
        next_states = torch.zeros(current_state.shape,dtype = self.dtype, device = self.device)

        if particle_pred == True:
            #  mean and variance of delta speed distribution
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            delta_vel_var = torch.cat(gp_output_var_list,1)
            # sample delta speed from distribution
            delta_speed_distribution = Normal(delta_vel_mean,torch.sqrt(delta_vel_var))  
            delta_speed_sample = delta_speed_distribution.rsample()
            # delta_speed_sample = delta_vel_mean + torch.sqrt(delta_vel_var)*torch.randn(delta_vel_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_vel_mean = torch.cat(gp_output_mean_list,1)
            delta_vel_var = None
            delta_speed_sample = delta_vel_mean

        # compute next states
        next_states[:,self.vel_indeces] = current_state[:,self.vel_indeces] + delta_speed_sample
        next_states[:,self.not_vel_indeces] = current_state[:,self.not_vel_indeces]+\
                                               self.T_sampling*current_state[:,self.vel_indeces]+\
                                               self.T_sampling/2*delta_speed_sample
        next_states[:, self.cnst_indices] = current_state[:, self.cnst_indices]
        return next_states, delta_vel_mean, delta_vel_var
