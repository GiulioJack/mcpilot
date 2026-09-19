# Copyright (C) 2023 Alberto Dalla Libera
# SPDX-License-Identifier: MIT

"""
This file contains the high level definition of the GP object.
"""

import torch
import numpy as np
import time

from matplotlib import pyplot as plt


class GP_prior(torch.nn.Module):
    """
    Superclass of each GP model (this class extends torch.nn.Module)
    """


    def __init__(self, active_dims,
                 sigma_n_init=None, flg_train_sigma_n=False,
                 scale_init=np.ones(1), flg_train_scale=False,
                 f_mean=None, f_mean_add_par_dict={},
                 pos_par_mean_init=None, flg_train_pos_par_mean=False,
                 free_par_mean_init=None, flg_train_free_par_mean=False,
                 norm_mean_coef=0., norm_scale_coef=1.,
                 name='', dtype=torch.float64, sigma_n_num=None, device=torch.device('cpu')):
        """
           Initialize the Module object and set flags regarding:
           - noise
            sigma_n_int = initial noise (None = no noise)
            flg_train_sigma_n = set to true to train the noise parameter
           - sigma_n_num = minimum value of sigma_n (usefull to enforce regularization)
           - active dims = indices of the input selected by the GP model to compute mean and covariance
           - data type
           - device
           - name
           - scaling parameters:
             scale_init = initialization of the scaling parameters
             flg_train_scale = set to true to train the scaling parameter
           -mean parameters:
             f_mean = mean function
             pos_par_mean_init = initialization of the f_mean positive parmeters (if present)
             free_par_mean_init = initialization of the f_mean free parameters (if present)
             f_mean_add_par_dict = additional parameters of f_mean (not treated as torch parameters)
        """
        # initilize the Module object
        super(GP_prior,self).__init__()
        # set name device and type
        self.name = name
        self.dtype = dtype
        self.device = device
        # active dims
        self.active_dims = torch.tensor(active_dims, requires_grad=False, device=device)
        # set sigma_n_log (the log of the noise standard deviation) 
        if sigma_n_init is None:
            self.GP_with_noise = False
        else:
            self.GP_with_noise = True
            self.sigma_n_log = torch.nn.Parameter(torch.tensor(np.log(sigma_n_init), dtype=self.dtype, device=self.device),
                                                  requires_grad=flg_train_sigma_n)
        # scaling parameters
        self.scale_log = torch.nn.Parameter(torch.tensor(np.log(scale_init), dtype=self.dtype, device=self.device),
                                            requires_grad=flg_train_scale)
        # set minimum value of sigma_n
        if sigma_n_num is not None:
            self.sigma_n_num = torch.tensor(sigma_n_num, dtype=self.dtype, device=self.device)
        else:
            self.sigma_n_num = torch.tensor(0., dtype=self.dtype, device=self.device)
        # check/set mean parameters
        self.check_mean(f_mean, f_mean_add_par_dict, pos_par_mean_init, flg_train_pos_par_mean,  free_par_mean_init, flg_train_free_par_mean)
        # set normalization parameters (used to normalize the mean function)
        self.norm_mean_coef = norm_mean_coef
        self.norm_scale_coef = norm_scale_coef


    def check_mean(self, f_mean, f_mean_add_par_dict, pos_par_mean_init, flg_train_pos_par_mean,  free_par_mean_init, flg_train_free_par_mean):
        """
        Set the mean function and its parameters
        """
        if f_mean is None:
            self.flg_mean = False
        else:
            self.flg_mean = True
            self.f_mean = f_mean
            self.f_mean_add_par_dict = f_mean_add_par_dict
            print('pos_par_mean_init', pos_par_mean_init)
            if pos_par_mean_init is None:
                self.flg_pos_par_mean = False
            else:
                self.flg_pos_par_mean = True
                self.pos_par_mean_log = torch.nn.Parameter(torch.tensor(np.log(pos_par_mean_init),
                                                                        dtype=self.dtype, device=self.device),
                                                                        requires_grad=flg_train_pos_par_mean)
            if free_par_mean_init is None:
                self.flg_free_par_mean = False
                self.free_par_mean = []
            else:
                self.flg_free_par_mean = True
                self.free_par_mean = torch.nn.Parameter(torch.tensor(free_par_mean_init,
                                                                     dtype=self.dtype, device=self.device),
                                                                     requires_grad=flg_train_free_par_mean)


    def to(self, dev):
        """
        Set the device and move the parameters
        """
        # set the new device
        super(GP_prior, self).to(dev)
        self.device = dev
        # move the model parameters in the new device
        self.sigma_n_num = self.sigma_n_num.to(dev)


    def set_eval_mode(self):
        """Set the model in eval mode"""
        self.flg_trainable_list = []
        for p in self.parameters():
            self.flg_trainable_list.append(p.requires_grad)
            p.requires_grad = False
    

    def set_training_mode(self):
        """Set the model in training mode"""
        for i, p in enumerate(self.parameters()):
            p.requires_grad = self.flg_trainable_list[i]  
    

    def get_sigma_n_2(self):
        """
        Returns the variance of the noise (sigma_n**2 + sigma_n_num**2)
        """
        return torch.exp(self.sigma_n_log)**2 + self.sigma_n_num**2


    def forward(self, X):
        """
        Returns:
        - prior distribution
        - the prior covariace inverse and log_det
        
        input:
        - X = training inputs (X has dimension [num_samples, num_features])
        
        output:
        - m_X = prior mean of X
        - K_X = prior covariance of X
        - K_X_inv = inverse of the prior covariance
        - log_det = log det of the prior covariance
        """
        # get the covariance
        N = X.shape[0]
        if self.GP_with_noise:
            K_X = self.get_covariance(X, flg_noise=True) 
        else:
            K_X = self.get_covariance(X)
        # get inverse and log_det with cholesky
        L = torch.linalg.cholesky(K_X)
        log_det = 2*torch.sum(torch.log(torch.diag(L)))
        K_X_inv = torch.cholesky_inverse(L)
        # get the mean
        m_X = self.get_mean(X)
        # return the values
        return m_X, K_X, K_X_inv, log_det


    def get_mean(self, X):
        """
        Returns the prior mean of X
        X is assumed to be a tensor of dimension [num_samples, num_features]
        """
        if self.flg_mean:
            if self.flg_pos_par_mean:
                if self.flg_free_par_mean:
                    m_X =  self.f_mean(X, torch.exp(self.pos_par_mean_log), self.free_par_mean, **self.f_mean_add_par_dict)
                else:
                    m_X =  self.f_mean(X, torch.exp(self.pos_par_mean_log), **self.f_mean_add_par_dict)
            else:
                if self.flg_free_par_mean:
                    m_X =  self.f_mean(X, self.free_par_mean, **self.f_mean_add_par_dict)
                else:
                    m_X =  self.f_mean(X, **self.f_mean_add_par_dict)
            return m_X/self.norm_scale_coef
        else:
            return torch.zeros([X.shape[0], 1], device=self.device, dtype=self.dtype)


    def get_covariance(self, X1, X2=None, flg_noise=False):
        """
        Returns the covariance betweeen the input locations X1 X2. 
        If X2 is None X2 is assumed to be equal to X1
        """
        raise NotImplementedError()


    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the diagonal elements of the covariance betweeen the input locations X
        """
        raise NotImplementedError()


    def get_alpha(self, X, Y):
        """
        Computes alpha, the vector of coefficients defining the posterior distribution

        inputs:
        - X = training input
        - Y = training output

        outputs:
        - alpha = vector defining the posterior distribution
        - m_X = prior mean of X
        - K_X_inv = inverse of the prior covariance of X
        """
        m_X, _, K_X_inv, _ = self.forward(X)
        alpha = torch.matmul(K_X_inv, Y-m_X)
        return alpha, m_X, K_X_inv


    def get_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None):
        """
        Compute the posterior distribution in X_test, given the alpha vector.
        
        input:
        - X = training input locations (used to compute alpha)
        - alpha = vector of coefficients defining the posterior
        - m_X = prior mean of X
        - K_X_inv = inverse of the prior covariance of X
        
        output:
        - Y_hat = posterior mean
        - var = diagonal elements of the posterior variance (If K_X_inv is given)
        - m_X_test = prior mean in X_test
        
        """ 
        # get covariance and prior mean
        K_X_test_X = self.get_covariance(X_test, X)
        m_X_test = self.get_mean(X_test)
        # get the estimate 
        Y_hat = m_X_test + torch.matmul(K_X_test_X, alpha)
        # if K_X_inv is given compute the confidence intervals
        if K_X_inv is not None:
            num_test = X_test.shape[0]
            var = self.get_diag_covariance(X_test) - torch.sum(torch.matmul(K_X_test_X, K_X_inv)*(K_X_test_X), dim=1)
        else:
            var = None

        # plt.figure(100)
        # #plt.plot(Y_hat.detach().cpu().numpy())
        # print(m_X_test.shape)
        # plt.plot(m_X_test.detach().cpu().numpy())
        # # plt.show()

        return Y_hat, var, m_X_test
         

    def get_estimate(self, X, Y, X_test, flg_return_K_X_inv=False):
        """
        Returns the posterior distribution in X_test, given the training samples X Y.
        
        input:
        - X = training input
        - Y = training output
        - X_test = test input
        
        output:
        - Y_hat = mean of the test posterior
        - var = diagonal elements of the variance posterior
        - alpha = coefficients defining the posterior
        - m_X = prior mean in X_test
        - K_X_inv = inverse of the training covariance

        The function returns:
           -a vector containing the sigma squared confidence intervals
           -the vector of the coefficient
           -the K_X inverse in case required through flg_return_K_X_inv
        """
        # get the coefficent and the mean
        alpha, m_X, K_X_inv = self.get_alpha(X, Y)
        # get the estimate and the confidence intervals
        Y_hat, var, m_X_test = self.get_estimate_from_alpha(X, X_test, alpha, K_X_inv=K_X_inv)
        #return the opportune values
        if flg_return_K_X_inv:
            return Y_hat, var, alpha, m_X_test, K_X_inv
        else:
            return Y_hat, var, alpha, m_X_test


    def print_model(self, flg_grad=True):
        """
        Print the model parameters
        if flg_grad is True the function prints only trainable parameters
        """
        print(self.name+' parameters:')
        for par_name, par in self.named_parameters():
            if flg_grad:
                if par.requires_grad:
                    print('-', par_name, ':', par.data)


    def fit_model(self,trainloader=None, 
                  optimizer=None, criterion=None,
                  N_epoch=1, N_epoch_print=1,
                  f_saving_model=None, f_print=None):
        """
        Optimize the model hyperparameters

        input:
        - trainloader = torch train loader object
        - optimizer = torch optimizer object
        - criterion = loss function
        - N_epoch = number of epochs
        - N_epoch_print = number of epoch between print two prints of the current loss and model parameters
        - f_saving_model = customizable function that save the model
        - f_print_model = customizable function that print the model (eventually with performance)
        """
        # print initial parametes and initial estimates
        print('\nInitial parameters:')
        self.print_model()  
        # iterate over the training data for N_epochs
        t_start = time.time()
        for epoch in range(0,N_epoch):
            # initialize loss grad and counter
            running_loss = 0.0
            N_btc = 0
            optimizer.zero_grad()
            #iterate over the training set
            for i, data in enumerate(trainloader, 0):
                # get the training data
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                out_GP_priors = self(inputs)
                loss = criterion(out_GP_priors, labels)
                loss.backward(retain_graph=False)
                optimizer.step()
                # update the running loss
                running_loss = running_loss + loss.item()
                N_btc = N_btc + 1
            # print statistics and save the model
            if epoch%N_epoch_print==0:
                print('\nEPOCH:', epoch)
                self.print_model()
                print('Running loss:', running_loss/N_btc)
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                if f_saving_model is not None:
                    f_saving_model(epoch)
                if f_print is not None:
                    f_print()
                t_start = time.time()
        # print the final parameters
        print('\nFinal parameters:')
        self.print_model()


    def get_SOD(self, X, Y, threshold, flg_permutation=False, criterion='error'):
        """
        Returns the SOD points with an online procedure
        SOD: most important subset of data
        - criterion == 'error' --> insert data if prediction error > threshold
        - criterion == 'std' --> insert data if prediction std > threshold
        """
        print('\nSelection of the inducing inputs...')
        print('Criterion:', criterion)
        t_start = time.time()
        # get number of samples
        num_samples = X.shape[0]
        # init the set of inducing inputs with the first sample
        SOD = X[0:1,:]
        inducing_inputs_indices = [0]
        # get a permuation of the inputs
        perm_indices = torch.arange(1,num_samples)
        if flg_permutation:
            perm_indices = perm_indices[torch.randperm(num_samples-1)]
        # init SOD and training model
        K_X_inv = torch.linalg.solve(self.get_covariance(X[inducing_inputs_indices,:], flg_noise=True), torch.eye(1,1, dtype=self.dtype, device=self.device))
        m_X_tr = self.get_mean(X[inducing_inputs_indices,:])
        Y_tr = Y[inducing_inputs_indices,:] 
        alpha = torch.matmul(K_X_inv, Y_tr-m_X_tr)
        X_tr = X[inducing_inputs_indices,:]
        # iterate over training samples
        for sample_index in perm_indices:
            # get the estimate 
            X_test = X[sample_index:sample_index+1,:]
            Y_test = Y[sample_index:sample_index+1,:]
            K_X_test_X = self.get_covariance(X_test, X_tr)
            # compute error
            if criterion == 'error':
                # compute test estimate
                m_X_test = self.get_mean(X[sample_index:sample_index+1,:])
                Y_test_hat = m_X_test + torch.matmul(K_X_test_X, alpha)
                E = torch.abs(Y_test-Y_test_hat)
            elif criterion == 'std':
                # compute std
                var = self.get_diag_covariance(X_test) - torch.sum((K_X_test_X@K_X_inv)*K_X_test_X, dim=1)
                E = torch.sqrt(var)
            # check error
            if E > threshold:
                # update SOD and training model
                SOD = torch.cat([SOD,X[sample_index:sample_index+1,:]],0)
                inducing_inputs_indices.append(sample_index)
                # update training model
                K_X_inv = incremental_inv(Q=K_X_test_X.t(),
                                          R=K_X_test_X,
                                          S=self.get_covariance(X_test, flg_noise=True),
                                          P_inv=K_X_inv)
                m_X_tr = self.get_mean(X[inducing_inputs_indices,:])
                Y_tr = Y[inducing_inputs_indices,:] 
                alpha = torch.matmul(K_X_inv, Y_tr-m_X_tr)
                X_tr = X[inducing_inputs_indices,:]

        t_stop = time.time()
        print('Time elapsed:', t_stop-t_start)
        print('Number of inputs before selection:', num_samples)
        print('Number of inputs after selection:', SOD.shape[0]) 
        return inducing_inputs_indices


    def get_SOD_projection_error(self, X, Y, threshold, num_pts, flg_permutation=False, return_alpha=False):
        """
        Returns the SOD points based on projection-induced error criterion (see sparse online gaussian process, Csato et al., 2001)
        SOD: most important subset of data
        """
        print('\nSelection of the inducing inputs...')
        print('Criterion: projection-induced error',)
        t_start = time.time()
        # get number of samples
        num_samples = X.shape[0]
        # init the set of inducing inputs with only the first sample
        inducing_inputs_indices = [0]
        # get a permuation of the inputs
        perm_indices = torch.arange(1,num_samples)
        if flg_permutation:
            perm_indices = perm_indices[torch.randperm(num_samples-1)]
        # init SOD and training model
        K_X_inv = torch.linalg.solve(self.get_covariance(X[inducing_inputs_indices,:], flg_noise=True), torch.eye(1,1, dtype=self.dtype, device=self.device))
        Q_true = torch.linalg.inv(self.get_covariance(X[inducing_inputs_indices,:], flg_noise=False),)
        Q = torch.linalg.inv(self.get_covariance(X[inducing_inputs_indices,:], flg_noise=False),)
        m_X_tr = self.get_mean(X[inducing_inputs_indices,:])
        Y_tr = Y[inducing_inputs_indices,:] 
        alpha = torch.matmul(K_X_inv, Y_tr-m_X_tr)
        X_tr = X[inducing_inputs_indices,:]
        # iterate over training samples
        for sample_index in perm_indices:
            # print('\nElaborating point with index', sample_index.detach().numpy(), 'out of', len(perm_indices))
            # current input-output pair
            X_test = X[sample_index:sample_index+1,:]
            Y_test = Y[sample_index:sample_index+1,:]
            # compute kernel related quantities
            K_tp1 = self.get_covariance(X_test, X_tr)
            k_star = self.get_covariance(X_test, X_test)
            # update alpha and C
            mu = self.get_sigma_n_2() - K_tp1 @ K_X_inv @ K_tp1.T + k_star 
            q = (Y_test - K_tp1@alpha)/mu
            r = 1/mu    # todo: do I need the minus?    
            # print('  q: ', q)       
            # compute RKHS projection error
            e_hat_tp1 = Q @ K_tp1.T 
            gamma_tp1 = k_star - K_tp1 @ e_hat_tp1
            tmp = - K_X_inv @ K_tp1.T
            # gamma_true = k_star - K_tp1 @ Q_true @ K_tp1.T
            # print('  gamma: ', gamma_tp1)
            # print('  gamma true: ', gamma_true)
            # check the error
            if gamma_tp1.cpu().detach().numpy() < threshold:
                # print('  point discarded')
                # perform reduced update and discard current input
                s = tmp + e_hat_tp1
                alpha += q * s
                K_X_inv += r * s @ s.T                
            else:
                # print('  point added')
                # add the current input-output pair to the dataset and perform the exact update
                inducing_inputs_indices.append(sample_index)
                Y_tr = Y[inducing_inputs_indices,:]
                X_tr = X[inducing_inputs_indices,:]
                # print('  number of selected points: ', len(inducing_inputs_indices))
                # iterative exact update
                s = torch.cat([tmp, torch.tensor([[1]], dtype=self.dtype, device=self.device)], axis = 0)
                # print('  s: ', s)
                alpha = torch.cat([alpha, torch.tensor([[0]], dtype=self.dtype, device=self.device)], axis = 0) 
                alpha += q*s 
                # print('  alpha: ', alpha)
                # update C
                K_X_inv = torch.cat([K_X_inv, torch.zeros((1, K_X_inv.size()[1]), dtype=self.dtype, device=self.device)], axis = 0)
                K_X_inv = torch.cat([K_X_inv, torch.zeros((K_X_inv.size()[0], 1), dtype=self.dtype, device=self.device)], axis = 1)
                K_X_inv += r * s @ s.T
                # update Q
                tmp_q = torch.cat([e_hat_tp1, torch.tensor([[-1]], dtype=self.dtype, device=self.device)], axis = 0)
                Q = torch.cat([Q, torch.zeros((1, Q.size()[1]), dtype=self.dtype, device=self.device)], axis = 0)
                Q = torch.cat([Q, torch.zeros((Q.size()[0], 1), dtype=self.dtype, device=self.device)], axis = 1)
                Q +=  torch.linalg.inv(gamma_tp1) * tmp_q @ tmp_q.T
                
            
            # delete basis vector
            C = -K_X_inv
            if num_pts is not None and len(inducing_inputs_indices)>num_pts:
                # print('\n  Too many points. Deleting one')
                # compute scores for each point added so far 
                scores = [(np.abs(alpha[idx,0].cpu().detach().numpy())/Q[idx,idx].cpu().detach().numpy()) for idx in range(len(inducing_inputs_indices))]
                min = np.min(scores)
                min_idx = np.argmin(scores)
                # print('  scores: ', scores)
                # print('  min', min)
                # print('  min_idx', min_idx)
                # remove less important
                inducing_inputs_indices.pop(min_idx)
                Y_tr = Y[inducing_inputs_indices,:]
                X_tr = X[inducing_inputs_indices,:]
                # update variables
                Q_t = torch.cat((Q[:min_idx], Q[min_idx+1:]), dim=0)
                Q_t = torch.cat((Q_t[:,:min_idx], Q_t[:,min_idx+1:]), dim=1)
                Q_star = torch.reshape(torch.cat((Q[:min_idx, min_idx], Q[min_idx+1:,min_idx]), dim=0), (-1,1))
                C_t = torch.cat((C[:min_idx], C[min_idx+1:]), dim=0)
                C_t = torch.cat((C_t[:,:min_idx], C_t[:,min_idx+1:]), dim=1)
                C_star = torch.reshape(torch.cat((C[:min_idx, min_idx], C[min_idx+1:,min_idx]), dim=0), (-1,1))
                q_star = Q[min_idx, min_idx]
                c_star = C[min_idx, min_idx]
                # print('  Q_t shape: ', Q_t.size())
                # print('  Q_star shape: ', Q_star.size())
                # print('  C_t shape: ', C_t.size())
                # print('  C_star shape: ', C_star.size())
                # print('  q_star shape: ', q_star.size())
                # print('  c_star shape: ', c_star.size())
                alpha_t = torch.reshape(torch.cat((alpha[:min_idx], alpha[min_idx+1:]), dim=0), (-1,1))
                alpha_star = alpha[min_idx]
                alpha = alpha_t - (alpha_star/q_star)*Q_star
                C = C_t + (c_star/q_star**2)*Q_star@Q_star.T - (1/q_star)*(Q_star @ C_star.T + C_star @ Q_star.T)
                Q = Q_t - (1/q_star)*Q_star@Q_star.T
            K_X_inv = -C    
            
            # Debug    
            # Q_true = torch.linalg.inv(self.get_covariance(X_tr, flg_noise=False),)
            # print('  max Q: ', torch.max(Q))
            # print('  max Q_true: ', torch.max(Q_true))
            # print('  max error on Q: ', torch.max(Q - Q_true))                
            # K_X_inv_true = torch.linalg.inv(self.get_covariance(X_tr, flg_noise=True),)
            # print('  max K_inv: ', torch.max(K_X_inv_true))
            # print('  max error on K_X_inv: ', torch.max(K_X_inv - K_X_inv_true))  
            # alpha_true = torch.matmul(K_X_inv_true, Y_tr-m_X_tr)
            # print('  max alpha: ', torch.max(alpha))
            # print('  max error on alpha: ', torch.max(alpha - alpha_true))

        t_stop = time.time()
        print('Time elapsed:', t_stop-t_start)
        print('Number of inputs before selection:', num_samples)
        print('Number of inputs after selection:', len(inducing_inputs_indices)) 

        if return_alpha:
            return inducing_inputs_indices, alpha, K_X_inv

        return inducing_inputs_indices


    def __add__(self, other_GP_prior):
        """
        Returns a new GP given by the sum of two GP
        """
        return Sum_Independent_GP(self, other_GP_prior)


    def __mul__(self, other_GP_prior):
        """
        Returns a new GP with mean given by the product of the means and covariances
        """
        return Multiply_GP_prior(self, other_GP_prior)




class Combine_GP(GP_prior):
    """
    Class that extend GP_prior and provide common utilities to combine GP
    """


    def __init__(self, *gp_priors_obj):
        """
        Initialize the multiple kernel object
        """
        # initialize a new GP object
        super(Combine_GP, self).__init__(active_dims=[], sigma_n_num=gp_priors_obj[0].sigma_n_num,
                                         scale_init=np.ones(1), flg_train_scale=False,
                                         dtype=gp_priors_obj[0].dtype, device=gp_priors_obj[0].device)
        # build a list with all the models
        self.gp_list = torch.nn.ModuleList(gp_priors_obj)
        # check the noise flag
        GP_with_noise = False
        for gp in self.gp_list:
            GP_with_noise = GP_with_noise or gp.GP_with_noise 
        self.GP_with_noise = GP_with_noise


    def to(self, dev):
        """
        Move all the models to the desired device
        """
        super(Combine_GP, self).to(dev)
        self.device = dev
        for gp in self.gp_list:
            gp.to(dev)


    def print_model(self):
        """
        Print the parameters of all the models in the gp_list
        """
        for gp in self.gp_list:
            gp.print_model()


    def get_sigma_n_2(self):
        """
        Iterate over all the models in the list and returns the global noise variance
        """
        sigma_n_2 = torch.zeros(1, dtype=self.dtype, device=self.device)
        for gp in self.gp_list:
            if gp.GP_with_noise:
                sigma_n_2 += gp.get_sigma_n_2()
        return sigma_n_2




class Sum_Independent_GP(Combine_GP):
    """
    Class that sum GP_priors objects
    """


    def __init__(self, *gp_priors_obj):
        """
        Initialize the gp list
        """
        super(Sum_Independent_GP, self).__init__(*gp_priors_obj)


    def get_mean(self, X):
        """
        Returns the sum of the means returned by the models in gp_list
        """
        N = X.shape[0]
        # mean = torch.zeros(N,1, dtype=self.dtype, device=self.device)
        # for gp in self.gp_list:
        #     mean += gp.get_mean(X)
        # return mean
        mean = torch.sum(torch.cat([(gp.get_mean(X)).reshape([N,1]) for gp in self.gp_list],1),1).reshape([N,1])
        return mean


    def get_covariance(self, X1, X2=None, flg_noise=False):
        """
        Returns the sum of the covariances returned by the mdoels in gp_list
        """
        # get dimensions
        N1 = X1.shape[0]
        if X2 is None:
            N2 = N1
        else:
            N2 = X2.shape[0]

        # # initialize the covariance
        # cov = torch.zeros(N1,N2, dtype=self.dtype, device=self.device)
        # sum all the covariances
        cov = torch.sum(torch.cat([(gp.get_covariance(X1,X2,flg_noise=False)).unsqueeze(0) for gp in self.gp_list],0),0)
        # add the noise
        if flg_noise & self.GP_with_noise:
            cov = cov + self.get_sigma_n_2()*torch.eye(N1, dtype=self.dtype, device=self.device)
        return cov


    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the sum of the diagonals of the covariances in gp list
        """
        # # initialize the vector
        # diag = torch.zeros(X.shape[0], dtype=self.dtype, device=self.device)
        # # iterate in the list and sum the diagonals
        # for gp in self.gp_list:
        #     diag += gp.get_diag_covariance(X, flg_noise=False)
        # get the number of input locations
        N = X.shape[0]
        # sum all the diagonal covariances
        diag = torch.sum(torch.cat([(gp.get_diag_covariance(X,flg_noise=False)).reshape([1,N]) for gp in self.gp_list],0),0)
        # add the noise
        if flg_noise & self.GP_with_noise:
            diag += self.get_sigma_n_2()
        return diag




class Multiply_GP_prior(Combine_GP):
    """
    Class that generates a GP_prior multiplying GP_priors objects
    """


    def __init__(self, *gp_priors_obj):
        """
        Initilize the GP list
        """
        super(Multiply_GP_prior, self).__init__(*gp_priors_obj)


    def get_mean(self, X):
        """
        Returns the product of the means returned by the models in gp_list
        """
        # initilize the mean vector
        N = X.shape[0]
        mean = torch.ones(N,1, dtype=self.dtype, device=self.device)
        # multiply all the means
        for gp in self.gp_list:
            mean = mean*gp.get_mean(X)
        return mean


    def get_covariance(self, X1, X2=None, flg_noise=False):
        """
        Returns the element-wise product of the covariances returned by the models in gp_list
        """
        # get size
        N1 = X1.shape[0]
        if X2 is None:
            N2 = N1
        else:
            N2 = X2.shape[0]
        # initilize the covariance
        cov = torch.ones(N1,N2, dtype=self.dtype, device=self.device)
        # multiply all the covariances
        for gp in self.gp_list:
            cov *=gp.get_covariance(X1,X2, flg_noise=False)
        # add the noise
        if flg_noise & self.GP_with_noise:
            cov = cov + self.get_sigma_n_2()*torch.eye(N1, dtype=self.dtype, device=self.device)
        return cov


    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the product of the diagonal elements of the covariance returned by the models in gp_list
        """
        # initilize the diagona
        N = X.shape[0]
        diag = torch.ones(N, dtype=self.dtype, device=self.device)
        # multiply all the diagonals
        for gp in self.gp_list:
            diag *= gp.get_diag_covariance(X, flg_noise=False)
        # add the nosie
        if flg_noise & self.GP_with_noise:
           diag +=self.get_sigma_n_2()
        return diag




def Scale_GP_prior(GP_prior_class, GP_prior_par_dict,
                   f_scale, active_dims_f_scale,
                   pos_par_f_init=None, flg_train_pos_par_f=True,
                   free_par_f_init=None, flg_train_free_par_f=True,
                   additional_par_f_list=[]):
    """
    Funciton that returns a GP_prior scaled. This class implement the following model:
    y(x) = a(x)f(x) + e, where f(x) is a GP and a(x) a deterministic function.
    The function a(x) can be parametrize with respect to a set of trainable prameters.
    This class retuns an instance of a new class defined dynamically in the following
    """

    # define dynamically the new class
    class Scaled_GP(GP_prior_class):
        """
        Class that extends the GP_prior_class with the scaling parameters
        """

        def __init__(self, GP_prior_par_dict,
                     f_scale, active_dims_f_scale,
                     pos_par_f_init, flg_train_pos_par_f,
                     free_par_f_init, flg_train_free_par_f,
                     additional_par_f_list):
            # initialize the object of the superclass
            super(Scaled_GP, self).__init__(**GP_prior_par_dict)
            # save the scaling info 
            self.f_scale = f_scale
            self.active_dims_f_scale = active_dims_f_scale
            self.additional_par_f_list = additional_par_f_list
            if pos_par_f_init is None:
                self.flg_pos_par = False
                self.pos_par_f_log = None
            else:
                self.flg_pos_par = True
                self.pos_par_f_log = torch.nn.Parameter(torch.tensor(np.log(pos_par_f_init), dtype=self.dtype, device=self.device), requires_grad=flg_train_pos_par_f)
            if free_par_f_init is None:
                self.flg_free_par = False
                self.free_par_f = None            
            else:
                self.flg_free_par = True
                self.free_par_f = torch.nn.Parameter(torch.tensor(free_par_f_init, dtype=self.dtype, device=self.device), requires_grad=flg_train_free_par_f)
        

        def get_scaling(self,X):
            """
            Returns the scaling funciton evaluated in X
            """
            if self.flg_pos_par:
                pos_par = torch.exp(self.pos_par_f_log)
            else:
                pos_par = None
            return self.f_scale(X[:, self.active_dims_f_scale], pos_par, self.free_par_f, *self.additional_par_f_list)


        def get_mean(self, X):
            """
            Calls the get_mean of the superclass and apply the scaling
            """
            # get the supercalss mean and scale the result
            return self.get_scaling(X)*super(Scaled_GP, self).get_mean(X)


        def get_covariance(self, X1, X2=None, flg_noise=False):
            """
            Calls the get_covariance of the superclass and apply the scaling
            """
            # get the scaling functions
            a_X1 = self.get_scaling(X1).reshape([-1,1])
            # if required evaluate the scaling function in X2 and get the covariance
            if X2 is None:
                K = a_X1*super(Scaled_GP, self).get_covariance(X1, X2, flg_noise=False)*(a_X1.transpose(0,1))
            else:
                a_X2 = self.get_scaling(X2).reshape([-1,1])
                K = a_X1*super(Scaled_GP, self).get_covariance(X1, X2, flg_noise=False)*(a_X2.transpose(0,1))
            # if required add the noise and return the covariance
            if flg_noise & self.GP_with_noise:
                N = K.shape[0]
                return K + self.get_sigma_n_2()*torch.eye(N, dtype=self.dtype, device=self.device)
            else:
                return K 


        def get_diag_covariance(self, X, flg_noise=False):
            """
            Calls the get_diag_covariance of the superclass and apply the scaling
            """
            # evaluate the scaling function in X1
            N = X.shape[0]
            a_X = self.get_scaling(X).reshape(N)
            diag = a_X**2*super(Scaled_GP, self).get_diag_covariance(X, flg_noise=False)
            # if required add the noise and return the covariance
            if flg_noise & self.GP_with_noise:
                return diag + self.get_sigma_n_2()
            else:
                return diag


    # return an object of the new class
    return Scaled_GP(GP_prior_par_dict,
                     f_scale, active_dims_f_scale,
                     pos_par_f_init, flg_train_pos_par_f,
                     free_par_f_init, flg_train_free_par_f,
                     additional_par_f_list)


def incremental_inv(Q, R, S, P_inv):
    """
    Returns the inverse of [[P,Q],[R, S]]
    """
    M = torch.linalg.solve((S -R@P_inv@Q), torch.eye(*S.shape, dtype=S.dtype, device=S.device))
    P_tilde = P_inv + P_inv@Q@M@R@P_inv
    Q_tilde = -P_inv@Q@M
    R_tilde = -M@R@P_inv
    return torch.cat([torch.cat([P_tilde, R_tilde], 0), torch.cat([Q_tilde, M], 0)], 1)
