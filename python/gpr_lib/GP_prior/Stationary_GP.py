# Copyright (C) 2023 Alberto Dalla Libera
# SPDX-License-Identifier: MIT

"""
This file contains the definition of GP with stationary covariance
"""

import torch
from . import GP_prior
import numpy as np




class Stationary_GP(GP_prior.GP_prior):
    """
    Superclass of the stationary GP:
    Defines common initializations and provides a function that
    computes the squared distances weighted by the lengthscales
    """


    def __init__(self, active_dims,
                 lengthscales_init=None, flg_train_lengthscales=True, 
                 sigma_n_init=None, flg_train_sigma_n=True,
                 f_mean=None, f_mean_add_par_dict={},
                 pos_par_mean_init=None, flg_train_pos_par_mean=False,
                 free_par_mean_init=None, flg_train_free_par_mean=False,
                 norm_mean_coef=0., norm_scale_coef=1.,
                 scale_init=np.ones(1), flg_train_scale=False,
                 name='', dtype=torch.float64, sigma_n_num=None, device=None):
        """
        Initialize the module and set the lengthscales parameters
        """
        # initialize the GP model
        super(Stationary_GP,self).__init__(active_dims,
                                           sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                                           f_mean=f_mean, f_mean_add_par_dict=f_mean_add_par_dict,
                                           pos_par_mean_init=pos_par_mean_init, flg_train_pos_par_mean=flg_train_pos_par_mean,
                                           free_par_mean_init=free_par_mean_init, flg_train_free_par_mean=flg_train_free_par_mean,
                                           norm_mean_coef=norm_mean_coef, norm_scale_coef=norm_scale_coef,
                                           scale_init=scale_init, flg_train_scale=flg_train_scale,
                                           name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
        # get the number of features
        if active_dims is None:
            raise RuntimeError('Stationary_GP obj require active_dims')
        else:
            self.num_features = active_dims.size
        # check the ARD flag and set the length scalesinitial value
        flg_ARD = True
        if lengthscales_init.size == 1:
                flg_ARD = False
        if lengthscales_init is None:
            lengthscales_init = np.ones(self.num_features)
        self.flg_ARD = flg_ARD
        # get the lengthscale
        self.log_lengthscales_par = torch.nn.Parameter(torch.tensor(np.log( lengthscales_init ),
                                                                    dtype=self.dtype, device=self.device),
                                                       requires_grad=flg_train_lengthscales)


    def get_weigted_distances(self, X1, X2, norm_coef_input=None):
        """
        Computes (X1-X2)^T*Sigma^-1*(X1-X2),
        where Sigma = diag(lengthscales**2)
        """
        # get the lengthscales
        if self.flg_ARD:
            lengthscales = torch.exp(self.log_lengthscales_par)
        else:
            lengthscales = torch.exp(self.log_lengthscales_par*torch.ones(self.num_features, dtype=self.dtype, device=self.device))
        if norm_coef_input is not None:
            lengthscales = lengthscales*norm_coef_input
        # get dimensions and if X1=X2
        N1,D1 = X1.shape
        if X2 is None:
            flg_single_input = True
            N2 = N1
            D2 = D1
        else:
            flg_single_input = False
            N2,D2 = X2.size()
        # slice the inputs and get the weighted distances
        X1_sliced = X1[:,self.active_dims]/lengthscales
        X1_squared = torch.sum(X1_sliced.mul(X1_sliced), dim=1, keepdim=True)
        if flg_single_input:
            dist = X1_squared + X1_squared.transpose(dim0=0, dim1=1) -2*torch.matmul(X1_sliced,X1_sliced.transpose(dim0=0, dim1=1))
        else:
            X2_sliced = X2[:,self.active_dims]/lengthscales
            X2_squared = torch.sum(X2_sliced.mul(X2_sliced), dim=1, keepdim=True)
            dist = X1_squared + X2_squared.transpose(dim0=0, dim1=1) -2*torch.matmul(X1_sliced,X2_sliced.transpose(dim0=0, dim1=1))
        return dist 




class RBF(Stationary_GP):
    """
    Implementation of the standard RBF GP with constant mean
    """
    

    def __init__(self, active_dims,
                 lengthscales_init=None, flg_train_lengthscales=True,
                 sigma_n_init=None, flg_train_sigma_n=True,
                 f_mean=None, f_mean_add_par_dict={},
                 pos_par_mean_init=None, flg_train_pos_par_mean=False,
                 free_par_mean_init=None, flg_train_free_par_mean=False,
                 norm_mean_coef=0., norm_scale_coef=1.,
                 scale_init=np.ones(1), flg_train_scale=False,
                 norm_coef_input=None,
                 name='', dtype=torch.float64, sigma_n_num=None, device=None):
        super(RBF, self).__init__(active_dims,
                                  lengthscales_init=lengthscales_init, flg_train_lengthscales=flg_train_lengthscales,
                                  sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                                  f_mean=f_mean, f_mean_add_par_dict=f_mean_add_par_dict,
                                  pos_par_mean_init=pos_par_mean_init, flg_train_pos_par_mean=flg_train_pos_par_mean,
                                  free_par_mean_init=free_par_mean_init, flg_train_free_par_mean=flg_train_free_par_mean,
                                  norm_mean_coef=norm_mean_coef, norm_scale_coef=norm_scale_coef,
                                  scale_init=scale_init, flg_train_scale=flg_train_scale,
                                  name=name, dtype=dtype, sigma_n_num=sigma_n_num,device=device)
        # set input normalization coefficients
        if norm_coef_input is None:
            norm_coef_input = np.ones(self.log_lengthscales_par.shape)
        self.norm_coef_input = torch.tensor(norm_coef_input, dtype=self.dtype, device=self.device)


    def get_covariance(self, X1, X2=None, flg_noise=False):
        """
        Compute the exponential of the negative squared weighted distance
        """
        if flg_noise & self.GP_with_noise:
            N = X1.size()[0]
            return torch.exp(self.scale_log)*torch.exp(-self.get_weigted_distances(X1, X2, norm_coef_input=self.norm_coef_input)/2) + self.get_sigma_n_2()*torch.eye(N, dtype=self.dtype, device=self.device)
        else:
            return torch.exp(self.scale_log)*torch.exp(-self.get_weigted_distances(X1, X2, norm_coef_input=self.norm_coef_input)/2)


    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the vector containing the elements along the diagonal of the covariance matrix
        """
        N = X.shape[0]
        if flg_noise:
            return torch.exp(self.scale_log)*torch.ones(N, dtype=self.dtype, device=self.device) + self.get_sigma_n_2()
        else:
            return torch.exp(self.scale_log)*torch.ones(N, dtype=self.dtype, device=self.device)
