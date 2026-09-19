# Copyright (C) 2023 Alberto Dalla Libera
# SPDX-License-Identifier: MIT

"""
This file contains the definition of:
- linear GP
- polynomial GP
- sparse GP
"""



import torch
import numpy as np
from . import GP_prior
from .. import Utils
import time
from .. Utils.Parameters_covariance_functions import diagonal_covariance_ARD

class Linear_GP(GP_prior.GP_prior):
    """
    Implementation of the GP with linear kernel (dot product covariance).
    f(X) = phi(X)*w, where the w is defined as a gaussian variable N(0, Sigma_w)
    f_transform is a function that apply transformation to the GP inputs
    """


    def __init__(self, active_dims, f_transform=lambda x:x, f_add_par_list=[],
                 sigma_n_init=None, flg_train_sigma_n=False,
                 f_mean=None, f_mean_add_par_dict={},
                 pos_par_mean_init=None, flg_train_pos_par_mean=False,
                 free_par_mean_init=None, flg_train_free_par_mean=False,
                 norm_mean_coef=0., norm_scale_coef=1.,
                 Sigma_function=None, Sigma_f_additional_par_list=None, 
                 Sigma_pos_par_init=None, flg_train_Sigma_pos_par=True,
                 Sigma_free_par_init=None, flg_train_Sigma_free_par=True,
                 flg_offset=False,
                 scale_init=np.ones(1), flg_train_scale=False,
                 name='', dtype=torch.float64, sigma_n_num=None, device=None):
        """
        Initialization of the object:
        - f_mean and Sigma_function define the prior on the model parameters w
        - Sigma_pos_par Sigma_free par are, respectively, the positive and free parameters of the w prior
        - if flg_offset is true a constant feature equal to 1 is added to the input X
        - active_dims defines the input dimension interested (after applying the f_transform)
        """
        # initilize the GP object
        super(Linear_GP, self).__init__(active_dims,
                                        sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                                        scale_init=scale_init, flg_train_scale=flg_train_scale,
                                        f_mean=f_mean, f_mean_add_par_dict=f_mean_add_par_dict,
                                        pos_par_mean_init=pos_par_mean_init, flg_train_pos_par_mean=flg_train_pos_par_mean,
                                        free_par_mean_init=free_par_mean_init, flg_train_free_par_mean=flg_train_free_par_mean,
                                        norm_mean_coef=norm_mean_coef, norm_scale_coef=norm_scale_coef,
                                        name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
        # save f_transform
        self.f_transform = f_transform
        self.f_add_par_list = f_add_par_list
        # check active dims
        if active_dims is None:
            raise RuntimeError("Active_dims are needed")
        self.num_features = active_dims.size
        # save flg_offset (flg_offset=True => ones added to the phi)
        self.flg_offset = flg_offset
        # check Sigma init
        self.check_sigma_function(Sigma_function, Sigma_f_additional_par_list,
                                  Sigma_pos_par_init, flg_train_Sigma_pos_par,
                                  Sigma_free_par_init, flg_train_Sigma_free_par)


    def check_sigma_function(self, Sigma_function, Sigma_f_additional_par_list,
                             Sigma_pos_par_init, flg_train_Sigma_pos_par,
                             Sigma_free_par_init, flg_train_Sigma_free_par):
        if Sigma_function is None:
            raise RuntimeError("Specify a Sigma function")
        self.Sigma_function = Sigma_function
        self.Sigma_f_additional_par_list = Sigma_f_additional_par_list
        if Sigma_pos_par_init is None:
            self.Sigma_pos_par = None
        else:
            self.Sigma_pos_par = torch.nn.Parameter(torch.tensor(np.log(Sigma_pos_par_init), dtype=self.dtype, device=self.device), requires_grad=flg_train_Sigma_pos_par)
        if Sigma_free_par_init is None:
            self.Sigma_free_par = None
        else:
            self.Sigma_free_par = torch.nn.Parameter(torch.tensor(Sigma_free_par_init, dtype=self.dtype, device=self.device), requires_grad=flg_train_Sigma_free_par)


    def get_phi(self,X):
        """
        Returns the regression matrix associated to the inputs X
        """
        num_samples = X.shape[0]
        # apply transform
        phi = self.f_transform(X, *self.f_add_par_list)
        if self.flg_offset:
            return torch.cat([phi[:,self.active_dims],torch.ones(num_samples,1, dtype=self.dtype, device=self.device)],1)
        else:
            return phi[:,self.active_dims]


    def get_Sigma(self):
        """
        Computes the Sigma matrix
        """
        if self.Sigma_pos_par is None:
            return self.Sigma_function( self.Sigma_pos_par, self.Sigma_free_par, *self.Sigma_f_additional_par_list)
        else:
            return self.Sigma_function( torch.exp(self.Sigma_pos_par), self.Sigma_free_par, *self.Sigma_f_additional_par_list)


    def get_covariance(self, X1, X2=None, flg_noise=False):
        """
        Returns scale*phi(X)^T*Sigma*phi(X)
        """
        # get the parameters variance
        Sigma = self.get_Sigma()
        # get the covariance
        phi_X1 = self.get_phi(X1)
        if X2 is None:
            K_X =  torch.exp(self.scale_log)*torch.matmul(phi_X1, torch.matmul(Sigma, phi_X1.transpose(0,1)))
            # check if we need to add the noise
            if flg_noise & self.GP_with_noise:
                N = X1.size()[0]
                return K_X + self.get_sigma_n_2()*torch.eye(N, dtype=self.dtype, device=self.device)
            else:
                return K_X
        else:
            return torch.exp(self.scale_log)*torch.matmul(phi_X1, torch.matmul(Sigma, self.get_phi(X2).transpose(0,1)))


    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the diag of the cov matrix
        """
        # Get the parameters the variance
        Sigma = self.get_Sigma()
        # get the diag of the covariance
        phi_X = self.get_phi(X)
        diag = torch.exp(self.scale_log)*torch.sum(torch.matmul(phi_X, Sigma)*(phi_X), dim=1)
        if flg_noise & self.GP_with_noise:
            return diag + self.get_sigma_n_2()
        else:
            return diag


    def get_parameters(self, X, Y, flg_print=False):
        """
        Returns the posterior estimate of w, the parameters of the 'linear' model.
        NB: the parameters returned are correct only if this GP is not combined with other GPs
        """
        # get the mean and the inverse of the kernel matrix using forward
        m_X, _, K_X_inv, _ = self.forward(X)
        Y = Y - m_X
        # get sigma and phi
        Sigma = self.get_Sigma()
        phi_X_T = torch.transpose(self.get_phi(X), 0, 1)
        # get the parameters
        w_hat = torch.matmul(Sigma, torch.matmul(phi_X_T, torch.matmul(K_X_inv, Y)))
        if flg_print:
            print(self.name+' linear parameters estimated: ', w_hat.data)
        return w_hat


    def get_parameters_inv_lemma(self, X, Y, flg_print=False):
        """
        Returns the posterior estimate of w, the parameters of the 'linear' model.
        This implementation exploit the sparsity of the covariance.
        NB: the parameters returned are correct only if this GP is not combined with other GPs
        """
        # get the mean, phi and the prior covariance of the parameters
        m_X = self.get_mean(X)
        Y = Y - m_X
        Phi_X = self.get_phi(X)
        Sigma = torch.exp(self.scale_log)*self.get_Sigma()
        # Sigma = self.get_Sigma()
        sigma_n_2 = self.get_sigma_n_2()
        # return the parameters
        A = torch.cholesky_inverse(torch.linalg.cholesky(Sigma)) + torch.matmul(Phi_X.transpose(0,1), Phi_X)/sigma_n_2
        A_inv = torch.cholesky_inverse(torch.linalg.cholesky(A))
        w_hat = torch.matmul(torch.matmul(A_inv, Phi_X.transpose(0,1)), Y)/sigma_n_2
        if flg_print:
            print(self.name+' linear parameters estimated: ', w_hat.data)
        return w_hat, A_inv


    def forward_inv_lemma(self, X):
        """
        Returns the prior covariace inverse and log_det
        
        input:
        - X = training inputs (X has dimension [num_samples, num_features])
        
        output:
        - m_X = prior mean of X
        - K_X = prior covariance of X
        - K_X_inv = inverse of the prior covariance
        - log_det = log det of the prior covariance
        """
        # get the mean
        m_X = self.get_mean(X)
        # get the covariance
        N = X.size()[0]
        # get the parameters prior covariance and its cholesky distribution
        Sigma = torch.exp(self.scale_log)*self.get_Sigma()
        L_Sigma = torch.linalg.cholesky(Sigma)
        # get phi
        phi_X = self.get_phi(X)
        # get sigma_n2
        sigma_n_2 = self.get_sigma_n_2()
        # get K_X
        # K_X =  torch.matmul(phi_X, torch.matmul(Sigma, phi_X.transpose(0,1))) + sigma_n_2*torch.eye(N, dtype=self.dtype, device=self.device)
        # get inverse and log_det with inv lemma
        tmp = torch.cholesky_inverse(L_Sigma) + torch.matmul(phi_X.t(), phi_X/sigma_n_2)
        L_tmp = torch.linalg.cholesky(tmp)
        K_X_inv = torch.eye(N, dtype=self.dtype, device=self.device)/sigma_n_2 - torch.matmul(phi_X, torch.matmul(torch.cholesky_inverse(L_tmp), phi_X.t()))/sigma_n_2**2
        log_det = 2*torch.sum(torch.log(torch.diag(L_Sigma))) + 2*torch.sum(torch.log(torch.diag(L_tmp))) + N*torch.log(sigma_n_2)
        # return the values
        return m_X, None, K_X_inv, log_det


    def fit_model_inv_lemma(self,trainloader=None, 
                            optimizer=None, criterion=None,
                            N_epoch=1, N_epoch_print=1,
                            f_saving_model=None, f_print=None):
        """
        Optimize the model hyperparameters
        This function exploits the fact that the kernel matrix is not full rank
        USE THIS FUNCTION WHEN NUMBER OF SAMPLES >> NUMBER OF PARAMETERS

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
                out_GP_priors = self.forward_inv_lemma(inputs)
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




class Poly_GP(Linear_GP):
    """
    GP with polynomial kernel. Implemented extending the Linear_GP
    """
    def __init__(self, active_dims, poly_deg,
                 sigma_n_init=None, flg_train_sigma_n=True,
                 f_mean=None, f_mean_add_par_dict={},
                 pos_par_mean_init=None, flg_train_pos_par_mean=False,
                 free_par_mean_init=None, flg_train_free_par_mean=False,
                 norm_mean_coef=0., norm_scale_coef=1.,
                 Sigma_function=None, Sigma_f_additional_par_list=None, 
                 Sigma_pos_par_init=None, flg_train_Sigma_pos_par=True,
                 Sigma_free_par_init=None, flg_train_Sigma_free_par=True,
                 flg_offset=True,
                 scale_init=np.ones(1), flg_train_scale=False,
                 name='', dtype=torch.float64, sigma_n_num=None, device=None):
        # initialize the linear model
        super(Poly_GP, self).__init__(active_dims=active_dims,
                                      sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                                      f_mean=f_mean, f_mean_add_par_dict=f_mean_add_par_dict,
                                      pos_par_mean_init=pos_par_mean_init, flg_train_pos_par_mean=flg_train_pos_par_mean,
                                      free_par_mean_init=free_par_mean_init, flg_train_free_par_mean=flg_train_free_par_mean,
                                      norm_mean_coef=norm_mean_coef, norm_scale_coef=norm_scale_coef,
                                      Sigma_function=Sigma_function, Sigma_f_additional_par_list=Sigma_f_additional_par_list, 
                                      Sigma_pos_par_init=Sigma_pos_par_init, flg_train_Sigma_pos_par=flg_train_Sigma_pos_par,
                                      Sigma_free_par_init=Sigma_free_par_init, flg_train_Sigma_free_par=flg_train_Sigma_free_par,
                                      flg_offset=flg_offset,
                                      scale_init=np.ones(1), flg_train_scale=False,
                                      name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
        # save the poly deg
        self.poly_deg = poly_deg
        # save the scaling parameter of the poly transformation
        self.scale_log = torch.nn.Parameter(torch.tensor(np.log(scale_init), dtype=self.dtype, device=self.device),
                                            requires_grad=flg_train_scale)
    

    def get_covariance(self, X1, X2=None, flg_noise=False):
        """
        Returns the linear covariance raised to self.poly_deg
        """
        return torch.exp(self.scale_log)*(super(Poly_GP, self).get_covariance(X1, X2, flg_noise))**self.poly_deg


    def get_diag_covariance(self, X, flg_noise=False):
        return torch.exp(self.scale_log)*(super(Poly_GP, self).get_diag_covariance(X,flg_noise=flg_noise))**self.poly_deg


    def get_parameters(self, X, Y, flg_print=False):
        raise NotImplementedError()


    def get_parameters_inv_lemma(self, X, Y, flg_print=False):
        raise NotImplementedError()




class MPK_GP(Linear_GP):
    """
    Implementation of the Multiplicaive Polynomial Kernel
    """
    def __init__(self, active_dims, poly_deg,
                 sigma_n_init=None, flg_train_sigma_n=True,
                 f_mean=None, f_mean_add_par_dict={},
                 pos_par_mean_init=None, flg_train_pos_par_mean=False,
                 free_par_mean_init=None, flg_train_free_par_mean=False,
                 norm_mean_coef=0., norm_scale_coef=1.,
                 Sigma_pos_par_init=None, flg_train_Sigma_pos_par=True,
                 flg_offset=True,
                 name='', dtype=torch.float64, sigma_n_num=None, device=None):
        # init the linear GP object
        Sigma_function = Utils.Parameters_covariance_functions.diagonal_covariance_ARD
        Sigma_f_additional_par_list = []
        super(MPK_GP, self).__init__(active_dims=active_dims,
                                     sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                                     f_mean=f_mean, f_mean_add_par_dict=f_mean_add_par_dict,
                                     pos_par_mean_init=pos_par_mean_init, flg_train_pos_par_mean=flg_train_pos_par_mean,
                                     free_par_mean_init=free_par_mean_init, flg_train_free_par_mean=flg_train_free_par_mean,
                                     norm_mean_coef=norm_mean_coef, norm_scale_coef=norm_scale_coef,
                                     Sigma_function=Sigma_function, Sigma_f_additional_par_list=Sigma_f_additional_par_list, 
                                     Sigma_pos_par_init=Sigma_pos_par_init, flg_train_Sigma_pos_par=flg_train_Sigma_pos_par,
                                     Sigma_free_par_init=None, flg_train_Sigma_free_par=False,
                                     flg_offset=flg_offset,
                                     name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
        self.poly_deg = poly_deg


    def get_covariance(self, X1, X2=None, flg_noise=False):
        #get sigma
        Sigma_matrices = torch.stack([self.Sigma_function(torch.exp(self.Sigma_pos_par[d,:]), None, *self.Sigma_f_additional_par_list)
                                      for d in range(0,self.poly_deg)])
        #get the kernel
        phi_X1 = self.get_phi(X1).unsqueeze_(0)
        if X2 is None:
            K_X =  torch.exp(self.scale_log)*torch.prod(torch.matmul(phi_X1, torch.matmul(Sigma_matrices, phi_X1.transpose(1,2))),0)
            # check if we need to add the noise
            if flg_noise & self.GP_with_noise:
                N = X1.size()[0]
                return K_X + self.get_sigma_n_2()*torch.eye(N, dtype=self.dtype, device=self.device)
            else:
                return K_X
        else:
            phi_X2 = self.get_phi(X2).unsqueeze_(0)
            return torch.exp(self.scale_log)*torch.prod(torch.matmul(phi_X1, torch.matmul(Sigma_matrices, phi_X2.transpose(1,2))),0)



    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the diag of the cov matrix
        """
        #get sigma
        Sigma_matrices = torch.stack([self.Sigma_function(torch.exp(self.Sigma_pos_par[d,:]), None, *self.Sigma_f_additional_par_list)
                                      for d in range(0,self.poly_deg)])
        phi_X = self.get_phi(X).unsqueeze_(0)
        diag = torch.exp(self.scale_log)*torch.sum(torch.matmul(phi_X, Sigma_matrices)*(phi_X), dim=2).prod(0)
        if flg_noise & self.GP_with_noise:
            return diag + self.get_sigma_n_2()
        else:
            return diag




def get_Volterra_MPK_GP(active_dims, poly_deg,
                        sigma_n_init=None, flg_train_sigma_n=True,
                        f_mean=None, f_mean_add_par_dict={},
                        pos_par_mean_init=None, flg_train_pos_par_mean=False,
                        free_par_mean_init=None, flg_train_free_par_mean=False,
                        norm_mean_coef=0., norm_scale_coef=1.,
                        Sigma_pos_par_init_list=[], flg_train_Sigma_pos_par_list=[],
                        name='', dtype=torch.float64, sigma_n_num=None, device=None):
    """
    Returns a Volterra MPK GP:
    the kernel is the sum of poly_deg MPK GP (one for each degree)
    """
    # init the GP list
    gp_list = []
    # get the first order contribution (with noise)
    gp_list.append(MPK_GP(active_dims, poly_deg=1,
                          sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
                          f_mean=f_mean, f_mean_add_par_dict=f_mean_add_par_dict,
                          pos_par_mean_init=pos_par_mean_init, flg_train_pos_par_mean=flg_train_pos_par_mean,
                          free_par_mean_init=free_par_mean_init, flg_train_free_par_mean=flg_train_free_par_mean,
                          norm_mean_coef=norm_mean_coef, norm_scale_coef=norm_scale_coef,
                          Sigma_pos_par_init=Sigma_pos_par_init_list[0], flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[0],
                          flg_offset=True,
                          name='MPK_1', dtype=dtype, sigma_n_num=sigma_n_num, device=device))
    # get the higher order contributions
    for deg in range(1, poly_deg):
        gp_list.append(MPK_GP(active_dims, poly_deg=deg+1,
                              sigma_n_init=None, flg_train_sigma_n=False,
                              Sigma_pos_par_init=Sigma_pos_par_init_list[deg], flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[deg],
                              flg_offset=False,
                              name='MPK_'+str(deg+1), dtype=dtype, sigma_n_num=None, device=device))
    #return the sum of the GPs
    return GP_prior.Sum_Independent_GP(*gp_list)




def get_SOR_GP(exact_GP_object):
    """
    function that returns a Subset of Regressors GP, given a GP object.
    This model is a low-rank approximation of an exact GP model.
    Given a GP model with kernel function k(x_1,x_2), SOR approximate its covariance
    with k_SOR(x_1,x_2) = k(x_1,U) K(U,U)^-1 k(U,x_2), where:
    - U = {set of inducing inputs}
    - K(U,U) = kernel matrix associated to the inducing inputs 
    """


    # create the SOR_GP class dynamically
    class SOR_GP(type(exact_GP_object)):
        """
        SOR_GP
        """
        def __init__(self, exact_GP_object):
            """Initialize the object inheriting all the exact_GP_object parameters"""
            # initialize the GP object randomly
            GP_prior.GP_prior.__init__(self,
                                       active_dims=[0],
                                       sigma_n_init=None,
                                       flg_train_sigma_n=False,
                                       scale_init=np.ones(1),
                                       flg_train_scale=False,
                                       name='',
                                       dtype=torch.float64,
                                       sigma_n_num=None,
                                       device=torch.device('cpu'))
            # assign all the variables of the exact_GP_object
            self.__dict__ = exact_GP_object.__dict__
            self.name = 'SOR_GP '+self.name


        def init_inducing_inputs(self, inducing_inputs, flg_train_inducing_inputs=False):
            """
            set the set U, a matrix of dimension (num_inducing_inputs, num_feature)
            which represent the initial value of the inducing inputs.
            If flg_train_inducing_inputs=True U is considered as a trainable hyperparameter
            """
            self.U = torch.nn.Parameter(torch.tensor(inducing_inputs, dtype=self.dtype, device=self.device),
                                        requires_grad=flg_train_inducing_inputs)


        def set_inducing_inputs_from_data(self, X, Y, threshold, flg_trainable, flg_permutation=False, criterion='std', num_pts = None):
            """
            Set the inducing inputs with an online procedure
            """
            # get number of samples
            num_samples = X.shape[0]
            # init the set of inducing inputs with the first sample
            #inducing_inputs_indices = self.get_SOD(X, Y, threshold, flg_permutation=flg_permutation, criterion=criterion)
            inducing_inputs_indices = self.get_SOD(X, Y, threshold, flg_permutation=flg_permutation, criterion=criterion) if criterion != 'projection_error' else self.get_SOD_projection_error(X, Y, threshold, flg_permutation=flg_permutation, num_pts=num_pts)
            self.U = torch.nn.Parameter(X[inducing_inputs_indices,:], requires_grad=flg_trainable)
            return inducing_inputs_indices


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
                E = threshold +1.
                if E > threshold:
                    # chec if Sigma_inv is invertible
                    U_tmp = torch.cat([SOD,X[sample_index:sample_index+1,:]],0)
                    K_XU_tmp = self.get_covariance(X,U_tmp)
                    K_UU_tmp = self.get_covariance(U_tmp)
                    sigma_n_square = self.get_sigma_n_2()
                    sigma_n_square_inv = 1/sigma_n_square
                    # compute the parameters
                    Sigma_inv_tmp = K_UU_tmp + sigma_n_square_inv*torch.matmul(K_XU_tmp.transpose(0,1), K_XU_tmp)
                    if torch.linalg.matrix_rank(Sigma_inv_tmp)==Sigma_inv_tmp.shape[0]:
                        # update SOD and training model
                        SOD = torch.cat([SOD,X[sample_index:sample_index+1,:]],0)
                        inducing_inputs_indices.append(sample_index)
                        # update training model
                        K_X_inv = GP_prior.incremental_inv(Q=K_X_test_X.t(),
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


        def get_SOR_alpha(self, X, Y):
            """
            Returns the coefficients that defines the SOR posterior distribution
            
            inputs:
            - X = training inputs
            - Y = training outputs

            outputs:
            - alpha = vector defining the SOR posterior distribution
            - m_X = prior mean of X
            - Sigma = inverse of (K_UU + sigma_n^-2*K_UX*K_XU)
            """
            # get the prior mean and the covariances
            m_X = self.get_mean(X)
            Y = Y - m_X
            K_XU = self.get_covariance(X,self.U)
            K_UU = self.get_covariance(self.U)
            sigma_n_square = self.get_sigma_n_2()
            sigma_n_square_inv = 1/sigma_n_square
            # compute the parameters
            Sigma_inv = K_UU + sigma_n_square_inv*torch.matmul(K_XU.transpose(0,1), K_XU)            
            L_Sigma_inv = torch.linalg.cholesky(Sigma_inv)
            Sigma = torch.cholesky_inverse(L_Sigma_inv)            
            SOR_alpha = sigma_n_square_inv*torch.matmul(torch.matmul(Sigma, K_XU.transpose(0,1)), Y)
            return SOR_alpha, m_X, Sigma


        def get_SOR_estimate_from_alpha(self, X_test, SOR_alpha, Sigma=None):
            """
            Compute the SOR posterior distribution in X_test, given the alpha vector.
            
            input:
            - X = training input locations (used to compute alpha)
            - X_test = test input locations
            - SOR_alpha = vector of coefficients defining the SOR posterior
            - Sigma = inverse of (K_UU + sigma_n^-2*K_UX*K_XU)
    
            output:
            - Y_hat = posterior mean
            - var = diagonal elements of the posterior variance (if SIgma is given)
            - m_X_test = prior mean in X_test
            """
            # get covariance and prior mean
            K_X_test_U = self.get_covariance(X_test, self.U)
            m_X_test = self.get_mean(X_test)
            # get the estimate 
            Y_hat = m_X_test + torch.matmul(K_X_test_U, SOR_alpha)
            # if Sigma is given compute the confidence intervals
            if Sigma is not None:
                var = torch.sum(torch.matmul(K_X_test_U, Sigma)*(K_X_test_U), dim=1)
                # var = self.get_sigma_n_2()*torch.sum(torch.matmul(K_X_test_U, Sigma)*(K_X_test_U), dim=1)
            else:
                var = None
            return Y_hat, var, m_X_test
         

        def get_SOR_estimate(self, X, Y, X_test, flg_return_Sigma=False):
            """
            Returns the SOR posterior distribution in X_test, given the training samples X Y and the inducing inputs self.U.
            
            input:
            - X = training input
            - Y = training output
            - X_test = test input
            
            output:
            - Y_hat = mean of the test posterior
            - var = diagonal elements of the variance posterior
            - SOR_alpha = coefficients defining the SOR posterior
            - m_X = prior mean of the training samples
            - Sigma = (K_UU + sigma_n^-2*K_UX*K_XU)
            """
            # get the coefficent and the mean
            SOR_alpha, m_X, Sigma = self.get_SOR_alpha(X, Y)
            # get the estimate and the confidence intervals
            Y_hat, var, m_X_test = self.get_SOR_estimate_from_alpha(X_test, SOR_alpha, Sigma=Sigma)
            # return the opportune values
            if flg_return_Sigma:
                return Y_hat, var, SOR_alpha, m_X, Sigma
            else:
                return Y_hat, var, SOR_alpha, m_X_test


        def SOR_forward(self, X):
            """
            Returns the elements that define the likelihood distribution of the model
            when considering the SOR approximation
            
            input:
            - X = training inputs (X has dimension [num_samples, num_features])
            
            output:
            - m_X = mean
            - K_X = None
            - K_X_inv = inverse of (K_XU*K_UU^-1*K_UX+sigma_n^2)
            - log_det = (K_XU*K_UU^-1*K_UX+sigma_n^2)
            """
            # get the mean
            m_X = self.get_mean(X)
            # get kernel matrices
            K_X = None
            N = X.shape[0]
            K_UU = self.get_covariance(self.U, None, True)
            #K_UU = self.get_covariance(self.U)
            K_XU = self.get_covariance(X, self.U)
            sigma_n_square_inv = 1/self.get_sigma_n_2()
            # compute the K_UU^-1 logdet
            L_K_UU = torch.linalg.cholesky(K_UU)
            K_UU_inv_log_det = -2*torch.sum(torch.log(torch.diag(L_K_UU)))
            # compute Sigma
            Sigma_inv = K_UU + sigma_n_square_inv*torch.matmul(K_XU.transpose(0,1), K_XU)
            # compute Sigma inverse and logdet
            L_Sigma_inv = torch.linalg.cholesky(Sigma_inv)
            Sigma_inv_log_det = 2*torch.sum(torch.log(torch.diag(L_Sigma_inv)))
            Sigma = K_X_inv = torch.cholesky_inverse(L_Sigma_inv)
            # compute K_X_inv
            K_X_inv = sigma_n_square_inv*torch.eye(N, dtype=self.dtype, device=self.device)
            K_X_inv -= sigma_n_square_inv**2*torch.matmul(K_XU, torch.matmul(Sigma, K_XU.transpose(0,1)))
            # compute the log_det
            log_det = N*torch.log(self.get_sigma_n_2()) + K_UU_inv_log_det + Sigma_inv_log_det
            return m_X, K_X, K_X_inv, log_det


        def fit_SOR_model(self,trainloader=None, 
                          optimizer=None, criterion=None,
                          N_epoch=1, N_epoch_print=1,
                          f_saving_model=None, f_print=None):
            """
            Optimize the SOR model hyperparameters
    
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
                    out_SOR_GP_priors = self.SOR_forward(inputs)
                    loss = criterion(out_SOR_GP_priors, labels)
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
                    print('Time elapsed:', t_stop-t_start)
                    if f_saving_model is not None:
                        f_saving_model(epoch)
                    if f_print is not None:
                        f_print()
                    t_start = time.time()
            # print the final parameters
            print('\nFinal parameters:')
            self.print_model()



    # init the object and return
    return SOR_GP(exact_GP_object)




def get_nystrom_GP(exact_GP_object, X_nystrom, R_max, init_dict, R_perc=-1):
    """
    Function that computes the nystrom approximation, given a GP object.
    This model is a low-rank approximation of an exact GP model.
    The algorithm derives a linear kernel approximating the 
    kernel matrix in X_nystrom with a rank-R matrix
    """
    with torch.no_grad():
        # Get the covariance in X_nystrom
        M, D = X_nystrom.shape
        K_X_nystrom = exact_GP_object.get_covariance(X_nystrom)
        # Get the eigendecomposition
        d, V = torch.linalg.eigh(K_X_nystrom)
        # Get the nystrom basis lambda function
        R = min(R_max, sum(d/d[-1]>=R_perc).detach().cpu().numpy())
        print('\nNystrom rank:', R)
        d_r_sqrt = torch.sqrt(d[M-R:M])
        V_r = V[:,M-R:M]
        f_nystrom = lambda X: torch.matmul(exact_GP_object.get_covariance(X,X_nystrom), V_r)/d_r_sqrt
        # Get a linear GP in the nystrom basis
        active_dims = np.arange(0,R)
    init_dict['Sigma_function']=Utils.Parameters_covariance_functions.diagonal_covariance_ARD
    init_dict['Sigma_f_additional_par_list']=[]
    init_dict['Sigma_pos_par_init'] = np.ones(R)
    init_dict['flg_train_Sigma_pos_par'] = True
    init_dict['Sigma_free_par_init'] = None
    init_dict['flg_train_Sigma_free_par'] = False
    return Linear_GP(active_dims,
                     f_transform=f_nystrom,
                     f_add_par_list=[],
                     name='Nystrom('+exact_GP_object.name+')',
                     dtype=exact_GP_object.dtype,
                     device=exact_GP_object.device,
                     **init_dict)
