# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import torch
import os
import pickle as pkl
import scipy
from tqdm import trange

from policy_learning.Policy import Policy, Sum_of_gaussians

class TossingPolicy(Sum_of_gaussians):
    """
        Considers an extended state: (state, target-state)

        Input dim is constrained to be a velocity vector oriented from current position to target
    """

    def __init__(self, state_dim, num_basis, target_indeces, alpha, pos_indeces,
                 flg_train_lengthscales=True, lengthscales_init=None,
                 flg_train_centers=True, centers_init=None, centers_init_min=-1, centers_init_max=1,
                 weight_init=None, flg_train_weight=True,
                 flg_bias=False, bias_init=None, flg_train_bias=False,
                 flg_squash=False, squash_name=None, u_max=1, flg_drop=True,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(TossingPolicy, self).__init__(state_dim=3, input_dim=1, num_basis=num_basis,
                                            flg_train_lengthscales=flg_train_lengthscales,
                                            lengthscales_init=lengthscales_init,
                                            flg_train_centers=flg_train_centers, centers_init=centers_init,
                                            centers_init_min=centers_init_min, centers_init_max=centers_init_max,
                                            weight_init=weight_init, flg_train_weight=flg_train_weight,
                                            flg_bias=flg_bias, bias_init=bias_init, flg_train_bias=flg_train_bias,
                                            flg_squash=flg_squash, u_max=u_max, flg_drop=flg_drop,
                                            dtype=dtype, device=device)
        if flg_squash:
            self.rectifier = torch.nn.Softplus()

        self.target_indeces = target_indeces
        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)
        self.pos_indeces = pos_indeces
        self.input_dim = 1

    def squashing(self, u, u_max):
        """
        Squash the inputs inside (0, +u_max)
        """
        if not np.isscalar(u_max):
            u_max = torch.tensor(u_max, dtype=self.dtype, device=self.device)

        # print('U: ')
        # print(u)

        return u_max * torch.tanh(self.rectifier(u) / u_max)

        # if squash_name is None or squash_name == 'sigmoid':
        #     return u_max*torch.sigmoid(u/u_max)
        # else:  # if squash_name == 'tanh':
        #     return u_max*(torch.tanh(u/u_max)+1)/2

    def get_v(self, states, t, p_dropout):
        """
            Tossing policy is only applied at starting state
        """
        #x = states[:, self.pos_indeces]
        target = states[:, self.target_indeces]

        #dist = target - x
        # print('Dist: ')
        # print(dist)
        # Orientation of tossing is just yaw to align x axis to distance vector
        gamma = torch.atan2(target[:, 1], target[:, 0])
        # print('Gamma')
        # print(gamma)

        if t > 0:
            return torch.zeros(states.size(dim=0), 1, device=self.device, dtype=self.dtype), gamma
        else:
            # return self.f_squash(super().forward(dist, t, p_dropout)), gamma  # one dimensional for each state
            return super().forward(target, t, p_dropout), gamma  # one dimensional for each state

    def forward(self, states, t=None, p_dropout=0.0):
        """
            Tossing is performed in the direction of the distance vector, with angle w.r.t. horizontal plane given
            by parameter alpha.
        """

        sample_size = states.size(dim=0)
        v, gamma = self.get_v(states, t, p_dropout)

        # print('V + gamma: ')
        # print(v)
        # print(gamma)

        v = v.reshape(sample_size)
        gamma = gamma.reshape(sample_size)

        out = torch.zeros(sample_size, 3, device=self.device, dtype=self.dtype)  # outputs a cartesian velocity vector
        out[:, 0] = v * torch.cos(self.alpha) * torch.cos(gamma)
        out[:, 1] = v * torch.cos(self.alpha) * torch.sin(gamma)
        out[:, 2] = v * torch.sin(self.alpha)

        # print('OUT')
        # print(out)

        return out


class TossingPolicyTarget(Sum_of_gaussians):
    """
        Considers an extended state: (state, target-state)

        Input dim is constrained to be a velocity vector oriented from current position to target
    """

    def __init__(self, state_dim, num_basis, target_indeces, alpha, pos_indeces,
                 flg_train_lengthscales=True, lengthscales_init=None,
                 flg_train_centers=True, centers_init=None, centers_init_min=-1, centers_init_max=1,
                 weight_init=None, flg_train_weight=True,
                 flg_bias=False, bias_init=None, flg_train_bias=False,
                 flg_squash=False, squash_name=None, u_max=1, flg_drop=True,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(TossingPolicyTarget, self).__init__(state_dim=len(target_indeces), input_dim=1, num_basis=num_basis,
                                                  flg_train_lengthscales=flg_train_lengthscales,
                                                  lengthscales_init=lengthscales_init,
                                                  flg_train_centers=flg_train_centers, centers_init=centers_init,
                                                  centers_init_min=centers_init_min, centers_init_max=centers_init_max,
                                                  weight_init=weight_init, flg_train_weight=flg_train_weight,
                                                  flg_bias=flg_bias, bias_init=bias_init, flg_train_bias=flg_train_bias,
                                                  flg_squash=flg_squash, u_max=u_max, flg_drop=flg_drop,
                                                  dtype=dtype, device=device)

        self.target_indeces = target_indeces
        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)
        self.pos_indeces = pos_indeces
        self.input_dim = 1

    def squashing(self, u, u_max):
        """
        Squash the inputs inside (0, +u_max)
        """
        if not np.isscalar(u_max):
            u_max = torch.tensor(u_max, dtype=self.dtype, device=self.device)

        return (u_max/2) * torch.tanh(u / u_max) + (u_max/2)

    def get_v(self, states, t=0.0, p_dropout=0.0):
        """
            Tossing policy is only applied at starting state
        """
        target = states[:, self.target_indeces]
        # Orientation of tossing is just yaw to align x axis to distance vector
        gamma = torch.atan2(target[:, 1], target[:, 0])

        if t > 0:
            return torch.zeros(states.size(dim=0), 1, device=self.device, dtype=self.dtype), gamma
        else:
            return super().forward(target, t, p_dropout), gamma  # one dimensional for each state

    def forward(self, states, t=None, p_dropout=0.0):
        """
            Tossing is performed in the direction of the distance vector, with angle w.r.t. horizontal plane given
            by parameter alpha.
        """

        sample_size = states.size(dim=0)
        v, gamma = self.get_v(states, t, p_dropout)

        # print('V + gamma: ')
        # print(v)
        # print(gamma)

        v = v.reshape(sample_size)
        gamma = gamma.reshape(sample_size)

        out = torch.zeros(sample_size, 3, device=self.device, dtype=self.dtype)  # outputs a cartesian velocity vector
        out[:, 0] = v * torch.cos(self.alpha) * torch.cos(gamma)
        out[:, 1] = v * torch.cos(self.alpha) * torch.sin(gamma)
        out[:, 2] = v * torch.sin(self.alpha)

        # print('OUT')
        # print(out)

        return out


class TossingPolicySimple(Sum_of_gaussians):
    """
        Considers an extended state: (state, target-state)

        Input dim is constrained to be a velocity vector oriented from current position to target
    """

    def __init__(self, state_dim, num_basis, target_indeces, alpha, pos_indeces,
                 flg_train_lengthscales=True, lengthscales_init=None,
                 flg_train_centers=True, centers_init=None, centers_init_min=-1, centers_init_max=1,
                 weight_init=None, flg_train_weight=True,
                 flg_bias=False, bias_init=None, flg_train_bias=False,
                 flg_squash=False, squash_name=None, u_max=1, flg_drop=True,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(TossingPolicySimple, self).__init__(state_dim=2, input_dim=1, num_basis=num_basis,
                                                  flg_train_lengthscales=flg_train_lengthscales,
                                                  lengthscales_init=lengthscales_init,
                                                  flg_train_centers=flg_train_centers, centers_init=centers_init,
                                                  centers_init_min=centers_init_min, centers_init_max=centers_init_max,
                                                  weight_init=weight_init, flg_train_weight=flg_train_weight,
                                                  flg_bias=flg_bias, bias_init=bias_init, flg_train_bias=flg_train_bias,
                                                  flg_squash=flg_squash, u_max=u_max, flg_drop=flg_drop,
                                                  dtype=dtype, device=device)
        if flg_squash:
            self.rectifier = torch.nn.Softplus()

        self.target_indeces = target_indeces
        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)
        self.pos_indeces = pos_indeces
        self.input_dim = 1

    def squashing(self, u, u_max):
        """
        Squash the inputs inside (0, +u_max)
        """
        if not np.isscalar(u_max):
            u_max = torch.tensor(u_max, dtype=self.dtype, device=self.device)

        #return u_max * torch.tanh(self.rectifier(u) / u_max)
        return self.rectifier(u_max * torch.tanh(u / u_max))

    def get_v(self, states, t, p_dropout):
        """
            Tossing policy is only applied at starting state
        """
        x = states[:, self.pos_indeces]
        target = states[:, self.target_indeces]

        dist = target - x
        # print('Dist: ')
        # print(dist)
        # Orientation of tossing is just yaw to align x axis to distance vector
        gamma = torch.atan2(target[:, 1], target[:, 0])
        # print('Gamma')
        # print(gamma)

        input = torch.zeros((dist.shape[0], 2), device=self.device, dtype=self.dtype)

        for i in range(dist.shape[0]):
            input[i, 0] = torch.linalg.norm(dist[i, :2])  # distance on the horizontal plane

        input[:, 1] = dist[:, 2]  # vertical distance

        #print(input)

        if t > 0:
            return torch.zeros(states.size(dim=0), 1, device=self.device, dtype=self.dtype), gamma
        else:
            # return self.f_squash(super().forward(dist, t, p_dropout)), gamma  # one dimensional for each state
            return super().forward(input, t, p_dropout), gamma  # one dimensional for each state

    def forward(self, states, t=None, p_dropout=0.0):
        """
            Tossing is performed in the direction of the distance vector, with angle w.r.t. horizontal plane given
            by parameter alpha.
        """

        sample_size = states.size(dim=0)
        v, gamma = self.get_v(states, t, p_dropout)

        # print('V + gamma: ')
        # print(v)
        # print(gamma)

        v = v.reshape(sample_size)
        gamma = gamma.reshape(sample_size)

        out = torch.zeros(sample_size, 3, device=self.device, dtype=self.dtype)  # outputs a cartesian velocity vector
        out[:, 0] = v * torch.cos(self.alpha) * torch.cos(gamma)
        out[:, 1] = v * torch.cos(self.alpha) * torch.sin(gamma)
        out[:, 2] = v * torch.sin(self.alpha)

        # print('OUT')
        # print(out)

        return out

class gravityPolicy(Policy):
    """
            Considers an extended state: (state, target-state)

            Input dim is constrained to be a velocity vector oriented from current position to target
        """

    def __init__(self, state_dim, alpha, release_pos, target_indeces, g=9.81,
                 flg_squash=False,  u_max=1, load_file='', v_mult=1.0,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(Policy, self).__init__()

        self.dtype = dtype
        self.device = device
        self.target_indeces = target_indeces
        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)
        self.release_pos = release_pos
        self.g = g
        self.v_mult = v_mult

        if not os.path.isfile(load_file):
            print('Setting identity function as model error estimate')
            self.f = lambda y: y * self.v_mult # identity
        else:
            print('Setting poly as model error estimate')
            poly_coeff = pkl.load(open(load_file, 'rb'))[0]
            print('Poly coeff: {}'.format(poly_coeff))
            poly = np.poly1d(poly_coeff)
            self.f = lambda y: poly(y) * self.v_mult  # estimated noise model

    def get_v(self, states, t=None, p_dropout=0.0):
        target = states[:, self.target_indeces]

        d = torch.sqrt(target[:, 0]**2 + target[:, 1]**2) - self.release_pos[0]
        h = self.release_pos[2] - target[:, 2]

        gamma = torch.atan2(target[:, 1], target[:, 0])

        # v_r = torch.sqrt(self.g * d**2 / (d + h)) # alpha = 45 deg
        v_r = torch.sqrt(0.5 * self.g * (d**2) / ((torch.cos(self.alpha))**2 * (torch.tan(self.alpha) * d + h)))

        v = torch.zeros_like(v_r)
        for i in range(v_r.size(0)):
            v[i] = self.f(v_r[i])

        return v, gamma

    def forward(self, states, t=None, p_dropout=0.0):
        return self.get_v(states, t, p_dropout)


class gravityPolicyFriction(Policy):
    """
            Considers an extended state: (state, target-state)

            Input dim is constrained to be a velocity vector oriented from current position to target

            Solves the equations of projectile motion under stokes drag
            b = 6 * pi * nabla * R

            (nabla= dyn viscosity of air @ 15°C)
        """

    def __init__(self, state_dim, alpha, release_pos, target_indeces, g=9.81, b=7.302883450681761e-06,
                 flg_squash=False,  u_max=1, v_mult=1.0,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(Policy, self).__init__()

        self.dtype = dtype
        self.device = device
        self.target_indeces = target_indeces
        self.alpha = alpha
        self.release_pos = release_pos
        self.g = g
        self.b = b
        self.u_max = u_max
        self.v_mult = v_mult

    def get_v(self, states, t=None, p_dropout=0.0):
        target = states[:, self.target_indeces].detach().cpu().numpy()
        tensor_target = states[:, self.target_indeces]

        dist = np.sqrt(target[:, 0]**2 + target[:, 1]**2) - self.release_pos[0]
        height = -(self.release_pos[2] - target[:, 2])
        print('dist: {}'.format(dist))
        print('height: {}'.format(height))

        gamma = torch.atan2(tensor_target[:, 1], tensor_target[:, 0]) #.detach().cpu().numpy()

        cos_alpha = np.cos(self.alpha)
        tan_alpha = np.tan(self.alpha)
        if cos_alpha <= 0.0:
            raise ValueError('gravityPolicyFriction requires cos(alpha) > 0')

        # Angle-aware vacuum solution.  The previous expression was the
        # special case for a 45-degree release and was also (incorrectly) used
        # as a lower bound on the Stokes solution.  That forced severe
        # overshoots when the current catapult trajectory changed to -1 deg.
        denominator = cos_alpha**2 * (tan_alpha * dist - height)
        if np.any(denominator <= 0.0):
            raise ValueError(
                'Target is not reachable by the configured release angle'
            )
        v_0 = np.sqrt(0.5 * self.g * dist**2 / denominator)

        v = np.zeros(gamma.size(0))

        for i in range(tensor_target.size(0)):
            d = float(dist[i])
            vacuum_speed = float(v_0[i])

            if d <= 0.0 or self.b == 0.0:
                solution = vacuum_speed
            else:
                if self.b < 0.0:
                    raise ValueError('Stokes drag coefficient must be non-negative')

                def log1p_plus_u(u):
                    # log(1-u) + u loses precision for the small values of u
                    # produced by the physical Stokes coefficient.  Evaluate
                    # its series directly in that regime.
                    if abs(u) < 1e-4:
                        return -(u**2 / 2.0 + u**3 / 3.0 +
                                 u**4 / 4.0 + u**5 / 5.0)
                    return np.log1p(-u) + u

                def vertical_residual(speed):
                    u = self.b * d / (speed * cos_alpha)
                    if u >= 1.0:
                        return -np.inf
                    z_fall = (
                        (self.g / self.b**2) * log1p_plus_u(u)
                        + d * tan_alpha
                    )
                    return z_fall - float(height[i])

                # The horizontal Stokes solution requires u < 1.  Use a
                # bounded scalar root solve instead of fsolve, then apply only
                # the actuator limit--not the obsolete 45-degree speed floor.
                lower = max(
                    np.finfo(float).tiny,
                    self.b * d / cos_alpha * (1.0 + 1e-12),
                )
                upper = max(vacuum_speed * 1.01, 1.0)
                while vertical_residual(upper) <= 0.0:
                    upper *= 2.0
                solution = scipy.optimize.brentq(
                    vertical_residual,
                    lower,
                    upper,
                    xtol=1e-12,
                    rtol=1e-12,
                )

            print(np.array([solution]))
            print(v_0)

            v[i] = np.clip(solution, 0.0, self.u_max) * self.v_mult
            print(v[i])

        return torch.tensor(v, dtype=self.dtype, device=self.device), gamma

    def forward(self, states, t=None, p_dropout=0.0):
        return self.get_v(states, t, p_dropout)


class TossingNNPolicy(Policy):
    """
        Considers an extended state: (state, target-state)

        Input dim is constrained to be a velocity vector oriented from current position to target
    """

    def __init__(self, state_dim, target_indeces, alpha, pos_indeces, num_hidden=1, hidden_dim=200, activation_name='ReLU',
                 flg_squash=False, u_max=1, dtype=torch.float64, device=torch.device('cpu')):
        super(TossingNNPolicy, self).__init__(state_dim=state_dim, input_dim=1, flg_squash=True, u_max=u_max, dtype=dtype,
                                              device=device)


        self.target_indeces = target_indeces
        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)
        self.pos_indeces = pos_indeces
        self.input_dim = 1
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.activation_name = activation_name
        self.u_max = u_max

        self.hidden = []
        self.hidden_activation = []
        self.hidden.append(torch.nn.Linear(len(target_indeces), self.hidden_dim, dtype=self.dtype, device=self.device))
        if self.activation_name == 'ReLU':
            self.hidden_activation.append(torch.nn.ReLU())
        else:
            raise NotImplementedError('only ReLU is currently implemented.')

        for i in range(self.num_hidden-1):
            self.hidden.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype, device=self.device))
            if self.activation_name == 'ReLU':
                self.hidden_activation.append(torch.nn.ReLU())
        self.hidden = torch.nn.ParameterList(self.hidden)
        self.hidden_activation = torch.nn.ParameterList(self.hidden_activation)

        self.out_layer = torch.nn.Linear(self.hidden_dim, self.input_dim, dtype=self.dtype, device=self.device)


    def squashing(self, u, u_max):
        return (u_max/2) + (u_max/2) * torch.tanh(u/u_max)


    def get_v(self, states, t=0, p_dropout=0.0):
        """
            Tossing policy is only applied at starting state
        """
        target = states[:, self.target_indeces]
        gamma = torch.atan2(target[:, 1], target[:, 0])

        if t > 0:
            return torch.zeros(states.size(dim=0), 1, device=self.device, dtype=self.dtype), gamma
        else:
            out = target
            for i in range(self.num_hidden):
                out = self.hidden[i](out)
                out = self.hidden_activation[i](out)

            out = self.out_layer(out)

            return self.squashing(out, self.u_max), gamma  # one dimensional for each state

    def forward(self, states, t=None, p_dropout=0.0):
        """
            Tossing is performed in the direction of the distance vector, with angle w.r.t. horizontal plane given
            by parameter alpha.
        """

        sample_size = states.size(dim=0)
        v, gamma = self.get_v(states, t, p_dropout)
        v = v.reshape(sample_size)
        gamma = gamma.reshape(sample_size)

        out = torch.zeros(sample_size, 3, device=self.device, dtype=self.dtype)  # outputs a cartesian velocity vector
        out[:, 0] = v * torch.cos(self.alpha) * torch.cos(gamma)
        out[:, 1] = v * torch.cos(self.alpha) * torch.sin(gamma)
        out[:, 2] = v * torch.sin(self.alpha)

        return out


    def to_np(self):
        parameters = {
            'hidden_layers': [[hidden_l.weight.data.detach().cpu().numpy(), hidden_l.bias.data.detach().cpu().numpy()]
                              for hidden_l in self.hidden],
            'hidden_activation': self.activation_name,
            'out_layer': [self.out_layer.weight.data.detach().cpu().numpy(),
                          self.out_layer.bias.data.detach().cpu().numpy()]}

        return parameters

class TossingResidualNNPolicy(TossingNNPolicy):
    """
        Considers an extended state: (state, target-state)

        Input dim is constrained to be a velocity vector oriented from current position to target

        Outputs gravity_policy(target) + NN(target)

        Trained on residuals of tossing trials
    """
    def __init__(self, state_dim, target_indeces, alpha, pos_indeces, num_hidden=1, hidden_dim=200, activation_name='ReLU',
                 flg_squash=False, u_max=1, dtype=torch.float64, device=torch.device('cpu')):
        super(TossingResidualNNPolicy, self).__init__(state_dim=state_dim, target_indeces=target_indeces, alpha=alpha,
                                                      pos_indeces=pos_indeces, num_hidden=num_hidden,
                                                      hidden_dim=hidden_dim, activation_name=activation_name,
                                                      flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device)

    def get_v(self, states, t=0, p_dropout=0.0):
        """
            Tossing policy is only applied at starting state
        """
        target = states[:, self.target_indeces]
        dist = np.sqrt(target[:, 0]**2 + target[:, 1]**2) - self.release_pos[0]
        height = self.release_pos[2] - target[:, 2]
        # get gravity
        v_0 = np.sqrt(self.g * dist**2 / (dist + height))
        delta_v, gamma = super().get_v(states, t, p_dropout)

        return v_0 + delta_v, gamma



def train_model(model, X, Y, optimizer, opt_steps, batch_size, loss_fn):
    """
        Train Tossing model
    """
    running_loss = 0.
    last_loss = 0.

    pbar = trange(opt_steps)
    for epoch in pbar:
        # Every data instance is an input + label pair
        batch_indices = np.random.choice(range(X.shape[0]), batch_size)
        inputs, labels = X[batch_indices,:], Y[batch_indices]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # outputs = torch.squeeze(model.get_v(inputs)[0])
        outputs = model.get_v(inputs)[0]

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        last_loss = loss.item()
        # Gather data and report
        pbar.set_description('  batch {} loss: {}'.format(epoch + 1, last_loss/batch_size))

    print(labels)
    print(outputs)
    return model
