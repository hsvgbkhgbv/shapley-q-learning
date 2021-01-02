import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ShapleyQMixer(nn.Module):
    def __init__(self, args):
        super(ShapleyQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.arch = args.arch
        self.embed_dim = args.mixing_embed_dim
        self.n_actions = args.n_actions

        self.sample_size = args.sample_size

        # w(s,u) f(s,u) g(s,u)
        if self.arch == "partial_action_observation":
            # w,f,g takes [state, u] as input
            w_input_size = self.state_dim + self.n_actions
            f_input_size = self.state_dim + self.n_actions
            g_input_size = self.state_dim + self.n_actions
        elif self.arch == "history_action_observation":
            # w,f,g takes [state, agent_action_observation_encodings]
            w_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions
            f_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions
            g_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions
        else:
            raise Exception("{} is not a valid ShapleyQ architecture".format(self.arch))

        if self.args.network_size == "small":
            w_list = [ nn.Sequential(nn.Linear(w_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1),
                                     nn.Tanh()
                                ) 
                                for _ in range(self.n_agents)
                            ]
            self.w_list = nn.ModuleList(w_list)
            
            f_list = [ nn.Sequential(nn.Linear(f_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1),
                                     nn.Sigmoid()
                                ) 
                                for _ in range(self.n_agents)
                            ]
            self.f_list = nn.ModuleList(f_list)

            g_list = [ nn.Sequential(nn.Linear(g_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1),
                                     nn.Sigmoid()
                                ) 
                                for _ in range(self.n_agents)
                            ]
            self.g_list = nn.ModuleList(g_list)

        elif self.args.network_size == "big":
            w_list = [ nn.Sequential(nn.Linear(w_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1)
                                ) 
                                for _ in range(self.n_agents)
                            ]
            self.w_list = nn.ModuleList(w_list)
            
            f_list = [ nn.Sequential(nn.Linear(f_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1)
                                ) 
                                for _ in range(self.n_agents)
                            ]
            self.f_list = nn.ModuleList(f_list)

            g_list = [ nn.Sequential(nn.Linear(g_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1)
                                ) 
                                for _ in range(self.n_agents)
                            ]
            self.g_list = nn.ModuleList(g_list)

        else:
            assert False

    def sample_grandcoalitions(self, batch_size):
        """
        E.g. batch_size = 2, n_agents = 3:

        >>> grand_coalitions
        tensor([[2, 0, 1],
                [1, 2, 0]])

        >>> subcoalition_map
        tensor([[[[1., 1., 1.],
                [1., 0., 0.],
                [1., 1., 0.]]],

                [[[1., 1., 0.],
                [1., 1., 1.],
                [1., 0., 0.]]]])

        >>> individual_map
        tensor([[[[0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.]]],

                [[[0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]]]])
        """
        seq_set = th.tril(th.ones(self.n_agents, self.n_agents).cuda(), diagonal=0, out=None)
        grand_coalitions = th.multinomial(th.ones(batch_size*self.sample_size, 
                                          self.n_agents).cuda()/self.n_agents, 
                                          self.n_agents, 
                                          replacement=False)
        individual_map = th.zeros(batch_size*self.sample_size*self.n_agents, self.n_agents).cuda()
        individual_map.scatter_(1, grand_coalitions.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_agents, self.n_agents)
        subcoalition_map = th.matmul(individual_map, seq_set)
        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, 
                                                                self.n_agents, 
                                                                self.n_agents).contiguous().view(batch_size, 
                                                                                                 self.sample_size, 
                                                                                                 self.n_agents, 
                                                                                                 self.n_agents) # shape = (b, n_s, n, n)
        return subcoalition_map, individual_map, grand_coalitions

    def get_f_estimate(self, states, actions):
        f_estimates = []
        for i in range(self.n_agents):
            inputs = th.cat([states, actions[:, i, :]], dim=1)
            f_estimates.append(self.f_list[i](inputs))
        f_estimates = th.stack(f_estimates, dim=1) # shape = (b, n, 1)
        return f_estimates
    
    def get_g_estimate(self, states, actions):
        g_estimates = []
        for i in range(self.n_agents):
            inputs = th.cat([states, actions[:, i, :]], dim=1)
            g_estimates.append(self.g_list[i](inputs))
        g_estimates = th.stack(g_estimates, dim=1) # shape = (b, n, 1)
        return g_estimates

    def get_w_estimate(self, states, actions):
        w_estimates = []
        for i in range(self.n_agents):
            inputs = th.cat([states, actions[:, i, :]], dim=1)
            w_estimates.append(self.w_list[i](inputs))
        w_estimates = th.stack(w_estimates, dim=1) # shape = (b, n, 1)
        return w_estimates

    def get_marginal_contribution(self, states, actions):
        batch_size = states.size(0)
        subcoalition_map, individual_map, grand_coalitions = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 1) # shape = (b, n_s, n, n, 1)
        
        w_est = self.get_w_estimate(states, actions) # shape = (b, n, 1)
        f_est = self.get_f_estimate(states, actions) # shape = (b, n, 1)
        g_est = self.get_g_estimate(states, actions) # shape = (b, n, 1)

        f_est = f_est.unsqueeze(1).unsqueeze(2).expand(batch_size, 
                                                       self.sample_size, 
                                                       self.n_agents, 
                                                       self.n_agents, 
                                                       1).gather(3, grand_coalitions) # shape = (b, n, 1) -> (b, 1, 1, n, 1) -> (b, n_s, n, n, 1)
        
        g_est = g_est.unsqueeze(1).unsqueeze(2).expand(batch_size, 
                                                       self.sample_size, 
                                                       self.n_agents, 
                                                       self.n_agents, 
                                                       1).gather(3, grand_coalitions) # shape = (b, n, 1) -> (b, 1, 1, n, 1) -> (b, n_s, n, n, 1)

        subcoalition_map = subcoalition_map.unsqueeze(-1).float() # shape = (b, n_s, n, n, 1)
        individual_map = individual_map.unsqueeze(-1).float() # shape = (b, n_s, n, n, 1)
        # remove agent i from the subcloation map
        subcoalition_map = subcoalition_map - individual_map
        # flip the subcoalition map
        mask_subcoalition_map = 1 - subcoalition_map

        f_est_subcoalition = f_est * subcoalition_map
        # TODO: check
        f_est_subcoalition = f_est_subcoalition + mask_subcoalition_map
        f_est_subcoalition = f_est_subcoalition.contiguous().view(batch_size, self.sample_size, self.n_agents, -1) # shape = (b, n_s, n, n)
        f_est_individual = f_est * individual_map
        f_est_individual = f_est_individual.contiguous().view(batch_size, self.sample_size, self.n_agents, -1) # shape = (b, n_s, n, n)
        g_est_individual = g_est * individual_map
        g_est_individual = g_est_individual.contiguous().view(batch_size, self.sample_size, self.n_agents, -1) # shape = (b, n_s, n, n)
        aux_ones = th.ones(self.n_agents).cuda()
        
        # normal_marginal_contribution
        # normal_marginal_contribution = f_est_subcoalition.prod(dim=-1) * (th.matmul(f_est_individual, aux_ones) - 1) - th.matmul(g_est_individual, aux_ones) # shape = (b, n_s, n)
        normal_marginal_contribution = f_est_subcoalition.prod(dim=-1) + th.matmul(f_est_individual, aux_ones) - th.matmul(g_est_individual, aux_ones) # shape = (b, n_s, n)
        normal_marginal_contribution = normal_marginal_contribution.unsqueeze(-1) # shape = (b, n_s, n, 1)

        # optimal_marginal_contribution
        # optimal_marginal_contribution = f_est_subcoalition.prod(dim=-1) * (th.matmul(f_est_individual, aux_ones) - 1) # shape = (b, n_s, n)
        optimal_marginal_contribution = f_est_subcoalition.prod(dim=-1) + th.matmul(f_est_individual, aux_ones) # shape = (b, n_s, n)
        optimal_marginal_contribution = optimal_marginal_contribution.unsqueeze(-1) # shape = (b, n_s, n, 1)

        # w_inverse_est
        w_inv_est = (w_est.squeeze(-1).unsqueeze(1).expand(batch_size, self.n_agents, self.n_agents) * th.eye(self.n_agents).cuda()).inverse().matmul(th.ones(self.n_agents).cuda()).unsqueeze(-1) # shape = (b, n, 1)
        
        return normal_marginal_contribution, optimal_marginal_contribution, w_inv_est, w_est

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.n_agents, self.n_actions).float()

        normal_marginal_contribution, optimal_marginal_contribution, w_inv_est, w_est = self.get_marginal_contribution(states, actions)
        shapley_q = normal_marginal_contribution.mean(dim=1) # shape = (b*t, n, 1)
        optimal_shapley_q = optimal_marginal_contribution.mean(dim=1) # shape = (b*t, n, 1)
        sum_optimal_shapley_q = optimal_shapley_q.sum(dim=1) # shape = (b*t, 1)
        # sum_shapley_q = shapley_q.sum(dim=1) # shape = (b*t, 1)

        inv_proj_shapley_q = w_inv_est * shapley_q # shape = (b*t, n, 1)
        # inv_proj_shapley_q = w_inv_est * optimal_shapley_q # shape = (b*t, n, 1)

        return inv_proj_shapley_q, sum_optimal_shapley_q, w_est, shapley_q
        # return inv_proj_shapley_q, sum_optimal_shapley_q, w_est, optimal_shapley_q