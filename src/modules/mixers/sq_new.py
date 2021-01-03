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

        # w(s,u)
        if self.arch == "observation_action":
            # w,f,g takes [state, u] as input
            w_input_size = self.state_dim + 2*self.n_actions
            # print (f"This is the w_input_size: {w_input_size}")
        else:
            raise Exception("{} is not a valid ShapleyQ architecture".format(self.arch))

        if self.args.network_size == "small":
            # w_list = [ nn.Sequential(nn.Linear(w_input_size, self.embed_dim),
            #                          nn.ReLU(),
            #                          nn.Linear(self.embed_dim, 1),
            #                          nn.ReLU(),
            #                          nn.Linear(self.embed_dim, 1),
            #                          nn.Tanh()
            #                     ) 
            #                     for _ in range(self.n_agents)
            #                 ]
            # self.w_list = nn.ModuleList(w_list)
            # self.coalition_actions_enc = nn.Sequential(
            #     nn.Linear(self.n_actions, self.embed_dim),
            #     nn.Sigmoid(),
            #     nn.Linear(self.embed_dim, self.embed_dim)
            # )

            # self.individual_actions_enc = nn.Sequential(
            #     nn.Linear(self.n_actions, self.embed_dim),
            #     nn.Sigmoid(),
            #     nn.Linear(self.embed_dim, self.embed_dim)
            # )

            # self.states_enc = nn.Sequential(
            #     nn.Linear(self.state_dim, self.embed_dim),
            #     nn.Sigmoid(),
            #     nn.Linear(self.embed_dim, self.embed_dim)
            # )

            self.w = nn.Sequential(nn.Linear(w_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1),
                                    #  nn.Tanh()
                            )
        elif self.args.network_size == "big":
            # w_list = [ nn.Sequential(nn.Linear(w_input_size, self.embed_dim),
            #                          nn.ReLU(),
            #                          nn.Linear(self.embed_dim, self.embed_dim),
            #                          nn.ReLU(),
            #                          nn.Linear(self.embed_dim, self.embed_dim),
            #                          nn.ReLU(),
            #                          nn.Linear(self.embed_dim, 1),
            #                          nn.Tanh()
            #                     ) 
            #                     for _ in range(self.n_agents)
            #                 ]
            # self.w_list = nn.ModuleList(w_list)
            self.w = nn.Sequential(nn.Linear(w_input_size, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1),
                                     nn.Tanh()
            )
        else:
            raise Exception("{} is not a valid ShapleyQ network size".format(self.args.network_size))

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

    def get_coalition_actions_enc(self, actions):
        inputs = actions.contiguous().view(-1, self.n_actions)
        coalition_actions_enc = self.coalition_actions_enc(inputs)
        coalition_actions_enc = coalition_actions_enc.contiguous().view(-1, self.n_agents, self.n_actions)
        return coalition_actions_enc

    def get_individual_actions_enc(self, actions):
        inputs = actions.contiguous().view(-1, self.n_actions)
        individual_actions_enc = self.individual_actions_enc(inputs)
        individual_actions_enc = individual_actions_enc.contiguous().view(-1, self.n_agents, self.n_actions)
        return individual_actions_enc

    def get_states_enc(self, states):
        inputs = states
        states_enc = self.states_enc(inputs)
        return states_enc

    def get_w_estimate(self, states, actions):
        batch_size = states.size(0)

        # get subcoalition map including agent i
        subcoalition_map, individual_map, grand_coalitions = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)

        # reshape the grand coalition map for rearranging the sequence of actions of agents
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 self.n_actions) # shape = (b, n_s, n, n, a)

        # remove agent i from the subcloation map
        subcoalition_map_no_i = subcoalition_map - individual_map
        subcoalition_map_no_i = subcoalition_map_no_i.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 self.n_actions) # shape = (b, n_s, n, n, a)
        
        # reshape actions for further process on coalitions
        reshape_actions = actions.unsqueeze(1).unsqueeze(2).expand(batch_size, 
                                                        self.sample_size, 
                                                        self.n_agents, 
                                                        self.n_agents, 
                                                        self.n_actions).gather(3, grand_coalitions) # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)

        # get actions of its coalition memebers for each agent
        actions_coalition = reshape_actions * subcoalition_map_no_i # shape = (b, n_s, n, n, a)

        # get actions vector of its coalition members for each agent
        actions_coalition_norm_vec = actions_coalition.mean(dim=-2) * subcoalition_map_no_i.sum(dim=-2) # shape = (b, n_s, n, a)
        actions_coalition_norm_vec = actions_coalition_norm_vec.mean(dim=1) # shape = (b, n, a)

        # get action vector of each agent
        actions_individual = actions / self.n_agents # shape = (b, n, a)

        reshape_actions_coalition_norm_vec = actions_coalition_norm_vec.contiguous().view(-1, self.n_actions) # shape = (b*n, a)
        reshape_actions_individual = actions_individual.contiguous().view(-1, self.n_actions) # shape = (b*n, a)
        reshape_states = states.unsqueeze(1).expand(batch_size, self.n_agents, self.state_dim).contiguous().view(-1, self.state_dim) / states.max()# shape = (b*n, s)

        # print (f"This is the reshape_actions_coalition_norm_vec: {reshape_actions_coalition_norm_vec.size()}")
        # print (f"This is the reshape_actions_individual: {reshape_actions_individual.size()}")
        # print (f"This is the reshape_states: {reshape_states.size()}")

        inputs = th.cat([reshape_states, reshape_actions_coalition_norm_vec, reshape_actions_individual], dim=-1) # shape = (b*n, s+2*a)

        w_estimates = self.w(inputs) # shape = (b*n, 1)
        w_estimates = w_estimates.squeeze(-1).contiguous().view(batch_size, self.n_agents) # shape = (b, n)

        return w_estimates

    def forward(self, states, actions, agent_qs, max_filter, target=True):
        # agent_qs, max_filter = (b, t, n)
        reshape_states = states.reshape(-1, self.state_dim)
        reshape_actions = actions.reshape(-1, self.n_agents, self.n_actions).float()
        if target:
            return th.sum(agent_qs, dim=2, keepdim=True)
        else:
            w_estimates = self.get_w_estimate(reshape_states, reshape_actions)
            w_estimates = w_estimates.contiguous().view(states.size(0), states.size(1), self.n_agents)
            # agent with non-max action will be given 1
            non_max_filter = 1 - max_filter

            # if the agent with the max-action then w=1
            # otherwise the agent will use the learned w
            return ( (w_estimates * non_max_filter + max_filter) * agent_qs).mean(dim=2, keepdim=True), w_estimates