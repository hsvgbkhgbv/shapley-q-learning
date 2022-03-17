import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .qmix_central_no_hyper import QMixerCentralFF
from .qmix import QMixer



class QMixerCentralFF(nn.Module):
    def __init__(self, args):
        super(QMixerCentralFF, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.input_dim = self.n_agents * self.args.n_actions + self.state_dim
        self.embed_dim = args.central_mixing_embed_dim
        self.n_actions = args.n_actions

        non_lin = nn.ReLU

        self.net = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, 1))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               non_lin(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents * self.n_actions)

        inputs = th.cat([states, agent_qs], dim=1)

        advs = self.net(inputs)
        vs = self.V(states)

        y = advs + vs

        q_tot = y.view(bs, -1, 1)
        return q_tot



class SQDDPGMixer(nn.Module):
    def __init__(self, args):
        super(SQDDPGMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        # self.embed_dim = args.mixing_embed_dim
        self.n_actions = args.n_actions

        self.sample_size = args.sample_size

        if args.marginal_contribution_type == 'ff':
            self.marginal_contribution = QMixerCentralFF(args)
        elif args.marginal_contribution_type == 'qmix':
            self.marginal_contribution = QMixer(args)
        else:
            raise ValueError("Marginal contribution type {} not recognised.".format(args.marginal_contribution_type))


    def sample_grandcoalitions(self, batch_size):
        """
        E.g. batch_size = 2, n_agents = 3:

        >>> grand_coalitions_pos
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
        grand_coalitions_pos = th.multinomial(th.ones(batch_size*self.sample_size, 
                                          self.n_agents).cuda()/self.n_agents, 
                                          self.n_agents, 
                                          replacement=False)
        individual_map = th.zeros(batch_size*self.sample_size*self.n_agents, self.n_agents).cuda()
        individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_agents, self.n_agents)
        subcoalition_map = th.matmul(individual_map, seq_set)


        # FIX: construct the grand coalition (in sequence by agent_idx) from the grand_coalitions_pos (e.g., pos_idx <- grand_coalitions_pos[agent_idx])
        # grand_coalitions = []
        # for grand_coalition_pos in grand_coalitions_pos:
        #     grand_coalition = th.zeros_like(grand_coalition_pos)
        #     for agent, pos in enumerate(grand_coalition_pos):
        #         grand_coalition[pos] = agent
        #     grand_coalitions.append(grand_coalition)
        # grand_coalitions = th.stack(grand_coalitions, dim=0).to(self.device)
        offset = (th.arange(batch_size*self.sample_size)*self.n_).reshape(-1, 1)
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions = th.zeros_like(grand_coalitions_pos_alter.flatten())
        grand_coalitions[grand_coalitions_pos_alter.flatten()] = th.arange(batch_size*self.sample_size*self.n_)
        grand_coalitions = grand_coalitions.reshape(batch_size*self.sample_size, self.n_) - offset

        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, 
                                                                self.n_agents, 
                                                                self.n_agents).contiguous().view(batch_size, 
                                                                                                 self.sample_size, 
                                                                                                 self.n_agents, 
                                                                                                 self.n_agents) # shape = (b, n_s, n, n)
        return subcoalition_map, individual_map, grand_coalitions


    def get_shapley_values(self, states, agent_qs):
        batch_size = states.size(0)

        # get subcoalition map including agent i
        subcoalition_map, individual_map, grand_coalitions = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)

        # reshape the grand coalition map for rearranging the sequence of actions of agents
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 self.n_actions) # shape = (b, n_s, n, n, 1)

        # remove agent i from the subcoalition map
        subcoalition_map_no_i = subcoalition_map - individual_map
        subcoalition_map_no_i = subcoalition_map_no_i.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 self.n_actions) # shape = (b, n_s, n, n, 1)
        individual_map = individual_map.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 self.n_actions) # shape = (b, n_s, n, n, 1)

        # reshape actions for further process on coalitions
        reshape_agent_qs = agent_qs.unsqueeze(1).unsqueeze(2).expand(batch_size, 
                                                        self.sample_size, 
                                                        self.n_agents, 
                                                        self.n_agents, 
                                                        self.n_actions).gather(3, grand_coalitions) # shape = (b, n, 1) -> (b, 1, 1, n, 1) -> (b, n_s, n, n, 1)

        # get actions of its coalition memebers for each agent
        agent_qs_coalition_no_i = reshape_agent_qs * subcoalition_map_no_i # shape = (b, n_s, n, n, 1)
        agent_qs_coalition_i = reshape_agent_qs * individual_map # shape = (b, n_s, n, n, 1)

        # keep u_{-i} no gradient backprop
        agent_qs_coalition = agent_qs_coalition_no_i.detach() + agent_qs_coalition_i # shape = (b, n_s, n, n, 1)
        # agent_qs_coalition = agent_qs_coalition_no_i + agent_qs_coalition_i # shape = (b, n_s, n, n, 1)

        reshape_agent_qs_coalition = agent_qs_coalition.contiguous().view(-1, self.n_agents*self.n_actions) # shape = (b*n_s*n, n)
        reshape_states = states.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_agents, self.state_dim).contiguous().view(-1, self.state_dim) # shape = (b*n_s*n, s)

        # inputs = th.cat([reshape_agent_qs_coalition, reshape_states], dim=-1) # shape = (b*n_s*n, n+s)

        marginal_contributions = self.marginal_contribution(reshape_agent_qs_coalition, reshape_states) # shape = (b*n_s*n, 1)
        marginal_contributions = marginal_contributions.contiguous().view(batch_size, self.sample_size, self.n_agents) # shape = (b, n_s, n)
        shapley_values = marginal_contributions.mean(dim=1) # shape = (b, n)

        return shapley_values


    def forward(self, states, agent_qs):
        # agent_qs = (b, t, n)
        reshape_states = states.contiguous().view(-1, self.state_dim)
        reshape_agent_qs = agent_qs.unsqueeze(-1).contiguous().view(-1, self.n_agents, self.n_actions)

        shapley_values = self.get_shapley_values(reshape_states, reshape_agent_qs) # shape = (b*t, n)
        shapley_values = shapley_values.contiguous().view(states.size(0), states.size(1), self.n_agents) # shape = (b, t, n)

        return shapley_values
