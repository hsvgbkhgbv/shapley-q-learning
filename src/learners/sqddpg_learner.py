import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.sqddpg import SQDDPGMixer
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
from collections import deque
from controllers import REGISTRY as mac_REGISTRY



class SQDDPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(self.mac.parameters())

        self.last_target_update_episode = 0
        self.last_actor_update_episode = 0
        self.last_critic_update_episode = 0

        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "sqddpg":
                self.mixer = SQDDPGMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            # self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.optimiser_mac = RMSprop(params=self.mac_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.optimiser_mixer = RMSprop(params=self.mixer_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # data structure = (b, t, agent, a)
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # actions for training critics
        policy_outs = F.softmax(mac_out[:, :-1], dim=-1)
        # chosen_action_qvals_agents = th.gather(policy_outs, dim=3, index=actions) 
        # chosen_action_qvals = chosen_action_qvals_agents # attention to the detach()
        chosen_action_qvals = policy_outs.detach()

        # for ddpg style policy training
        mac_out_clone = mac_out.clone()
        mac_out_clone[avail_actions == 0] = -9999999

        if self.args.gumbel_softmax:
            # gumbel softmax
            gumbel_actions_distr = F.gumbel_softmax(mac_out_clone[:, :-1], hard=False, dim=-1, tau=self.args.policy_temp)
            # gumbel_actions_label = gumbel_actions_distr.clone().detach().max(dim=-1, keepdim=True)[1]
            # actor_actions = th.gather(gumbel_actions_distr, 3, gumbel_actions_label)
            actor_actions = gumbel_actions_distr
        else:
            # greedy
            greedy_actions_distr = F.softmax(mac_out_clone[:, :-1], dim=-1)
            # greedy_actions_label = greedy_actions_distr.clone().detach().max(dim=-1, keepdim=True)[1]
            # actor_actions = th.gather(greedy_actions_distr, 3, greedy_actions_label)
            actor_actions = greedy_actions_distr
        
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Max over target Q-Values
        if self.args.double_q:
            raise Exception("No double q for DDPG")
        else:
            # target_mac_out_detach = mac_out.clone().detach()
            target_mac_out_detach = target_mac_out.clone().detach()
            target_mac_out_detach[avail_actions == 0] = -9999999
            # target_mac_out_detach = target_mac_out.clone().detach()
            # target_mac_out_detach[avail_actions == 0] = -9999999

            if self.args.gumbel_softmax:
                # gumbel softmax
                target_actions_distr = F.gumbel_softmax(target_mac_out_detach[:, 1:], hard=False, dim=-1, tau=self.args.policy_temp)
                # target_max_actions_label = target_actions_distr.clone().detach().max(dim=-1, keepdim=True)[1]
                # target_max_qvals = th.gather(target_actions_distr, 3, target_max_actions_label)
                target_max_qvals = target_actions_distr.detach()
            else:
                # greedy
                target_actions_distr = F.softmax(target_mac_out_detach[:, 1:], dim=-1)
                # target_max_actions_label = target_actions_distr.clone().detach().max(dim=-1, keepdim=True)[1]
                # target_max_qvals = th.gather(target_actions_distr, 3, target_max_actions_label)
                target_max_qvals = target_actions_distr.detach()

        # get shapley values
        shapley_values_sum = self.mixer(batch["state"][:, :-1], chosen_action_qvals.detach()).sum(dim=2, keepdim=True)
        # shapley_values_sum = self.mixer(batch["state"][:, :-1], chosen_action_qvals).sum(dim=2, keepdim=True)
        target_shapley_values_sum = self.target_mixer(batch["state"][:, 1:], target_max_qvals).sum(dim=2, keepdim=True)
        shapley_value_actor = self.mixer(batch["state"][:, :-1], actor_actions)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_shapley_values_sum

        # Logit entropy (correct)
        ps = F.softmax(mac_out[:, :-1], dim=3) * avail_actions[:, :-1]
        log_ps = F.log_softmax(mac_out[:, :-1], dim=3) * avail_actions[:, :-1]
        logit_entropy = -(((ps * log_ps).sum(dim=3) * mask).sum() / mask.sum())

        shapley_td_error = (shapley_values_sum - targets.detach())
        shapley_mask = mask.expand_as(shapley_td_error)
        shapley_masked_td_error = shapley_td_error * shapley_mask
        shapley_loss = 0.5 * shapley_masked_td_error.pow(2).sum() / mask.sum()

        actor_loss = - (shapley_value_actor * shapley_mask).sum() / mask.sum()

        loss = self.args.actor_loss * actor_loss + self.args.shapley_loss * shapley_loss + -self.args.logit_entropy * logit_entropy

        # self.optimiser_mac.zero_grad()
        # self.optimiser_mixer.zero_grad()
        # loss.backward()

        if (episode_num - self.last_critic_update_episode) / self.args.critic_update_interval >= 1.0:
            self.optimiser_mixer.zero_grad()
            (self.args.shapley_loss * shapley_loss).backward()
            mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
            self.optimiser_mixer.step()
            self.last_critic_update_episode = episode_num
        else:
            self.optimiser_mixer.zero_grad()
            (self.args.shapley_loss * shapley_loss).backward()
            mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)

        if (episode_num - self.last_actor_update_episode) / self.args.actor_update_interval >= 1.0:
            self.optimiser_mac.zero_grad()
            (self.args.actor_loss * actor_loss - self.args.logit_entropy * logit_entropy).backward()
            mac_grad_norm = th.nn.utils.clip_grad_norm_(self.mac_params, self.args.grad_norm_clip)
            self.optimiser_mac.step()
            self.last_actor_update_episode = episode_num
        else:
            self.optimiser_mac.zero_grad()
            (self.args.actor_loss * actor_loss - self.args.logit_entropy * logit_entropy).backward()
            mac_grad_norm = th.nn.utils.clip_grad_norm_(self.mac_params, self.args.grad_norm_clip)


        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("mac_grad_norm", mac_grad_norm, t_env)
            self.logger.log_stat("mixer_grad_norm", mixer_grad_norm, t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("q_taken_mean", (chosen_action_qvals.squeeze(3) * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("shapley_loss", shapley_loss.item(), t_env)
            self.logger.log_stat("logit_entropy", logit_entropy.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        # self.target_mac.load_state(self.mac)
        for name, param in self.target_mac.agent.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.mac.agent.state_dict()[name]
            self.target_mac.agent.state_dict()[name].copy_(update_params)
        if self.mixer is not None:
            # self.target_mixer.load_state_dict(self.mixer.state_dict())
            for name, param in self.target_mixer.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.mixer.state_dict()[name]
                self.target_mixer.state_dict()[name].copy_(update_params)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            th.save(self.target_mixer.state_dict(), "{}/target_mixer.th".format(path))
        th.save(self.optimiser_mac.state_dict(), "{}/opt_mac.th".format(path))
        th.save(self.optimiser_mixer.state_dict(), "{}/opt_mixer.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/target_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_mac.load_state_dict(th.load("{}/opt_mac.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_mixer.load_state_dict(th.load("{}/opt_mixer.th".format(path), map_location=lambda storage, loc: storage))
