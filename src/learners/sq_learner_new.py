import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop


class SQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        if args.name == "sq":
            from modules.mixers.sq_new import ShapleyQMixer
        elif args.name == "sq_enc":
            from modules.mixers.sq_new_enc import ShapleyQMixer
        elif args.name == "sqmix_v1":
            from modules.mixers.sqmix import ShapleyQMixer
        elif args.name == "sqmix_v2":
            from modules.mixers.sqmix_v2 import ShapleyQMixer
        elif args.name == "sqmix_v3":
            from modules.mixers.sqmix_v3 import ShapleyQMixer
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.last_mixer_update_episode = 0
        self.last_sample_coalition_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = ShapleyQMixer(args)
            self.params_mixer = list(self.mixer.parameters())
            # self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_mixer = RMSprop(params=self.params_mixer, lr=args.mixer_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        one_hot_actions = th.nn.functional.one_hot(actions, num_classes=self.args.n_actions)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        
        # generate a filter for selecting the agents with the max-action
        if self.args.max_filter:
            _mac_out_detach = mac_out.clone().detach()
            _mac_out_detach[avail_actions == 0] = -9999999
            _cur_max_actions = _mac_out_detach[:, :-1].max(dim=3, keepdim=True)[1].squeeze(3)
            # print (f"This is the size of actions: {actions.size()}")
            max_filter = (actions.detach().squeeze(3)==_cur_max_actions).float()
        else:
            max_filter = None

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals, w_est = self.mixer(batch["state"][:, :-1], one_hot_actions, chosen_action_qvals, max_filter, target=False)
            target_max_qvals = self.mixer(batch["state"][:, 1:], one_hot_actions, target_max_qvals, max_filter, target=True)
            # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        N = getattr(self.args, "n_step", 1)
        if N == 1:
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            # N step Q-Learning targets
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma**i for i in range(N)], dtype=th.float, device=n_rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                n_rewards[:,i,0] = ((rewards * mask)[:,i:i+N,0] * gamma_tensor[:(batch.max_seq_length - 1 - i)]).sum(dim=1)
            indices = th.linspace(0, batch.max_seq_length-2, steps=batch.max_seq_length-1, device=steps.device).unsqueeze(1).long()
            n_targets_terminated = th.gather(target_max_qvals*(1-terminated),dim=1,index=steps.long()+indices-1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.w_constraint_coef:
            # w_est L2 loss
            w_est_error = (w_est*max_filter - max_filter)
            mask_ = mask.expand_as(w_est_error)
            masked_w_est_error = w_est_error * mask_

            # print (f"This is the size of max_filter: {max_filter.size()}")
            # print (f"This is the sum of max_filter: {max_filter.sum()}")
            loss_w_constraint = (masked_w_est_error ** 2).sum() / mask_.sum()
            loss += self.args.w_constraint_coef * loss_w_constraint

        if self.args.w_contrast_coef:
            non_max_filter = (1 - max_filter)
            w_contrast_error = (w_est*non_max_filter - non_max_filter)
            mask_ = mask.expand_as(w_contrast_error)
            masked_w_contrast_error = w_contrast_error * mask_
            loss_w_contrast = (masked_w_contrast_error ** 2).sum() / mask_.sum()
            loss -= self.args.w_contrast_coef * loss_w_contrast

        # Optimise
        self.optimiser.zero_grad()
        self.optimiser_mixer.zero_grad()
        loss.backward()
        # if self.args.w_constraint_coef:
        #     loss_w.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        grad_norm_mixer = th.nn.utils.clip_grad_norm_(self.params_mixer, self.args.grad_norm_clip)
        
        if not self.args.mixer_update_interval:
            self.optimiser.step()
            self.optimiser_mixer.step()
        else:
            if (episode_num - self.last_mixer_update_episode) / self.args.mixer_update_interval >= 1.0:
                self.last_mixer_update_episode = episode_num
                self.optimiser_mixer.step()
            else:
                self.optimiser.step()
            

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.args.sample_coalition_interval > 0:
            if (episode_num - self.last_sample_coalition_episode) / self.args.sample_coalition_interval >= 1.0:
                self._update_coalitions()
                self.last_sample_coalition_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            if self.args.w_constraint_coef:
                self.logger.log_stat("loss_w_constraint", loss_w_constraint.item(), t_env)
            if self.args.w_contrast_coef:
                self.logger.log_stat("loss_w_contrast", loss_w_contrast.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("grad_norm_mixer", grad_norm_mixer, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            agent_utils = (th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("agent_utils", agent_utils, t_env)
            if self.args.max_filter:
                self.logger.log_stat("w_est", ( w_est * (1 - max_filter) * mask.expand_as(w_est) ).sum().item() / ( ( (1 - max_filter) * mask.expand_as(w_est) ).sum().item() ), t_env)
            else:
                self.logger.log_stat("w_est", ( w_est * mask.expand_as(w_est) ).sum().item() / mask.expand_as(w_est).sum().item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        # if self.args.name == "sqmix_v3":
        #     self.mixer.sample_grandcoalitions()
        self.logger.console_logger.info("Updated target network")

    def _update_coalitions(self):
        self.mixer.sample_grandcoalitions()
        # self.logger.console_logger.info("Updated coalitions")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            # self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
