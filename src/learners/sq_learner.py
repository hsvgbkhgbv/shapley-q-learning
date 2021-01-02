import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.sq import ShapleyQMixer
import torch as th
from torch.optim import RMSprop


class SQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = ShapleyQMixer(args)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.td_loss_coef = args.td_loss_coef
        self.proj_coef = args.proj_coef
        self.w_reg_coef = args.w_reg_coef

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        one_hot_actions = th.nn.functional.one_hot(actions, num_classes=self.args.n_actions)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float() # shape = (b, t, 1)
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
            one_hot_cur_max_actions = th.nn.functional.one_hot(cur_max_actions, num_classes=self.args.n_actions)
            # target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            # target_max_qvals = target_mac_out.max(dim=3)[0]
            cur_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
            one_hot_cur_max_actions = th.nn.functional.one_hot(cur_max_actions, num_classes=self.args.n_actions)

        # Mix
        if self.mixer is not None:
            inv_proj_shapley_q, _, w_est, shapley_q = self.mixer(batch["state"][:, :-1], one_hot_actions)
            _, target_sum_optimal_shapley_q, _, _ = self.target_mixer(batch["state"][:, 1:], one_hot_cur_max_actions)

        N = getattr(self.args, "n_step", 1)
        # print (f"This is the shape of rewards: {rewards.size()}.")
        # print (f"This is the shape of sum_shapley_q: {target_sum_optimal_shapley_q.size()}.")
        # print (f"This is the shape of termination: {terminated.size()}.")
        # print (f"This is the shape of gamma: {self.args.gamma.size()}.")
        # print (f"This is the shape of mask: {mask.shape}.")
        # mask = mask.expand_as(rewards).contiguous().view(-1, 1)

        if N == 1:
            rewards = rewards.contiguous().view(-1, 1)
            terminated = terminated.contiguous().view(-1, 1)
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_sum_optimal_shapley_q
        else:
            # N step Q-Learning targets
            target_sum_optimal_shapley_q = target_sum_optimal_shapley_q.contiguous().view(rewards.size()) # shape = (b, t, 1)
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma**i for i in range(N)], dtype=th.float, device=n_rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                n_rewards[:,i,0] = ((rewards * mask)[:,i:i+N,0] * gamma_tensor[:(batch.max_seq_length - 1 - i)]).sum(dim=1)
            indices = th.linspace(0, batch.max_seq_length-2, steps=batch.max_seq_length-1, device=steps.device).unsqueeze(1).long()
            n_targets_terminated = th.gather(target_sum_optimal_shapley_q*(1-terminated),dim=1,index=steps.long()+indices-1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated # shape = (b, t, 1)
            targets = targets.contiguous().view(-1, 1)

        # Td-error for shapley q
        targets = targets.unsqueeze(1).expand_as(inv_proj_shapley_q)
        td_error = (inv_proj_shapley_q - targets.detach()).sum(dim=1)
        
        # print (f"This is the mask size: {mask.size()}.")

        # local q projection to shapley q loss
        # print (f"This is the shape of chosen_action_qvals: {chosen_action_qvals.size()}")
        # print (f"This is the shape of shapley_q: {shapley_q.size()}")
        chosen_action_qvals = chosen_action_qvals.unsqueeze(-1).contiguous().view(-1, 1)
        proj_error = chosen_action_qvals - shapley_q.contiguous().view(-1, 1).detach()

        # 0-out the targets that came from padded data
        mask_td = mask.contiguous().view(-1, 1)
        masked_td_error = td_error * mask_td
        # print (f"This is the shape of w_est: {w_est.size()}.")
        # print (f"This is the shape of mask: {mask.size()}.")
        mask_w_est = mask.contiguous().view(-1, 1).unsqueeze(1).expand_as(w_est)
        masked_w_est = w_est * mask_w_est
        mask_proj = mask.contiguous().view(-1, 1).unsqueeze(1).expand_as(shapley_q).contiguous().view(-1, 1)
        masked_proj_error = proj_error * mask_proj

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask_td.sum()
        w_est_reg = (masked_w_est ** 2).sum() / mask_w_est.sum()
        proj_loss = (masked_proj_error ** 2).sum() / mask_proj.sum()
        loss = self.td_loss_coef*td_loss + self.w_reg_coef*w_est_reg + self.proj_coef*proj_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_td_elems = mask_td.sum().item()
            mask_w_est_elems = mask_w_est.sum().item()
            mask_proj_elems = mask_proj.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item()/mask_td_elems, t_env)
            self.logger.log_stat("w_est_reg", w_est_reg.item(), t_env)
            self.logger.log_stat("proj_loss", proj_loss.item(), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask_proj).sum().item()/mask_proj_elems, t_env)
            self.logger.log_stat("target_mean", (targets.contiguous().view(-1, 1) * mask_proj).sum().item()/mask_proj_elems, t_env)
            agent_utils = (th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) * mask).sum().item() / mask_proj_elems
            self.logger.log_stat("agent_utils", agent_utils, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
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
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
