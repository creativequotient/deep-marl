import torch as T
import torch.nn as nn
from torch.nn.functional import gumbel_softmax

from deepmarl.algorithms.maddpg import MADDPGAgent

MSELoss = nn.MSELoss()

T.autograd.set_detect_anomaly(True)

class M3DDPGAgent(MADDPGAgent):
    """
    M3DDPG Implementation from (Li et al 2019)

    This class inherits from MADDPGAgent because it shares ALL methods and functions except for the update function with
    has the mini-max property injection
    """

    def __init__(self, agent_name, agent_idx, model, obs_shape_n, act_shape_n, args, perturbation_factors):
        # local_q is set to FALSE because M3DDPG algorithm is ONLY for multi-agent setting. For independent agent,
        # use DDPG
        super().__init__(agent_name, agent_idx, model, obs_shape_n, act_shape_n, args, False)
        self.perturbation_factors = perturbation_factors

    @staticmethod
    def calculate_perturbation(raw_perturbation, action, perturbation_factor):
        act_norm = action.norm(dim=-1, keepdim=True)
        perturbation_norm = raw_perturbation.norm(dim=-1, keepdim=True)
        normalized_perturbation = raw_perturbation / perturbation_norm
        adaptive_perturbation = perturbation_factor * act_norm * normalized_perturbation
        return adaptive_perturbation

    def update(self, agents, i):
        assert agents[i] == self
        # Check if it is time to update
        if self.replay_buffer.get_size() < self.min_replay_size:
            return None

        # Update Q function
        sampled_idx = self.replay_buffer.make_index(self.batch_size)

        # Collate experiences
        obs_n = []
        act_n = []
        next_obs_n = []
        for agent in agents:
            o, a, _, o_, _ = agent.replay_buffer.sample_index(sampled_idx)
            o, a, o_ = list(map(lambda x: T.tensor(x, dtype=T.double, device=self.device),
                                [o, a, o_]))  # Convert all to tensors first
            obs_n.append(o)
            act_n.append(a)
            next_obs_n.append(o_)

        # Update Q-network
        target_logits_n = [agent.pi_target(obs) for agent, obs in zip(agents, next_obs_n)]
        if not self.discrete:  # Continuous action space
            target_act_n = T.cat(target_logits_n, 1)
        else:
            if not self.multi_discrete:  # Discrete action space
                target_act_n = [gumbel_softmax(logit, hard=True) for logit in target_logits_n]
            else:  # Multi-Discrete action space
                target_act_n = []
                for agent, logit in zip(agents, target_logits_n):
                    act = logit.split_with_sizes(agent.multi_discrete, dim=-1)
                    act = list(map(lambda logits: gumbel_softmax(logits, hard=True), act))
                    target_act_n.append(T.cat(act, dim=-1))
            target_act_n = T.cat(target_act_n, 1)
            target_logits_n = T.cat(target_logits_n, 1)

        target_q_in = T.cat([*next_obs_n, target_act_n], 1)
        target_q_in.retain_grad()
        target_q_values = self.q_target(target_q_in)
        target_q_values.backward(T.ones_like(target_q_values))
        action_shape_n = list(map(lambda agent: agent.act_shape[0], agents))
        split_idx = target_q_in.shape[1] - sum(action_shape_n)
        perturbations = target_q_in.grad.split(split_idx, dim=-1)[1]

        if not self.discrete:
            target_act_n = target_act_n.split_with_sizes(action_shape_n, dim=-1)
            perturbations = perturbations.split_with_sizes(action_shape_n, dim=-1)
            for agent, act, perturb, perturb_factor in zip(agents, target_act_n, perturbations, self.perturbation_factors):
                if agent == self:
                    continue
                act -= M3DDPGAgent.calculate_perturbation(perturb, act, perturb_factor)
        else:
            target_logits_n = target_logits_n.split_with_sizes(action_shape_n, dim=-1)
            perturbations = perturbations.split_with_sizes(action_shape_n, dim=-1)
            if not self.multi_discrete:
                for agent, logits, perturb, perturb_factor in zip(agents, target_logits_n, perturbations, self.perturbation_factors):
                    if agent == self:
                        continue
                    logits -= M3DDPGAgent.calculate_perturbation(perturb, logits, perturb_factor)
                target_act_n = list(map(lambda logits: gumbel_softmax(logits, hard=True), target_logits_n))
            else:
                target_act_n = []
                for logits, agent, perturb, perturb_factor in zip(target_logits_n, agents, perturbations, self.perturbation_factors):
                    if agent != self:
                        logits -= M3DDPGAgent.calculate_perturbation(perturb, logits, perturb_factor)
                    logits = logits.split_with_sizes(agent.multi_discrete, dim=-1)
                    act = list(map(lambda logits_: gumbel_softmax(logits_, hard=True), logits))
                    target_act_n.append(T.cat(act, dim=-1))
        target_q_in = T.cat((*next_obs_n, *target_act_n), 1)
        target_q_values = self.q_target(target_q_in)

        obs, _, rew, _, done = self.replay_buffer.sample_index(sampled_idx)
        obs, rew, done = list(map(lambda x: T.tensor(x, dtype=T.double, device=self.device), [obs, rew, done]))
        # Calculate target q-values
        with T.no_grad():
            target_q_value = rew + self.gamma * (1 - done) * target_q_values
        # Calculate predicted q-values
        actual_q_in = T.cat((*obs_n, *act_n), 1)
        actual_q_value = self.q(actual_q_in)
        # Compute MSE loss
        q_loss = MSELoss(actual_q_value, target_q_value.detach())
        # Update gradients
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 0.5)
        self.q_optimizer.step()

        # Update Pi network
        logits_n = [agent.pi(obs) for agent, obs in zip(agents, next_obs_n)]
        if not self.discrete:
            act_n = T.cat(logits_n)
        else:
            if not self.multi_discrete:  # Discrete action space
                act_n = [gumbel_softmax(logit, hard=True) for logit in logits_n]
            else:  # Multi-discrete action space
                act_n = []
                for agent, logits in zip(agents, logits_n):
                    logits = logits.split_with_sizes(agent.multi_discrete, dim=-1)
                    act = list(map(lambda logits_: gumbel_softmax(logits_, hard=True), logits))
                    act_n.append(T.cat(act, dim=-1))
            act_n = T.cat(act_n, 1)
            logits_n = T.cat(logits_n, 1)

        q_in = T.cat((*obs_n, act_n), 1)
        q_in.retain_grad()
        q_values = self.q(q_in)
        q_values.backward(T.ones_like(q_values), retain_graph=True)
        perturbations = q_in.grad.split(split_idx, dim=-1)[1]

        if not self.discrete:  # Continuous actions
            act_n = act_n.split_with_sizes(action_shape_n, dim=-1)
            perturbations = perturbations.split_with_sizes(action_shape_n, dim=-1)
            perturbed_act_n = []
            for agent, act, perturb, perturb_factor in zip(agents, act_n, perturbations, self.perturbation_factors):
                if agent != self:
                    perturbed_act = act - M3DDPGAgent.calculate_perturbation(perturb, act, perturb_factor)
                    perturbed_act_n.append(perturbed_act)
                else:
                    perturbed_act = act
                    reg_logits = perturbed_act
            act_n = perturbed_act_n
        else:
            logits_n = logits_n.split_with_sizes(action_shape_n, dim=-1)
            perturbations = perturbations.split_with_sizes(action_shape_n, dim=-1)
            if not self.multi_discrete:  # Discrete actions
                perturbed_logits_n = []
                for agent, logits, perturb, perturb_factor in zip(agents, logits_n, perturbations, self.perturbation_factors):
                    if agent != self:
                        perturbed_logits = logits - M3DDPGAgent.calculate_perturbation(perturb, logits, perturb_factor)
                        perturbed_logits_n.append(perturbed_logits)
                    else:
                        perturbed_logits_n.append(logits)
                        reg_logits = logits
                act_n = list(map(lambda logits: gumbel_softmax(logits, hard=True), logits_n))
            else:  # Multi-discrete actions
                act_n = []
                for agent, logits, perturb, perturb_factor in zip(agents, logits_n, perturbations, self.perturbation_factors):
                    if agent != self:
                        perturbed_logits = logits - M3DDPGAgent.calculate_perturbation(perturb, logits, perturb_factor)
                    else:
                        perturbed_logits = logits
                        reg_logits = perturbed_logits
                    perturbed_logits = perturbed_logits.split_with_sizes(agent.multi_discrete, dim=-1)
                    act = list(map(lambda logits_: gumbel_softmax(logits_, hard=True), perturbed_logits))
                    act_n.append(T.cat(act, dim=-1))

        q_in = T.cat((*obs_n, *act_n), 1)
        q_values = self.q(q_in)

        pi_loss = -q_values.mean()
        pi_loss += (reg_logits ** 2).mean() * 1e-3  # Regularization term
        # Update gradients
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
        self.pi_optimizer.step()

        return q_loss.cpu().detach().numpy(), pi_loss.cpu().detach().numpy()
