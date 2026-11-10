"""One learning agent: a small actor-critic network trained with PPO.

The same network is copied into every parallel world, so experience from all
of them updates one individual. The reward is the agent's own wellbeing at
each step (see Worlds.wellbeing); nothing rewards food as such.

The task never ends on its own: the agent lives until it collapses, and a
collapse is the only point where the future is cut off.

Rewards are multiplied by (1 - gamma) before learning, so the value the critic
predicts stays between 0 and 1 (roughly the wellbeing the agent expects to
keep). Without this the critic's error is in the hundreds and swamps the
policy's share of each update. Scaling all rewards by one constant does not
change which behaviour is best.

MemoryAgent is the same agent with a memory: the second hidden layer is a recurrent (GRU) cell, so what
the agent does can depend on what it sensed earlier, not only on what it senses now.
"""
import numpy as np
import torch
import torch.nn as nn


def layer(i, o, gain=np.sqrt(2)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, gain)
    nn.init.zeros_(l.bias)
    return l


class Net(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.body = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.policy = layer(hidden, n_actions, 0.01)
        self.value = layer(hidden, 1, 1.0)

    def forward(self, obs):
        h = self.body(obs)
        return self.policy(h), self.value(h).squeeze(-1)


class Agent:
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2,
                 epochs=4, minibatches=4, entropy=0.01, value_coef=0.5, max_grad=0.5, hidden=128):
        self.net = Net(obs_dim, n_actions, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.entropy, self.value_coef, self.max_grad = entropy, value_coef, max_grad

    def begin(self, n):
        """Called when the agent is put into a new set of `n` worlds. Only an agent with memory uses it."""

    def after_step(self, collapsed):
        """Called after every step with the worlds where the agent collapsed. Only an agent with memory uses it."""

    @torch.no_grad()
    def act(self, obs):
        logits, value = self.net(torch.as_tensor(obs))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), value.numpy()

    @torch.no_grad()
    def value(self, obs):
        return self.net(torch.as_tensor(obs))[1].numpy()

    def learn(self, obs, actions, logp, values, rewards, dones, last_value):
        """One PPO update from a rollout of shape (steps, worlds)."""
        steps = len(rewards)
        rewards = rewards * (1 - self.gamma)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(steps)):
            nxt = last_value if t == steps - 1 else values[t + 1]
            alive = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nxt * alive - values[t]
            gae = delta + self.gamma * self.lam * alive * gae
            adv[t] = gae
        ret = adv + values

        flat = lambda a: torch.as_tensor(a.reshape(steps * a.shape[1], *a.shape[2:]))
        obs, actions, logp, adv, ret = map(flat, (obs, actions, logp, adv, ret))
        actions = actions.long()
        n = len(adv)
        size = n // self.minibatches
        for _ in range(self.epochs):
            order = torch.randperm(n)
            for start in range(0, n, size):
                idx = order[start:start + size]
                logits, v = self.net(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(actions[idx])
                ratio = (new_logp - logp[idx]).exp()
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                value_loss = ((v - ret[idx]) ** 2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.opt.step()


class MemoryNet(nn.Module):
    """As Net, but the second hidden layer is a GRU cell that carries a state from step to step."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.encode = nn.Sequential(layer(obs_dim, hidden), nn.Tanh())
        self.cell = nn.GRUCell(hidden, hidden)
        for name, w in self.cell.named_parameters():
            nn.init.orthogonal_(w, 1.0) if "weight" in name else nn.init.zeros_(w)
        self.policy = layer(hidden, n_actions, 0.01)
        self.value = layer(hidden, 1, 1.0)

    def forward(self, obs, memory):
        """One step: what the agent senses and the memory it came with -> action scores, value, new memory."""
        return self.step(self.encode(obs), memory)

    def step(self, encoded, memory):
        memory = self.cell(encoded, memory)
        return self.policy(memory), self.value(memory).squeeze(-1), memory


class MemoryAgent(Agent):
    """PPO with the same settings as Agent, trained through time over each whole rollout.

    The memory is empty when the agent enters a new set of worlds and after a collapse, when it wakes with
    a fresh body. In learning, a minibatch is a group of worlds (their whole rollouts), not single steps.
    """

    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2,
                 epochs=4, minibatches=4, entropy=0.01, value_coef=0.5, max_grad=0.5, hidden=128):
        self.net = MemoryNet(obs_dim, n_actions, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.entropy, self.value_coef, self.max_grad = entropy, value_coef, max_grad
        self.hidden = hidden
        self.memory = self.rollout_start = None

    def begin(self, n):
        self.memory = torch.zeros(n, self.hidden)
        self.rollout_start = self.memory.clone()

    def after_step(self, collapsed):
        self.memory[torch.as_tensor(collapsed)] = 0.0

    @torch.no_grad()
    def act(self, obs):
        logits, value, self.memory = self.net(torch.as_tensor(obs), self.memory)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), value.numpy()

    @torch.no_grad()
    def value(self, obs):
        return self.net(torch.as_tensor(obs), self.memory)[1].numpy()

    def learn(self, obs, actions, logp, values, rewards, dones, last_value):
        """One PPO update from a rollout of shape (steps, worlds), replayed from the memory it started with."""
        steps, n = rewards.shape
        rewards = rewards * (1 - self.gamma)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(steps)):
            nxt = last_value if t == steps - 1 else values[t + 1]
            alive = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nxt * alive - values[t]
            gae = delta + self.gamma * self.lam * alive * gae
            adv[t] = gae
        ret = adv + values

        obs, actions, logp, adv, ret = map(torch.as_tensor, (obs, actions, logp, adv, ret))
        actions = actions.long()
        keep = torch.as_tensor(1.0 - dones)          # 0 where the agent collapsed: its memory starts empty again
        size = n // self.minibatches
        for _ in range(self.epochs):
            order = torch.randperm(n)
            for start in range(0, n, size):
                idx = order[start:start + size]
                encoded = self.net.encode(obs[:, idx])
                kept = keep[:, idx, None]
                memory = self.rollout_start[idx]
                states = []
                for t in range(steps):
                    memory = self.net.cell(encoded[t], memory)
                    states.append(memory)
                    memory = memory * kept[t]
                states = torch.stack(states)
                logits, v = self.net.policy(states), self.net.value(states).squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(actions[:, idx])
                ratio = (new_logp - logp[:, idx]).exp()
                a = adv[:, idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                value_loss = ((v - ret[:, idx]) ** 2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.opt.step()
        self.rollout_start = self.memory.clone()     # the next rollout starts from the memory this one ended with
