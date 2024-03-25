import logging

import numpy as np
import torch as T

from .deep_q_network import DeepQNetwork
from .replay_memory import ReplayBuffer


class DQNAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=500, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0  # Count no of times of learn function so we know when we can udpate the weights of the target network with the weights of the evaluation network?

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DeepQNetwork(learning_rate=self.lr,n_actions= self.n_actions, input_dims=self.input_dims,
                                   name=f"{self.env_name}_{self.algo}_q_eval", chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(learning_rate=self.lr, n_actions=self.n_actions, input_dims=self.input_dims,
                                   name=f"{self.env_name}_{self.algo}_q_next", chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation, action_mask):
        if np.random.random() > self.epsilon:

            state = T.tensor(observation[np.newaxis, :], dtype=T.float,
                             device=self.q_eval.device)  # CNN expects input as batchsize * input_size, so we have to add an extra dimension, we can do that by converting it to a list and then convert it to a tensor?
            actions = self.q_eval.forward(state)
            logging.info(f"ACTIONS VALUES BEFORE MASK\n: {actions}")
            # Anpassung hier: Erweitere `action_mask` um eine Dimension, um sie kompatibel zu machen
            action_mask_tensor = T.tensor(action_mask, dtype=T.float, device=self.q_eval.device).unsqueeze(0)

            adjusted_action_mask = actions.clone()
            adjusted_action_mask[action_mask_tensor == 0] = T.tensor(-np.inf, device=self.q_eval.device)
            action = T.argmax(adjusted_action_mask).item()
            logging.info(f"ACTION VLAUES AFTER MASK:\n {adjusted_action_mask}")
        else:
            valid_actions = [i for i, available in enumerate(action_mask) if available]
            action = np.random.choice(valid_actions)

        return action

    def store_transition(self, state, action, reward, state_, terminal):
        self.memory.store_transition(state, action, reward, state_, terminal)

    def sample_memory(self):
        state, action, reward, new_state, terminal = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state, device=self.q_eval.device)
        rewards = T.tensor(reward, device=self.q_eval.device)
        terminals = T.tensor(terminal, device=self.q_eval.device)
        actions = T.tensor(action, device=self.q_eval.device)
        states_ = T.tensor(new_state, device=self.q_eval.device)

        return states, actions, rewards, states_, terminals

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, terminal = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[
            0]  # dim 1 = action dimension, 0tes Element ist der value, [1] wäre der index

        terminal_mask = (terminal == 1).to(self.q_eval.device)  # Create boolean mask
        q_next[terminal_mask] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
