from net import *
import os 
import numpy.random as rd 
from copy import deepcopy

class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = None
        self.trajectory_list = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        # explict call self.init() for multiprocessing
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.act(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu().numpy()
    def explore_env(self, env, target_step):
        trajectory = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            
            state,next_state, reward, done, = env.step(action)

            trajectory.append((state, (reward, done, *action)))
            state = env.reset() if done else next_state
        self.state = state
        return trajectory
    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # explore noise of action
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = Critic
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)#critic loss 
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()# actor loss, makes it bigger
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_actor.item(), obj_critic.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

class AgentTD3(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = CriticTwin
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_use_per=False, gpu_id=0, env_num=1):
        super().init(net_dim, state_dim, action_dim, learning_rate, _if_use_per, gpu_id)

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)

    def select_action(self, state):
        states = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = self.act.get_action(states)
        return actions.detach().cpu().numpy()[0]

    def explore_env(self, env, target_step):
        trajectory = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            
            state,next_state, reward, done, = env.step(action)

            trajectory.append((state, (reward, done, *action)))
            state = env.reset() if done else next_state
        self.state = state
        return trajectory

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        alpha = self.alpha_log.exp().detach()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_a, next_log_prob = self.act_target.get_action_logprob(next_s)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_log_prob * alpha)
            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor)

            self.soft_update(self.act_target, self.act, soft_update_tau)

        return obj_critic.item(), obj_actor.item(), alpha.item()

class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = CriticAdv
        self.ClassAct = ActorPPO

        self.if_off_policy = False
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.01~0.05
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0, env_num=1):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_use_gae, gpu_id)
        self.trajectory_list = list()
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def select_action(self, state):
        states = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step):
        state = self.state

        trajectory_temp = list()
        last_done = -1
        for i in range(target_step):
            action, noise = self.select_action(state)
            current_state, next_state, reward, done = env.step(np.tanh(action))
            trajectory_temp.append((current_state, reward, done, action, noise))
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        if last_done < 0:
            self.trajectory_list.extend(trajectory_temp)
            return list()

        trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        self.trajectory_list = trajectory_temp[last_done + 1:]
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:]

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, dist_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate - dist_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1))
        return obj_critic.item(), obj_actor.item(), a_std_log.mean().item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value):
        'tensor, tensor '
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])  # fix a bug here
            pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage



