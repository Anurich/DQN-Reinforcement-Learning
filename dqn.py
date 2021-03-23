import torch
import gym
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
from collections import deque
env = gym.make('MountainCar-v0')

class policyNetwork(nn.Module):
    def __init__(self, in_features, out_action):
        super(policyNetwork, self).__init__()
        self.dense =  nn.Linear(in_features = in_features, out_features = 512 )
        self.dense2 = nn.Linear(in_features= 512, out_features=64)
        self.output = nn.Linear(in_features =64, out_features=out_action)
    def forward(self, x):
        x = torch.relu(self.dense(x))
        x = torch.relu(self.dense2(x))
        output = self.output(x)
        return output

class network(nn.Module):
    def __init__(self, **kwg):
        super(network, self).__init__()
        self.state_space = kwg["state"]
        self.action_space = kwg["action"]
        self.policy_net  = policyNetwork(self.state_space, self.action_space)
        self.target_net  =policyNetwork(self.state_space, self.action_space)
        '''
        for param in self.target_net.parameters():
            param.requires_grad = False
        '''

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer= torch.optim.Adam(self.policy_net.parameters())

    def forward(self, x, type_="policy"):
        if type_ is "policy":
            return self.policy_net(x)
        else:
            return self.target_net(x)
# this class will store the logic of
# storing and taking action
class main(nn.Module):
    def __init__(self, **kwgs):
        super(main,self).__init__()
        self.net  = network(**kwgs)
        self.maxLengthDeque = 10000
        self.dq   = deque(maxlen=self.maxLengthDeque)
        self.discount_factor = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0
    def storage(self,state, action, next_state, done, reward):
        if len(self.dq) >= self.maxLengthDeque:
            self.dq.pop()
        else:
            # we save the value in dq
            self.dq.append((state, action, next_state, done, reward))
    def take_action(self,state):
        # we take action
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *np.math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        action =None
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float32)
                action = torch.argmax(self.net.policy_net(state)).item()
        else:
             action = random.randrange(self.net.action_space)
        return action

    def sample(self, batch_size):
        return random.sample(self.dq, batch_size)

    def transfer_weight(self):
        self.net.target_net.load_state_dict(self.net.policy_net.state_dict())

def fill_memory(agent):
    done=False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        agent.storage(state,action,next_state,done, reward)

def learn(agent, batch_size):
    if len(agent.dq)<batch_size:
        return
    sampled_data= agent.sample(batch_size)
    states = []
    actions= []
    next_states =[]
    dones = []
    rewards = []
    for state, act, next_st, compl, rew in sampled_data:
        states.append(state)
        actions.append(act)
        next_states.append(next_st)
        dones.append(compl)
        rewards.append(rew)
    states = torch.from_numpy(np.array(states))
    next_states = torch.from_numpy(np.array(next_states))
    dones  = torch.from_numpy(np.array(dones))
    dones = dones.type(torch.FloatTensor)
    actions = torch.from_numpy(np.array(actions))
    rewards = torch.from_numpy(np.array(rewards))

    policy_value = agent.net(states.type(torch.FloatTensor)).gather(1,actions.view(-1,1))
    q_target  = agent.net(next_states.type(torch.FloatTensor),type_="target").max(1).values
    checker = torch.ones(dones.shape[0])
    target = rewards + agent.discount_factor*torch.mul(q_target,(torch.sub(checker,dones)))
    loss_ = loss(target.view(-1,1), policy_value)
    agent.net.optimizer.zero_grad()
    loss_.backward()
    for param in agent.net.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    agent.net.optimizer.step()
# loss
loss = nn.SmoothL1Loss()
def train():
    state_size = env.reset().shape[0]
    action_size = env.action_space.n
    agent = main(state=state_size, action=action_size)
    fill_memory(agent)
    episode = 1000
    batch_size = 128
    render = True
    update_target = 10
    reward_history = []
    total_reward_episode = 0.0
    total_len = env.reset().shape[0]
    for cnt in tqdm(range(episode)):
        initial_state = env.reset()
        done =False
        while not done:
            if render:
                env.render()
            # first take the action from the
            action = agent.take_action(initial_state)
            next_state, reward, done, _ = env.step(action)
            reward = 100*((np.math.sin(3*next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) - (np.math.sin(3*initial_state[0]) * 0.0025 + 0.5 * initial_state[1] * initial_state[1]))
            agent.storage(initial_state, action, next_state, done, reward)
            # learning
            # we sample the batch
            if cnt%update_target == 0 and cnt!= 0:
                agent.transfer_weight()
            learn(agent, batch_size)
            total_reward_episode += reward
            initial_state = next_state
        if cnt % 100 == 0 and cnt!=0:
            print("Total reward after epoch {} is {}".format(str(cnt), str(total_reward_episode/100)))
            reward_history.append(total_reward_episode/100)
            total_reward_episode = 0.0
        if cnt % 300 == 0:
            torch.save({
                "EPOCH": cnt,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": agent.net.optimizer.state_dict(),
                "loss": total_reward_episode
            },"model.pt")
        #agent.update_epsilon()
    print("Average reward", str(np.mean(reward_history)))
#train()

