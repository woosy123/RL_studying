# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# Lib
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import os

# Files
from noise import OrnsteinUhlenbeckActionNoise as OUNoise
from replaybuffer import Buffer
from actorcritic import Actor, Critic

PLOT_FIG = False

# Hyperparameters
ACTOR_LR = 0.0003
CRITIC_LR = 0.003
MINIBATCH_SIZE = 64
NUM_EPISODES = 90
NUM_TIMESTEPS = 3
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './checkpoints/manipulator/'
BUFFER_SIZE = 100000
DISCOUNT = 0.9
TAU = 0.001
WARMUP = 70
EPSILON = 1.0
EPSILON_DECAY = 1e-6

NUM_ACTIONS = 15
NUM_STATES = 5+2+3
NUM_RESOURCES = 5
ID = 'default'

# converts observation dictionary to state tensor
# TODO: currently it's conversion between list and state tensor
def obs2state(state_list):
    #l1 = [val.tolist() for val in list(observation.values())]
    #l2 = []
    #for sublist in l1:
    #    try:
    #        l2.extend(sublist)
    #    except:
    #        l2.append(sublist)
    return torch.FloatTensor(state_list).view(1, -1)

class DDPG:
    def __init__(self):
        # self.env = env
        self.stateDim = NUM_STATES
        self.actionDim = NUM_ACTIONS
        self.actor = Actor(self.stateDim, self.actionDim)
        self.critic = Critic(self.stateDim, self.actionDim)
        self.targetActor = deepcopy(Actor(self.stateDim, self.actionDim))
        self.targetCritic = deepcopy(Critic(self.stateDim, self.actionDim))
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA)
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.rewardgraph = []
        self.start = 0
        self.end = NUM_EPISODES

    # Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
    # Target Q-value <- reward and bootstraped Q-value of next state via the target actor and target critic
    # Output: Batch of Q-value targets
    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):       
        targetBatch = torch.FloatTensor(rewardBatch)
        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s != True, terminalBatch)))
        nextStateBatch = torch.cat(nextStateBatch)
        nextActionBatch = self.targetActor(nextStateBatch)
        qNext = self.targetCritic(nextStateBatch, nextActionBatch)  
        
        nonFinalMask = self.discount * nonFinalMask.type(torch.FloatTensor)
        targetBatch += nonFinalMask * qNext.squeeze().data
        
        return Variable(targetBatch)

    # weighted average update of the target network and original network
    # Inputs: target actor(critic) and original actor(critic)
    def updateTargets(self, target, original):
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)

    # Inputs: Current state of the episode
    # Output: the action which maximizes the Q-value of the current state-action pair
    def getMaxAction(self, curState):
        noise = self.epsilon * Variable(torch.FloatTensor(self.noise()))
        action = self.actor(curState)
        actionNoise = action + noise
        
        # get the max
        action_list = actionNoise.tolist()[0]
        max_action = max(action_list)
        max_index = action_list.index(max_action)

        return max_index, actionNoise

    # training of the original and target actor-critic networks
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        print('Training started...')
        
        action_step = 10
        available_actions = [0, action_step, -action_step]
        all_rewards = []
        avg_rewards = []
        # for each episode 
        for episode in range(self.start, self.end):
            # state = self.env.new_reset(ID)
            self.curr_arrival_rate = random.uniform(0.0, 1.0)
            self.cpu_limit = 1 # state.limit.cpu
            self.mem_limit = 1 # state.limit.memory
            self.llc_limit = 1 # state.limit.llc
            self.io_limit = 1 # state.limit.io
            self.net_limit = 1 # state.limit.network
            self.curr_cpu_util = random.randint(0,1) # state.usage.cpu
            self.curr_mem_util = random.randint(0,1) # state.usage.memory
            self.curr_llc_util = random.randint(0,1) # state.usage.llc
            self.curr_io_util = random.randint(0,1) # state.usage.io
            self.curr_net_util = random.randint(0,1) # state.usage.network
            self.slo_retainment = random.randint(0,1) # state.other.slo_retainment
            self.rate_ratio = random.randint(1,3) # state.other.rate_ratio
            self.percentages = [random.randint(1,3),random.randint(1,3), random.randint(1,3)] # state.other.percentages
            state = {
            'curr_arrival_rate': self.curr_arrival_rate,
            'cpu_limit': self.cpu_limit,
            'mem_limit': self.mem_limit,
            'llc_limit': self.llc_limit,
            'io_limit': self.io_limit,
            'net_limit': self.net_limit,
            'curr_cpu_util': self.curr_cpu_util,
            'curr_mem_util': self.curr_mem_util,
            'curr_llc_util': self.curr_llc_util,
            'curr_io_util': self.curr_io_util,
            'curr_net_util': self.curr_net_util,
            'slo_retainment': self.slo_retainment,
            'rate_ratio': self.rate_ratio,
            'percentages': self.percentages
            }
            ep_reward = 0
            
            for step in range(NUM_TIMESTEPS):
            # while not time_step.last():

                cpu_limit = state['cpu_limit']
                mem_limit = state['mem_limit']
                llc_limit = state['llc_limit']
                io_limit = state['io_limit']
                net_limit = state['net_limit']
                curr_cpu_util = state['curr_cpu_util']
                curr_mem_util = state['curr_mem_util']
                curr_llc_util = state['curr_llc_util']
                curr_io_util = state['curr_io_util']
                curr_net_util = state['curr_net_util']
                curr_arrival_rate = state['curr_arrival_rate']
                slo_retainment = state['slo_retainment']
                rate_ratio = state['rate_ratio']
                percentages = state['percentages']

                # print each time step only at the last EPISODE
                if episode == NUM_EPISODES-1:
                    print("EP:", episode, " | Step:", step)
                    print("Update - Current SLO Retainment:", slo_retainment)
                    print("Update - Current Util:", str(curr_cpu_util)+'/'+str(cpu_limit), str(curr_mem_util)+'/'+str(mem_limit), str(curr_llc_util)+'/'+str(llc_limit), str(curr_io_util)+'/'+str(io_limit), str(curr_net_util)+'/'+str(net_limit))

                # get maximizing action
                currStateTensor = Variable(obs2state([curr_cpu_util/cpu_limit,curr_mem_util/mem_limit,curr_llc_util/llc_limit,curr_io_util/io_limit,curr_net_util/net_limit,slo_retainment,rate_ratio,percentages[0],percentages[1],percentages[2]])) 
                self.actor.eval()     
                action, actionToBuffer = self.getMaxAction(currStateTensor)

                cpu_action = 0
                if action < 3:
                    cpu_action = available_actions[action]
                mem_action = 0
                if action >= 3 and action < 6:
                    mem_action = available_actions[action-3]
                llc_action = 0
                if action >= 6 and action < 9:
                    llc_action = available_actions[action-6]
                io_action = 0
                if action >= 9 and action < 12:
                    io_action = available_actions[action-9]
                net_action = 0
                if action >= 12:
                    net_action = available_actions[action-12]

                if episode == NUM_EPISODES-1:
                    print("Update - Actions to take:", cpu_action, mem_action, llc_action, io_action, net_action)

                self.actor.train()
                
                # step episode
                # state, reward, done = self.env.new_step(cpu_action, mem_action, llc_action, io_action, net_action, ID)
                
                curr_arrival_rate = random.randint(0, 1)
                cpu_limit += cpu_action  # response.limit.cpu
                mem_limit += mem_action # response.limit.memory
                llc_limit += llc_action # response.limit.llc
                io_limit += io_action # response.limit.io
                net_limit += net_action # response.limit.network
                curr_cpu_util = random.randint(0,1) # response.usage.cpu
                curr_mem_util = random.randint(0,1) # response.usage.memory
                curr_llc_util = random.randint(0,1) # response.usage.llc
                curr_io_util = random.randint(0,1) # response.usage.io
                curr_net_util = random.randint(0,1) # response.usage.network
                slo_retainment = random.randint(0,1) # state.other.slo_retainment
                rate_ratio = random.randint(1,3) # response.usage.rate
                percentages = [random.randint(1,3),random.randint(1,3), random.randint(1,3)] #response.usage.percentages
                reward = NUM_RESOURCES*self.slo_retainment + self.curr_cpu_util/self.cpu_limit + self.curr_mem_util/self.mem_limit + self.curr_llc_util/self.llc_limit + self.curr_io_util/self.io_limit + self.curr_net_util/self.net_limit
                state = {
                    'cpu_limit': cpu_limit,
                    'mem_limit': mem_limit,
                    'llc_limit': llc_limit,
                    'io_limit': io_limit,
                    'net_limit': net_limit,
                    'curr_cpu_util': curr_cpu_util,
                    'curr_mem_util': curr_mem_util,
                    'curr_llc_util': curr_llc_util,
                    'curr_io_util': curr_io_util,
                    'curr_net_util': curr_net_util,
                    'slo_retainment': slo_retainment,
                    'curr_arrival_rate': curr_arrival_rate, # workload
                    'rate_ratio': rate_ratio,               # workload
                    'percentages': percentages              # workload
                }
                print('Reward: {}'.format(reward))
                curr_arrival_rate = state['curr_arrival_rate']
                cpu_limit = state['cpu_limit']
                mem_limit = state['mem_limit']
                llc_limit = state['llc_limit']
                io_limit = state['io_limit']
                net_limit = state['net_limit']
                curr_cpu_util = state['curr_cpu_util']
                curr_mem_util = state['curr_mem_util']
                curr_llc_util = state['curr_llc_util']
                curr_io_util = state['curr_io_util']
                curr_net_util = state['curr_net_util']
                slo_retainment = state['slo_retainment']
                rate_ratio = state['rate_ratio']
                percentages = state['percentages']                
                
                nextState = Variable(obs2state([curr_cpu_util/cpu_limit,curr_mem_util/mem_limit,curr_llc_util/llc_limit,curr_io_util/io_limit,curr_net_util/net_limit,slo_retainment,rate_ratio,percentages[0],percentages[1],percentages[2]]))
                ep_reward += reward
                done = random.randint(0,3)
                # Update replay bufer
                self.replayBuffer.append((currStateTensor, actionToBuffer, nextState, reward, done))
                
                # Training loop
                if len(self.replayBuffer) >= self.warmup:
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)
                    curStateBatch = torch.cat(curStateBatch)
                    actionBatch = torch.cat(actionBatch)
                    
                    qPredBatch = self.critic(curStateBatch, actionBatch)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)
                    
                    # Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                    print('Critic Loss: {}'.format(criticLoss))
                    # criticLoss.backward(retain_graph=True)
                    self.criticOptim.step()
            
                    # Actor update
                    self.actorOptim.zero_grad()
                    actorLoss = -torch.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                    print('Actor Loss: {}'.format(actorLoss))
                    # actorLoss.backward(retain_graph=True)
                    self.actorOptim.step()
                    
                    # Update Targets                        
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)
                    self.epsilon -= self.epsilon_decay
            print("EP -", episode, "| Total Reward -", ep_reward)
   
            # save to checkpoints
            if episode % 20 == 0:
                self.save_checkpoint(episode)
            self.rewardgraph.append(ep_reward)

            if PLOT_FIG:
                if episode % 1000 ==0 and episode != 0:
                    plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Episodes')
                    plt.savefig('ep'+str(episode)+'.png')
        if PLOT_FIG:
            plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
            # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.savefig('final.png')

    def save_checkpoint(self, episode_num):
        checkpointName = self.checkpoint_dir + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetActor': self.targetActor.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOpt': self.actorOptim.state_dict(),
            'criticOpt': self.criticOptim.state_dict(),
            'replayBuffer': self.replayBuffer,
            'rewardgraph': self.rewardgraph,
            'epsilon': self.epsilon
            
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.targetActor.load_state_dict(checkpoint['targetActor'])
            self.targetCritic.load_state_dict(checkpoint['targetCritic'])
            self.actorOptim.load_state_dict(checkpoint['actorOpt'])
            self.criticOptim.load_state_dict(checkpoint['criticOpt'])
            self.replayBuffer = checkpoint['replayBuffer']
            self.rewardgraph = checkpoint['rewardgraph']
            self.epsilon = checkpoint['epsilon']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')
