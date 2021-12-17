# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import copy
from multiprocessing import Pool
#import torch.multiprocessing as mp
import math
import timeit
import itertools
#import cProfile
import argparse
import pickle

parser = argparse.ArgumentParser(description='Parallel DRL training')
parser.add_argument('--train_times', metavar='N', type=int, default=204,
                    help='DRL training times/epoches')
parser.add_argument('--ap', metavar='%', type=float, default=0.3,
                    help='attack probability')
parser.add_argument('--l', metavar='N', type=int, default=5,
                    help='number of software versions')
parser.add_argument('--x', metavar='%', type=float, default=0.5,
                    help='degree of sv')
parser.add_argument('--b', metavar='%', type=int, default=500,
                    help='budget')
parser.add_argument('--gpu_num', metavar='%', type=int, default=2,
                    help='gpu_num')
parser.add_argument('--process_num', metavar='%', type=int, default=3,
                    help='process_num')
parser.add_argument('--p_s', metavar='%', type=float, default=0,
                    help='state_manupulation_attack')
parser.add_argument('--d_r', metavar='%', type=float, default=1,
                    help='detection rate for state_manupulation_attack')
parser.add_argument('--ev', metavar='%', type=int, default=0,
                    help='eval indicator')

args = parser.parse_args()

if args.ev == 1:
    evals = np.load('evals/evals_new_%.2f,%.1f,%d.npy'%(args.ap,args.x,args.l))
elif args.ev == 2:
    evals = np.load('evals/evals_nnew_%.2f,%.1f,%d.npy'%(args.ap,args.x,args.l))
else:
    evals = np.load('evals/evals_%.2f,%.1f,%d.npy'%(args.ap,args.x,args.l))
    
def get_next(area,action):
    x1, y1, x2, y2, x3, y3 = area
    if action == 0:
        x1, y1, x2, y2, x3, y3 = (x1+x2)/2, (y1+y2)/2, x2, y2, (x2+x3)/2, (y2+y3)/2
        area = [x1, y1, x2, y2, x3, y3]
        state = [(x1+x2+x3)/3, (y1+y2+y3)/3]
        return area, state
    elif action == 1:
        x1, y1, x2, y2, x3, y3 = x1, y1, (x1+x2)/2, (y1+y2)/2, (x1+x3)/2, (y1+y3)/2
        area = [x1, y1, x2, y2, x3, y3]
        state = [(x1+x2+x3)/3, (y1+y2+y3)/3]
        return area, state
    elif action == 2:
        x1, y1, x2, y2, x3, y3 = (x2+x3)/2, (y2+y3)/2, (x1+x3)/2, (y1+y3)/2, (x1+x2)/2, (y1+y2)/2
        area = [x1, y1, x2, y2, x3, y3]
        state = [(x1+x2+x3)/3, (y1+y2+y3)/3]
        return area, state
    elif action == 3:
        x1, y1, x2, y2, x3, y3 = (x1+x3)/2, (y1+y3)/2, (x2+x3)/2, (y2+y3)/2, x3, y3
        area = [x1, y1, x2, y2, x3, y3]
        state = [(x1+x2+x3)/3, (y1+y2+y3)/3]
        return area, state

p_s = args.p_s
detection_rate = args.d_r
indicator = np.random.rand(args.b+1, args.b+1)

class ENV(object):
    def __init__(self, length):
        self.length = length
        self.area = [0,self.length,0,0,self.length,0]
        self.center = np.array([self.length/3,self.length/3],dtype=float)
        self.state = np.array([0]*math.ceil(math.log(self.length,2)))
        self.steps = 0
        self.action_space = np.array([0,1,2,3],dtype=int)
        
    def reset(self):
        self.area = [0,self.length,0,0,self.length,0]
        self.center = np.array([self.length/3,self.length/3],dtype=int)
        self.steps = 0
        self.state = np.array([0]*math.ceil(math.log(self.length,2)))
        return self.state
    
    def step(self,action):
        done = False
        x,y = round(self.center[0]), round(self.center[1])
        real_cf = evals[x,y]
        if random.random() < p_s:
            if random.random() < detection_rate:
                current_f = indicator[x,y]
            else:
                indicator[x,y] = random.random()
                current_f = indicator[x,y]
        else:
            indicator[x,y] = evals[x,y]
            current_f = indicator[x,y]
        #print(self.area)
        self.area,self.center = get_next(self.area,action)
        x,y = round(self.center[0]), round(self.center[1])
        real_nf = evals[x,y]
        if random.random() < p_s:
            if random.random() < detection_rate:
                next_f = indicator[x,y]
            else:
                indicator[x,y] = random.random()
                next_f = indicator[x,y]
        else:
            indicator[x,y] = evals[x,y]
            next_f = indicator[x,y]
        self.state[self.steps] = action + 1
        self.steps += 1
        if abs(self.area[2]-self.area[4])<=1:
            done = True
        #print(self.area)
        real_r = real_nf - real_cf
        r = next_f - current_f
        return self.state, r, real_r, done
    
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0, a, r, s1, done))

    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        s0 = torch.tensor(s0, dtype=torch.float).cuda()
        s1 = torch.tensor(s1, dtype=torch.float).cuda()
        a = torch.tensor(a, dtype=torch.long).cuda()
        r = torch.tensor(r, dtype=torch.float).cuda()
        done = torch.tensor(done, dtype=torch.float).cuda()
        return s0, a, r, s1, done

    def size(self):
        return len(self.buffer)

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.memory = ReplayBuffer(2000)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9
        self.learning_rate = 0.02
        size = 256
        #224:optimal:5/100,better than greedy:55/100, at least greedy:76/100,320.0211285299997s
        #280:optimal:12/100,better than greedy:64/100, at least greedy:86/100,434.8715323809997s

        self.nn = nn.Sequential(
            nn.Linear(self.state_size, size),
            nn.ReLU(),
            nn.Linear(size, 2*size),
            nn.ReLU(),
            #nn.Linear(2*size, 2*size),
            #nn.ReLU(),
            nn.Linear(2*size, size),
            nn.ReLU(),
            nn.Linear(size, self.action_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.85, 0.99), weight_decay=1e-4)#no need cuda
        self.is_training = True

    def model(self, x):
        return self.nn(x)
    
    def act(self, state):
        if random.random() > self.epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
            q_value = self.model(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)
        return action
    
    def remember(self, state, action, reward, next_state, done):#, batch_size):
        self.memory.add(state, action, reward, next_state, done)
        #if self.memory.size() < batch_size:
            #self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        s0, a, r, s1, done = self.memory.sample(batch_size)

        q_values = self.model(s0)
        next_q_values = self.model(s1)
        next_q_value = next_q_values.max(1)[0]

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def greedy(length):
    total = 0
    for i in range(args.train_times*100):
        area = [0,length,0,0,length,0]
        while abs(area[2]-area[4])>1:
            max_area,max_state = get_next(area,0)
            x,y = round(max_state[0]), round(max_state[1])
            if random.random() > p_s:
                max_f = evals[x,y]
            else:
                max_f = random.random()
            max_action = 0
            for action in range(1,4):
                temp_area,state = get_next(area,action)
                x,y = round(state[0]), round(state[1])
                if random.random() > p_s:
                    temp = evals[x,y]
                else:
                    temp = random.random()
                if temp>max_f:
                    max_state = state
                    max_f = temp
                    max_area = temp_area
                    max_action = action
            area = max_area
        x,y = round(max_state[0]), round(max_state[1])
        real_f = evals[x,y]
        total += real_f-evals[round(length/3),round(length/3)]
    return total/args.train_times/100
    
greedy_result = greedy(args.b)
print("p_s=%.2f,greedy_result:%f" % (p_s, greedy_result))

def train(steps, device):
    torch.cuda.set_device(device)
    episodes = 1000#150#120
    max_reward_list = []
    max_episode = [-np.inf]*episodes
    min_episode = [np.inf]*episodes
    avg_episode = [0]*episodes
    bc_list, ba_list = [], []
    for i in range(steps):
        bc, ba = 0, 0
        max_reward = -np.inf
        env = ENV(args.b)
        state_size = len(env.state)
        action_size = len(env.action_space)
        done = False
        agent = DQNAgent(state_size, action_size)
        agent.cuda()
        batch_size = 7*8*9#7*8*7#7*8*5
        for e in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while done == False:
                action = agent.act(state)
                next_state, reward, real_r, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)#, batch_size)
                state = next_state
                total_reward += real_r
                if agent.memory.size() > batch_size:
                    agent.replay(batch_size)
            max_episode[e] = max(max_episode[e],total_reward)
            min_episode[e] = min(min_episode[e],total_reward)
            avg_episode[e] += total_reward
            if total_reward > max_reward:
                max_reward = total_reward
                bc, ba = round(env.center[0]), round(env.center[1])
            #print(e,total_reward)    
        max_reward_list.append(max_reward)
        bc_list.append(bc)
        ba_list.append(ba)
        #print(i,max_reward)
    #print("better than greedy:%d/%d, at least greedy:%d/%d"%(sum([i>greedy_result for i in max_reward_list), steps, sum([i>=greedy_result for i in max_reward_list]),steps))
    return max_reward_list, max_episode, min_episode, avg_episode, bc_list, ba_list

def multi_train(train_times, process_num, gpu_num):#for best performance, please set train_times % 6 == 0
    temp = math.floor(train_times/gpu_num/process_num)
    remain = train_times/gpu_num-temp*process_num
    if remain > 0:
        process_num += 1
        input_array = [temp for i in range(process_num)] + [remain]
    else:
        input_array = [temp for i in range(process_num)]
    input_array = [[i,j] for j in range(gpu_num) for i in input_array]
    #print(input_array)
    pool = Pool()
    #print("Separated results:")
    results = pool.starmap(train, input_array)
    pool.close()
    pool.join()
    result = [list(itertools.chain.from_iterable([results[i][j] for i in range(process_num*gpu_num)])) for j in range(6)]
    max_reward_list = result[0]
    print("Merged results:")
    print("Average:%f"%(sum(max_reward_list)/train_times))
    print("better than greedy:%d/%d, at least greedy:%d/%d"%(sum([i>greedy_result for i in max_reward_list]), train_times, sum([i>=greedy_result for i in max_reward_list]),train_times))
    if args.ev == 1:
        with open("results/dqn/dqn_results_sgmd_%.2f,%.1f,%d,%d,%.2f,%.3f.txt"%(args.ap,args.x,args.l,args.b,args.p_s,args.d_r), "wb") as fp:   #fc, sgc
            pickle.dump(result, fp)
    elif args.ev == 2:
        with open("results/dqn/dqn_results_md_%.2f,%.1f,%d,%d,%.2f,%.3f.txt"%(args.ap,args.x,args.l,args.b,args.p_s,args.d_r), "wb") as fp:   #fc, sgc
            pickle.dump(result, fp)
    else:
        with open("results/dqn/dqn_results_sg_%.2f,%.1f,%d,%d,%.2f,%.3f.txt"%(args.ap,args.x,args.l,args.b,args.p_s,args.d_r), "wb") as fp:   #fc, sgc
            pickle.dump(result, fp)
    #return result
    
#torch.multiprocessing.set_start_method('spawn', force=True)
gpu_num = args.gpu_num
process_num = args.process_num
train_times = args.train_times
#for best performance, please make train_times % (gpu_num*process_num) == 0
elapsed_time = timeit.timeit('multi_train(%d, %d, %d)'%(train_times, process_num, gpu_num), 'from __main__ import multi_train', number=1)/1
print("Total training times:%d, Time cost:%fs." % (train_times,elapsed_time))