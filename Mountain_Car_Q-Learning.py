# MOUNTAIN CAR PROBLEM 
import gym
from IPython.display import clear_output
from time import sleep
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

env = gym.make('MountainCar-v0')                 #Initilizes the Moutain car problem and then resets it
env.reset()

def QLearning(env, learning, discount, epsilon, min_eps, episodes):              #Q-Learning function 
    
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])                                          # Determin the state space of the Mountain Car
    num_states = np.round(num_states, 0).astype(int) + 1
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1],                  # Creates a Q-Table
                                  env.action_space.n))
    reward_list = []                             # Initiliases the reward list, will be appended with the learning
    ave_reward_list = []                         # Creates a list for the Average Rewards                                    
    reduction = (epsilon - min_eps)/episodes     # Reduces the Epsilon by subtracting the minimum episode and dividing this by the episodes
    
    for i in range(episodes):                    # Runs the Q-learning for indicated amount of episodes  
        done = False                             # Initialize parameters
        tot_reward, reward = 0,0                 # Sets the intial reward feilds to 0 
        state = env.reset()                      # Resets the environemt
        
        state_adj = (state - env.observation_space.low)*np.array([10, 100])     # States the dicretized state data 1st value *10 second *100 that wil be ajusted with Q-Learning later
        state_adj = np.round(state_adj, 0).astype(int)                          # Dicretizes the data by rounding it to the nearest whole number( Note 1.5 will round down to 1)
    
        while done != True:   
            if np.random.random() < 1 - epsilon:                                # Finds next action using the epsilon (Greedy) value 
                action = np.argmax(Q[state_adj[0], state_adj[1]])               # Next Action that is found with the Q-table
            else:
                action = np.random.randint(0, env.action_space.n)               # When the random value dosent meet the greedy stament it creats a random value between 0 and the current action space
                
            state2, reward, done, info = env.step(action)                       # Defines the parameters for the next step
        
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])      # Discretized state2 value
            state2_adj = np.round(state2_adj, 0).astype(int)                           # Rounds the state2_adj to the nearest 0 as an integer type
            

            if done and state2[0] >= 0.5:                                              # Assures the car start position is at or above 0.5
                Q[state_adj[0], state_adj[1], action] = reward                         # Creates the reward for the change from stae 0 to state 1 
                
            else:                                                                      # Makes changes to the Q value for the current state                                                               
                delta = learning*(reward + discount*np.max(Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
            tot_reward += reward                            # Incrimetaly adds each reward to the total 
            state_adj = state2_adj                          # Defines state_adj to stae2_adj
        if epsilon > min_eps:                               # Reduces Epsilon when it is greater then minimum episodes
            epsilon -= reduction                            # Epsilon is subtracted by reduction value (Defined on line 19)
        
        reward_list.append(tot_reward)                      # Add the total rewards to the List of rewards 
        
        if (i+1) % 100 == 0:                                # As the episode ends in 99, the 100 increment is averaged(100,200-1900)
            ave_reward = np.mean(reward_list)               # Average reward at the 00'th episode 
            ave_reward_list.append(ave_reward)              # Added to the total reward list 
            reward_list = []
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))   # Will Print the Episore and average reward every 100 episodes 
        
    return ave_reward_list

rewards = QLearning(env, 0.1, 0.9, 0.8, 0, 2000)     # Q-Learningwith fields: environment,learning rate,discount value ,epsilon,min_eps, episodes run 

# Plot Rewards
plt.plot((100*np.arange(len(rewards)) +1 )  , rewards) #Plots the Rewards 100* makes the episode numbers accurate, 
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()  
