# Reinforcement_Learning_Project
Code the works through my project implemtation with Open Ai gym and tackles these using q_Learning.

Taxi Problem : https://github.com/mpfletcher1/Reinforcement_Learning_Project/blob/main/Taxi_Q-Learning_and_Random_Search
Cart Pole : https://github.com/mpfletcher1/Reinforcement_Learning_Project/blob/main/Cart_Pole_Q-Learning_and_Random
Moutain Car : https://github.com/mpfletcher1/Reinforcement_Learning_Project/blob/main/Mountain_Car_Q-Learning.py

All Codes sources are Referenced here: https://github.com/mpfletcher1/Reinforcement_Learning_Project/blob/main/References 

The Only thing that is required to be isntalled is the Atari Environment, this was simply done in Juipter notebook by adding:

!pip install cmake 'gym[atari]' scipy

The Code was originalty in one large source file, all the imports were taken over to each individual file to the gethub all the resources needed are as follows (These are all in the code too):

import gym
from IPython.display import clear_output
from time import sleep
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


                   
