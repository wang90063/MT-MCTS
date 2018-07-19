from Atari import Atari
import gym
env=gym.make('Qbert-v4')
# env=Atari('qbert')
task_env = {'Qbert':env}

task_list = ['Qbert']

real_act_dim_dict={'Qbert':env.action_space.n}