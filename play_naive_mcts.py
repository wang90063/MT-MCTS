from multiprocessing import Pool,Process
episode_num=2
import time
import os
from copy import copy
def play():
    from MCTS import MCTSPlayer
    # import gym
    # env = gym.make('Qbert-v4')
    from Atari import Atari
    env=Atari('qbert')
    player = MCTSPlayer('Qbert')
    reward_tot = []

    terminal = True
    episode = 0
    step = 0
    reward_epi_tot = 0
    while episode < episode_num:
        if terminal:
            episode += 1
            ob=env.reset()
            player.reset_root()
            reward_tot.append(reward_epi_tot)
            step = 0
            reward_epi_tot = 0
        print('process_id:%i,episode:%i,step:%i'%(os.getpid(),episode, step))
        step += 1
        state=env.clone_state()
        start = time.time()
        action = player.get_action(state,ob)
        print(time.time() - start)
        ob, reward, terminal, info = env.step(int(action))
        player.update_root(action)
        reward_epi_tot += reward

        print(action)

    with open("test_data/navie_mcts"+str(os.getpid())+".txt", 'a') as k:
        for ts in reward_tot:
            k.write(str(ts) + '\n')
        k.close()


if __name__ =='__main__':
    print(os.getpid())

    pl = [Process(target=play,args=()) for _ in range(4)]
    for p in pl:
        p.start()

    for p in pl:
        p.join()











