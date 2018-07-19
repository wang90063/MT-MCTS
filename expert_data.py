##########################################################
# MCTS experts generates data.
#TODO: constrain aggregated data at every iteration
##########################################################
import numpy as np
from MCTS import MCTSPlayer
from task_list_dict import task_env, real_act_dim_dict,task_list
from policy_value_net import network, BatchManager
from skimage.color import rgb2gray
from skimage.transform import resize
import random
from  multiprocessing import Process,Event,Queue
import json
import tensorflow as tf
import psutil
def preprocess(observation):
    processed_observation = np.uint8(
        resize(rgb2gray(observation), (84, 84), mode='constant') * 255)
    return processed_observation


class config:
    episode_num = 100
    epsilon=0.05
    Dagger_ite=20
    Mome_ther=100*1024*1024


class expert_data:
    def __init__(self,task,queue):
        self.task=task
        self.queue=queue
        self.env=task_env[self.task]
        self.player = MCTSPlayer(self.task)

    def data_generation(self):

        import tensorflow as tf

        self.net = network()

        checkpoint = tf.train.get_checkpoint_state('saved_nn')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.is_guide = True
        else:
            self.is_guide = False

        terminal=True
        episode=0
        while not self.queue.full():

            if terminal:
                ob = self.env.reset()
                self.player.reset_root()
                ram = ob['ram']
                rgb_frame = ob['image']
                grey_frame = preprocess(rgb_frame)
                frame = np.stack((grey_frame, grey_frame, grey_frame, grey_frame), axis=2)
                episode+=1
                step=0
            step+=1
            print('episode: %f, step: %f'%(episode,step))

            state = self.env.unwrapped.clone_state()
            action = self.player.get_action(state,ram)
            value = self.player.get_value_vec()
            policy = self.player.get_policy()
            self.player.update_root(action)

            # Note that ram is a vector, and policy and value are one-dimension matrix
            data_point = {'task': self.task, 'ram': ram,
                                      'frame': frame, 'value': value, 'policy': policy}
            self.queue.put(data_point)

            if not self.is_guide:
                nextob, reward, terminal, info = self.env.step(int(action))
            else:
                value,_= self.net.get_image_pred(self.sess, self.task, frame)

                if random.random() <= config.epsilon:
                    action_stud = random.randrange(real_act_dim_dict[self.task])
                else:
                    action_stud = np.argmax(value)

                nextob, reward, terminal, info = self.env.step(action_stud)
            next_frame= preprocess(nextob['image'])
            next_frame = np.reshape(next_frame,[next_frame.shape[0],next_frame.shape[1],1])
            frame = np.append(frame[:, :, 1:], next_frame, axis=2)
            ram = nextob['ram']

        if self.is_guide:
            self.sess.close()


if __name__=="__main__":
    config=config()
    for ite in range(config.Dagger_ite):

        ##########################################################################
        # generate the data using experts
        ###########################################################################

        for task in task_list:
            queue = Queue(maxsize=100000)
            process = Process()
            expert = expert_data(task, queue)
            # expert.data_generation()
            pl = [Process(target=expert.data_generation,args=()) for _ in range(4)]
            for p in pl:
                p.start()

            for p in pl:
                p.join()


            # store the epoch obtained at every iteration
            epoch = {'task': task, 'ram_epoch': [], 'frame_epoch': [], 'value_epoch': [], 'policy_epoch': []}#}
            while not queue.empty():
                data_point = queue.get()

                epoch['ram_epoch'].append(data_point['ram'].tolist())
                epoch['frame_epoch'].append(data_point['frame'].tolist())
                epoch['value_epoch'].append(data_point['value'].tolist())
                epoch['policy_epoch'].append(data_point['policy'].tolist())

            with open('data/data_'+task+str(ite) + '.json', 'w') as f:
                json.dump(epoch,f)

            epoch['ram_epoch']=np.array(epoch['ram_epoch'])
            epoch['frame_epoch']=np.array(epoch['frame_epoch'])
            epoch['value_epoch']=np.array(epoch['value_epoch']).reshape((-1,real_act_dim_dict[task]))
            epoch['policy_epoch']=np.array(epoch['policy_epoch']).reshape((-1,real_act_dim_dict[task]))


            # aggeragate the data in the 'Dagger_data_(task)'
            try:
                with open('data/Dagger_data_'+task+'.json', 'r') as f:
                    dagger_epoch = json.load(f)
                dagger_epoch['ram_epoch']=np.concatenate((epoch['ram_epoch'],dagger_epoch['ram_epoch']),axis=0)
                dagger_epoch['frame_epoch']=np.concatenate((epoch['frame_epoch'], dagger_epoch['frame_epoch']), axis=0)
                dagger_epoch['value_epoch']=np.concatenate((epoch['value_epoch'], dagger_epoch['value_epoch']), axis=0)
                dagger_epoch['policy_epoch'] =np.concatenate((epoch['policy_epoch'], dagger_epoch['policy_epoch']), axis=0)
            except:
                print('No dagger data yet!')
                dagger_epoch=epoch

            # "json" cannot dump array, so I convert the arrays to lists.
            dagger_epoch_store={}
            dagger_epoch_store['task'] = task
            dagger_epoch_store['ram_epoch']=dagger_epoch['ram_epoch'].tolist()
            dagger_epoch_store['frame_epoch'] = dagger_epoch['frame_epoch'].tolist()
            dagger_epoch_store['value_epoch'] = dagger_epoch['value_epoch'].tolist()
            dagger_epoch_store['policy_epoch'] = dagger_epoch['policy_epoch'].tolist()
            with open('data/Dagger/Dagger_data_'+task+'.json', 'w') as f:
                json.dump(dagger_epoch_store, f)

        ##############################################################################
        # Utilizing the data to do the imitation learning
        ##############################################################################
        net=network()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state('saved_nn')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        queue_empty_indicator=False
        batch_generator = BatchManager()

        batch_generator.start_processes()

        train_step=0
        while not queue_empty_indicator:
            mt_batch,queue_empty_indicator=batch_generator.get_mt_batch()
            net.train_on_batch(sess,mt_batch)

            mem=psutil.virtual_memory()
            if mem.available<config.Mome_ther:
                batch_generator.close_processes()
                batch_generator.start_processes()

        batch_generator.close_processes()

        saver.save(sess,'saved_nn/' + 'network', global_step = train_step)

        sess.close()



















