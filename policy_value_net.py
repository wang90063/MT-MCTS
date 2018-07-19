
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from multiprocessing import Process, Queue, Event
import json
import math
from task_list_dict import task_list
from task_list_dict import real_act_dim_dict

########################################################################################
#Train the network
########################################################################################

# "Class network" sets up the calculation on the "GPU:0"
class config:
    batch_size=256
    train_iter=20
    lr=0.0075
    temperature=1

class network:

    def __init__(self):
        # "task_keys" is list storing all the task names
        self.real_act_dim_dict=real_act_dim_dict
        self.task_list=task_list
        self.num_task=len(self.task_list)
        self.config = config
        self.build()


    def add_placeholders(self):
        state=tf.placeholder('float',[None,84,84,4],name='state')
        ram=tf.placeholder('float',[None,128],name='ram')
        policy_label_dict={}
        value_label_dict={}
        for task in self.task_list:
            policy_label = tf.placeholder('float',[None,self.real_act_dim_dict[task]],name='plabel_'+task)
            value_label = tf.placeholder('float',[None,self.real_act_dim_dict[task]],name='qvlabel_'+task)
            policy_label_dict[task]=policy_label
            value_label_dict[task]=value_label
        return state,ram, value_label_dict,policy_label_dict

    def add_pred_op(self,state,ram):


        self.global_step = tf.Variable(0)

        with tf.variable_scope('image_net'):
            out = layers.conv2d(state, num_outputs=32, kernel_size=[8, 8], stride=4, padding='SAME',
                                activation_fn=tf.nn.relu,
                                normalizer_fn=tf.layers.batch_normalization,
                                normalizer_params={'training': False, 'reuse': False}, scope='conv1')
            out = layers.conv2d(out, num_outputs=64, kernel_size=[4, 4], stride=2, padding='SAME',
                                activation_fn=tf.nn.relu,
                                normalizer_fn=tf.layers.batch_normalization,
                                normalizer_params={'training': False, 'reuse': False}, scope='conv2')
            out = layers.conv2d(out, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu,
                                normalizer_fn=tf.layers.batch_normalization,
                                normalizer_params={'training': False, 'reuse': False}, scope='conv3')
            out = layers.flatten(out)
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.layers.batch_normalization,
                                         normalizer_params={'training': False, 'reuse': False}, scope='fc4')

            values_image_dict = {}
            policies_image_dict = {}

            for task in self.task_list:
                value = layers.fully_connected(out, num_outputs=self.real_act_dim_dict[task], activation_fn=None,
                                               normalizer_fn=None, scope="value_" + task)
                values_image_dict[task] = value
                policy = tf.nn.softmax(layers.fully_connected(out, num_outputs=self.real_act_dim_dict[task], activation_fn=None,
                                                normalizer_fn=None, scope="policy_" + task))
                policies_image_dict[task] = policy

        with tf.variable_scope('ram_net'):

            out_ram = layers.fully_connected(ram, num_outputs=128, activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.layers.batch_normalization,
                                         normalizer_params={'training': False, 'reuse': False}, scope='fc1')
            out_ram = layers.fully_connected(out_ram, num_outputs=128, activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.layers.batch_normalization,
                                         normalizer_params={'training': False, 'reuse': False}, scope='fc2')
            values_ram_dict = {}
            policies_ram_dict = {}
            for task in self.task_list:
                value_ram = layers.fully_connected(out_ram, num_outputs=self.real_act_dim_dict[task], activation_fn=None,
                                               normalizer_fn=None, scope="value_" + task)
                values_ram_dict[task] = value_ram
                policy_ram = tf.nn.softmax(layers.fully_connected(out, num_outputs=self.real_act_dim_dict[task], activation_fn=None,
                                                normalizer_fn=None, scope="policy_" + task))
                policies_ram_dict[task] = policy_ram


        return values_image_dict, policies_image_dict,values_ram_dict, policies_ram_dict


    def add_loss_op(self,values_image_dict,policies_image_dict,values_ram_dict,policies_ram_dict):
        losses_image_dict={}
        losses_ram_dict = {}
        for task in self.task_list:
            policy_image_loss= tf.reduce_mean(-tf.reduce_sum(self.policy_label_dict[task]*tf.log(policies_image_dict[task]),axis=1))
            value_image_loss=tf.reduce_mean(-tf.nn.softmax(self.value_label_dict[task]/self.config.temperature,axis=1)*tf.log(tf.nn.softmax(values_image_dict[task],axis=1)))
            losses_image_dict[task]=value_image_loss+policy_image_loss

            policy_ram_loss= tf.reduce_mean(-tf.reduce_sum(self.policy_label_dict[task]*tf.log(policies_ram_dict[task]),axis=1))
            value_ram_loss=tf.reduce_mean(-tf.nn.softmax(self.value_label_dict[task]/self.config.temperature,axis=1)*tf.log(tf.nn.softmax(values_ram_dict[task],axis=1)))
            losses_ram_dict[task]=value_ram_loss+policy_ram_loss

        return losses_image_dict,losses_ram_dict

    def add_training_op(self,losses_image_dict,losses_ram_dict):
        train_ops_image_dict={}
        train_ops_ram_dict = {}
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):

            learning_rate=tf.train.exponential_decay(learning_rate=self.config.lr,global_step=self.global_step,decay_steps=1000,decay_rate=0.8,staircase=True)
            for task in self.task_list:
                train_image_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=losses_image_dict[task],global_step=self.global_step)
                train_ops_image_dict[task]=train_image_op

                train_ram_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=losses_ram_dict[task], global_step=self.global_step)
                train_ops_ram_dict[task] = train_ram_op
        return train_ops_image_dict,train_ops_ram_dict

    def build(self):
        with tf.device('/device:GPU:0'):
            with tf.name_scope('inputs'):
                self.state, self.ram, self.value_label_dict,self.policy_label_dict = self.add_placeholders()
            with tf.variable_scope('network'):
                self.values_image_dict, self.policies_image_dict,self.values_ram_dict, self.policies_ram_dict = self.add_pred_op(self.state,self.ram)
            with tf.name_scope('add_loss'):
                self.losses_image_dict,self.losses_ram_dict=self.add_loss_op(self.values_image_dict, self.policies_image_dict,self.values_ram_dict, self.policies_ram_dict)
            with tf.name_scope('add_training'):
                self.train_ops_image_dict, self.train_ops_ram_dict = self.add_training_op(self.losses_image_dict,self.losses_ram_dict)

    def creat_feed_dict(self, mt_batch,is_training=True):
        feed_mt_dict={}
        if is_training == True:
            for task in self.task_list:
                feed_dict ={self.state:mt_batch[task]['frame_batch'],self.ram:mt_batch[task]['ram_batch'],
                            self.value_label_dict[task]:mt_batch[task]['value_batch']
                    ,self.policy_label_dict[task]:mt_batch[task]['policy_batch']}
                feed_mt_dict[task]=feed_dict
        return feed_mt_dict

    def train_on_batch(self,sess,mt_batch,is_training=True):
        feed_mt_dict = self.creat_feed_dict(mt_batch,is_training=is_training)
        loss_image_list=[]
        loss_ram_list=[]
        for task in self.task_list:
           _,_,loss_image,loss_ram= sess.run([self.train_ops_image_dict[task],self.train_ops_ram_dict[task],self.losses_image_dict[task],self.losses_ram_dict[task]],feed_mt_dict[task])
           loss_image_list.append(loss_image)
           loss_ram_list.append(loss_ram)
        return loss_image_list,loss_ram_list

    def get_image_pred(self,sess,task,state):
        value_image,policy_image = sess.run(self.values_image_dict[task],self.policies_image_dict[task], feed_dict={self.state:[state]})
        return value_image,policy_image

    def get_ram_pred(self, sess, task, ram):
        value_ram, policy_ram = sess.run(self.values_ram_dict[task],self.policies_ram_dict[task],feed_dict={self.ram: [ram]})
        return value_ram, policy_ram


#################################################################################
#Data Feeding
##################################################################################
# Using multiple processes to feed datas from all the tasks.
# The structure is from "https://hanxiao.github.io/2017/07/07/Get-10x-Speedup-in-Tensorflow-Multi-Task-Learning-using-Python-Multiprocessing/"

class single_batch_generator(Process):
    def __init__(self,task,single_task_queue,stop_event,seed_dict,seed_queue):
        super().__init__()
        self.stop_event=stop_event
        self.single_task_queue=single_task_queue
        self.task_list=task_list
        self.real_act_dim_dict=real_act_dim_dict
        self.config=config
        self.task=task
        self.load_data()
        self.seed=seed_dict[task]['seed']
        self.seed_queue=seed_queue

    def run(self):
        while (not self.stop_event.is_set()) and \
                (self.seed<int(math.ceil(self.config.train_iter*self.epoch_frame.shape[0]/self.config.batch_size))):
            if not self.single_task_queue.full():
                self.single_task_queue.put(self.next_batch(self.seed))
                self.seed+=1
        self.seed_queue.put({'task':self.task,'seed':self.seed})


    def load_data(self):
        # obtain data from "task_i.json"
        with open('data/Dagger_data_' + self.task + '.json', 'r') as f:
            raw_data = json.load(f)
        single_task_epoch = self.epoch_data(raw_data)
        self.epoch_ram, self.epoch_frame, self.epoch_value_label, self.epoch_policy_label \
            = self.data_shuffle(single_task_epoch)

    def next_batch(self,seed):
        offset = (seed*self.config.batch_size)% self.epoch_frame.shape[0]
        batch_frame=self.epoch_frame[offset:(offset+self.config.batch_size),:]
        batch_policy_label = self.epoch_policy_label[offset:(offset + self.config.batch_size), :]
        batch_value_label = self.epoch_value_label[offset:(offset + self.config.batch_size), :]
        batch={'job_seed':seed,'task':self.task,'frame_batch':batch_frame,
               'value_batch':batch_value_label,'policy_batch':batch_policy_label}
        return batch

    def data_shuffle(self,epoch):
        a=epoch['ram_epoch'].reshape((epoch['ram_epoch'].shape[0],-1))
        b=epoch['frame_epoch'].reshape((epoch['frame_epoch'].shape[0],-1))
        c=np.concatenate((a,b,epoch['value_epoch'],epoch['policy_epoch']),axis=1)#
        np.random.shuffle(c)
        X=c[:,:a.shape[1]]
        X=X.reshape(epoch['ram_epoch'])
        Y=c[:,a.shape[1]:a.shape[1]+b.shape[1]]
        Y=Y.reshape(epoch['frame_epoch'])
        y=c[:,a.shape[1]+b.shape[1]:a.shape[1]+b.shape[1]+epoch['value_epoch'].shape[1]]
        z=c[:,a.shape[1]+b.shape[1]+epoch['value_epoch'].shape[1]:]
        return X,Y,y,z

    def epoch_data(self,raw_data):
        # convert the list into array
        epoch={}
        for key in raw_data:
            if key=='task':
                epoch[key]=raw_data[key]
            else:
                epoch[key]=np.array(raw_data[key])
        return epoch

class multi_task_batch_generator(Process):
    def __int__(self,stop_event,single_task_queue,multi_task_queue):
        super().__init__()
        self.stop_event = stop_event
        self.multi_task_queue=multi_task_queue
        self.single_task_queue=single_task_queue
        self.task_list=task_list
        self.real_act_dim_dict=real_act_dim_dict
        self.mt_batch={}

    def is_complete(self,mt_batch):
        keys_list=[]
        for key in mt_batch.keys():
            keys_list.append(key)
        return len(keys_list)==len(self.task_list)

    def run(self):
        while not self.stop_event.is_set():
            if not self.multi_task_queue.full():
                st_batch=self.single_task_queue.get()
                job_task=st_batch['task']
                job_seed=st_batch['job_seed']
                self.mt_batch.setdefault(job_seed,{})[job_task]=st_batch
                if self.is_complete(self.mt_batch[job_seed]):
                    self.multi_task_queue.put(self.mt_batch.pop(job_seed))

class BatchManager:
     def __init__(self):

        self.task_list=task_list
        self.real_act_dim_dict=real_act_dim_dict
        self.seed_dict={}
        for task in task_list:
            self.seed_dict.setdefault(task,{})['seed']=0

     def start_processes(self):
         self.stop_event = Event()
         self.st_queue = Queue()
         self.mt_queue = Queue()
         self.seed_queue=Queue()
         self.st_batch_generator = {task: single_batch_generator(task, self.st_queue, self.stop_event,self.seed_dict,self.seed_queue) for task in
                                    self.task_list}
         for w in self.st_batch_generator.values():
             w.start()
         self.mt_batch_generator = multi_task_batch_generator(self.stop_event, self.st_queue, self.mt_queue,
                                                              self.real_act_dim_dict)
         self.mt_batch_generator.start()


     def get_mt_batch(self):
         return self.mt_queue.get(),self.mt_queue.empty()

     def close_processes(self):
         self.stop_event.set()
         for w in self.st_batch_generator:
             w.join()
             w.terminate()
         self.mt_batch_generator.join()
         self.mt_batch_generator.terminate()
         dict={}
         while not self.seed_queue.empty():
             dict_get=self.seed_queue.get()
             dict.setdefault(dict_get['task'],{})['seed']=dict_get['seed']
         self.seed_dict=dict




