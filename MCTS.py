import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from task_list_dict import real_act_dim_dict,task_env
import time
import random
########################################################################################
# MCTS planning in Atari games
#TODO: Constraint the queue capacity
########################################################################################

class TreeNode:
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        # This value for u will be overwritten in the first call to update(), but is useful for
        # choosing the first action from this node.
        self._u = prior_p
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.

        Arguments:
        action_priors -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.

        Returns:
        None
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self,c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).

        Returns:
        A tuple of (action, next_node)
        """

        return max(self._children.items(), key=lambda act_node: act_node[1].get_value()
                    +c_puct*act_node[1]._P * np.sqrt(act_node[1]._parent._n_visits) / (1 + act_node[1]._n_visits))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.

        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.

        Returns:
        None
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += (leaf_value - self._Q) / self._n_visits
        # Update u, the prior weighted by an exploration hyperparameter c_puct and the number of
        # visits. Note that u is not normalized to be a distribution.


    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.

        Note: it is important that this happens from the root downward so that 'parent' visit
        counts are correct.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        """
        return self._Q

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class net:
    "Restore the trained value and policy networks to do the prediction in the MCTS"
    def __init__(self,task):
        self.task=task
        self.real_act_dim_dict=real_act_dim_dict
        self.build()


    def add_placeholder(self):
        ram = tf.placeholder(dtype='float', shape=[None, 128])
        return ram

    def add_pred_op(self, ram):
        with tf.variable_scope('network'):
            out = layers.fully_connected(ram, num_outputs=128, activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.layers.batch_normalization,
                                         normalizer_params={'training': False, 'reuse': False}, scope='fc1')
            out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.layers.batch_normalization,
                                         normalizer_params={'training': False, 'reuse': False}, scope='fc2')

            value = layers.fully_connected(out, num_outputs=self.real_act_dim_dict[self.task], activation_fn=None,
                                           normalizer_fn=None, scope="value_" + self.task)
            policy = tf.nn.softmax(layers.fully_connected(out, num_outputs=self.real_act_dim_dict[self.task],
                                            activation_fn=None,normalizer_fn=None, scope="policy_" + self.task))

        return value,policy

    def build(self):
        with tf.name_scope('placeholder_ram'):
            self.ram = self.add_placeholder()
        with tf.variable_scope('network_ram'):
            self.value, self.policy = self.add_pred_op(self.ram)

    def get_pred(self,sess,input_ram):
        value_pred,policy_pred=sess.run([self.value,self.policy],feed_dict={self.ram:[input_ram]})
        return value_pred,policy_pred


class MCTS:
    def __init__(self,task,rollout_limit, playout_depth, n_playout, temperature=1):
        # the 'state' is the ram.
        self.task=task
        self.temerature=temperature
        self.net=net(self.task)
        self.real_act_dim_dict=real_act_dim_dict
        self._c_puct = 5
        checkpoint = tf.train.get_checkpoint_state('saved_nn')
        if checkpoint:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network/ram_net'))
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.is_guide=True
        else:
            self.is_guide=False

        if self.is_guide:
            self._lmbda=0.5
        else:
            self._lmbda=1.0

        self.reset_root_node()

        self._rollout_limit = rollout_limit
        self._L = playout_depth
        self._n_playout = n_playout

    def _playout(self, ob, leaf_depth):
        """Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. env_copy is modified in-place, so a copy must be
        provided.

        Arguments:
        root_state -- copy of env : 'self.env.unwrapped.clone_state()'
        leaf_depth -- after this many moves, leaves are evaluated.

        Returns:
        None
        """
        node = self._root
        state=ob['ram']

        for i in range(leaf_depth):
        # Only expand node if it has not already been done. Existing nodes already know their prior.
        #     print('leaf%i'%i)
            if node.is_leaf():
                if self.is_guide:
                    _,action_probs = self.net.get_pred(self.sess, state)
                else:
                    action_probs = np.ones([1,self.real_act_dim_dict[self.task]])#/self.real_act_dim_dict[self.task]
                action_probs_dict = zip(np.arange(self.real_act_dim_dict[self.task]).tolist(),
                                        action_probs.reshape((self.real_act_dim_dict[self.task])).tolist())
                node.expand(action_probs_dict)
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            ob,reward,terminal,info=self.env.step(int(action))
            state=ob['ram']
            if terminal:
                break

        # Evaluate the leaf using a weighted combination of the value network, v, and the game's
        # winner, z, according to the rollout policy. If lmbda is equal to 0 or 1, only one of
        # these contributes and the other may be skipped.
        if self._lmbda < 1:
            value_vec, action_vec = self.net.get_pred(self.sess, state)
            value = np.sum(value_vec * action_vec)
        else:
            value = 0

        # Use the random policy to play until the end of the game, returning the
        # accumulated future reward.
        z=0

        for i in range(self._rollout_limit):
            action = random.randrange(self.env.action_len)#self.env.action_space.sample()
            _, reward, terminal1,info1 = self.env.step(action)
            z += np.clip(reward,-1,1)
            if terminal1:
                break

        else:
            #If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")

        leaf_value = (1 - self._lmbda) * value + self._lmbda * z
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def get_move(self, state, ob):
        """Runs all playouts sequentially and returns the most visited action.

        Arguments:
        env -- The envrionment this class is working on.

        Returns:
        the selected action
        """
        # import gym
        # self.env=gym.make(self.task)
        from Atari import Atari
        self.env=Atari('qbert')
        self.env.reset()

        for n in range(self._n_playout):
            self.env.restore_state(state)
            self._playout(ob, self._L)

        # chosen action is the *most visited child*, not the highest-value one
        # (they are the same as self._n_playout gets large).
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def get_value(self):
        """Get the value of the root node after the action is achieved

        Return:
        value -- the value the root
        """
        return self._root._Q

    def get_policy(self):
        tot=0
        policy=np.zeros((1,self.real_act_dim_dict[self.task]))
        for act_node in self._root._children.items():
            policy[:,int(act_node[0])]=act_node[1]._n_visits
            tot+=act_node[1]._n_visits
        return policy/tot

    def get_value_vec(self):
        value_vec=np.zeros((1,self.real_act_dim_dict[self.task]))
        for act_node in self._root._children.items():
            value_vec[:,int(act_node[0])]=act_node[1].get_value()
        return value_vec

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def reset_root_node(self):
        self._root=TreeNode(None,1.0)


class MCTSPlayer:
    """
    This class is used to interact with the environment.
    """
    def __init__(self, task, rollout_limit=10000, playout_depth=20, n_playout=300):
        self.mcts = MCTS(task, rollout_limit, playout_depth, n_playout)
    def get_action(self, state,ob):
        """
        Arguments:
        env -- environment
        Return:
        action -- the action from MCTS planning
        """
        action = self.mcts.get_move(state,ob)
        return action
    def update_root(self,action):
        self.mcts.update_with_move(action)
    def get_value_vec(self):
        return self.mcts.get_value_vec()
    def get_policy(self):
        return self.mcts.get_policy()
    def reset_root(self):
        self.mcts.reset_root_node()






