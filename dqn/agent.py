import tensorflow as tf 
import numpy as np 

class Model:
    def __init__(self, scope, num_actions):
        self.scope = scope
        self.num_actions = num_actions

    def forward(self, state):
        X = tf.contrib.layers.fullyconnected(state, 32, activation_fn = relu, scope = self.scope)
        out = tf.contrib.layers.fullyconnected(X, self.num_actions, activation_fn = None, scope = self.scope)
        return out

def copy_var(sess, current, target):
    '''
    Copies the parameters of the current network to the target network
    '''
    col1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope = current)
    col2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope = target)

    col_curr = {}
    for var in col1:
        var_name = var.name.split('/')[-1]
        col_curr[var_name] = var

    for var in col2:
        var_name = var.name.split('/')[-1]
        sess.run(tf.assign(col_curr[var_name], var))

def best_action(env, q_state, steps, eps_begin = 1.0, eps_end = 0.25, num_steps = 100):
    eps = (eps_begin - eps_end) * steps/num_steps
    p = random.random()
    if p < eps:
        return env.action_space.sample()
    else:
        return tf.max(q_state, axis = 1)
def compute_loss(batch, current, target, gamma):
    pass
    return loss_op

def eval(batch):
    pass
    return eval_op
