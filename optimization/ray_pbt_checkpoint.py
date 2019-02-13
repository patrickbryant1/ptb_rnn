#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:25:58 2019

@author: patbry
"""

import ray
import ray.tune as tune
import reader
import numpy as np
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf
import pdb
import random


global train_data
global valid_data
global test_data

ray.init()

#Get data - no labels
data_path = '/Users/patbry/Documents/Tensorflow/RNN/simple-examples/data'
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, vocabulary = raw_data



def get_data(num_unrollings, batch_size):

  train_feed_inputs = []
  train_feed_labels = []
  for i in range(num_unrollings):
    #Has to be sequential - otherwise how can it learn what should come after?
    train_start = random.randint(0,len(train_data)-2*batch_size)
    train_feed_inputs.append(train_data[train_start:train_start+batch_size])
    train_feed_labels.append(train_data[train_start+batch_size:train_start+2*batch_size])
              

  train_feed_inputs = np.array(train_feed_inputs)
  train_feed_labels = np.array(train_feed_labels)

  #Valid data and labels
  valid_start = random.randint(0,len(valid_data)-2*batch_size)
  valid_feed_inputs = np.array(list(valid_data[valid_start:valid_start+batch_size]))
  valid_feed_labels = np.array(list(valid_data[valid_start+batch_size:valid_start+(2*batch_size)]))

  #print len(train_feed_inputs), len(train_feed_labels), len(valid_feed_inputs), len(valid_feed_labels)

  return train_feed_inputs, train_feed_labels, valid_feed_inputs, valid_feed_labels


class Model:

  def __init__(self, params):

    self.params = params
      #Parameters
    start_learning_rate = params['start_learning_rate']
    decay_steps = params['decay_steps']
    decay_rate = params['decay_rate']
    number_of_layers = params['number_of_layers']
    num_unrollings = params['num_unrollings']
    batch_size = params['batch_size']
    vocab_size = params['vocab_size']
    forget_bias = params['forget_bias']
    num_nodes = params['num_nodes']
    embedding_size = params['embedding_size'] # Dimension of the embedding vector.
    keep_prob = params['keep_prob']
    softmax_dev = params['softmax_dev']
    embedding_dev = params['embedding_dev']
  #Inputs
     #Train data
    self.train_inputs = tf.placeholder(tf.int32, shape=[num_unrollings, batch_size])
    self.train_labels = tf.placeholder(tf.int32, shape=[num_unrollings, batch_size])  
    train_labels_hot = tf.one_hot(self.train_labels, vocab_size)  

    #Valid data
    self.valid_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    self.valid_labels = tf.placeholder(tf.int32, shape=[batch_size])
    valid_hot = tf.one_hot(self.valid_labels, vocab_size)


    # Classifier weights and biases. Must be vocab size, otherwise all words will not be represented in logits
    softmax_w = tf.Variable(tf.truncated_normal([num_nodes, vocab_size], stddev=softmax_dev))
    softmax_b = tf.Variable(tf.zeros([vocab_size]))
  
    #Embedding vector
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -embedding_dev, embedding_dev))

    #Get LSTM network
    stacked_lstm = self._create_network(forget_bias, number_of_layers, num_nodes)
    state = stacked_lstm.zero_state(batch_size, tf.float32)

    #Unroll lstm (for training)
    outputs, state = self._unroll_loop(num_unrollings, embeddings, self.train_inputs, stacked_lstm, state)
    final_state = state
    #Calculate train perplexity
    train_perplexity, logits = self._train_perplexity(outputs, softmax_w, softmax_b, train_labels_hot, keep_prob, num_unrollings, batch_size, vocab_size)
    

    #Train predictions
    self.train_predictions = tf.argmax(tf.nn.softmax(logits), axis = -1)

    #Learning rate
    global_step = tf.Variable(0, trainable=False)
  
    learning_rate = tf.train.exponential_decay(
    start_learning_rate,     # Base learning rate.
    global_step,             # Current index into the dataset.
    decay_steps,             # Decay step. How often to decay
    decay_rate,              # Decay rate.
    staircase=True)

    #Optimize
    self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_perplexity)

    #Validate
    valid_perplexity, valid_predictions = self._validate(self.valid_inputs, valid_hot, embeddings, stacked_lstm, final_state, softmax_w, softmax_b)

    self.valid_perplexity = valid_perplexity
    self.valid_predictions = valid_predictions

  def _create_network(self, forget_bias, number_of_layers, num_nodes):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(num_nodes, forget_bias=1.0) #tf.contrib.rnn.LSTMBlockCell(num_nodes, forget_bias=forget_bias)# tf.nn.rnn_cell.LSTMCell(num_nodes, forget_bias=forget_bias) #, use_peepholes=True)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)])

    return stacked_lstm

  def _unroll_loop(self, num_unrollings, embeddings, train_inputs, stacked_lstm, state):

    outputs = [] #Store outputs
    for i in range(num_unrollings):
      embed = tf.nn.embedding_lookup(embeddings, train_inputs[i])
      # The value of state is updated after processing each batch of words.
      # Look up embeddings for inputs.
      output, state = stacked_lstm(embed, state)

      outputs.append(output)

    return outputs, state

  def _train_perplexity(self, outputs, softmax_w, softmax_b, train_labels_hot, keep_prob, num_unrollings, batch_size, vocab_size):#, outputs, softmax_w, softmax_b, train_labels_hot):

    logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
    logits = tf.nn.dropout(logits, keep_prob) #Dropout to reduce overfitting - still overfits AF though
    logits = (1/keep_prob)*logits #Normalize by multiplying with 2. Then simply remove for validation.
    logits = tf.reshape(logits, [num_unrollings ,batch_size,vocab_size])    


    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.concat(train_labels_hot, 0), logits=logits)
  
    train_perplexity = tf.math.exp(tf.reduce_mean(loss)) #Reduce mean is a very "strong" mean
  
    return train_perplexity, logits

  def _validate(self, valid_inputs, valid_hot, embeddings, stacked_lstm, final_state, softmax_w, softmax_b):

    valid_embed = tf.nn.embedding_lookup(embeddings, valid_inputs)
    valid_output, state = stacked_lstm(valid_embed, final_state)
    valid_logits = tf.nn.xw_plus_b(valid_output, softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
    valid_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=valid_hot, logits=valid_logits)
    valid_perplexity = tf.math.exp(tf.reduce_mean(valid_loss)) #Take mean of softmax probabilities and then the exp --> perplexity. Use e as base, as tf measures the cross-entropy loss with the natural logarithm.
    valid_predictions = tf.argmax(tf.nn.softmax(valid_logits), axis = -1)
 
    return valid_perplexity, valid_predictions


class MyTrainable(tune.Trainable):

  def _setup(self, config):
    # Setup your tensorflow model
    # Hyperparameters for this trial can be accessed in dictionary self.config
    self.model = Model(self.config)
    # To save and restore your model
    self.saver = tf.train.Saver()
    # Start a tensorflow session
    self.sess = tf.Session()
    #Initialize variables
    self.sess.run(tf.global_variables_initializer())


  def _train(self):
    # Run your training op for n iterations
    num_unrollings = self.config['num_unrollings']
    batch_size = self.config['batch_size']
    #for step in range(10001):
      # Load your data
      
    train_feed_inputs, train_feed_labels, valid_feed_inputs, valid_feed_labels = get_data(num_unrollings, batch_size) #Must generate new random data for each step - a bottleneck
    feed_dict= {self.model.train_inputs: train_feed_inputs, self.model.train_labels: train_feed_labels, self.model.valid_inputs: valid_feed_inputs, self.model.valid_labels: valid_feed_labels }
    self.sess.run(self.model.optimize, feed_dict = feed_dict)

    # Report a performance metric to be used in your hyperparameter search
    v_perplexity = self.sess.run(self.model.valid_perplexity, feed_dict = feed_dict)
    return {"mean_loss":v_perplexity} #tune.TrainingResult(timesteps_this_iter=n, mean_loss=validation_loss)

  def _stop(self):
    self.sess.close()

  # This function will be called if a population member
  # is good enough to be exploited
  def _save(self, checkpoint_dir):
    path = checkpoint_dir + '/save'
    return self.saver.save(self.sess, path, global_step=self._timesteps_total)

  # Population members that perform very well will be
  # exploited (restored) from their checkpoint
  def _restore(self, checkpoint_path):
    return self.saver.restore(self.sess, checkpoint_path)




#Register trainable function
#tune.register_trainable('MyTrainable', MyTrainable)


#Specify parameters for training
train_spec = {
  'run': MyTrainable,
  # Specify the number of CPU cores and GPUs each trial requires
  'trial_resources': {'cpu': 5},
  'stop': {'timesteps_total': 10000},
  # All your hyperparameters (variable and static ones)
  'config': {
    'start_learning_rate': tune.grid_search([1e-3, 1e-4]),
    'decay_steps' : 1000,
    'decay_rate' : 0.95,
    'number_of_layers' : 3,
    'num_unrollings' : 20, 
    'batch_size' : 20,
    'vocab_size' : 10000,
    'forget_bias' : 1.0,
    'num_nodes': tune.grid_search([200, 500]),
    'embedding_size' : 128,
    'keep_prob': tune.grid_search([0.3, 0.5]),
    'softmax_dev' : 0.1,
    'embedding_dev' : 1.0
  },
  # Number of trials
  'num_samples': 4
}

#Population based training
pbt = PopulationBasedTraining(
  time_attr='training_iteration',
  reward_attr='mean_loss',
  perturbation_interval=1000,
  resample_probability=0.25,
  hyperparam_mutations={
    'keep_prob': [0.3, 0.4, 0.5],
    'start_learning_rate': [1e-2, 1e-3, 1e-4]
  }
)

#Run the experiment. Follow progress by: tensorboard --logdir ~/ray_results/population_based_training
tune.run_experiments({'population_based_training': train_spec}, scheduler=pbt)

