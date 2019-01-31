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



#Get data - no labels
data_path = '/Users/patbry/Documents/Tensorflow/RNN/simple-examples/data'
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, vocabulary = raw_data

global train_data
global valid_data
global test_data

ray.init()


def MyTrainable(config, reporter):
    # Setup your tensorflow model
    # Hyperparameters for this trial can be accessed in dictionary config
    params = config

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


    graph = tf.Graph()
    
    with graph.as_default():
    
        # Classifier weights and biases. Must be vocab size, otherwise all words will not be represented in logits
      softmax_w = tf.Variable(tf.truncated_normal([num_nodes, vocab_size], stddev=softmax_dev))
      softmax_b = tf.Variable(tf.zeros([vocab_size]))
      
      	#Embedding vector
      train_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -embedding_dev, embedding_dev))
      valid_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -embedding_dev, embedding_dev))
        
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
    
        # Input data. Create sturcutre for input data
      #Train data
      train_inputs = tf.placeholder(tf.int32, shape=[num_unrollings, batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[num_unrollings, batch_size])  
      train_labels_hot = tf.one_hot(train_labels, 10000)  
    
      #Valid data
      valid_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      valid_labels = tf.placeholder(tf.int32, shape=[batch_size])
      valid_hot = tf.one_hot(valid_labels, 10000)
    
      
    
      
      def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_nodes, forget_bias=forget_bias) #tf.contrib.rnn.LSTMBlockCell(num_nodes, forget_bias=forget_bias)# tf.nn.rnn_cell.LSTMCell(num_nodes, forget_bias=forget_bias) #, use_peepholes=True)
      stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)])
    
    
      #LSTM
      # Initial state of the LSTM memory.
      state = stacked_lstm.zero_state(batch_size, tf.float32) #Initial state. Return zero-filled state tensor(s).
      
      outputs = [] #Store outputs
        
      #Unrolled lstm loop  
      for i in range(num_unrollings):
          
          embed = tf.nn.embedding_lookup(train_embeddings, train_inputs[i])
        	# The value of state is updated after processing each batch of words.
        	# Look up embeddings for inputs.
          output, state = stacked_lstm(embed, state)
    
          outputs.append(output)
    
      #Save final state for validation and testing
      final_state = state
      
      logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
      logits = tf.nn.dropout(logits, keep_prob) #Dropout to reduce overfitting - still overfits AF though
      #logits = (1/keep_prob)*logits #Normalize by multiplying with 2. Then simply remove for validation.
      logits = tf.reshape(logits, [num_unrollings ,batch_size,vocab_size])    
    
      #Returns 1D batch-sized float Tensor: The log-perplexity for each sequence.
      #The labels are encoded using a unique int for each word in the vocabulary, but the proabilities
      #are one hot. This is why the train labels have to be one hot as well. 
      
      loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.concat(train_labels_hot, 0), logits=logits)
      
      train_perplexity = tf.math.exp(tf.reduce_mean(loss)) #Reduce mean is a very "strong" mean
      
      train_predictions = tf.argmax(tf.nn.softmax(logits), axis = -1)
    
      
    
      #Learning rate
      global_step = tf.Variable(0, trainable=False)
      
      learning_rate = tf.train.exponential_decay(
      start_learning_rate,     # Base learning rate.
      global_step,             # Current index into the dataset.
      decay_steps,             # Decay step. How often to decay
      decay_rate,              # Decay rate.
      staircase=True)
      
      #Optimizer
      optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_perplexity)
     
     #Validation
      
      valid_embed = tf.nn.embedding_lookup(valid_embeddings, valid_inputs)
      valid_output, state = stacked_lstm(valid_embed, final_state)
    
      #valid_output, state = tf.nn.dynamic_rnn(stacked_lstm, embed, final_state)
      valid_logits = tf.nn.xw_plus_b(valid_output, softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
      valid_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=valid_hot, logits=valid_logits)
      valid_perplexity = tf.math.exp(tf.reduce_mean(valid_loss)) #Take mean of softmax probabilities and then the exp --> perplexity. Use e as base, as tf measures the cross-entropy loss with the natural logarithm.
      valid_predictions = tf.argmax(tf.nn.softmax(valid_logits), axis = -1)



    #saver = tf.train.Saver()
    # Start a tensorflow session
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        for step in range(1000):
            #Get train data and labels
            train_feed_inputs = [] #Lists to store train inputs
            train_feed_labels = []

            #Now the structure for the train input is created. Now we have to feed the train_data
            for i in range(params['num_unrollings']):
            #Has to be sequential - otherwise how can it learn what should come after?
              train_start = random.randint(0,len(train_data)-batch_size)
              train_feed_inputs.append(train_data[train_start:train_start+batch_size])
              train_feed_labels.append(train_data[train_start+batch_size:train_start+2*batch_size])
              

            train_feed_inputs = np.array(train_feed_inputs)
            train_feed_labels = np.array(train_feed_labels)

            #Valid data and labels
            valid_start = random.randint(0,len(valid_data)-2*batch_size)
            valid_feed_inputs = np.array(list(valid_data[valid_start:valid_start+batch_size]))
            valid_feed_labels = np.array(list(valid_data[valid_start+batch_size:valid_start+(2*batch_size)]))


            #Feed dict
            feed_dict= {train_inputs: train_feed_inputs, train_labels: train_feed_labels, valid_inputs: valid_feed_inputs, valid_labels: valid_feed_labels }

            _, t_perplexity, train_pred, v_perplexity, valid_pred = session.run(
              [optimizer, train_perplexity, train_predictions, valid_perplexity, valid_predictions], feed_dict= feed_dict)

            reporter(timesteps_total=step, mean_loss=v_perplexity)
        
    	   

tune.register_trainable('MyTrainable', MyTrainable)

train_spec = {
  'run': MyTrainable,
  # Specify the number of CPU cores and GPUs each trial requires
  'trial_resources': {'cpu': 2},
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


pbt = PopulationBasedTraining(
  time_attr='training_iteration',
  reward_attr='mean_loss',
  perturbation_interval=1,
  hyperparam_mutations={
    'keep_prob': [0.3, 0.4, 0.5],
    'start_learning_rate': [1e-2, 1e-3, 1e-4]
  }
)
tune.run_experiments({'population_based_training': train_spec}, scheduler=pbt)


