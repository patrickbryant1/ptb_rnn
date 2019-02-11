#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''Predicting text using a RNN of LSTMs on the Penn Treebank Dataset.
'''

import pdb
import reader
import tensorflow as tf
import numpy as np
import random


#Get data - no labels
data_path = '/Users/patbry/Documents/Tensorflow/RNN/simple-examples/data'
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, vocabulary = raw_data

#Parameters
start_learning_rate = 0.001
decay_steps = 10000 #1 epoch = 1000000/(batch_size*num_unrollings) steps
decay_rate = 0.95

max_grad_norm = 5.0

number_of_layers = 3

num_unrollings = 20
batch_size = 20
vocab_size = 10000

forget_bias = 1.0
num_nodes = 200
embedding_size = 128 # Dimension of the embedding vector.
keep_prob = 0.5

num_steps = 50000

#Graph
graph = tf.Graph()

with graph.as_default():
    
    
    # Classifier weights and biases. Must be vocab size, otherwise all words will not be represented in logits
  softmax_w = tf.Variable(tf.truncated_normal([num_nodes, vocab_size], stddev=0.1))
  softmax_b = tf.Variable(tf.zeros([vocab_size]))
  
  	#Embedding vector
  train_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
  valid_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    
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

  #Keep prob
  keep_probability = tf.placeholder(tf.float32)

  

  
  def lstm_cell(keep_probability):
    cell = tf.contrib.rnn.BasicLSTMCell(num_nodes, forget_bias=forget_bias, activation=tf.nn.leaky_relu) 
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_probability) #Only applies dropout to output weights. Can also apply to state, input, forget. Not compatible with clipping gradients later.
    #tf.contrib.rnn.LSTMBlockCell(num_nodes, forget_bias=forget_bias)# tf.nn.rnn_cell.LSTMCell(num_nodes, forget_bias=forget_bias) #, use_peepholes=True)
  
  stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(keep_probability) for _ in range(number_of_layers)])

  #LSTM
  # Initial state of the LSTM memory.
  state = stacked_lstm.zero_state(batch_size, tf.float32) #Initial state. Return zero-filled state tensor(s).
  
  outputs = [] #Store outputs
    
  #Unrolled lstm loop  
  for i in range(num_unrollings):
    
    	# The value of state is updated after processing each batch of words.
    	# Look up embeddings for inputs.
      embed = tf.nn.embedding_lookup(train_embeddings, train_inputs[i])
      #embed = tf.nn.dropout(embed, keep_prob)
      #embed = (1/keep_prob)*embed
      output, state = stacked_lstm(embed, state)

      outputs.append(output)

  #Save final state for validation and testing
  final_state = state
  
  logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
  logits = tf.layers.batch_normalization(logits, training=True) #Batch normalize to avoid vanishing gradients
  logits = tf.nn.dropout(logits, keep_prob) #Dropout to reduce overfitting. Causes problem with norm later. Apply dropout btw RNN layers.
  logits = (1/keep_prob)*logits #Normalize by multiplying with 2. Then simply remove for validation.
  logits = tf.reshape(logits, [num_unrollings ,batch_size,vocab_size])    

  
  #Returns 1D batch-sized float Tensor: The log-perplexity for each sequence.
  #The labels are encoded using a unique int for each word in the vocabulary, but the proabilities
  #are one hot. This is why the train labels have to be one hot as well. 
  
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.concat(train_labels_hot, 0), logits=logits)
  
  train_perplexity = tf.math.exp(tf.reduce_mean(loss)) #Reduce mean is a very "strong" mean
  
  train_predictions = tf.argmax(tf.nn.softmax(logits), axis = -1)

  

  #Learning rate
  global_step = tf.Variable(0, trainable=False)
  
  learning_rate = tf.train.exponential_decay( #Lowers learning rate as training progresses. lr*dr^(global_step/decay_step)
  start_learning_rate,     # Base learning rate.
  global_step,             # Current batch index in the dataset.
  decay_steps,             # Decay step. How often to decay
  decay_rate,              # Decay rate.
  staircase=True)
  
  #Optimizer
  optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate) #.minimize(train_perplexity)
  save_gradients, variables = zip(*optimizer.compute_gradients(train_perplexity))
  gradients, _ = tf.clip_by_global_norm(save_gradients, max_grad_norm)
  optimize = optimizer.apply_gradients(zip(gradients, variables))

 #Validation
  
  valid_embed = tf.nn.embedding_lookup(valid_embeddings, valid_inputs)
  valid_output, state = stacked_lstm(valid_embed, final_state)

  #valid_output, state = tf.nn.dynamic_rnn(stacked_lstm, embed, final_state)
  valid_logits = tf.nn.xw_plus_b(valid_output, softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
  valid_logits = tf.layers.batch_normalization(valid_logits, training=False) #Batch normalize to avoid vanishing gradients
  valid_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=valid_hot, logits=valid_logits)
  valid_perplexity = tf.math.exp(tf.reduce_mean(valid_loss)) #Take mean of softmax probabilities and then the exp --> perplexity. Use e as base, as tf measures the cross-entropy loss with the natural logarithm.
  valid_predictions = tf.argmax(tf.nn.softmax(valid_logits), axis = -1)
 

  # add a summary to store the perplexity
  tf.summary.scalar('train_perplexity', train_perplexity)
  tf.summary.scalar('validation_perplexity', valid_perplexity)
  #tf.summary.histogram('gradients', save_gradients)
  merged = tf.summary.merge_all()
 
 #Evaluate
def get_words(predictions):
    """A function to get the predicted words and concatenate them into
    a string that can be printed.
    """
    
    pred_words = ''
    
    for i in predictions:       
        for key in vocabulary:
            if vocabulary[key] == i:
                pred_words += key + ' '
    
    return pred_words
            

#Graph
STORE_PATH = '/Users/patbry/Documents/Tensorflow/RNN/visual/run_8'



#Run model
#Store perplexities at each step
store_perplexities = []

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  writer = tf.summary.FileWriter(STORE_PATH, session.graph)
 
  for step in range(num_steps):
  #Get train data and labels
    train_feed_inputs = [] #Lists to store train inputs
    train_feed_labels = []
    
  #Now the structure for the train input is created. Now we have to feed the train_data
    train_start = random.randint(0,len(train_data)-21*batch_size)
    for i in range(num_unrollings):
    #Has to be sequential - otherwise how can it learn what should come after?
      
      train_feed_inputs.append(train_data[train_start:train_start+batch_size])
      train_feed_labels.append(train_data[train_start+batch_size:train_start+2*batch_size])

      train_start+=2*batch_size

    train_feed_inputs = np.array(train_feed_inputs)
    train_feed_labels = np.array(train_feed_labels)

    #Valid data and labels
    valid_start = random.randint(0,len(valid_data)-2*batch_size)
    valid_feed_inputs = np.array(list(valid_data[valid_start:valid_start+batch_size]))
    valid_feed_labels = np.array(list(valid_data[valid_start+batch_size:valid_start+(2*batch_size)]))

    #pdb.set_trace()
    #Feed dict
    if step%100 == 0:
      keep_prob = 1.0
    else:
      keep_prob = 0.5
    feed_dict= {train_inputs: train_feed_inputs, train_labels: train_feed_labels, valid_inputs: valid_feed_inputs, valid_labels: valid_feed_labels, global_step: step, keep_probability: keep_prob}

    _, t_perplexity, train_pred, v_perplexity, valid_pred, summary = session.run([optimize, train_perplexity, train_predictions, valid_perplexity, valid_predictions, merged], feed_dict= feed_dict)

    writer.add_summary(summary, step) 
    #Print preds
    if step%100 == 0:
      print('Train perplexity at step %d: %f' % (step, t_perplexity)) #, valid_perplexity.eval()
      print('Validation perplexity at step %d: %f' % (step, v_perplexity))

      print 'train_input: ' + get_words(train_feed_inputs[0])
      print 'train_pred: ' + get_words(train_pred[0]) 
          
      print 'valid_input: ' + get_words(valid_feed_inputs)
      print 'valid_pred: ' + get_words(valid_pred) + '\n'

    #Print labels for final validation run
  print 'valid_labels: ' + get_words(valid_feed_labels) + '\n'


  





      
      

