#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pdb
import reader
import tensorflow as tf
import numpy as np
import random
import plot_training

#Get data - no labels
data_path = '/Users/patbry/Documents/Tensorflow/RNN/simple-examples/data'
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, vocabulary = raw_data

#Parameters
start_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.95

number_of_layers = 3

num_unrollings = 20
batch_size = 20
vocab_size = 10000

forget_bias = 1.0
num_nodes = 200
embedding_size = 128 # Dimension of the embedding vector.
keep_prob = 0.5

num_epochs = 14
#Graph
graph = tf.Graph()

with graph.as_default():
    
    
    # Classifier weights and biases. Must be vocab size, otherwise all words will not be represented in logits
  softmax_w = tf.Variable(tf.truncated_normal([num_nodes, vocab_size], -0.1, 0.1))
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

  

  
  def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(num_nodes, forget_bias=1.0) #tf.contrib.rnn.LSTMBlockCell(num_nodes, forget_bias=forget_bias)# tf.nn.rnn_cell.LSTMCell(num_nodes, forget_bias=forget_bias) #, use_peepholes=True)
  stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)])


  #LSTM
  #lstm = tf.contrib.rnn.BasicLSTMCell(num_nodes, forget_bias=1.0)
  # Initial state of the LSTM memory.
  state = stacked_lstm.zero_state(batch_size, tf.float32) #Initial state. Return zero-filled state tensor(s).
 


  
  outputs = []
    
  #Unrolled lstm loop  
  for i in range(num_unrollings):
      
      embed = tf.nn.embedding_lookup(train_embeddings, train_inputs[i])
    	# The value of state is updated after processing each batch of words.
    	# Look up embeddings for inputs.
      output, state = stacked_lstm(embed, state)

      #output, state = tf.nn.dynamic_rnn(stacked_lstm, embed, state)

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
  
  loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.concat(train_labels_hot, 0), logits=logits)
  
  train_perplexity = tf.math.exp(tf.reduce_mean(loss)) #Reduce mean is a very "strong" mean
  
  train_predictions = tf.argmax(tf.nn.softmax(logits), axis = -1)
  

  
  #targets = tf.to_float(tf.concat(train_labels, 0))
  #loss = tf.contrib.legacy_seq2seq.sequence_loss(logits=logits, targets = targets, weights = tf.ones([batch_size, num_unrollings, 1], dtype=tf.float32))
  #total_loss += loss

  global_step = tf.Variable(0, trainable=False)
  
  learning_rate = tf.train.exponential_decay(
  start_learning_rate,     # Base learning rate.
  global_step,             # Current index into the dataset.
  decay_steps,             # Decay step. How often to decay
  decay_rate,              # Decay rate.
  staircase=True)
  
  #Optimizer
  optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_perplexity)

  
  




#Can also stack
#def lstm_cell():
 # return tf.contrib.rnn.BasicLSTMCell(lstm_size)
#stacked_lstm = tf.contrib.rnn.MultiRNNCell(
 #   [lstm_cell() for _ in range(number_of_layers)])
 
 #Validation
  
  valid_embed = tf.nn.embedding_lookup(valid_embeddings, valid_inputs)
  valid_output, state = stacked_lstm(valid_embed, final_state)

  #valid_output, state = tf.nn.dynamic_rnn(stacked_lstm, embed, final_state)
  valid_logits = tf.nn.xw_plus_b(valid_output, softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
  valid_loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=valid_hot, logits=valid_logits)
  valid_perplexity = tf.math.exp(tf.reduce_mean(valid_loss)) #Take mean of softmax probabilities and then the exp --> perplexity. Use e as base, as tf measures the cross-entropy loss with the natural logarithm.
  valid_predictions = tf.argmax(tf.nn.softmax(valid_logits), axis = -1)
 

  # add a summary to store the perplexity
  tf.summary.scalar('train_perplexity', train_perplexity)
  tf.summary.scalar('validation_perplexity', valid_perplexity)
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
STORE_PATH = '/Users/patbry/Documents/Tensorflow/RNN'



#Run model
#Store perplexities at each step
store_perplexities = []
epoch = len(train_data)/(num_unrollings*batch_size)
num_steps = epoch*num_epochs #They use max_max_epoch = 55, which means they will go through all data 55 times --> 55*len(train_data)/(num_unrollings*batch_size) steps
num_steps = (num_steps/100)*100 #Get congruent with 100

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  writer = tf.summary.FileWriter(STORE_PATH, session.graph)
 
  for step in range(num_steps):
  #Get train data and labels
    train_feed_inputs = [] #Lists to store train inputs
    train_feed_labels = []
    
  #Now the structure for the train input is created. Now we have to feed the train_data
    for i in range(num_unrollings):
    #Has to be sequential - otherwise how can it learn what should come after?
      train_start = random.randint(0,vocab_size-batch_size)
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

    _, t_perplexity, train_pred, v_perplexity, valid_pred, summary = session.run(
      [optimizer, train_perplexity, train_predictions, valid_perplexity, valid_predictions, merged], feed_dict= feed_dict)

    writer.add_summary(summary, step)
    #Print preds
    if step%100 == 0:
      print('Train perplexity at step %d: %f' % (step, t_perplexity)) #, valid_perplexity.eval()
      print('Validation perplexity at step %d: %f' % (step, v_perplexity))

      print 'train_input: ' + get_words(train_feed_inputs[0])
      print 'train_pred: ' + get_words(train_pred[0]) 
          
      print 'valid_input: ' + get_words(valid_feed_inputs)
      print 'valid_pred: ' + get_words(valid_pred) + '\n'

      store_perplexities.append(str(step) + '/' + str(t_perplexity) + '/' + str(v_perplexity))

    #Print labels for final validation run
  print 'valid_labels: ' + get_words(valid_feed_labels) + '\n'
  plot_training.plot_training(store_perplexities)

  





      
      

