#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pdb
import reader
import tensorflow as tf
import numpy as np




#Get data - no labels
data_path = '/Users/patbry/Documents/Tensorflow/RNN/simple-examples/data'
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, vocabulary = raw_data

#Parameters
number_of_layers = 3
num_unrollings = 40
batch_size = 20
vocab_size = 10000

forget_bias = 0.0 #Bias for LSTMs forget gate, reduce forgetting in beginning of training
num_nodes = 300
embedding_size = num_nodes # Dimension of the embedding vector. 1:1 ratio with hidden nodes
init_scale = 0.1
epoch_length = len(train_data)/(num_unrollings*batch_size)

start_learning_rate = 5.0
decay_steps = epoch_length 
decay_rate = 0.5
max_grad_norm = 0.25

keep_prob = 0.9
num_steps = 50000
epsilon = 0.0000001
penalty = 1.3



#Graph
graph = tf.Graph()

with graph.as_default():
    
    
    # Classifier weights and biases. Must be vocab size, otherwise all words will not be represented in logits
  softmax_w = tf.Variable(tf.random_uniform(shape = [num_nodes, vocab_size], minval = -init_scale, maxval = init_scale, name = 'softmax_w'))
  softmax_b = tf.Variable(tf.zeros([vocab_size]), name = 'softmax_b')
  
    #Embedding vector
  embeddings = tf.Variable(tf.random_uniform(shape = [vocab_size, embedding_size], minval = -init_scale, maxval = init_scale), name = 'embeddings')
    
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

  #Embed scaling
  embed_scaling = tf.constant(1/(1-keep_prob))

  

  #Define the LSTM cell erchitecture to be used  
  def lstm_cell(keep_probability):
    cell = tf.contrib.rnn.BasicLSTMCell(num_nodes, forget_bias=forget_bias, activation = tf.nn.elu)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_probability, variational_recurrent = True, dtype = tf.float32) #Only applies dropout to output weights. Can also apply to state, input, forget.
    #The variational_recurrent = True applies the same dropout mask in every step - allowing more long-term dependencies to be learned   
  
  stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(keep_probability) for _ in range(number_of_layers)])

  #LSTM
  # Initial state of the LSTM memory.
  initial_state = stacked_lstm.zero_state(batch_size, tf.float32) #Initial state. Return zero-filled state tensor(s).
  state = initial_state
  outputs = [] #Store outputs
    
  #Unrolled lstm loop  
  for i in range(num_unrollings):
    
      # The value of state is updated after processing each batch of words.
      # Look up embeddings for inputs.
      embed = tf.nn.embedding_lookup(embeddings, train_inputs[i])
      Embed dropout and scaling
      embed = tf.nn.dropout(embed, keep_prob = keep_probability)
      embed = tf.math.scalar_mul(embed_scaling, embed)
      #Output, state of  LSTM
      output, state = stacked_lstm(embed, state)

      outputs.append(output)

  #Save final state for validation and testing
  final_state = state

  logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
  logits = tf.layers.batch_normalization(logits, training=True) #Batch normalize to avoid vanishing gradients
  logits = tf.reshape(logits, [num_unrollings ,batch_size,vocab_size])   


  
  #Returns 1D batch-sized float Tensor: The log-perplexity for each sequence.
  #The labels are encoded using a unique int for each word in the vocabulary, but the proabilities
  #are one hot. This is why the train labels have to be one hot as well. 
  
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.concat(train_labels_hot, 0), logits=logits) 
  #Linearly constrained weights to reduce angle bias
  true_train_perplexity = tf.math.exp(tf.reduce_mean(loss)) #Train perplexity before addition of penalty
  loss = tf.cond(tf.abs(tf.reduce_sum(softmax_w)) > epsilon, lambda:tf.multiply(penalty, loss), lambda:tf.add(loss, 0)) #condition, TRUE, FALSE
  train_perplexity = tf.math.exp(tf.reduce_mean(loss)) #Reduce mean is a very "strong" mean
  train_predictions = tf.argmax(tf.nn.softmax(logits), axis = -1)

  

  #Learning rate
  global_step = tf.Variable(0, trainable=False)
  decay_rate = tf.train.exponential_decay( #Lowers learning rate as training progresses. lr*dr^(global_step/decay_step)
  start_learning_rate,     # Base learning rate.
  global_step,             # Current batch index in the dataset.
  decay_steps,             # Decay step. How often to decay
  decay_rate,              # Decay rate.
  staircase=True)
  
  lr=tf.get_variable('lr',initializer=start_learning_rate,trainable=False)
  learning_rate_decay=lr.assign(decay_rate) #Run when decay is to start


  
  
  #Optimizer
  optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr) #.minimize(train_perplexity) #regular SGD has been found to outpeform adaptive
  gradients, variables = zip(*optimizer.compute_gradients(train_perplexity))
  gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
  optimize = optimizer.apply_gradients(zip(gradients, variables))

 #Validation
  
  valid_embed = tf.nn.embedding_lookup(embeddings, valid_inputs)
  valid_output, state = stacked_lstm(valid_embed, final_state)

  valid_logits = tf.nn.xw_plus_b(valid_output, softmax_w, softmax_b) #Computes matmul, need to have this tf concat, any other and it complains
  valid_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=valid_hot, logits=valid_logits)
  valid_perplexity = tf.math.exp(tf.reduce_mean(valid_loss)) #Take mean of softmax probabilities and then the exp --> perplexity. Use e as base, as tf measures the cross-entropy loss with the natural logarithm.
  valid_predictions = tf.argmax(tf.nn.softmax(valid_logits), axis = -1)
 

  #Get variable names
  variables_names =[v.name for v in tf.trainable_variables()]
  # add a summary to store the perplexity
  tf.summary.scalar('train_perplexity', train_perplexity)
  tf.summary.scalar('true_train_perplexity', true_train_perplexity)
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
            

#Visualizing output
STORE_PATH = '/Users/patbry/Documents/Tensorflow/outputs/visual/run_1'
from tensorboard_logging import Logger
logger = Logger(STORE_PATH+'/activations')



#Run model
#Store perplexities at each step
store_perplexities = []
ti = 0 #Train index
vi = 0 #Valid index
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  writer = tf.summary.FileWriter(STORE_PATH, session.graph)
  #Assign start learning rate
  learning_rate = start_learning_rate

  for step in range(num_steps):
  #Get train data and labels
    train_feed_inputs = [] #Lists to store train inputs
    train_feed_labels = []
    
    if ti >= (len(train_data)-(batch_size*(num_unrollings*2))): #Reset
      ti = 0

  #Now the structure for the train input is created. Now we have to feed the train_data
    for i in range(num_unrollings):
    #Has to be sequential - otherwise how can it learn what should come after?
      train_feed_inputs.append(train_data[ti:ti+batch_size])
      train_feed_labels.append(train_data[ti+batch_size:ti+2*batch_size])
      ti += (2*batch_size)
    
    train_feed_inputs = np.array(train_feed_inputs)
    train_feed_labels = np.array(train_feed_labels)

    #Valid data and labels
    if vi >= (len(valid_data)-(3*batch_size)): #Reset
      vi = 0

    valid_feed_inputs = np.array(list(valid_data[vi:vi+batch_size]))
    valid_feed_labels = np.array(list(valid_data[vi+batch_size:vi+(2*batch_size)]))
   
    vi += (2*batch_size)



    #Feed dict
    feed_dict= {train_inputs: train_feed_inputs, train_labels: train_feed_labels, valid_inputs: valid_feed_inputs, valid_labels: valid_feed_labels, global_step: step, keep_probability: keep_prob}

    if step < 14*epoch_length: 
      _, t_perplexity, train_pred, summary = session.run([optimize, train_perplexity, train_predictions, merged], feed_dict= feed_dict)
    else:
      _, t_perplexity, train_pred, summary, _ = session.run([optimize, train_perplexity, train_predictions, merged, learning_rate_decay], feed_dict= feed_dict)

    #Write monitored parameters to disk for visualization in tensorboard
    writer.add_summary(summary, step) 


    #Print preds
    if step%100 == 0:
      feed_dict= {train_inputs: train_feed_inputs, train_labels: train_feed_labels, valid_inputs: valid_feed_inputs, valid_labels: valid_feed_labels, global_step: step, keep_probability: 1.0}

      v_perplexity, valid_pred, summary = session.run([valid_perplexity, valid_predictions, merged], feed_dict= feed_dict)
      

      #Get all trainable variables
      variables_names =[v.name for v in tf.trainable_variables()]
      values = session.run(variables_names, feed_dict= feed_dict)
      #Write hidden activations to tensorboard for visualization
      for i in range(len(variables_names)):
        if '/lstm_cell' in (variables_names[i]):
          logger.log_histogram(variables_names[i], values[i], step)
         
      
      writer.add_summary(summary, step)
      print('Train perplexity at step %d: %f' % (step, t_perplexity)) #, valid_perplexity.eval()
      print('Validation perplexity at step %d: %f' % (step, v_perplexity))

      print 'train_input: ' + get_words(train_feed_inputs[0])
      print 'train_pred: ' + get_words(train_pred[0]) 
      print 'train_labels: ' + get_words(train_feed_labels[0])
          
      print 'valid_input: ' + get_words(valid_feed_inputs)
      print 'valid_pred: ' + get_words(valid_pred)
      print 'valid_labels: ' + get_words(valid_feed_labels) + '\n'
  


