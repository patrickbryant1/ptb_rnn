#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pdb
import reader
import tensorflow as tf
from tensorflow import keras
import numpy as np



#Get data - no labels
data_path = '/Users/patbry/Documents/Tensorflow/RNN/simple-examples/data'
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, vocabulary = raw_data


#Parameters
num_unrollings = 20
seqlen = 20
vocab_size = 10000

hidden_size = 300
embedding_size = num_nodes # Dimension of the embedding vector. 1:1 ratio with hidden nodes
init_scale = 0.1
epoch_length = len(train_data)/(num_unrollings*batch_size)
keep_prob = 0.5


#Function for generating input
def generate(self):
    x = np.zeros((self.num_unrollings, self.seqlen))
    y = np.zeros((self.num_unrollings, self.seqlen, self.vocab_size))
    while True:
        for i in range(self.num_unrollings):
            if self.current_idx + self.seqlen >= len(self.data):
                # reset the index back to the start of the data set
                self.current_idx = 0
            x[i, :] = self.data[self.current_idx:self.current_idx + self.seqlen]
            temp_y = self.data[self.current_idx + 1:self.current_idx + self.seqlen + 1]
            # convert all of temp_y into a one hot representation
            y[i, :, :] = to_categorical(temp_y, num_classes=self.vocab_size)
            self.current_idx += self.skip_step
        yield x, y



train_data_generator = KerasBatchGenerator(train_data, seqlen, num_unrollings, vocab_size,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, seqlen, num_unrollings, vocab_size,
                                           skip_step=num_steps)

#Define model using keras
model = keras.models.Sequential() #Define model type, then add aspects using model.add
#Embedding lookup
model.add(keras.layers.Embedding(vocab_size, hidden_size, input_length=seqlen))

model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(keras.layers.Dropout(keep_prob))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(vocab_size)))
model.add(keras.layers.Activation('softmax'))


#Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']) #cross entropy applied in cases where there are many classes or categories, of which only one is true. 

#Save model using Keras callback
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

#Fit
model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

dummy_iters = 10
print("Training data:")
for i in range(dummy_iters):
    dummy = next(example_training_generator.generate())
num_predict = 10
true_print_out = "Actual words: "
pred_print_out = "Predicted words: "
for i in range(num_predict):
    data = next(example_training_generator.generate())
    prediction = model.predict(data[0])
    predict_word = np.argmax(prediction[:, num_steps-1, :])
    true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
    pred_print_out += reversed_dictionary[predict_word] + " "
print(true_print_out)
print(pred_print_out)


