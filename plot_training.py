#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pdb

def plot_training(store_perplexities):
	'''A function that plots the training and validation perplexities 
	at each step.
	Input: list of str containing 'str(step) + '/' + str(t_perplexity) + '/' + str(v_perplexity)'
	'''

	steps = []
	train_perplexities = []
	valid_perplexities = []
	for item in store_perplexities:
		item = item.split('/')
		steps.append(int(item[0]))
		train_perplexities.append(float(item[1]))
		valid_perplexities.append(float(item[2]))



	plt.plot(steps, train_perplexities, 'r', steps, valid_perplexities, 'g')
	plt.show()

	pdb.set_trace()

	
