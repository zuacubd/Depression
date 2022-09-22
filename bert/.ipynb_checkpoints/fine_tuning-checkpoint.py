import sys
import os
import collections
import csv
import argparse
import random
import re
import emoji
import pickle

import xml.etree.ElementTree as et
import numpy as np
import pandas as pd

import tensorflow as tf
import torch
import torch.nn as nn
import transformers

from datetime import datetime
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from load import *

from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class BERT_Arch(nn.Module):
	'''
		defining the architecture
	'''
	def __init__(self, bert):
		super(BERT_Arch, self).__init__()
		self.bert = bert 
		
		# dropout layer
		self.dropout = nn.Dropout(0.1)
		
		# relu activation function
		self.relu =  nn.ReLU()
		
		# dense layer 1
		self.fc1 = nn.Linear(768,512)
		
		# dense layer 2 (Output layer)
		self.fc2 = nn.Linear(512,2)
		
		#softmax activation function
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, sent_id, mask):
		'''
			define the forward pass
		'''
		#pass the inputs to the model  
		_, cls_hs = self.bert(sent_id, attention_mask=mask)
		
		x = self.fc1(cls_hs)
		
		x = self.relu(x)
		
		x = self.dropout(x)

		# output layer
		x = self.fc2(x)
		
		# apply softmax activation
		x = self.softmax(x)
		
		return x

	
def train():
	'''
		# function to train the model

	'''  
	model.train()
	
	total_loss, total_accuracy = 0, 0
	
	# empty list to save model predictions
	total_preds=[]
	
	# iterate over batches
	for step,batch in enumerate(train_dataloader):
		# progress update after every 50 batches.
		if step % 50 == 0 and not step == 0:
			print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
			
		# push the batch to gpu
		batch = [r.to(device) for r in batch]
		
		sent_id, mask, labels = batch
		
		# clear previously calculated gradients 
		model.zero_grad()        
		
		# get model predictions for the current batch
		preds = model(sent_id, mask)
		
		# compute the loss between actual and predicted values
		loss = cross_entropy(preds, labels)
		
		# add on to the total loss
		total_loss = total_loss + loss.item()
		
		# backward pass to calculate the gradients
		loss.backward()
		
		# clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		
		# update parameters
		optimizer.step()
		
		# model predictions are stored on GPU. So, push it to CPU
		preds=preds.detach().cpu().numpy()
		
		# append the model predictions
		total_preds.append(preds)
	
	# compute the training loss of the epoch
	avg_loss = total_loss / len(train_dataloader)
  
	# predictions are in the form of (no. of batches, size of batch, no. of classes).
	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)
	
	#returns the loss and predictions
	return avg_loss, total_preds


def evaluate():
	'''
		# function for evaluating the model
	'''
	print("\nEvaluating...")
	
	# deactivate dropout layers
	model.eval()
	
	total_loss, total_accuracy = 0, 0
	
	# empty list to save the model predictions
	total_preds = []
	
	# iterate over batches
	for step,batch in enumerate(val_dataloader):
		# Progress update every 50 batches.
		if step % 50 == 0 and not step == 0:
			# Calculate elapsed time in minutes.
			#elapsed = format_time(time.time() - t0)
			
			# Report progress.
			print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

		# push the batch to gpu
		batch = [t.to(device) for t in batch]

		sent_id, mask, labels = batch

		# deactivate autograd
		with torch.no_grad():
			# model predictions
			preds = model(sent_id, mask)

			# compute the validation loss between actual and predicted values
			loss = cross_entropy(preds,labels)
			
			total_loss = total_loss + loss.item()
			preds = preds.detach().cpu().numpy()
			total_preds.append(preds)
	# compute the validation loss of the epoch
	avg_loss = total_loss / len(val_dataloader) 
	
	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)
	return avg_loss, total_preds


def get_bert_tokenizer(model_name):
	#import bert pre-trained model
	bert = AutoModel.from_pretrained(model_name)
	#load the bert tokenizer
	tokenizer = BertTokenizerFast.from_pretrained(model_name)

	return bert, tokenizer


def freeze_bert_parameters():
	'''
		freeze all the parameters
	'''
	for param in bert.parameters():
		param.requires_grad = False
	
	
def initialize_model(bert_base):
	'''
		# pass the pre-trained BERT to our define architecture
	'''	
	model = BERT_Arch(bert_base)	
	return model


def get_Adamw_optimizer(learning_rate):
	''' 
		define the optimizer
	'''
	optimizer = AdamW(model.parameters(), lr = learning_rate) # learning rate
	return optimizer


def get_cross_entropy_loss():
	'''
		defining the loss function
	'''
	cross_entropy  = nn.NLLLoss()
	return cross_entropy


def get_cross_entropy_loss_weighted(weights):
	'''
		define the loss function
	'''
	cross_entropy = nn.NLLLoss(weight=weights)
	return cross_entropy


def get_tokenize_encoded_sequence(data_df, max_len, padding=True, truncation=True):
	'''
		tokenize and encode sequences in the training set
	'''
	tokens_encoded = tokenizer.batch_encode_plus(
		data_df.tolist(),
		max_length = max_len,
		padding=padding,
		truncation=truncation
	)
	return tokens_encoded


def get_list_to_tensors(data_list):
	'''
		convert lists to tensors
	'''
	data_seq = torch.tensor(data_list)
	return data_seq


def get_list_to_tensors_float(data_list):
	'''
		converting list of class weights to a tensor
	'''
	data_tensor= torch.tensor(data_list, dtype=torch.float)
	return data_tensor


def get_tensor_dataset(seq, mask, y):
	# wrap tensors
	tensor_data = TensorDataset(seq, mask, y)
	return tensor_data


def get_random_sampler(data):
	'''
		random sampler for sampling the data during training
	'''
	data_sampler = RandomSampler(data)
	return data_sampler


def get_sequential_sampler(data)
	'''
		sequential sampler for sampling the data during evaluation
	'''
	data_sampler = SequentialSampler(data)
	return data_sampler


def get_data_loader(data, data_sampler, bs):
	'''
		dataLoader for train set
	'''
	dataloader = DataLoader(data, sampler=data_sampler, batch_size=bs)
	return dataloader


def load_data(data_path):
	'''
		loading data with pandas
	'''
	df = pd.read_csv(data_path)
	return df
	

def get_train_test_split(df, rs, ts):
	''' 
		split train dataset into train, validation and test sets
	'''
	train_text, val_text, train_labels, val_labels = train_test_split(df['texts'], df['labels'], 
                                                                    random_state=42, 
                                                                    test_size=0.2, 
                                                                    stratify=df['labels'])
	return train_text, val_text, train_labels, val_labels
	

def compute_class_weights(labels):
	'''
		compute the class weights
	'''
	class_weights = compute_class_weight('balanced', np.unique(labels), labels)
	#print("Class Weights:",class_weights)
	return class_weights
	

if __name__ == '__main__':

    print ("Starting ...")
    year = int(sys.argv[1])
	
	device = torch.device("cuda") # push the model to GPU
	
	bert, tokenizer = get_bert_tokenizer('bert-base-uncased')
	freeze_bert_parameters() #freezing all the parameters in pre-trained model
	model = initialize_model(bert)
	model = model.to(device) # push the model to GPU
	optimizer = get_Adamw_optimizer(1e-5)

	#Setting the datapath
	processed_train_dir = "/projets/sig/mullah/nlp/depression/data/processed/"+str(year)+"/train/"
	processed_test_dir = "/projets/sig/mullah/nlp/depression/data/processed/"+str(year)+"/test/"
	models_dir = "/projets/sig/mullah/nlp/depression/models/"+str(year)+"/bert"

	data_path = os.path.join(processed_train_dir, "train_texts")
	data_df = load_data(data_path)
	train_text, val_text, train_labels, val_labels = get_train_test_split(data_df, 42, 0.2)

	print ("Total train: {}".format(len(train_text)))
	print (train_text.head())
	print ("Total validation: {}".format(len(val_text)))
	print (val_text.head())

	max_len = 128
	tokens_train = get_tokenize_encoded_sequence(train_text, max_len, True, True)
	tokens_val = get_tokenize_encoded_sequence(val_text, max_len, True, True)
	##tokens_test = get_tokenize_encoded_sequence(test_text, max_len, True, True)

	train_seq = get_list_to_tensors(tokens_train['input_ids'])
	train_mask = get_list_to_tensors(tokens_train['attention_mask'])
	train_y = get_list_to_tensors(train_labels.tolist())

	val_seq = get_list_to_tensors(tokens_val['input_ids'])
	val_mask = get_list_to_tensors(tokens_val['attention_mask'])
	val_y = get_list_to_tensors(val_labels.tolist())

	#test_seq = get_list_to_tensors(tokens_test['input_ids'])
	#test_mask = get_list_to_tensors(tokens_test['attention_mask'])
	#test_y = get_list_to_tensors(test_labels.tolist())

	batch_size = 32 #define a batch size
	train_data = get_tensor_dataset(train_seq, train_mask, train_y) # wrap tensors
	train_sampler = get_random_sampler(train_data) # sampler for sampling the data during training
	train_dataloader = get_data_loader(train_data, train_sampler, batch_size) # dataLoader for train set

	val_data = get_tensor_dataset(val_seq, val_mask, val_y) # wrap tensors
	val_sampler = get_sequential_sampler(val_data)
	val_dataloader = get_data_loader(val_data, val_sampler, batch_size)# dataLoader for validation set

	#class_weights = compute_class_weights(train_labels) ##compute the class weights
	#weights = get_list_to_tensors_float(class_weights) # converting list of class weights to a tensor
	#weights = weights.to(device) # push to GPU
	cross_entropy = get_cross_entropy_loss() # define the loss function

	########################################################
	epochs = 10 # number of training epochs
	best_valid_loss = float('inf') # set initial loss to infinite

	# empty lists to store training and validation loss of each epoch
	train_losses=[]
	valid_losses=[]

	for epoch in range(epochs):

		print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

		#train model
		train_loss, _ = train()

		#evaluate model
		valid_loss, _ = evaluate()

		#save the best model
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			model_path = os.path.join(models_dir, 'saved_weights.pt')
			torch.save(model.state_dict(), model_path)

		# append training and validation loss
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		print(f'\nTraining Loss: {train_loss:.3f}')
		print(f'Validation Loss: {valid_loss:.3f}')

	print(f'Best Validation Loss: {best_valid_loss:.3f}')